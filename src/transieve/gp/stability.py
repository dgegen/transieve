from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fit import GPFamily, SHOGPFamily
from .match import MatchedFilter
from ..lightcurve import LightCurve
from ..utils import as_1d_array

__all__ = ["GPStabilityMap", "scan_gp_stability_map"]


@dataclass
class GPStabilityMap:
    """2D retrievability map over GP hyperparameters.

    The first axis is sigma and the second axis is timescale
    (period for SHO or scale for exponential kernel).
    """

    sigma_grid: np.ndarray
    timescale_grid: np.ndarray
    timescale_name: str
    z_score: np.ndarray
    z_white_noise: np.ndarray
    recovery_fraction: np.ndarray
    relative_capacity: np.ndarray
    local_drop: np.ndarray
    reference_gp_params: dict[str, float]
    log_marginal_likelihood: np.ndarray | None = None
    reference_gp_theta_dict: dict[str, float] | None = None
    # Per-grid-point null-distribution summary statistics (populated by
    # scan_gp_stability_map via z_score_map_fft).
    z_median: np.ndarray | None = None
    z_mad: np.ndarray | None = None
    z_mean: np.ndarray | None = None
    z_std: np.ndarray | None = None
    p_value: np.ndarray | None = None

    def get_plateau_mask(
        self,
        z_threshold: float,
        max_fragility: float = 0.1,
    ) -> np.ndarray:
        """Return a boolean mask of stable, above-threshold grid cells.

        A cell qualifies when its z-score meets ``z_threshold`` and the
        relative drop to its worst 4-neighbor is at most ``max_fragility``.
        """
        return (self.z_score >= z_threshold) & (self.local_drop <= max_fragility)

    def strongest_point(self) -> dict[str, float]:
        idx = np.unravel_index(np.nanargmax(self.z_score), self.z_score.shape)
        i_sigma, i_timescale = int(idx[0]), int(idx[1])
        return {
            "sigma": float(self.sigma_grid[i_sigma]),
            self.timescale_name: float(self.timescale_grid[i_timescale]),
            "z_score": float(self.z_score[i_sigma, i_timescale]),
            "recovery_fraction": float(self.recovery_fraction[i_sigma, i_timescale]),
            "relative_capacity": float(self.relative_capacity[i_sigma, i_timescale]),
        }

    def summary(
        self,
        z_threshold: float,
        max_fragility: float = 0.1,
    ) -> dict[str, float]:
        plateau_mask = self.get_plateau_mask(z_threshold, max_fragility)
        plateau_points = int(np.sum(plateau_mask))
        total_points = int(plateau_mask.size)
        return {
            "plateau_points": plateau_points,
            "plateau_fraction": float(plateau_points / max(total_points, 1)),
            "log_space_plateau_area": self._log_space_plateau_area(plateau_mask),
            "max_z_score": float(np.nanmax(self.z_score)),
            "median_recovery_fraction": float(np.nanmedian(self.recovery_fraction)),
        }

    def robust_features(
        self,
        z_threshold: float = 5.0,
        max_fragility: float = 0.1,
    ) -> dict[str, float]:
        """Return grid-invariant scalar features from the stability map.

        Unlike ``summary()``, these metrics do not depend on the choice of grid
        bounds or resolution.

        Parameters
        ----------
        z_threshold:
            Minimum z-score for a cell to belong to the plateau.
        max_fragility:
            Maximum ``local_drop`` for a cell to be considered stable.

        Returns
        -------
        dict with keys:

        - ``log_space_plateau_area``: sum of log-space cell areas within the
          plateau mask — a grid-invariant replacement for ``plateau_fraction``.
        - ``capacity_bounded_peak_z``: maximum raw z-score within the
          physically valid region (log_marginal_likelihood within 10 of its
          maximum, recovery_fraction > 0.8, and 0.8 < relative_capacity <
          1.2).  The likelihood gate excludes corners where the GP noise model
          has broken down; the remaining constraints are hard domain priors.
          Retains standard-normal properties because no multiplicative
          weighting is applied.
        - ``max_z_plateau``: maximum z-score within the plateau mask.
        - ``peak_brittleness``: ``local_drop`` at the global z-score maximum.
        - ``is_safe_harbor``: 1.0 if the plateau is non-empty and
          ``peak_brittleness <= max_fragility``, else 0.0.
        """
        mask = self.get_plateau_mask(z_threshold, max_fragility)

        log_plateau_area = self._log_space_plateau_area(mask)

        log_L = self.log_marginal_likelihood
        likelihood_ok = np.isfinite(log_L) & (log_L >= np.nanmax(log_L) - 10.0)
        valid = (
            likelihood_ok
            & (self.recovery_fraction > 0.8)
            & (self.relative_capacity > 0.8)
            & (self.relative_capacity < 1.2)
        )
        capacity_bounded_peak_z = (
            float(np.nanmax(self.z_score[valid])) if np.any(valid) else 0.0
        )

        masked_z = np.where(mask, self.z_score, np.nan)
        max_z_plateau = float(np.nanmax(masked_z)) if np.any(mask) else 0.0

        idx_peak = np.unravel_index(np.nanargmax(self.z_score), self.z_score.shape)
        peak_brittleness = float(self.local_drop[idx_peak])

        is_safe_harbor = (
            1.0 if (np.any(mask) and peak_brittleness <= max_fragility) else 0.0
        )

        return {
            "log_space_plateau_area": log_plateau_area,
            "capacity_bounded_peak_z": capacity_bounded_peak_z,
            "max_z_plateau": max_z_plateau,
            "peak_brittleness": peak_brittleness,
            "is_safe_harbor": is_safe_harbor,
        }

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _marginalise(
        self,
        grid: np.ndarray,
        delta_log_L: float = np.inf,
    ) -> float:
        """Posterior-weighted marginalisation of an arbitrary 2D grid.

        Shares the same weight computation as :meth:`marginalized_z`:
        log-marginal-likelihood weights multiplied by log-space Voronoi cell
        areas, with a relative likelihood gate of ``delta_log_L``.

        Parameters
        ----------
        grid:
            2D array of the same shape as ``z_score``.
        delta_log_L:
            Relative likelihood gate; cells more than ``delta_log_L`` below the
            maximum are excluded.
        """
        if self.log_marginal_likelihood is None:
            raise ValueError(
                "log_marginal_likelihood is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        log_L = self.log_marginal_likelihood
        finite_mask = np.isfinite(log_L) & np.isfinite(grid)
        if not np.any(finite_mask):
            return float(np.nanmax(grid))
        log_L_stable = np.where(
            finite_mask, log_L - np.nanmax(log_L[finite_mask]), -np.inf
        )
        valid = finite_mask & (log_L_stable >= -delta_log_L)
        if not np.any(valid):
            return float(np.nanmax(grid))
        w = np.exp(np.where(valid, log_L_stable, -np.inf))
        cell_areas = _compute_log_space_cell_areas(self.sigma_grid, self.timescale_grid)
        combined = w * cell_areas
        total = np.sum(combined[valid])
        if total <= 0.0:
            return float(np.nanmax(grid))
        return float(np.sum(grid[valid] * combined[valid]) / total)

    # ------------------------------------------------------------------
    # Marginalised scalar summaries
    # ------------------------------------------------------------------

    def marginalized_z(self, delta_log_L: float = np.inf) -> float:
        """Posterior-weighted Z-score marginalised over GP hyperparameters.

        Computes

        .. math::

            Z_{\\text{marg}} = \\frac{\\sum_{ij} Z_{ij}\\, w_{ij}\\, A_{ij}}
                                     {\\sum_{ij} w_{ij}\\, A_{ij}}

        where :math:`w_{ij} = \\exp(\\log L_{ij} - \\max \\log L)` are the
        (unnormalised) posterior weights from the GP marginal likelihood and
        :math:`A_{ij}` are the log-space cell areas (Voronoi assignment so
        that the result is grid-invariant).

        A relative likelihood weight gate is applied: cells more than
        ``delta_log_L`` below the maximum log-likelihood are assigned weight
        zero, preventing numerical failures in large broken grid regions from
        leaking into the integral.

        Requires that the map was built with ``scan_gp_stability_map``, which
        now stores ``log_marginal_likelihood``.

        Parameters
        ----------
        delta_log_L:
            Gate threshold in log-likelihood units relative to the maximum.
            Cells with ``log_L < max(log_L) - delta_log_L`` are excluded.
            Default 16.0 (≈ 4σ equivalent confidence boundary).

        Raises
        ------
        ValueError
            If ``log_marginal_likelihood`` is ``None`` (map built before this
            feature was added).
        """
        return self._marginalise(self.z_score, delta_log_L)

    def marginalized_median_z(self, delta_log_L: float = np.inf) -> float:
        """Posterior-weighted marginalisation of the per-grid-point Z-score median."""
        if self.z_median is None:
            raise ValueError(
                "z_median is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        return self._marginalise(self.z_median, delta_log_L)

    def marginalized_mad_z(self, delta_log_L: float = np.inf) -> float:
        """Posterior-weighted marginalisation of the per-grid-point Z-score MAD."""
        if self.z_mad is None:
            raise ValueError(
                "z_mad is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        return self._marginalise(self.z_mad, delta_log_L)

    def marginalized_mean_z(self, delta_log_L: float = np.inf) -> float:
        """Posterior-weighted marginalisation of the per-grid-point Z-score mean."""
        if self.z_mean is None:
            raise ValueError(
                "z_mean is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        return self._marginalise(self.z_mean, delta_log_L)

    def marginalized_std_z(self, delta_log_L: float = np.inf) -> float:
        """Posterior-weighted marginalisation of the per-grid-point Z-score std."""
        if self.z_std is None:
            raise ValueError(
                "z_std is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        return self._marginalise(self.z_std, delta_log_L)

    def marginalized_p_value(self, delta_log_L: float = np.inf) -> float:
        """Posterior-weighted marginalisation of the per-grid-point empirical p-value.

        Each grid-point p-value is the fraction of the FFT Z-score time series
        that equals or exceeds the exact peak Z-score at the target epoch.
        """
        if self.p_value is None:
            raise ValueError(
                "p_value is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        return self._marginalise(self.p_value, delta_log_L)

    def marginalized_empirical_z(self, delta_log_L: float = np.inf) -> float:
        """Empirical Z-score derived from the marginalized peak, mean, and std.

        Computes

        .. math::

            Z_{\\text{emp}} = \\frac{Z_{\\text{peak,marg}} - Z_{\\text{mean,marg}}}
                                    {Z_{\\text{std,marg}}}

        where each quantity is independently marginalized over the GP
        hyperparameter grid using the posterior likelihood weights.
        """
        peak = self.marginalized_z(delta_log_L)
        mean = self.marginalized_mean_z(delta_log_L)
        std = self.marginalized_std_z(delta_log_L)
        return (peak - mean) / (std if std > 0.0 else 1.0)

    def posterior_z_integrand(self, delta_log_L: float = np.inf) -> np.ndarray:
        """Normalised posterior integrand: the per-cell contribution to marginalized_z.

        Returns a 2D array of the same shape as ``z_score`` where each entry is

        .. math::

            I_{ij} = Z_{ij} \\cdot \\frac{w_{ij}\\, A_{ij}}{\\sum_{kl} w_{kl}\\, A_{kl}}

        Cells excluded by the relative likelihood weight gate (more than
        ``delta_log_L`` below the maximum) are set to NaN.
        The array integrates to ``marginalized_z(delta_log_L)``.

        Parameters
        ----------
        delta_log_L:
            Gate threshold passed through to ``marginalized_z``. Default 16.0.
        """
        if self.log_marginal_likelihood is None:
            raise ValueError(
                "log_marginal_likelihood is not stored on this GPStabilityMap."
            )
        log_L = self.log_marginal_likelihood
        finite_mask = np.isfinite(log_L) & np.isfinite(self.z_score)
        log_L_stable = np.where(
            finite_mask, log_L - np.nanmax(log_L[finite_mask]), -np.inf
        )
        valid = finite_mask & (log_L_stable >= -delta_log_L)
        w = np.exp(np.where(valid, log_L_stable, -np.inf))
        cell_areas = _compute_log_space_cell_areas(self.sigma_grid, self.timescale_grid)
        combined = w * cell_areas
        total = np.sum(combined[valid])
        integrand = np.where(valid, self.z_score * combined / total, np.nan)
        return integrand

    def peak_integrand_theta_dict(self) -> dict[str, float]:
        """Return the log-space GP parameter dict for the cell with the highest posterior integrand.

        Overrides ``log_sigma`` and the timescale key (``log_omega`` or ``log_scale``) in the
        reference theta dict with the values at the argmax of ``posterior_z_integrand()``.
        All other hyperparameters (quality, jitter, …) are kept at the reference MLE values.
        """
        if self.reference_gp_theta_dict is None:
            raise ValueError(
                "reference_gp_theta_dict is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        integrand = self.posterior_z_integrand()
        finite = np.where(np.isfinite(integrand), integrand, -np.inf)
        i, j = np.unravel_index(int(np.argmax(finite)), integrand.shape)
        params = dict(self.reference_gp_theta_dict)
        params["log_sigma"] = float(np.log(self.sigma_grid[i]))
        ts = float(self.timescale_grid[j])
        if "log_omega" in params:
            params["log_omega"] = float(np.log(2 * np.pi / ts))
        elif "log_scale" in params:
            params["log_scale"] = float(np.log(ts))
        return params

    def _log_space_plateau_area(self, plateau_mask: np.ndarray) -> float:
        """Sum of log-space cell areas within ``plateau_mask``."""
        d_log_sigma = np.diff(np.log(self.sigma_grid))
        d_log_tau = np.diff(np.log(self.timescale_grid))
        if d_log_sigma.size == 0 or d_log_tau.size == 0:
            return 0.0
        d_sigma_mesh, d_tau_mesh = np.meshgrid(d_log_sigma, d_log_tau, indexing="ij")
        cell_areas = d_sigma_mesh * d_tau_mesh
        return float(np.sum(cell_areas[plateau_mask[:-1, :-1]]))


def _compute_log_space_cell_areas(
    sigma_grid: np.ndarray,
    timescale_grid: np.ndarray,
) -> np.ndarray:
    """Return per-grid-point log-space cell areas with the same shape as the 2D maps.

    Each grid point is assigned the area of its Voronoi cell in
    (log sigma, log tau) space — the midpoints between neighbouring grid
    points define the cell boundaries.  Edge points use the nearest spacing
    as a one-sided extension so that the total area sums to the full
    log-space bounding box.
    """

    def _voronoi_widths(log_grid: np.ndarray) -> np.ndarray:
        if len(log_grid) == 1:
            return np.ones(1)
        mid = (log_grid[:-1] + log_grid[1:]) / 2.0
        left = np.concatenate([[log_grid[0] - (log_grid[1] - log_grid[0]) / 2.0], mid])
        right = np.concatenate(
            [mid, [log_grid[-1] + (log_grid[-1] - log_grid[-2]) / 2.0]]
        )
        return right - left

    w_sigma = _voronoi_widths(np.log(sigma_grid))
    w_tau = _voronoi_widths(np.log(timescale_grid))
    w_s_mesh, w_t_mesh = np.meshgrid(w_sigma, w_tau, indexing="ij")
    return w_s_mesh * w_t_mesh


def _infer_timescale_from_theta(
    gp_family: GPFamily,
    theta_dict: dict[str, float],
) -> tuple[str, float]:
    if "log_omega" in theta_dict:
        return "period", 2 * np.pi / np.exp(theta_dict["log_omega"])
    if "log_scale" in theta_dict:
        return "scale", np.exp(theta_dict["log_scale"])
    raise ValueError("Cannot infer timescale parameter from GP theta names.")


def _timescale_to_theta(
    gp_family: GPFamily,
    params: dict[str, float],
    timescale: float,
) -> str:
    if "log_omega" in gp_family.theta_names:
        params["log_omega"] = np.log(2 * np.pi / timescale)
        return "period"
    if "log_scale" in gp_family.theta_names:
        params["log_scale"] = np.log(timescale)
        return "scale"
    raise ValueError("GP family must expose log_omega or log_scale for stability map.")


def _compute_local_drop(z_values: np.ndarray) -> np.ndarray:
    """Compute local fragility proxy from neighboring relative Z-score drops."""
    padded = np.pad(z_values, 1, constant_values=np.nan)
    neighbors = np.stack(
        [
            padded[:-2, 1:-1],  # up
            padded[2:, 1:-1],  # down
            padded[1:-1, :-2],  # left
            padded[1:-1, 2:],  # right
        ]
    )
    worst_neighbor = np.nanmin(neighbors, axis=0)

    valid = np.isfinite(z_values) & (z_values > 0) & np.isfinite(worst_neighbor)
    local_drop = np.full_like(z_values, np.nan, dtype=float)
    local_drop[valid] = np.clip(
        (z_values[valid] - worst_neighbor[valid]) / z_values[valid], 0.0, np.inf
    )
    return local_drop


def scan_gp_stability_map(
    time: np.ndarray,
    flux: np.ndarray,
    template: np.ndarray,
    flux_err: np.ndarray | None = None,
    gp_family: GPFamily | None = None,
    sigma_grid: np.ndarray | None = None,
    timescale_grid: np.ndarray | None = None,
    fit_mean: float = 0.0,
    fit_method: str = "differential_evolution",
    fit_max_retries: int = 2,
    fit_kwargs: dict[str, Any] | None = None,
    center_flux: bool = True,
    reference_time: np.ndarray | None = None,
    reference_flux: np.ndarray | None = None,
    reference_flux_err: np.ndarray | None = None,
) -> GPStabilityMap:
    """Scan retrievability over a 2D GP hyperparameter grid.

    The map is evaluated for one fixed template and stores the strongest
    matched-filter response at each grid point.

    When ``reference_time/flux/flux_err`` are provided, the initial GP fit
    (used to derive grid boundaries and ``reference_gp_theta_dict``) and the
    ``log_marginal_likelihood`` grid are computed on the reference data rather
    than on ``time/flux``.  This lets the weights used for marginalisation
    reflect the noise properties of a clean, uninjected baseline light curve
    while the z-scores are still evaluated on the injection sequence.
    """

    template = as_1d_array(template)
    light_curve = LightCurve.from_arrays(
        time=time,
        flux=flux,
        flux_err=flux_err,
        mean=fit_mean,
    )
    time = light_curve.time
    flux = light_curve.flux
    flux_err = light_curve.flux_err

    if len(template) != len(time):
        raise ValueError("time, flux, and template must have the same length.")

    gp_family = SHOGPFamily() if gp_family is None else gp_family

    use_reference = reference_time is not None and reference_flux is not None
    if use_reference:
        ref_lc = LightCurve.from_arrays(
            time=reference_time,
            flux=reference_flux,
            flux_err=reference_flux_err,
            mean=fit_mean,
        )
        ref_time = ref_lc.time
        ref_flux = ref_lc.flux
        ref_flux_err = ref_lc.flux_err
        gp_family.validate_white_noise_baseline(ref_flux_err)
        fit_lc = ref_lc
    else:
        ref_time = ref_flux = ref_flux_err = None
        gp_family.validate_white_noise_baseline(flux_err)
        fit_lc = light_curve

    opt_result = gp_family.fit_light_curve(
        fit_lc,
        method=fit_method,
        max_retries=fit_max_retries,
        **({} if fit_kwargs is None else dict(fit_kwargs)),
    )

    theta_dict = gp_family.theta_to_dict(opt_result.x)
    timescale_name, timescale_ref = _infer_timescale_from_theta(gp_family, theta_dict)
    sigma_ref = np.exp(theta_dict["log_sigma"])

    if sigma_grid is None:
        sigma_upper = float(np.nanstd(ref_flux if use_reference else flux))
        sigma_upper = max(sigma_upper, sigma_ref)
        sigma_lower = max(sigma_ref / 6.0, sigma_upper / 50.0, 1e-7)
        sigma_grid = np.geomspace(sigma_lower, sigma_upper, 16)
    else:
        sigma_grid = as_1d_array(sigma_grid)

    if timescale_grid is None:
        lower = max(timescale_ref / 3.0, 0.02)
        upper = timescale_ref * 3.0
        timescale_grid = np.geomspace(lower, upper, 16)
    else:
        timescale_grid = as_1d_array(timescale_grid)

    centered = light_curve.centered_flux(center=center_flux)

    z_map = np.full((len(sigma_grid), len(timescale_grid)), np.nan, dtype=float)
    z_white_noise_map = np.full_like(z_map, np.nan)
    recovery_fraction_map = np.full_like(z_map, np.nan)
    relative_capacity_map = np.full_like(z_map, np.nan)
    log_marginal_likelihood_map = np.full_like(z_map, np.nan)
    # Null-distribution summary statistics (populated via z_score_map_fft)
    z_median_map = np.full_like(z_map, np.nan)
    z_mad_map = np.full_like(z_map, np.nan)
    z_mean_map = np.full_like(z_map, np.nan)
    z_std_map = np.full_like(z_map, np.nan)
    p_value_map = np.full_like(z_map, np.nan)

    for i, sigma in enumerate(sigma_grid):
        for j, ts in enumerate(timescale_grid):
            params = dict(theta_dict)
            params["log_sigma"] = np.log(float(sigma))
            _timescale_to_theta(gp_family, params, float(ts))

            gp = gp_family.build(params, time=time, flux_err=flux_err, mean=fit_mean)
            mf = MatchedFilter(gp=gp, flux=centered, check_zero_centered=True)

            m = mf.template_metrics(template)
            z_map[i, j] = m["z_score"]
            z_white_noise_map[i, j] = m["z_white_noise"]
            recovery_fraction_map[i, j] = m["recovery_fraction"]
            relative_capacity_map[i, j] = m["relative_capacity"]

            # Full Z-score time series via FFT convolution — used to characterise
            # the empirical null distribution for this GP configuration.
            mf_stats = mf.z_score_map_fft(template)
            z_array = mf_stats.z_score
            median_z = float(np.nanmedian(z_array))
            z_median_map[i, j] = median_z
            z_mad_map[i, j] = float(np.nanmedian(np.abs(z_array - median_z)))
            z_mean_map[i, j] = float(np.nanmean(z_array))
            z_std_map[i, j] = float(np.nanstd(z_array))
            # Empirical p-value: fraction of the light curve where the FFT
            # Z-score equals or exceeds the exact peak at the target epoch.
            peak_z = z_map[i, j]
            n_finite = int(np.sum(np.isfinite(z_array)))
            p_value_map[i, j] = (
                float(np.nansum(z_array >= peak_z) / n_finite)
                if n_finite > 0
                else np.nan
            )
            if use_reference:
                gp_ref = gp_family.build(
                    params, time=ref_time, flux_err=ref_flux_err, mean=fit_mean
                )
                log_marginal_likelihood_map[i, j] = gp_ref.log_likelihood(ref_flux)
            else:
                log_marginal_likelihood_map[i, j] = gp.log_likelihood(flux)

            # if np.isfinite(log_marginal_likelihood_map[i, j] and):
            #     print(f"DEBUG CELL (ts={ts:.2}, sigma={sigma:.2e}):")
            #     print(f"  Local Z at epoch: {z_map[i, j]}")
            #     print(
            #         f"  Log Marginal Likelihood: {log_marginal_likelihood_map[i, j]}"
            #     )

    return GPStabilityMap(
        sigma_grid=np.asarray(sigma_grid, dtype=float),
        timescale_grid=np.asarray(timescale_grid, dtype=float),
        timescale_name=timescale_name,
        z_score=z_map,
        z_white_noise=z_white_noise_map,
        recovery_fraction=recovery_fraction_map,
        relative_capacity=relative_capacity_map,
        local_drop=_compute_local_drop(z_map),
        reference_gp_params=gp_family.theta_to_physical(opt_result.x),
        log_marginal_likelihood=log_marginal_likelihood_map,
        reference_gp_theta_dict=theta_dict,
        z_median=z_median_map,
        z_mad=z_mad_map,
        z_mean=z_mean_map,
        z_std=z_std_map,
        p_value=p_value_map,
    )
