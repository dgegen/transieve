from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

from .fit import GPFamily, SHOGPFamily
from .match import MatchedFilter
from ..lightcurve import LightCurve
from ..utils import as_1d_array

__all__ = ["GPStabilityMap", "scan_gp_stability_map"]


@dataclass
class GPStabilityMap:
    """2D scan over GP hyperparameters evaluating template significance and evidence.

    The first axis maps out the process amplitude amplitude (sigma) and the second axis
    tracks the process timescale (period for SHO or scale for exponential kernels).
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
    log_bf_conditional: np.ndarray | None = None
    log_bf_global: np.ndarray | None = None
    template_bank_size: int | None = None

    # Hyperspace tensors populated during Global template matrix searches
    z_score_tensor: np.ndarray | None = None  # (S, T, M, t_transit)
    log_bf_tensor: np.ndarray | None = None  # (S, T, M, t_transit)

    # Sensitivity limit boundaries
    ideal_sensitivity: np.ndarray | None = None
    realization_adjusted_sensitivity: np.ndarray | None = None

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

    # ------------------------------------------------------------------
    # Multi-dimensional reduction properties
    # ------------------------------------------------------------------

    @property
    def profiled_z_score_map(self) -> np.ndarray:
        """(S, T, t_transit): frequentist profile over the duration axis (nanmax over M)."""
        if self.z_score_tensor is None:
            raise ValueError(
                "z_score_tensor is not available. Use context='global' with a 3-D "
                "template_matrix of shape (N, M, t_transit)."
            )
        return np.nanmax(self.z_score_tensor, axis=2)

    @property
    def marginalized_log_bf_map(self) -> np.ndarray:
        """(S, T, t_transit): Bayesian marginalization over duration (logsumexp over M, uniform prior)."""
        if self.log_bf_tensor is None:
            raise ValueError(
                "log_bf_tensor is not available. Use context='global' with "
                "operational_mode='inference' and a 3-D template_matrix of shape "
                "(N, M, t_transit)."
            )
        M = self.log_bf_tensor.shape[2]
        return logsumexp(self.log_bf_tensor, axis=2) - np.log(M)

    # ------------------------------------------------------------------
    # Profiled Snapshot Metrics
    # ------------------------------------------------------------------

    def profiled_max_z(self) -> float:
        """Z-score evaluated at the Maximum Likelihood Estimate (MLE) of the GP hyperparameters.

        Extracts the Z-score from the grid cell where log_marginal_likelihood is maximized.
        """
        if self.log_marginal_likelihood is None:
            raise ValueError(
                "log_marginal_likelihood is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )
        idx = np.nanargmax(self.log_marginal_likelihood)
        return float(self.z_score.flat[idx])

    def profiled_mle_metrics(self) -> dict[str, float]:
        """Profile all 2D grid metrics at the Maximum Likelihood Estimate (MLE).

        Extracts the scalar value of every hyperparameter grid array at the exact
        coordinate cell where `log_marginal_likelihood` is maximized.
        """
        if self.log_marginal_likelihood is None:
            raise ValueError(
                "log_marginal_likelihood is not stored on this GPStabilityMap. "
                "Recompute the map with scan_gp_stability_map to include it."
            )

        idx = np.nanargmax(self.log_marginal_likelihood)

        # Gather all core 2D grid attributes that match the grid shape
        grid_shape = self.z_score.shape
        metrics = {}

        target_attributes = [
            "z_score",
            "z_white_noise",
            "recovery_fraction",
            "relative_capacity",
            "local_drop",
            "log_bf_conditional",
            "log_bf_global",
        ]

        for attr_name in target_attributes:
            attr = getattr(self, attr_name, None)
            if (
                attr is not None
                and isinstance(attr, np.ndarray)
                and attr.shape == grid_shape
            ):
                metrics[attr_name] = float(attr.flat[idx])

        return metrics

    # ------------------------------------------------------------------
    # Marginalised scalar summaries
    # ------------------------------------------------------------------

    def _marginalise_bayes_factor(
        self,
        log_bf_input: np.ndarray,
        delta_log_L: float,
        axis: int | tuple[int, ...] = (0, 1),
    ) -> float | np.ndarray:
        """Grid-integrated log Bayes factor using posterior likelihood cell volumes.

        Supports both 2D scalar integration and 3D tensor-timeline integration
        by broadcasting log_marginal_likelihood over the remaining axes.
        """
        if self.log_marginal_likelihood is None:
            raise ValueError("log_marginal_likelihood is not stored on this map.")

        log_L0 = self.log_marginal_likelihood  # Shape: (S, T)

        # If the input has a time dimension (S, T, t_transit), we expand log_L0
        # to (S, T, 1) so it broadcasts natively across time.
        if log_bf_input.ndim == 3:
            log_L0_expanded = log_L0[..., np.newaxis]
            log_area = np.log(
                _compute_log_space_cell_areas(self.sigma_grid, self.timescale_grid)
            )[..., np.newaxis]
        else:
            log_L0_expanded = log_L0
            log_area = np.log(
                _compute_log_space_cell_areas(self.sigma_grid, self.timescale_grid)
            )

        log_L1 = log_L0_expanded + log_bf_input
        finite_mask = np.isfinite(log_L0_expanded) & np.isfinite(log_L1)

        if not np.any(finite_mask):
            return (
                np.full(log_bf_input.shape[2:], np.nan)
                if log_bf_input.ndim == 3
                else float("nan")
            )

        # Stability Gate: Use the 2D baseline max to mask bad pixels globally
        max_log_L0 = np.nanmax(log_L0)
        log_L0_stable = np.where(finite_mask, log_L0_expanded - max_log_L0, -np.inf)
        valid = finite_mask & (log_L0_stable >= -delta_log_L)

        if not np.any(valid):
            return (
                np.full(log_bf_input.shape[2:], np.nan)
                if log_bf_input.ndim == 3
                else float("nan")
            )

        # Build integration bounds
        log_L0_integrand = np.where(valid, log_L0_expanded + log_area, -np.inf)
        log_L1_integrand = np.where(valid, log_L1 + log_area, -np.inf)

        # Marginalize over specified axes (Default: (0, 1) to eliminate S and T)
        denom = logsumexp(log_L0_integrand, axis=axis)
        numer = logsumexp(log_L1_integrand, axis=axis)

        return numer - denom

    def marginalized_conditional_log_bf(self, delta_log_L: float = np.inf) -> float:
        r"""Posterior-weighted conditional Bayes factor marginalised over GP hyperparameters.

        Computes

        .. math::

            \ln B_{10}^{\text{cond}} = \log\sum \exp(\ln L_1 + \ln A)
            - \log\sum \exp(\ln L_0 + \ln A)

        where :math:`\ln L_1 = \ln L_0 + \ln B_{10}^{\text{cond}}` and :math:`\ln A` are
        log-space Voronoi cell areas acting as the prior over the grid.
        """
        if self.log_bf_conditional is None:
            raise ValueError("log_bf_conditional is not stored on this GPStabilityMap.")
        return float(
            self._marginalise_bayes_factor(self.log_bf_conditional, delta_log_L)
        )

    def marginalized_bayes_factor(self, delta_log_L: float = np.inf) -> float:
        r"""Posterior-weighted Bayes factor marginalised over GP hyperparameters.

        Computes

        .. math::

            \ln B_{10}^{\text{global}} = \log\sum \exp(\ln L_1 + \ln A)
            - \log\sum \exp(\ln L_0 + \ln A)

        where :math:`\ln L_1 = \ln L_0 + \ln B_{10}` and :math:`\ln A` are
        log-space Voronoi cell areas acting as the prior over the grid.
        """
        if self.log_bf_global is None:
            raise ValueError("log_bf_global is not stored on this GPStabilityMap.")
        return float(self._marginalise_bayes_factor(self.log_bf_global, delta_log_L))

    def marginalized_log_bf_timeline(self, delta_log_L: float = 16.0) -> np.ndarray:
        """(t_transit,): Integrated log Bayes factor timeline marginalized over
        all GP noise parameters and template duration variants.
        """
        if self.log_bf_tensor is None:
            raise ValueError("log_bf_tensor is not available.")

        M = self.log_bf_tensor.shape[2]
        log_bf_st = logsumexp(self.log_bf_tensor, axis=2) - np.log(M)

        return self._marginalise_bayes_factor(
            log_bf_st, delta_log_L=delta_log_L, axis=(0, 1)
        )  # type: ignore


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


def _logsumexp_with_weights(
    log_values: np.ndarray,
    log_weights: np.ndarray | None,
) -> float:
    if log_weights is None:
        return float(logsumexp(log_values) - np.log(log_values.size))
    log_weights = np.asarray(log_weights)
    if log_weights.shape != log_values.shape:
        raise ValueError("log_weights must have the same shape as log_values.")
    return float(logsumexp(log_values + log_weights) - logsumexp(log_weights))


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
    template: np.ndarray | None = None,
    flux_err: np.ndarray | None = None,
    gp_family: GPFamily | None = None,
    sigma_grid: np.ndarray | None = None,
    timescale_grid: np.ndarray | None = None,
    fit_mean: float = 0.0,
    fit_method: str = "differential_evolution",
    fit_max_retries: int = 2,
    fit_kwargs: dict[str, Any] | None = None,
    theta_dict: dict[str, Any] | None = None,
    center_flux: bool = True,
    reference_time: np.ndarray | None = None,
    reference_flux: np.ndarray | None = None,
    reference_flux_err: np.ndarray | None = None,
    context: Literal["local", "global"] = "local",
    template_matrix: np.ndarray | None = None,
    sigma_a: float | None = 1,
    prior: str = "gaussian",
    amplitude_bounds: tuple[float, float] | None = None,
    log_template_weights: np.ndarray | None = None,
    operational_mode: Literal["inference", "sensitivity"] = "inference",
    z_threshold: float | None = None,
) -> GPStabilityMap:
    """Scan retrievability over a 2D GP hyperparameter grid.

    Two orthogonal axes control what is computed at each grid point:

    **Search Context** (``context``):
        ``"local"`` evaluates matched-filter statistics for a single fixed
        ``template`` at each GP grid point (recoverability / characterization).
        ``"global"`` evaluates statistics for a pre-materialized
        ``template_matrix`` of shape ``(N, M)`` or ``(N, M_dur, t_transit)``
        and profiles or marginalises over templates (plausibility / completeness).

    **Operational Mode** (``operational_mode``):
        ``"inference"`` projects the observed data vector ``y`` against
        covariance-weighted templates to quantify the prominence of a candidate.
        ``"sensitivity"`` extracts the deterministic template energy profile
        (``sigma_s = sqrt(s^T C^{-1} s)``) to map minimum detectable depths,
        isolating the noise manifold from the current data realization.

    The four cells of this 2×2 design:

    +-----------------------+------------------------------+-------------------------------+
    |                       | Local Context (fixed t0)     | Global Context (free t0 ∈ T)  |
    +=======================+==============================+===============================+
    | **Inference Mode**    | Conditional Z-score and BF   | Look-elsewhere max Z-score    |
    | *(uses y)*            | at the fixed (s, t0)         | and marginalized Bayes factor |
    +-----------------------+------------------------------+-------------------------------+
    | **Sensitivity Mode**  | Minimum detectable depth at  | Completeness map: min depth   |
    | *(uses C^{-1} only)*  | this exact (s, t0)           | across all (s, t0) windows    |
    +-----------------------+------------------------------+-------------------------------+

    Parameters
    ----------
    context : "local" or "global"
        Search context controlling the scope of the template evaluation.
    template : ndarray, required when context="local"
        Single fixed transit template of length N (cadences).
    template_matrix : ndarray, required when context="global"
        Pre-materialized template array of shape ``(N, M)`` or
        ``(N, M_dur, t_transit)``.  When 3-D, ``z_score_tensor`` and
        ``log_bf_tensor`` of shape ``(S, T, M_dur, t_transit)`` are populated;
        use the ``profiled_z_score_map`` and ``marginalized_log_bf_map`` properties to
        reduce the duration axis.
    operational_mode : "inference" or "sensitivity"
        Operational mode controlling what question is answered at each grid point.
    z_threshold : float, required when operational_mode="sensitivity"
        Detection significance threshold used to derive the minimum detectable
        transit depth.  Populates ``ideal_sensitivity`` and
        ``realization_adjusted_sensitivity`` on the returned map.
    sigma_a : float, required when operational_mode="inference"
        Standard deviation of the Gaussian amplitude prior for log Bayes factor
        computation.
    prior : str
        Amplitude prior for Bayes factor computation: ``"gaussian"``,
        ``"half-normal"``, or ``"uniform"``.  Only used when
        ``operational_mode="inference"`` and ``context="global"``.
    amplitude_bounds : tuple[float, float], optional
        Required when ``prior="uniform"``.
    log_template_weights : ndarray, optional
        Log-space prior weights over the template bank for the global
        log-sum-exp marginalization.  Shape must match ``template_matrix.shape[1:]``
        flattened.  Uniform prior if ``None``.
    reference_time, reference_flux, reference_flux_err : optional
        When provided, the initial GP fit and ``log_marginal_likelihood`` grid
        are computed on this reference sequence rather than on
        ``time/flux``.  Useful when z-scores should reflect an injected signal
        while the posterior weights should reflect a clean baseline.
    theta_dict : dict, optional
        GP hyperparameters to use directly, bypassing the optimisation step.
        When provided, ``fit_method``, ``fit_max_retries``, and ``fit_kwargs``
        are ignored.  Keys must match those produced by
        ``gp_family.theta_to_dict``.
    """
    # ------------------------------------------------------------------
    # Validate search context
    # ------------------------------------------------------------------
    if context == "local":
        if template is None:
            raise ValueError("context='local' requires a template array.")
    elif context == "global":
        if template_matrix is None:
            raise ValueError(
                "context='global' requires a pre-materialized template_matrix of shape "
                "(N, M) or (N, M_dur, t_transit)."
            )
    else:
        raise ValueError(f"context must be 'local' or 'global', got {context!r}.")

    # ------------------------------------------------------------------
    # Validate operational mode
    # ------------------------------------------------------------------
    if operational_mode == "inference":
        if sigma_a is None:
            raise ValueError(
                "operational_mode='inference' requires sigma_a (amplitude prior std)."
            )
    elif operational_mode == "sensitivity":
        if z_threshold is None:
            raise ValueError("operational_mode='sensitivity' requires z_threshold.")
    else:
        raise ValueError(
            f"operational_mode must be 'inference' or 'sensitivity', "
            f"got {operational_mode!r}."
        )

    if template is not None:
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

    if context == "local" and len(template) != len(time):
        raise ValueError("time, flux, and template must have the same length.")
    if context == "global" and template_matrix.shape[0] != len(time):
        raise ValueError(
            "template_matrix.shape[0] must match the number of time points."
        )

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

    if theta_dict is None:
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
    if center_flux:
        thinning = max(1, len(centered) // 100)
        median_flux_estimate = np.nanmedian(centered[::thinning])
        if np.abs(median_flux_estimate - 1) < np.abs(median_flux_estimate):
            raise ValueError(
                f"Flux must be zero-centered for matched filtering. "
                f"Estimated median flux is {median_flux_estimate:.2e}. "
            )

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------
    S, T = len(sigma_grid), len(timescale_grid)
    z_map = np.full((S, T), np.nan, dtype=float)
    z_white_noise_map = np.full_like(z_map, np.nan)
    recovery_fraction_map = np.full_like(z_map, np.nan)
    relative_capacity_map = np.full_like(z_map, np.nan)
    log_marginal_likelihood_map = np.full_like(z_map, np.nan)

    # Inference: conditional BF always; global BF and 4-D tensor only for global context
    if operational_mode == "inference":
        log_bf_conditional_map = np.full_like(z_map, np.nan)
        if context == "global":
            log_bf_global_map = np.full_like(z_map, np.nan)
            is_3d_template = template_matrix.ndim == 3
            if is_3d_template:
                tensor_shape = (S, T) + template_matrix.shape[1:]
                z_score_tensor_full = np.full(tensor_shape, np.nan, dtype=float)
                log_bf_tensor_full = np.full(tensor_shape, np.nan, dtype=float)
            else:
                z_score_tensor_full = log_bf_tensor_full = None
        else:  # local context: t0 is fixed, no global BF or tensor
            log_bf_global_map = None
            is_3d_template = False
            z_score_tensor_full = log_bf_tensor_full = None
    else:
        log_bf_conditional_map = log_bf_global_map = None
        is_3d_template = False
        z_score_tensor_full = log_bf_tensor_full = None

    # Global context + sensitivity: full z_score tensor (no BF) and sensitivity tensors
    if context == "global" and operational_mode == "sensitivity":
        sensitivity_shape = (S, T) + template_matrix.shape[1:]
        ideal_sensitivity_arr = np.full(sensitivity_shape, np.nan, dtype=float)
        realization_adjusted_arr = np.full(sensitivity_shape, np.nan, dtype=float)
        if template_matrix.ndim == 3:
            tensor_shape = (S, T) + template_matrix.shape[1:]
            z_score_tensor_full = np.full(tensor_shape, np.nan, dtype=float)
        else:
            z_score_tensor_full = None
    elif context == "local" and operational_mode == "sensitivity":
        ideal_sensitivity_arr = np.full((S, T), np.nan, dtype=float)
        realization_adjusted_arr = np.full((S, T), np.nan, dtype=float)
    else:
        ideal_sensitivity_arr = realization_adjusted_arr = None

    # ------------------------------------------------------------------
    # Main grid loop
    # ------------------------------------------------------------------
    for i, sigma in enumerate(sigma_grid):
        for j, ts in enumerate(timescale_grid):
            params = dict(theta_dict)
            params["log_sigma"] = np.log(float(sigma))
            _timescale_to_theta(gp_family, params, float(ts))

            gp = gp_family.build(params, time=time, flux_err=flux_err, mean=fit_mean)
            mf = MatchedFilter(gp=gp, flux=centered, check_zero_centered=False)

            if context == "local":
                m = mf.template_metrics(template)
                z_map[i, j] = m["z_score"]
                z_white_noise_map[i, j] = m["z_white_noise"]
                recovery_fraction_map[i, j] = m["recovery_fraction"]
                relative_capacity_map[i, j] = m["relative_capacity"]

                if operational_mode == "sensitivity":
                    ideal_sensitivity_arr[i, j] = z_threshold / m["template_norm"]
                    realization_adjusted_arr[i, j] = (z_threshold - m["z_score"]) / m[
                        "template_norm"
                    ]
                elif operational_mode == "inference":
                    z_proj = m["z_score"] * m["template_norm"]
                    log_bf_conditional_map[i, j] = float(
                        MatchedFilter._log_bayes_factor_from_projection(
                            np.array([z_proj]),
                            np.array([m["template_norm"]]),
                            sigma_a,
                            prior=prior,
                            amplitude_bounds=amplitude_bounds,
                        )[0]
                    )

            else:  # context == "global"
                m = mf.matrix_metrics(template_matrix)
                z_scores = m["z_score"]  # shape: template_matrix.shape[1:]

                z_scores_flat = z_scores.ravel()
                idx_max = np.nanargmax(z_scores_flat)

                z_map[i, j] = z_scores_flat[idx_max]
                z_white_noise_map[i, j] = m["z_white_noise"].ravel()[idx_max]
                recovery_fraction_map[i, j] = m["recovery_fraction"].ravel()[idx_max]
                relative_capacity_map[i, j] = m["relative_capacity"].ravel()[idx_max]

                if operational_mode == "inference":
                    log_bf_values = MatchedFilter._log_bayes_factor_from_projection(
                        z_scores.ravel() * m["template_norm"].ravel(),
                        m["template_norm"].ravel(),
                        sigma_a,
                        prior=prior,
                        amplitude_bounds=amplitude_bounds,
                    )
                    log_bf_conditional_map[i, j] = float(np.nanmax(log_bf_values))
                    log_bf_global_map[i, j] = _logsumexp_with_weights(
                        log_bf_values, log_template_weights
                    )
                    if is_3d_template:
                        z_score_tensor_full[i, j] = z_scores
                        log_bf_tensor_full[i, j] = log_bf_values.reshape(z_scores.shape)

                else:  # operational_mode == "sensitivity"
                    norms = m["template_norm"]  # shape: template_matrix.shape[1:]
                    ideal_sensitivity_arr[i, j] = z_threshold / norms
                    realization_adjusted_arr[i, j] = (
                        z_threshold - m["z_score"]
                    ) / norms
                    if z_score_tensor_full is not None:
                        z_score_tensor_full[i, j] = z_scores

            if use_reference:
                gp_ref = gp_family.build(
                    params, time=ref_time, flux_err=ref_flux_err, mean=fit_mean
                )
                log_marginal_likelihood_map[i, j] = gp_ref.log_likelihood(ref_flux)
            else:
                log_marginal_likelihood_map[i, j] = gp.log_likelihood(flux)

    # ------------------------------------------------------------------
    # Build and return the result
    # ------------------------------------------------------------------
    common = dict(
        sigma_grid=np.asarray(sigma_grid, dtype=float),
        timescale_grid=np.asarray(timescale_grid, dtype=float),
        timescale_name=timescale_name,
        z_score=z_map,
        z_white_noise=z_white_noise_map,
        recovery_fraction=recovery_fraction_map,
        relative_capacity=relative_capacity_map,
        local_drop=_compute_local_drop(z_map),
        reference_gp_params=gp_family.theta_to_physical(
            np.array([theta_dict[n] for n in gp_family.theta_names])
        ),
        log_marginal_likelihood=log_marginal_likelihood_map,
        reference_gp_theta_dict=theta_dict,
        ideal_sensitivity=ideal_sensitivity_arr,
        realization_adjusted_sensitivity=realization_adjusted_arr,
    )

    if context == "local":
        return GPStabilityMap(
            **common,
            # z_median=z_median_map,
            # z_mad=z_mad_map,
            # z_mean=z_mean_map,
            # z_std=z_std_map,
            # p_value=p_value_map,
            log_bf_conditional=log_bf_conditional_map,
        )
    else:  # context == "global"
        return GPStabilityMap(
            **common,
            log_bf_conditional=log_bf_conditional_map,
            log_bf_global=log_bf_global_map,
            template_bank_size=int(np.prod(template_matrix.shape[1:])),
            z_score_tensor=z_score_tensor_full,
            log_bf_tensor=log_bf_tensor_full,
        )
