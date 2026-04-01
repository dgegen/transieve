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
            "max_z_score": float(np.nanmax(self.z_score)),
            "median_recovery_fraction": float(np.nanmedian(self.recovery_fraction)),
        }


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
) -> GPStabilityMap:
    """Scan retrievability over a 2D GP hyperparameter grid.

    The map is evaluated for one fixed template and stores the strongest
    matched-filter response at each grid point.
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
    gp_family.validate_white_noise_baseline(flux_err)

    opt_result = gp_family.fit_light_curve(
        light_curve,
        method=fit_method,
        max_retries=fit_max_retries,
        **({} if fit_kwargs is None else dict(fit_kwargs)),
    )

    theta_dict = gp_family.theta_to_dict(opt_result.x)
    timescale_name, timescale_ref = _infer_timescale_from_theta(gp_family, theta_dict)
    sigma_ref = np.exp(theta_dict["log_sigma"])

    if sigma_grid is None:
        sigma_upper = float(np.nanstd(flux))
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
    )
