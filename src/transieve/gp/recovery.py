from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from collections.abc import Callable

from .fit import GPFamily, SHOGPFamily
from ..lightcurve import LightCurve
from .match import MatchedFilter, MatchedFilterStatistics, TemplateBank
from .stability import GPStabilityMap

__all__ = [
    "RetrievabilityResult",
    "assess_retrievability",
    "check_retrievability",
]


@dataclass
class RetrievabilityResult:
    """Matched-filter retrievability summary for one template bank.

    Arrays are indexed in the same order as the template bank evaluation.
    For callable template banks this is the order of the provided time array.
    """

    time: np.ndarray
    z_score: np.ndarray
    template_norm: np.ndarray
    z_white_noise: np.ndarray
    recovery_fraction: np.ndarray
    relative_capacity: np.ndarray
    depth_to_threshold: np.ndarray
    sensitivity: np.ndarray
    detection_threshold: float
    gp_theta: np.ndarray
    gp_physical_params: dict[str, float]
    gp_optimization_result: Any

    @property
    def white_template_norm(self) -> np.ndarray:
        """Baseline template norm in white noise."""
        return self.template_norm / self.relative_capacity

    def _peak_idx(self) -> int:
        return int(np.nanargmax(self.z_score))

    def strongest_match(self) -> tuple[int, float]:
        idx = self._peak_idx()
        return idx, float(self.z_score[idx])

    @property
    def peak_recovery_fraction(self) -> float:
        return float(self.recovery_fraction[self._peak_idx()])

    @property
    def peak_relative_capacity(self) -> float:
        return float(self.relative_capacity[self._peak_idx()])


def assess_retrievability(
    time: np.ndarray,
    flux: np.ndarray,
    template_bank: Callable[[float], np.ndarray] | TemplateBank,
    flux_err: np.ndarray | None = None,
    gp_family: GPFamily | None = None,
    fit_mean: float = 0.0,
    fit_method: str = "differential_evolution",
    fit_max_retries: int = 2,
    fit_kwargs: dict[str, Any] | None = None,
    center_flux: bool = True,
    threshold: float | None = None,
    robust_threshold: bool = True,
) -> RetrievabilityResult:
    """Assess retrievability for a template bank in correlated noise.

    This function fits a GP to the provided light curve, computes the exact
    matched-filter Z-score for each template, and returns both red-noise and
    white-noise baseline diagnostics.

    Parameters
    ----------
    time : np.ndarray
            Observation times.
    flux : np.ndarray
            Input flux values. When ``center_flux=True`` the median is removed before
            matched filtering.
    template_bank : callable or TemplateBank
            Template source. If callable, evaluated at every point in ``time``.
            Pass a :class:`TemplateBank` to use a custom epoch grid.
    flux_err : np.ndarray, optional
            Per-sample uncertainty.
    gp_family : GPFamily, optional
            GP family used for fitting. Defaults to SHOGPFamily.
    fit_mean : float, optional
            GP mean used when optimising the kernel. Does not affect matched-filter
            output — use ``center_flux`` to remove the photometric baseline.
    fit_method : str, optional
            Optimizer name used by ``GPFamily.fit_light_curve``.
    fit_max_retries : int, optional
            Number of optimizer retries.
    fit_kwargs : dict, optional
            Extra keyword arguments forwarded to ``fit_light_curve``.
    center_flux : bool, optional
            If True, subtract the median from flux before filtering.
    threshold : float, optional
            Detectability threshold. Defaults to ``MatchedFilterStatistics.THRESHOLD``.
    robust_threshold : bool, optional
            If True, scale threshold by robust noise floor estimate.
    """

    light_curve = LightCurve.from_arrays(
        time=time,
        flux=flux,
        flux_err=flux_err,
        mean=fit_mean,
    )
    time = light_curve.time
    flux_err = light_curve.flux_err

    bank = TemplateBank._coerce(template_bank, default_epochs=time)

    gp_family = SHOGPFamily() if gp_family is None else gp_family
    gp_family.validate_white_noise_baseline(flux_err)

    opt_result = gp_family.fit_light_curve(
        light_curve,
        method=fit_method,
        max_retries=fit_max_retries,
        **({} if fit_kwargs is None else dict(fit_kwargs)),
    )

    gp = gp_family.build_gp_from_theta(
        opt_result.x,
        light_curve,
    )

    centered = light_curve.centered_flux(center=center_flux)
    matched_filter = MatchedFilter(gp=gp, flux=centered, check_zero_centered=True)

    z_scores = []
    template_norms = []
    z_white_noise_list = []
    recovery_fraction = []
    relative_capacity = []

    for template in bank:
        if len(template) != len(time):
            raise ValueError("Template length must match time/flux length.")

        m = matched_filter.template_metrics(template)
        z_scores.append(m["z_score"])
        template_norms.append(m["template_norm"])
        z_white_noise_list.append(m["z_white_noise"])
        recovery_fraction.append(m["recovery_fraction"])
        relative_capacity.append(m["relative_capacity"])

    z_scores = np.asarray(z_scores, dtype=float)
    template_norms = np.asarray(template_norms, dtype=float)
    z_white_noise_arr = np.asarray(z_white_noise_list, dtype=float)
    recovery_fraction = np.asarray(recovery_fraction, dtype=float)
    relative_capacity = np.asarray(relative_capacity, dtype=float)

    stats = MatchedFilterStatistics(z_score=z_scores, template_norm=template_norms)
    depth_to_threshold, sensitivity, effective_threshold = stats.get_detectability(
        threshold=threshold,
        robust=robust_threshold,
    )

    return RetrievabilityResult(
        time=bank.epochs,
        z_score=z_scores,
        template_norm=template_norms,
        z_white_noise=z_white_noise_arr,
        recovery_fraction=recovery_fraction,
        relative_capacity=relative_capacity,
        depth_to_threshold=np.asarray(depth_to_threshold, dtype=float),
        sensitivity=np.asarray(sensitivity, dtype=float),
        detection_threshold=effective_threshold,
        gp_theta=np.asarray(opt_result.x, dtype=float),
        gp_physical_params=gp_family.theta_to_physical(opt_result.x),
        gp_optimization_result=opt_result,
    )


def check_retrievability(
    retr: RetrievabilityResult,
    injected_time: float | None = None,
    duration: float | None = None,
    z_threshold: float | None = None,
    recovery_cutoff: float = 0.5,
    capacity_bounds: tuple[float, float] = (0.5, 1.5),
    stability_map: GPStabilityMap | None = None,
    localization_tol: float | None = None,
):
    """Decide if an injected transit is retrievable.

    Returns (bool, diagnostics_dict).

    The decision applies the 4-gate rule used in the notebook:
      - significance (Z >= threshold)
      - recovery fraction (>= recovery_cutoff)
      - relative capacity within bounds
      - stability / plateau presence (if `stability_map` provided)

    If `injected_time` is None the function evaluates at the strongest-match index.
    """
    if z_threshold is None:
        z_threshold = retr.detection_threshold

    if injected_time is None:
        idx = int(np.nanargmax(retr.z_score))
    else:
        t = np.asarray(retr.time, dtype=float)
        idx = int(np.argmin(np.abs(t - float(injected_time))))

    z = float(retr.z_score[idx])
    z_white_noise = float(retr.z_white_noise[idx])
    recovery = float(retr.recovery_fraction[idx])
    capacity = float(retr.relative_capacity[idx])

    sig_ok = z >= z_threshold
    rec_ok = recovery >= recovery_cutoff
    cap_ok = capacity_bounds[0] <= capacity <= capacity_bounds[1]

    loc_ok = True
    global_peak_near_epoch = True
    peak_idx, _ = retr.strongest_match()
    if injected_time is not None and duration is not None:
        half_dur = 0.5 * float(duration)
        peak_time = float(retr.time[peak_idx])
        tol = half_dur if localization_tol is None else float(localization_tol)
        # Additional diagnostic: is the global peak near the injection?
        global_peak_near_epoch = abs(peak_time - float(injected_time)) <= tol
        # Verdict gate: is z at the epoch a local maximum within ±half_dur?
        t = np.asarray(retr.time, dtype=float)
        window = np.abs(t - float(injected_time)) <= tol
        local_max_z = float(np.nanmax(retr.z_score[window])) if np.any(window) else z
        loc_ok = z >= local_max_z

    stab_ok = True
    if stability_map is not None:
        plateau_mask = stability_map.get_plateau_mask(z_threshold)
        gp_ref = stability_map.reference_gp_params
        if "sigma" in gp_ref and stability_map.timescale_name in gp_ref:
            sigma_idx = int(
                np.argmin(np.abs(stability_map.sigma_grid - float(gp_ref["sigma"])))
            )
            timescale_idx = int(
                np.argmin(
                    np.abs(
                        stability_map.timescale_grid
                        - float(gp_ref[stability_map.timescale_name])
                    )
                )
            )
            stab_ok = bool(plateau_mask[sigma_idx, timescale_idx])
        else:
            stab_ok = False

    verdict = bool(sig_ok and rec_ok and cap_ok and stab_ok)

    diagnostics = dict(
        index=idx,
        z=z,
        z_threshold=z_threshold,
        z_white_noise=z_white_noise,
        recovery=recovery,
        recovery_cutoff=recovery_cutoff,
        capacity=capacity,
        capacity_bounds=capacity_bounds,
        significance_ok=sig_ok,
        recovery_ok=rec_ok,
        capacity_ok=cap_ok,
        localization_ok=loc_ok,
        global_peak_near_epoch=global_peak_near_epoch,
        stability_ok=stab_ok,
    )

    return verdict, diagnostics
