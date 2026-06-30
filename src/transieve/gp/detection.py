from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from collections.abc import Callable

from .fit import GPFamily, SHOGPFamily
from ..lightcurve import LightCurve
from .match import MatchedFilter, TemplateBank

_DEFAULT_THRESHOLD = 7.0


def compute_detectability(
    z_score: np.ndarray,
    template_norm: np.ndarray,
    threshold: float | None = None,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    if robust:
        mad = np.nanmedian(np.abs(z_score - np.nanmedian(z_score)))
        noise_floor = mad * 1.4826
    else:
        noise_floor = 1.0

    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    if noise_floor <= 0:
        raise ValueError("Noise floor must be positive for robust thresholding.")

    effective_threshold = threshold * noise_floor
    depth_to_threshold = (effective_threshold - z_score) / template_norm
    sensitivity = effective_threshold / template_norm
    return depth_to_threshold, sensitivity, float(effective_threshold)


__all__ = [
    "FrequentistDetectionResult",
    "MultiProfileFrequentistDetectionResult",
    "BayesianDetectionResult",
    "MultiProfileBayesianDetectionResult",
    "NoiseContext",
    "compute_detectability",
    "pipeline_evaluate_frequentist",
    "pipeline_evaluate_bayesian",
    "evaluate_frequentist_detection",
    "evaluate_bayesian_detection",
]


@dataclass
class FrequentistDetectionResult:
    """Matched-filter retrievability summary for a single parameter profile.

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


@dataclass
class MultiProfileFrequentistDetectionResult(FrequentistDetectionResult):
    """Matched-filter summary evaluating M nuisance-parameter profile realizations.

    ``z_score_tensor`` and ``template_norm_tensor`` carry the full
    ``(M, t_transit)`` tensor.  ``z_score`` / ``template_norm`` hold the
    profile-maximized ``(t_transit,)`` summary (nanmax over M).
    """

    z_score_tensor: np.ndarray  # (M, t_transit)
    template_norm_tensor: np.ndarray  # (M, t_transit)

    @property
    def profiled_z_score(self) -> np.ndarray:
        """(t_transit,): peak-envelope collapse over the profile axis M."""
        return np.nanmax(self.z_score_tensor, axis=0)


@dataclass
class BayesianDetectionResult:
    """Bayes-factor retrievability summary for a single parameter profile.

    Arrays are indexed in the same order as the template bank evaluation.
    For callable template banks this is the order of the provided time array.
    """

    time: np.ndarray
    log_bayes_factor: np.ndarray
    template_norm: np.ndarray
    log_bayes_factor_global: float
    gp_theta: np.ndarray
    gp_physical_params: dict[str, float]
    gp_optimization_result: Any

    def _peak_idx(self) -> int:
        return int(np.nanargmax(self.log_bayes_factor))

    def strongest_match(self) -> tuple[int, float]:
        idx = self._peak_idx()
        return idx, float(self.log_bayes_factor[idx])

    @property
    def peak_log_bayes_factor(self) -> float:
        return float(self.log_bayes_factor[self._peak_idx()])


@dataclass
class MultiProfileBayesianDetectionResult(BayesianDetectionResult):
    """Bayesian summary evaluating M nuisance-parameter profile realizations.

    ``log_bayes_factor_tensor`` carries the full ``(M, t_transit)`` tensor.
    ``log_bayes_factor`` holds the profile-marginalized ``(t_transit,)``
    summary (logsumexp over M with a uniform profile prior).
    """

    log_bayes_factor_tensor: np.ndarray  # (M, t_transit)

    @property
    def marginalized_log_bf(self) -> np.ndarray:
        """(t_transit,): uniform-prior marginalization over the profile axis M."""
        M = self.log_bayes_factor_tensor.shape[0]
        return logsumexp(self.log_bayes_factor_tensor, axis=0) - np.log(M)


class NoiseContext:
    """Encapsulates the fitted GP and matched-filter state for a light curve.

    Separates the non-linear hyperparameter optimization from downstream
    linear matched-filter evaluations, so a single optimized noise model can
    be reused across multiple template banks without redundant GP fits.
    """

    def __init__(
        self,
        gp: Any,
        light_curve: LightCurve,
        opt_result: Any = None,
        gp_family: GPFamily | None = None,
        center_flux: bool = True,
    ) -> None:
        self.gp = gp
        self.light_curve = light_curve
        self.opt_result = opt_result
        self.gp_family = gp_family
        centered = light_curve.centered_flux(center=center_flux)
        self.matched_filter = MatchedFilter(
            gp=gp, flux=centered, check_zero_centered=True
        )

    @classmethod
    def fit_and_resolve(
        cls,
        light_curve: LightCurve,
        gp_family: GPFamily | None = None,
        fit_method: str = "differential_evolution",
        fit_max_retries: int = 2,
        center_flux: bool = True,
        **fit_kwargs: Any,
    ) -> NoiseContext:
        """Optimize GP hyperparameters from data and return a ready NoiseContext."""
        gp_family = SHOGPFamily() if gp_family is None else gp_family
        gp_family.validate_white_noise_baseline(light_curve.flux_err)
        opt_result = gp_family.fit_light_curve(
            light_curve, method=fit_method, max_retries=fit_max_retries, **fit_kwargs
        )
        gp = gp_family.build_gp_from_theta(opt_result.x, light_curve)
        return cls(
            gp=gp,
            light_curve=light_curve,
            opt_result=opt_result,
            gp_family=gp_family,
            center_flux=center_flux,
        )

    @classmethod
    def from_preset_gp(
        cls,
        gp: Any,
        light_curve: LightCurve,
        center_flux: bool = True,
    ) -> NoiseContext:
        """Build a NoiseContext from a pre-built GP, bypassing optimization."""
        return cls(gp=gp, light_curve=light_curve, center_flux=center_flux)

    def _gp_meta(self) -> tuple[np.ndarray, dict[str, float]]:
        gp_theta = (
            np.asarray(self.opt_result.x, dtype=float)
            if self.opt_result is not None
            else np.array([])
        )
        gp_phys = (
            self.gp_family.theta_to_physical(gp_theta)
            if self.gp_family is not None
            else {}
        )
        return gp_theta, gp_phys


def pipeline_evaluate_frequentist(
    context: NoiseContext,
    template_bank: TemplateBank | list[TemplateBank],
    threshold: float | None = None,
    robust_threshold: bool = True,
) -> FrequentistDetectionResult | MultiProfileFrequentistDetectionResult:
    """Run frequentist matched filtering against a pre-built NoiseContext.

    Parameters
    ----------
    context:
        A resolved :class:`NoiseContext` carrying the fitted GP and centered
        flux residuals.
    template_bank:
        Single :class:`TemplateBank` for a flat evaluation, or a
        ``list[TemplateBank]`` for multi-profile mode.  In multi-profile mode
        all banks must share the same epoch grid; the list axis becomes the
        profile axis M.
    threshold:
        Detectability threshold.  Defaults to ``_DEFAULT_THRESHOLD``.
    robust_threshold:
        If True, scale threshold by robust noise floor estimate.
    """
    time = context.light_curve.time
    mf = context.matched_filter
    gp_theta, gp_phys = context._gp_meta()

    if isinstance(template_bank, list):
        banks = [TemplateBank._coerce(b, default_epochs=time) for b in template_bank]
        epochs = banks[0].epochs
        for b in banks[1:]:
            if not np.array_equal(b.epochs, epochs):
                raise ValueError(
                    "All TemplateBanks in the list must share the same epoch grid."
                )

        template_tensor = np.stack(
            [np.column_stack(list(b)) for b in banks], axis=1
        )  # (N, M, t_transit)
        m = mf.matrix_metrics(template_tensor)

        z_score_tensor = m["z_score"]  # (M, t_transit)
        template_norm_tensor = m["template_norm"]

        best_m = np.nanargmax(z_score_tensor, axis=0)  # (t_transit,)
        col_idx = np.arange(len(epochs))
        z_scores = z_score_tensor[best_m, col_idx]
        template_norms = template_norm_tensor[best_m, col_idx]
        z_white_noise = m["z_white_noise"][best_m, col_idx]
        recovery_fraction = m["recovery_fraction"][best_m, col_idx]
        relative_capacity = m["relative_capacity"][best_m, col_idx]

        depth_to_threshold, sensitivity, effective_threshold = compute_detectability(
            z_scores, template_norms, threshold=threshold, robust=robust_threshold
        )
        return MultiProfileFrequentistDetectionResult(
            time=epochs,
            z_score=z_scores,
            template_norm=template_norms,
            z_white_noise=z_white_noise,
            recovery_fraction=recovery_fraction,
            relative_capacity=relative_capacity,
            depth_to_threshold=np.asarray(depth_to_threshold, dtype=float),
            sensitivity=np.asarray(sensitivity, dtype=float),
            detection_threshold=effective_threshold,
            gp_theta=gp_theta,
            gp_physical_params=gp_phys,
            gp_optimization_result=context.opt_result,
            z_score_tensor=z_score_tensor,
            template_norm_tensor=template_norm_tensor,
        )
    else:
        bank = TemplateBank._coerce(template_bank, default_epochs=time)
        epochs = bank.epochs

        z_list, norm_list, wn_list, rec_list, cap_list = [], [], [], [], []
        for template in bank:
            if len(template) != len(time):
                raise ValueError("Template length must match time/flux length.")
            metrics = mf.template_metrics(template)
            z_list.append(metrics["z_score"])
            norm_list.append(metrics["template_norm"])
            wn_list.append(metrics["z_white_noise"])
            rec_list.append(metrics["recovery_fraction"])
            cap_list.append(metrics["relative_capacity"])

        z_scores = np.asarray(z_list, dtype=float)
        template_norms = np.asarray(norm_list, dtype=float)
        z_white_noise = np.asarray(wn_list, dtype=float)
        recovery_fraction = np.asarray(rec_list, dtype=float)
        relative_capacity = np.asarray(cap_list, dtype=float)

        depth_to_threshold, sensitivity, effective_threshold = compute_detectability(
            z_scores, template_norms, threshold=threshold, robust=robust_threshold
        )
        return FrequentistDetectionResult(
            time=epochs,
            z_score=z_scores,
            template_norm=template_norms,
            z_white_noise=z_white_noise,
            recovery_fraction=recovery_fraction,
            relative_capacity=relative_capacity,
            depth_to_threshold=np.asarray(depth_to_threshold, dtype=float),
            sensitivity=np.asarray(sensitivity, dtype=float),
            detection_threshold=effective_threshold,
            gp_theta=gp_theta,
            gp_physical_params=gp_phys,
            gp_optimization_result=context.opt_result,
        )


def pipeline_evaluate_bayesian(
    context: NoiseContext,
    template_bank: TemplateBank | list[TemplateBank],
    sigma_a: float,
    prior: str = "gaussian",
    amplitude_bounds: tuple[float, float] | None = None,
    log_template_weights: np.ndarray | None = None,
) -> BayesianDetectionResult | MultiProfileBayesianDetectionResult:
    """Run Bayesian log Bayes factor evaluation against a pre-built NoiseContext.

    Parameters
    ----------
    context:
        A resolved :class:`NoiseContext` carrying the fitted GP and centered
        flux residuals.
    template_bank:
        Single :class:`TemplateBank` or a ``list[TemplateBank]`` for
        multi-profile mode.  In multi-profile mode all banks must share the
        same epoch grid; the list axis becomes the profile axis M.
    sigma_a:
        Prior amplitude scale.
    prior:
        Prior form passed to the Bayes factor computation.
    amplitude_bounds:
        Optional bounds for the amplitude integration.
    log_template_weights:
        Optional log-weights over templates for the global BF.
    """
    time = context.light_curve.time
    mf = context.matched_filter
    gp_theta, gp_phys = context._gp_meta()

    if isinstance(template_bank, list):
        banks = [TemplateBank._coerce(b, default_epochs=time) for b in template_bank]
        epochs = banks[0].epochs
        for b in banks[1:]:
            if not np.array_equal(b.epochs, epochs):
                raise ValueError(
                    "All TemplateBanks in the list must share the same epoch grid."
                )

        template_tensor = np.stack(
            [np.column_stack(list(b)) for b in banks], axis=1
        )  # (N, M, t_transit)
        M, t_transit = template_tensor.shape[1], template_tensor.shape[2]
        flat = template_tensor.reshape(len(time), -1)

        projections_flat, norms_flat = mf._bank_stats(flat)
        log_bfs_flat = MatchedFilter._log_bayes_factor_from_projection(
            projections_flat,
            norms_flat,
            sigma_a,
            prior=prior,
            amplitude_bounds=amplitude_bounds,
        )
        log_bayes_factor_tensor = log_bfs_flat.reshape(M, t_transit)

        if log_template_weights is None:
            log_bf_global = float(logsumexp(log_bfs_flat) - np.log(len(log_bfs_flat)))
        else:
            log_template_weights = np.asarray(log_template_weights, dtype=float)
            if log_template_weights.shape != log_bfs_flat.shape:
                raise ValueError(
                    "log_template_weights must have length M * t_transit "
                    f"({len(log_bfs_flat)}) in multi-profile mode."
                )
            log_bf_global = float(
                logsumexp(log_bfs_flat + log_template_weights)
                - logsumexp(log_template_weights)
            )

        log_bayes_factors = logsumexp(log_bayes_factor_tensor, axis=0) - np.log(M)
        best_m = np.nanargmax(log_bayes_factor_tensor, axis=0)
        col_idx = np.arange(t_transit)
        template_norms = norms_flat.reshape(M, t_transit)[best_m, col_idx]

        return MultiProfileBayesianDetectionResult(
            time=epochs,
            log_bayes_factor=log_bayes_factors,
            template_norm=template_norms,
            log_bayes_factor_global=log_bf_global,
            gp_theta=gp_theta,
            gp_physical_params=gp_phys,
            gp_optimization_result=context.opt_result,
            log_bayes_factor_tensor=log_bayes_factor_tensor,
        )
    else:
        bank = TemplateBank._coerce(template_bank, default_epochs=time)
        epochs = bank.epochs

        projections, template_norms = mf._bank_stats(bank)
        log_bayes_factors = MatchedFilter._log_bayes_factor_from_projection(
            projections,
            template_norms,
            sigma_a,
            prior=prior,
            amplitude_bounds=amplitude_bounds,
        )

        if log_template_weights is None:
            log_bf_global = float(
                logsumexp(log_bayes_factors) - np.log(len(log_bayes_factors))
            )
        else:
            log_template_weights = np.asarray(log_template_weights, dtype=float)
            if log_template_weights.shape != log_bayes_factors.shape:
                raise ValueError("log_template_weights must match template bank size.")
            log_bf_global = float(
                logsumexp(log_bayes_factors + log_template_weights)
                - logsumexp(log_template_weights)
            )

        return BayesianDetectionResult(
            time=epochs,
            log_bayes_factor=log_bayes_factors,
            template_norm=template_norms,
            log_bayes_factor_global=log_bf_global,
            gp_theta=gp_theta,
            gp_physical_params=gp_phys,
            gp_optimization_result=context.opt_result,
        )


def evaluate_frequentist_detection(
    time: np.ndarray,
    flux: np.ndarray,
    template_bank: Callable[[float], np.ndarray] | TemplateBank | list[TemplateBank],
    flux_err: np.ndarray | None = None,
    gp_family: GPFamily | None = None,
    fit_mean: float = 0.0,
    fit_method: str = "differential_evolution",
    fit_max_retries: int = 2,
    fit_kwargs: dict[str, Any] | None = None,
    center_flux: bool = True,
    threshold: float | None = None,
    robust_threshold: bool = True,
) -> FrequentistDetectionResult | MultiProfileFrequentistDetectionResult:
    """Assess retrievability for a template bank in correlated noise.

    Fits a GP to the provided light curve, computes the exact matched-filter
    Z-score for each template, and returns both red-noise and white-noise
    baseline diagnostics.

    Parameters
    ----------
    time : np.ndarray
            Observation times.
    flux : np.ndarray
            Input flux values. When ``center_flux=True`` the median is removed before
            matched filtering.
    template_bank : callable, TemplateBank, or list[TemplateBank]
            Template source. If callable, evaluated at every point in ``time``.
            Pass a :class:`TemplateBank` to use a custom epoch grid.
            Pass a ``list[TemplateBank]`` for multi-profile mode: each bank
            must share the same epoch grid; the list axis becomes the profile
            axis M.  Returns a :class:`MultiProfileFrequentistDetectionResult`
            carrying ``z_score_tensor`` of shape ``(M, t_transit)``.
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
            Detectability threshold. Defaults to ``detection._DEFAULT_THRESHOLD``.
    robust_threshold : bool, optional
            If True, scale threshold by robust noise floor estimate.
    """
    if isinstance(template_bank, list):
        _epochs = [b.epochs for b in template_bank if isinstance(b, TemplateBank)]
        if len(_epochs) > 1 and not all(
            np.array_equal(_epochs[0], e) for e in _epochs[1:]
        ):
            raise ValueError(
                "All TemplateBanks in the list must share the same epoch grid."
            )

    light_curve = LightCurve.from_arrays(
        time=time, flux=flux, flux_err=flux_err, mean=fit_mean
    )
    context = NoiseContext.fit_and_resolve(
        light_curve=light_curve,
        gp_family=gp_family,
        fit_method=fit_method,
        fit_max_retries=fit_max_retries,
        center_flux=center_flux,
        **({} if fit_kwargs is None else fit_kwargs),
    )
    coerced: TemplateBank | list[TemplateBank] = (
        template_bank
        if isinstance(template_bank, (TemplateBank, list))
        else TemplateBank._coerce(template_bank, default_epochs=light_curve.time)
    )
    return pipeline_evaluate_frequentist(
        context=context,
        template_bank=coerced,
        threshold=threshold,
        robust_threshold=robust_threshold,
    )


def evaluate_bayesian_detection(
    time: np.ndarray,
    flux: np.ndarray,
    template_bank: Callable[[float], np.ndarray] | TemplateBank | list[TemplateBank],
    sigma_a: float,
    flux_err: np.ndarray | None = None,
    gp_family: GPFamily | None = None,
    fit_mean: float = 0.0,
    fit_method: str = "differential_evolution",
    fit_max_retries: int = 2,
    fit_kwargs: dict[str, Any] | None = None,
    center_flux: bool = True,
    prior: str = "gaussian",
    amplitude_bounds: tuple[float, float] | None = None,
    log_template_weights: np.ndarray | None = None,
) -> BayesianDetectionResult | MultiProfileBayesianDetectionResult:
    """Assess retrievability for a template bank using log Bayes factors.

    Fits a GP to the provided light curve, computes the analytic log Bayes
    factor for each template, and marginalizes across the template bank using
    log-sum-exp.
    """
    if isinstance(template_bank, list):
        _epochs = [b.epochs for b in template_bank if isinstance(b, TemplateBank)]
        if len(_epochs) > 1 and not all(
            np.array_equal(_epochs[0], e) for e in _epochs[1:]
        ):
            raise ValueError(
                "All TemplateBanks in the list must share the same epoch grid."
            )

    light_curve = LightCurve.from_arrays(
        time=time, flux=flux, flux_err=flux_err, mean=fit_mean
    )
    context = NoiseContext.fit_and_resolve(
        light_curve=light_curve,
        gp_family=gp_family,
        fit_method=fit_method,
        fit_max_retries=fit_max_retries,
        center_flux=center_flux,
        **({} if fit_kwargs is None else fit_kwargs),
    )
    coerced: TemplateBank | list[TemplateBank] = (
        template_bank
        if isinstance(template_bank, (TemplateBank, list))
        else TemplateBank._coerce(template_bank, default_epochs=light_curve.time)
    )
    return pipeline_evaluate_bayesian(
        context=context,
        template_bank=coerced,
        sigma_a=sigma_a,
        prior=prior,
        amplitude_bounds=amplitude_bounds,
        log_template_weights=log_template_weights,
    )
