from __future__ import annotations
from typing import Any

import celerite2.driver as driver
import numpy as np
from celerite2 import GaussianProcess
from collections.abc import Callable
from dataclasses import dataclass, field
from scipy.signal import fftconvolve
from scipy.special import log_ndtr
from scipy.stats import norm


__all__ = ["MatchedFilter", "TemplateBank"]


@dataclass
class TemplateBank:
    """A callable template factory paired with its evaluation epochs.

    Parameters
    ----------
    make : Callable[[float], np.ndarray]
        Template factory. Called as ``make(epoch)`` and must return an array
        of the same length as the observation time vector.
    epochs : np.ndarray
        Transit-center times (or other parameter values) at which to evaluate
        ``func``. These become the time axis of any result statistics.
    """

    make: Callable[[float], np.ndarray]
    epochs: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        self.epochs = np.asarray(self.epochs, dtype=float)

    def __iter__(self):
        return (np.asarray(self.make(t), dtype=float) for t in self.epochs)

    def __len__(self) -> int:
        return len(self.epochs)

    @classmethod
    def _coerce(
        cls,
        template_bank: Callable[[float], np.ndarray] | TemplateBank,
        default_epochs: np.ndarray,
    ) -> TemplateBank:
        """Return a TemplateBank, wrapping a bare callable with ``default_epochs``."""
        if isinstance(template_bank, TemplateBank):
            return template_bank
        return cls(make=template_bank, epochs=default_epochs)


class MatchedFilter:
    """
    Matched-filter Z-score utilities for correlated-noise light curves.

    The noise model is represented by a pre-computed `celerite2.GaussianProcess`.
    Given a template ``s`` and observed data ``y``, this class computes
    statistics based on the standard matched-filter expressions:

    - template amplitude (norm): ``sqrt(s^T C^-1 s)``
    - projected template: ``y^T C^-1 s``
    - Z-score: ``(y^T C^-1 s) / sqrt(s^T C^-1 s)``

    Notes
    -----
    - `z_score_map` evaluates templates exactly.
    - `z_score_map_fft` is a fast convolution-based approximation for nearly
        uniform cadence and stationary noise assumptions.

    Template assumptions
    --------------------
    Templates are assumed to be **locally compact** — e.g. a single transit dip
    that decays to zero well within the observation baseline.
    """

    """Small constant to prevent division by zero."""
    _EPS = 1e-12

    def __init__(
        self, gp: GaussianProcess, flux: np.ndarray, check_zero_centered: bool = True
    ):
        """Store the GP model and flux."""
        self.gp = gp
        self.flux = flux

        if np.any(np.isnan(flux)):
            raise ValueError(
                "flux contains NaN values. Remove NaN cadences before constructing "
                "MatchedFilter (e.g. use inject_gaps with mode='remove')."
            )

        if check_zero_centered:
            self._ensure_zero_centered()

    def _ensure_zero_centered(self):
        """Check that flux is zero-centered to avoid biasing the matched filter."""
        thinning = max(1, len(self.flux) // 100)
        median_flux_estimate = np.nanmedian(self.flux[::thinning])
        if np.abs(median_flux_estimate - 1) < np.abs(median_flux_estimate):
            raise ValueError(
                f"Flux must be zero-centered for matched filtering. "
                f"Estimated median flux is {median_flux_estimate:.2e}. "
            )

    def apply_inverse_variance(self, vector: np.ndarray) -> np.ndarray:
        """Apply the GP inverse covariance operator to `vector` (C^-1 * v)."""
        return self.gp.apply_inverse(vector)

    def whiten(self, vector: np.ndarray) -> np.ndarray:
        """Apply the GP whitening transform ``z = D^{-1/2} L^{-1} v``.

        Decorrelates and normalizes `vector` to produce independent, unit-variance
        samples (innovations). When applied to the residual flux ``y - mu``, the
        output is the sequence of one-step prediction errors (innovations) scaled
        by their standard deviations — i.e. each entry is the surprise at time ``t``
        given all previous observations, in units of its own standard deviation.

        A standard deviation near 1.0 indicates the GP is well-calibrated. Values
        substantially below 1.0 suggest overfitting; values above 1.0 suggest
        underfitting.
        """
        y = self.gp._process_input(vector, inplace=False)
        is_vector = y.ndim == 1
        y_work = y[:, None] if is_vector else y

        z = driver.solve_lower(
            self.gp._t, self.gp._c, self.gp._U, self.gp._W, y_work, y_work.copy()
        )

        z /= np.sqrt(self.gp._d[:, None] + self._EPS)  # Scale to unit variance

        return z[:, 0] if is_vector else z

    def template_norm_and_projection(
        self, template: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return template norm and whitened template for `template`."""
        precision_weighted_template = self.apply_inverse_variance(template)
        template_norm = np.sqrt(
            np.maximum(
                np.dot(template, precision_weighted_template), MatchedFilter._EPS
            )
        )
        return template_norm, precision_weighted_template

    def get_search_profile(
        self, bank: Callable[[float], np.ndarray] | TemplateBank
    ) -> SearchProfile:
        bank = TemplateBank._coerce(bank, default_epochs=self.gp._t)
        z_stats = self.z_score_map(bank)
        bf_map = self.log_bayes_factor_map(bank, sigma_a=1.0, prior="half-normal")
        return SearchProfile(
            epochs=bank.epochs,
            z_score=z_stats.z_score,
            template_norm=z_stats.template_norm,
            log_bayes_factor=bf_map,
        )

    def template_projection_and_norm(
        self, template: np.ndarray, inject_template: bool = False
    ) -> tuple[float, float]:
        """Return (y^T C^-1 s, sqrt(s^T C^-1 s)) for `template`."""
        template_norm, precision_weighted_template = self.template_norm_and_projection(
            template
        )
        flux = self.flux + template if inject_template else self.flux
        projection = float(np.dot(flux, precision_weighted_template))
        return projection, template_norm

    @staticmethod
    def _logdiffexp(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
        """Return log(exp(log_a) - exp(log_b)) in a stable way."""
        log_a = np.asarray(log_a)
        log_b = np.asarray(log_b)
        swap = log_b > log_a
        hi = np.where(swap, log_b, log_a)
        lo = np.where(swap, log_a, log_b)
        return hi + np.log1p(-np.exp(lo - hi))

    @staticmethod
    def _log_bayes_factor_from_projection(
        projection: np.ndarray,
        template_norm: np.ndarray,
        sigma_a: float,
        prior: str = "gaussian",
        amplitude_bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Compute log Bayes factor using sufficient statistics.

        Parameters
        ----------
        projection:
            The matched-filter projection, ``y^T C^{-1} s``.
        template_norm:
            The matched-filter norm, ``sqrt(s^T C^{-1} s)``.
        sigma_a:
            Standard deviation of the Gaussian amplitude prior (ignored for
            ``prior='uniform'``).
        prior:
            One of ``'gaussian'``, ``'half-normal'``, or ``'uniform'``.
        amplitude_bounds:
            (low, high) bounds for the uniform prior. Required when
            ``prior='uniform'``.
        """
        s2 = np.asarray(template_norm) ** 2
        projection = np.asarray(projection)
        if np.any(s2 <= 0):
            raise ValueError("Template norm must be positive to compute Bayes factor.")

        prior_key = str(prior).lower().replace("_", "-")
        if prior_key in {"gaussian", "normal"}:
            if sigma_a <= 0:
                raise ValueError("sigma_a must be positive for a Gaussian prior.")
            sigma_a2 = float(sigma_a) ** 2
            denom = 1.0 + sigma_a2 * s2
            return 0.5 * (sigma_a2 * projection**2 / denom) - 0.5 * np.log(denom)

        if prior_key in {"half-normal", "halfnormal", "halfnormal"}:
            if sigma_a <= 0:
                raise ValueError("sigma_a must be positive for a half-normal prior.")
            sigma_a2 = float(sigma_a) ** 2
            denom = 1.0 + sigma_a2 * s2
            log_bf = 0.5 * (sigma_a2 * projection**2 / denom) - 0.5 * np.log(denom)
            kappa = projection * sigma_a / np.sqrt(denom)
            log_factor = np.log(2.0) + norm.logcdf(kappa)
            return log_bf + log_factor

        if prior_key == "uniform":
            if amplitude_bounds is None:
                raise ValueError("amplitude_bounds are required for uniform prior.")
            low, high = amplitude_bounds
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError("amplitude_bounds must be finite with low < high.")
            mu = projection / s2
            sigma = 1.0 / np.sqrt(s2)
            z_low = (low - mu) / sigma
            z_high = (high - mu) / sigma
            log_cdf_low = log_ndtr(z_low)
            log_cdf_high = log_ndtr(z_high)
            log_diff = MatchedFilter._logdiffexp(log_cdf_high, log_cdf_low)
            log_prefactor = 0.5 * np.log(2.0 * np.pi) - 0.5 * np.log(s2)
            log_exp = 0.5 * (projection**2 / s2)
            return log_exp + log_prefactor + log_diff - np.log(high - low)

        raise ValueError(f"Unknown amplitude prior: {prior!r}.")

    def z_score(self, template, inject_template=False) -> tuple[float, float]:
        """Compute scalar matched-filter Z-score for a single template.

        Parameters
        ----------
        template : array-like
            Template to match against self.flux.
        inject_template : bool, default False
            If True, evaluate on ``self.flux + template``.
        """
        flux = self.flux + template if inject_template else self.flux

        template_norm, precision_weighted_template = self.template_norm_and_projection(
            template
        )
        projected_template = np.dot(flux, precision_weighted_template)
        z_score = projected_template / template_norm

        return z_score, template_norm

    def _bank_stats(
        self, bank: TemplateBank | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (projections, template_norms) for all templates in `bank`.

        Builds the full [N, M] template matrix and solves C^{-1} S in a single
        celerite2 call, avoiding a Python loop over templates.

        `bank` may be a :class:`TemplateBank` or a pre-built ``[N, M]`` array
        (pass the array to avoid re-materializing templates on repeated calls).
        """
        template_matrix = (
            bank if isinstance(bank, np.ndarray) else np.column_stack(list(bank))
        )
        if template_matrix.shape[0] != len(self.flux):
            raise ValueError("Template length must match time/flux length.")
        precision_weighted = self.gp.apply_inverse(template_matrix)  # [N, M]
        template_norms = np.sqrt(
            np.maximum(
                np.einsum("ij,ij->j", template_matrix, precision_weighted), self._EPS
            )
        )
        projections = self.flux @ precision_weighted
        return projections, template_norms

    def z_score_map(
        self,
        template_bank: Callable[[float], np.ndarray] | TemplateBank,
        epochs: np.ndarray | None = None,
    ) -> SearchProfile:
        """Compute Z-scores for a template bank over a grid of epochs.

        Parameters
        ----------
        template_bank : callable or TemplateBank
            If callable, evaluated at every point in the GP's stored time array.
            Pass a :class:`TemplateBank` to use a custom epoch grid.

        Returns
        -------
        SearchProfile
            Z-scores and template norms, one entry per epoch.
        """
        bank = TemplateBank._coerce(template_bank, default_epochs=self.gp._t)
        projections, template_norms = self._bank_stats(bank)
        if epochs is None:
            if hasattr(bank, "epochs"):
                epochs = bank.epochs
            elif isinstance(template_bank, TemplateBank):
                epochs = template_bank.epochs
            elif len(self.gp._t) == len(projections):
                epochs = self.gp._t
            else:
                raise ValueError(
                    "Unable to infer epochs for z_score_map. Please provide an explicit "
                    "epochs array or pass a TemplateBank with an epochs attribute."
                )
        return SearchProfile(
            epochs=epochs,
            z_score=projections / template_norms,
            template_norm=template_norms,
        )

    def log_bayes_factor(
        self,
        template: np.ndarray,
        sigma_a: float,
        prior: str = "gaussian",
        amplitude_bounds: tuple[float, float] | None = None,
        inject_template: bool = False,
    ) -> float:
        """Compute log Bayes factor for a single template.

        The amplitude prior is Gaussian by default (two-sided, mean 0). Use
        ``prior='half-normal'`` for positive-only depths or ``prior='uniform'``
        with ``amplitude_bounds`` to define a bounded prior.
        """
        projection, template_norm = self.template_projection_and_norm(
            template, inject_template=inject_template
        )
        log_bf = MatchedFilter._log_bayes_factor_from_projection(
            projection,
            template_norm,
            sigma_a,
            prior=prior,
            amplitude_bounds=amplitude_bounds,
        )
        return float(log_bf)

    def log_bayes_factor_map(
        self,
        template_bank: Callable[[float], np.ndarray] | TemplateBank | np.ndarray,
        sigma_a: float,
        prior: str = "gaussian",
        amplitude_bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Compute log Bayes factors for a template bank over epochs."""
        if isinstance(template_bank, np.ndarray):
            projections, template_norms = self._bank_stats(template_bank)
        else:
            bank = TemplateBank._coerce(template_bank, default_epochs=self.gp._t)
            projections, template_norms = self._bank_stats(bank=bank)

        return MatchedFilter._log_bayes_factor_from_projection(
            projections,
            template_norms,
            sigma_a,
            prior=prior,
            amplitude_bounds=amplitude_bounds,
        )

    def z_score_map_fft(self, template):
        """Compute an FFT-based Z-score time series for one reference template.

        This method assumes that the noise is approximately stationary and that the
        time sampling is uniform, allowing us to use convolution to compute the
        matched-filter projection efficiently. The template is assumed to be centered
        at time zero, and the resulting Z-score map will be aligned with the original
        time array.
        """
        template_norm, precision_weighted_template = self.template_norm_and_projection(
            template
        )

        projected_template = fftconvolve(
            self.flux, precision_weighted_template[::-1], mode="same"
        )
        z_score = projected_template / template_norm

        return SearchProfile(
            epochs=self.gp._t,
            z_score=z_score,
            template_norm=template_norm * np.ones_like(z_score),
        )

    def template_metrics(self, template: np.ndarray) -> dict:
        """Compute matched-filter and white-noise metrics for a single template.

        Returns
        -------
        dict with keys:
            z_score, template_norm, z_white_noise, white_template_norm,
            recovery_fraction, relative_capacity
        """
        z_score, template_norm = self.z_score(template)
        z_white_noise, white_template_norm = self.white_noise_z_score(template)
        recovery_fraction = np.maximum(z_score, 0.0) / white_template_norm
        relative_capacity = template_norm / white_template_norm
        return {
            "z_score": z_score,
            "template_norm": template_norm,
            "z_white_noise": z_white_noise,
            "white_template_norm": white_template_norm,
            "recovery_fraction": recovery_fraction,
            "relative_capacity": relative_capacity,
        }

    def matrix_metrics(self, template_matrix: np.ndarray) -> dict:
        """Vectorized template_metrics for a pre-materialized template array.

        Parameters
        ----------
        template_matrix : ndarray, shape (N, M) or (N, M_dur, t_transit)
            Columns/slices are individual templates, all of length N.
            For multi-dimensional template spaces (e.g. duration × epoch),
            pass a 3-D array; output arrays preserve the trailing shape.

        Returns
        -------
        dict with same keys as template_metrics() but each value has shape
        ``template_matrix.shape[1:]``:
            z_score, template_norm, z_white_noise, white_template_norm,
            recovery_fraction, relative_capacity
        """
        if template_matrix.shape[0] != len(self.flux):
            raise ValueError(
                f"template_matrix.shape[0] ({template_matrix.shape[0]}) must equal "
                f"len(self.flux) ({len(self.flux)})."
            )
        output_shape = template_matrix.shape[1:]
        flat = template_matrix.reshape(len(self.flux), -1)
        projections, template_norms = self._bank_stats(flat)
        variance = np.maximum(self.gp._diag, self._EPS)
        white_template_norms = np.sqrt(
            np.maximum(
                np.einsum("ij,ij->j", flat, flat / variance[:, None]),
                self._EPS,
            )
        )
        z_score = projections / template_norms
        z_white_noise = (self.flux / variance) @ flat / white_template_norms
        recovery_fraction = np.maximum(z_score, 0.0) / white_template_norms
        relative_capacity = template_norms / white_template_norms
        return {
            "z_score": z_score.reshape(output_shape),
            "template_norm": template_norms.reshape(output_shape),
            "z_white_noise": z_white_noise.reshape(output_shape),
            "white_template_norm": white_template_norms.reshape(output_shape),
            "recovery_fraction": recovery_fraction.reshape(output_shape),
            "relative_capacity": relative_capacity.reshape(output_shape),
        }

    def log_bayes_factor_matrix(
        self,
        template_matrix: np.ndarray,
        sigma_a: float,
        prior: str = "gaussian",
        amplitude_bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Batch log Bayes factors for a pre-materialized template array.

        Parameters
        ----------
        template_matrix : ndarray, shape (N, M) or (N, M_dur, t_transit)
            For multi-dimensional template spaces pass a 3-D array; output
            preserves the trailing shape.
        sigma_a : float
            Standard deviation of the amplitude prior.
        prior : str
            "gaussian", "half-normal", or "uniform".
        amplitude_bounds : tuple[float, float] | None
            Required when prior="uniform".

        Returns
        -------
        ndarray, shape ``template_matrix.shape[1:]``
            Log Bayes factor for each template.
        """
        output_shape = template_matrix.shape[1:]
        flat = template_matrix.reshape(len(self.flux), -1)
        projections, template_norms = self._bank_stats(flat)
        log_bf = MatchedFilter._log_bayes_factor_from_projection(
            projections,
            template_norms,
            sigma_a,
            prior=prior,
            amplitude_bounds=amplitude_bounds,
        )
        return log_bf.reshape(output_shape)

    def compute_sensitivity_limits(
        self, template_or_matrix: np.ndarray, z_threshold: float
    ) -> dict:
        """Minimum detectable transit depth under ideal and realization-adjusted conditions.

        Parameters
        ----------
        template_or_matrix : ndarray, shape (N,) or (N, M)
            Single template (1D) or pre-materialized matrix (2D).
        z_threshold : float
            Target detection significance.

        Returns
        -------
        dict with keys:
            ideal : scalar or (M,) array
                z_threshold / sigma_s — the depth needed assuming zero background noise.
            realization_adjusted : scalar or (M,) array
                (z_threshold - Z_bg) / sigma_s — additional depth needed given the
                current noise fluctuation at each epoch.
        """
        if template_or_matrix.ndim == 1:
            z_bg, template_norm = self.z_score(template_or_matrix)
        else:
            projections, template_norms = self._bank_stats(template_or_matrix)
            z_bg = projections / template_norms
            template_norm = template_norms
        return {
            "ideal": z_threshold / template_norm,
            "realization_adjusted": (z_threshold - z_bg) / template_norm,
        }

    def white_noise_template_norm(self, template: np.ndarray) -> float:
        """Compute the precision-weighted template norm under a white-noise approximation."""
        variance = np.maximum(self.gp._diag, self._EPS)
        return np.sqrt(np.sum(template**2 / variance))

    def white_noise_z_score(
        self, template: np.ndarray, baseline: float = 0.0, inject_template: bool = False
    ) -> tuple[float, float]:
        r"""
        Matched-filter Z-score under a diagonal (white-noise) approximation.

        Uses the GP's diagonal variance estimate (`self.gp._diag`) as the
        per-sample variance and computes the matched-filter projection
        $$Z = \frac{(y - baseline)^T C^{-1} s}{\sqrt{s^T C^{-1} s}}$$ where
        $y$ is the stored flux and `s` is `template`.

        Parameters
        ----------
        template : ndarray
            Template vector (same length as `self.flux`).
        baseline : float, optional
            Null-level subtracted from the data.
        inject_template : bool, optional
            If True, evaluate on `y + template`.

        Returns
        -------
        z_score : float
            Matched-filter Z-score under the diagonal-noise assumption.
        template_norm : float
            The precision-weighted template norm (sqrt(s^T C^{-1} s)).
        """
        variance = np.maximum(self.gp._diag, self._EPS)
        inv_var = 1.0 / variance

        precision_weighted_template = template * inv_var

        template_norm = np.sqrt(
            np.maximum(np.dot(template, precision_weighted_template), self._EPS)
        )

        flux = self.flux + template if inject_template else self.flux

        numerator = np.dot(flux - baseline, precision_weighted_template)
        z_score = numerator / template_norm

        return z_score, template_norm


@dataclass(frozen=True)
class SearchProfile:
    """A unified diagnostic container for 1D matched-filter search profiles.

    Can hold pure frequentist metrics, pure Bayesian metrics, or a joint evaluation.
    Properties are lazy-evaluated to prevent unnecessary memory or compute overhead.
    """

    epochs: np.ndarray

    # Optional parameters depending on which scan paradigm was executed
    z_score: np.ndarray | None = None
    template_norm: np.ndarray | None = None
    log_bayes_factor: np.ndarray | None = None
    log_bayes_factor_global: float | None = None
    log_evidence_marg: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Strict validation only on the arrays that are actively provided
        n_epochs = len(self.epochs)
        if self.z_score is not None and len(self.z_score) != n_epochs:
            raise ValueError("Dimensions of epochs and Z-scores must match.")
        if self.template_norm is not None and len(self.template_norm) != n_epochs:
            raise ValueError("Dimensions of epochs and template norms must match.")
        if self.log_bayes_factor is not None and len(self.log_bayes_factor) != n_epochs:
            raise ValueError("Dimensions of epochs and Log Bayes Factors must match.")
        if (
            self.log_evidence_marg is not None
            and len(self.log_evidence_marg) != n_epochs
        ):
            raise ValueError("Dimensions of epochs and log evidence (marg) must match.")

    @property
    def best_fit_scale(self) -> np.ndarray:
        """The Maximum Likelihood Estimate (MLE) of the template norm profile."""
        if self.z_score is None or self.template_norm is None:
            raise AttributeError(
                "best_fit_scale requires both z_score and template_norm."
            )
        return self.z_score / self.template_norm

    @property
    def log_likelihood_ratio(self) -> np.ndarray:
        """The log-likelihood ratio (Delta ln L) timeline profile."""
        if self.z_score is None:
            raise AttributeError("log_likelihood_ratio requires z_score.")
        return 0.5 * np.maximum(self.z_score, 0) ** 2

    def strongest_frequentist_match(self) -> dict[str, Any]:
        if self.z_score is None:
            raise AttributeError(
                "No frequentist Z-score data available in this profile."
            )
        idx = int(np.nanargmax(self.z_score))
        return {
            "index": idx,
            "epoch": float(self.epochs[idx]),
            "value": float(self.z_score[idx]),
            "metric": "Z-score",
        }

    def strongest_bayesian_match(self) -> dict[str, Any]:
        if self.log_bayes_factor is None:
            raise AttributeError(
                "No Bayesian log Bayes factor data available in this profile."
            )
        idx = int(np.nanargmax(self.log_bayes_factor))
        return {
            "index": idx,
            "epoch": float(self.epochs[idx]),
            "value": float(self.log_bayes_factor[idx]),
            "metric": "ln B10",
        }

    def plot(
        self,
        threshold_z: float | None = 3.0,
        threshold_ln_bf: float | None = 5.0,
        axes=None,
    ) -> Any:
        """Generates a perfectly synchronized layout matching your diagnostic needs."""
        import matplotlib.pyplot as plt

        has_bf = self.log_bayes_factor is not None
        has_z = self.z_score is not None

        if not (has_bf or has_z):
            raise ValueError("Profile contains no data arrays to plot.")

        n_rows = int(has_bf) + int(has_z)
        if axes is None:
            fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=True)
        else:
            if hasattr(axes, "figure"):
                fig = axes.figure
            elif isinstance(axes, (list, np.ndarray)) and len(axes) > 0 and hasattr(axes[0], "figure"):
                fig = axes[0].figure
            else:
                fig = plt.gcf()
        ax_list = [axes] if n_rows == 1 else list(axes)

        curr_ax = 0
        # Panel 1: Bayesian Evidence Map
        if has_bf:
            ax = ax_list[curr_ax]
            ax.plot(
                self.epochs,
                self.log_bayes_factor,
                color="black",
                lw=1.5,
                label=r"$\ln B_{10}$",
            )
            if threshold_ln_bf is not None:
                ax.axhline(
                    threshold_ln_bf,
                    color="teal",
                    ls="-.",
                    label=f"Strong Evidence ({threshold_ln_bf})",
                )
            ax.set_ylabel(r"$\ln B_{10}$")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            curr_ax += 1

        # Panel 2: Frequentist Map
        if has_z:
            ax = ax_list[curr_ax]
            ax.plot(
                self.epochs,
                self.z_score,
                color="crimson",
                lw=1.5,
                label="MLE $Z$-score",
            )
            if threshold_z is not None:
                ax.axhline(
                    threshold_z,
                    color="orange",
                    ls="--",
                    label=rf"{threshold_z}$\sigma$ Threshold",
                )
            ax.set_ylabel("Matched Filter Z-score")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        ax_list[-1].set_xlabel(r"Transit search epoch time ($t_0$)")
        plt.tight_layout()
        return fig, ax_list
