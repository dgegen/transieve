from __future__ import annotations

import celerite2.driver as driver
import numpy as np
from celerite2 import GaussianProcess
from collections.abc import Callable
from dataclasses import dataclass, field
from scipy.signal import fftconvolve
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

    def z_score_map(
        self,
        template_bank: Callable[[float], np.ndarray] | TemplateBank,
    ) -> MatchedFilterStatistics:
        """Compute Z-scores for a template bank over a grid of epochs.

        Parameters
        ----------
        template_bank : callable or TemplateBank
            If callable, evaluated at every point in the GP's stored time array.
            Pass a :class:`TemplateBank` to use a custom epoch grid.

        Returns
        -------
        MatchedFilterStatistics
            Z-scores and template norms, one entry per epoch.
        """
        bank = TemplateBank._coerce(template_bank, default_epochs=self.gp._t)
        z_scores, template_norms = zip(*[self.z_score(t) for t in bank])
        return MatchedFilterStatistics(
            z_score=np.array(z_scores), template_norm=np.array(template_norms)
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

        return MatchedFilterStatistics(
            z_score=z_score, template_norm=template_norm * np.ones_like(z_score)
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


@dataclass
class MatchedFilterStatistics:
    """Results from a MatchedFilter."""

    THRESHOLD = 7.0

    z_score: np.ndarray
    template_norm: np.ndarray

    @property
    def best_fit_scale(self):
        """The Maximum Likelihood Estimate (MLE) of the template norm."""
        return self.z_score / self.template_norm

    @property
    def log_likelihood_ratio(self):
        """
        The log-likelihood ratio (Delta ln L) of the match.
        For Gaussian noise, this is 1/2 * Z^2.
        """
        # We usually care about the LLR of the positive match
        return 0.5 * np.maximum(self.z_score, 0) ** 2

    def get_detectability(self, threshold=None, robust=True):
        """
        Calculates depth-related thresholds based on a target Z-score.

        Returns
        -------
        depth_to_threshold : np.ndarray
            The additional depth needed to reach the threshold given
            current noise fluctuations.
        sensitivity : np.ndarray
            The depth required to achieve the threshold z-score
            under nominal noise (50% recovery probability).
        """
        if robust:
            mad = np.nanmedian(np.abs(self.z_score - np.nanmedian(self.z_score)))
            noise_floor = mad * 1.4826
        else:
            noise_floor = 1.0

        if threshold is None:
            threshold = MatchedFilterStatistics.THRESHOLD

        if noise_floor <= 0:
            raise ValueError("Noise floor must be positive for robust thresholding.")

        threshold *= noise_floor

        depth_to_threshold = (threshold - self.z_score) / self.template_norm

        sensitivity = threshold / self.template_norm

        return depth_to_threshold, sensitivity, float(threshold)

    def theoretical_fap(self):
        """Probability of finding at least one noise peak >= max Z-score by chance."""
        max_z_score = np.nanmax(self.z_score)
        n_trials = len(self.z_score)
        p_single = 1 - norm.cdf(max_z_score)
        return 1 - (1 - p_single) ** n_trials

    def empirical_significance(self, peak_index=None, window_size=None):
        """Percentile rank of the peak relative to the rest of the Z-score map."""
        if peak_index is None:
            peak_index = np.nanargmax(self.z_score)
        peak_value = self.z_score[peak_index]

        if window_size:
            mask = np.ones(len(self.z_score), dtype=bool)
            start = max(0, peak_index - window_size)
            end = min(len(self.z_score), peak_index + window_size)
            mask[start:end] = False
            null_dist = self.z_score[mask]
        else:
            null_dist = self.z_score

        p_value = np.sum(null_dist >= peak_value) / len(null_dist)
        z_score = (peak_value - np.mean(null_dist)) / np.std(null_dist)
        return {"p_value": p_value, "empirical_z": z_score}

    def strongest_match(self):
        """Return the index and value of the strongest match."""
        idx = np.nanargmax(self.z_score)
        return idx, self.z_score[idx]

    def __repr__(self):
        return (
            f"MatchedFilterStatistics(max_z_score={np.nanmax(self.z_score):.2f}, "
            f"mean_template_norm={np.nanmean(self.template_norm):.2f})"
        )

    def __str__(self):
        return self.__repr__()

    def plot(self, time=None, axes=None, threshold=None, **kwargs):
        import matplotlib.pyplot as plt

        depth_to_threshold, sensitivity, _ = self.get_detectability(threshold=threshold)

        if axes is None:
            _, axes = plt.subplots(3, 1, sharex=True)

        if time is None:
            time = np.arange(len(self.z_score))

        axes[0].plot(time, self.z_score, **kwargs)
        axes[1].plot(time, self.template_norm, **kwargs)
        axes[2].plot(time, depth_to_threshold, label="Depth to Threshold", **kwargs)
        axes[2].plot(time, sensitivity, label="Sensitivity", **kwargs)

        if np.any(sensitivity < 0):
            axes[2].axhline(0, color="black", linestyle="--")

        axes[0].set_ylabel("Z-score")
        axes[1].set_ylabel("Template norm")
        axes[2].set_ylabel("Sensitivity")
        axes[-1].set_xlabel("Time [days]")
        axes[2].legend()
