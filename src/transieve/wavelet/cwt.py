"""Continuous wavelet transform (CWT) diagnostics for light-curve analysis.

Provides a thin, validated wrapper around ``pywt.cwt`` that returns structured
result objects with power, global spectrum, and per-time salience. A separate
transit-vetting path computes duration-matched CWT scales and evaluates
epoch-consistency, scale-consistency, edge-safety, and dip-sign flags against
a specific candidate epoch and duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

from ..utils import as_1d_array


@dataclass
class CWTResult:
    """Continuous-wavelet diagnostics for one light curve.

    Produced by `cwt_diagnostics`. Wraps the raw ``pywt.cwt`` output with
    convenience properties for power, global spectrum, and per-time salience.

    Attributes
    ----------
    time:
        Time array of length N passed to `cwt_diagnostics`.
    scales:
        1-D array of CWT scales, shape ``(S,)``.
    coefficients:
        Complex CWT coefficient array, shape ``(S, N)``.
    frequencies:
        Pseudo-frequencies corresponding to each scale (in 1/time_unit),
        shape ``(S,)``. Computed by PyWavelets from the scale grid and the
        sampling period.
    wavelet:
        PyWavelets continuous-wavelet name used for the decomposition.
    """

    time: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    frequencies: np.ndarray
    wavelet: str

    @property
    def power(self) -> np.ndarray:
        """CWT power map ``|coefficients|²``, shape ``(S, N)``."""
        return np.abs(self.coefficients) ** 2

    @property
    def global_spectrum(self) -> np.ndarray:
        """Time-averaged power at each scale, shape ``(S,)``."""
        return np.nanmean(self.power, axis=1)

    @property
    def salience(self) -> np.ndarray:
        """Per-time salience: maximum power across all scales, shape ``(N,)``."""
        return np.nanmax(self.power, axis=0)

    def strongest_match(self) -> tuple[int, float]:
        """Return the time index and salience value of the peak power.

        Returns
        -------
        tuple[int, float]
            ``(index, salience[index])`` of the maximum salience sample.
        """
        idx = int(np.nanargmax(self.salience))
        return idx, float(self.salience[idx])


@dataclass
class CWTTransitVettingResult:
    """Transit-focused CWT diagnostics around an expected epoch and duration.

    Produced by `cwt_transit_vetting`. Bundles the full `CWTResult` with
    localised vetting flags that compare the CWT response near the expected
    transit epoch against global behaviour.

    Attributes
    ----------
    cwt:
        Full `CWTResult` computed over the entire light curve.
    epoch:
        Expected transit midpoint time (same units as ``cwt.time``).
    duration:
        Expected transit duration (same units as ``cwt.time``).
    cadence:
        Median cadence inferred from ``cwt.time`` (same units as ``cwt.time``).
    local_idx:
        Index of the peak salience sample within the local epoch window.
    global_idx:
        Index of the peak salience sample across the entire light curve.
    local_salience:
        Salience at ``local_idx``.
    global_salience:
        Salience at ``global_idx``.
    best_scale:
        CWT scale (in samples) with the strongest response at ``local_idx``.
    best_timescale:
        ``best_scale * cadence`` — the dominant timescale in time units.
    epoch_consistent:
        True if ``|time[local_idx] - epoch| <= duration``.
    scale_consistent:
        True if ``best_timescale`` falls within the duration bracket
        defined by ``scale_multipliers``.
    edge_safe:
        True if ``local_idx`` is sufficiently far from both array edges
        (controlled by ``edge_guard_duration``).
    dip_sign_consistent:
        True if the median flux near ``local_idx`` is below the global
        median — consistent with a transit dip rather than a brightening.
    """

    cwt: CWTResult
    epoch: float
    duration: float
    cadence: float
    local_idx: int
    global_idx: int
    local_salience: float
    global_salience: float
    best_scale: float
    best_timescale: float
    epoch_consistent: bool
    scale_consistent: bool
    edge_safe: bool
    dip_sign_consistent: bool

    @property
    def local_time(self) -> float:
        return float(self.cwt.time[self.local_idx])

    @property
    def global_time(self) -> float:
        return float(self.cwt.time[self.global_idx])

    @property
    def local_to_global_ratio(self) -> float:
        den = max(float(self.global_salience), 1e-16)
        return float(self.local_salience / den)


def default_cwt_scales(
    n_samples: int,
    min_scale: float = 1.0,
    max_scale: float | None = None,
    n_scales: int = 32,
) -> np.ndarray:
    """Create a logarithmically-spaced scale grid for CWT diagnostics.

    Generates ``n_scales`` scales in geometric progression from ``min_scale``
    to ``max_scale`` (defaulting to ``sqrt(n_samples)``). Geometric spacing
    gives equal relative resolution at every scale, which is appropriate for
    signals whose characteristic width varies over orders of magnitude.

    Parameters
    ----------
    n_samples:
        Number of samples in the light curve. Used to set the default
        upper scale limit.
    min_scale:
        Smallest scale (in samples). Must be positive. Default 1.0.
    max_scale:
        Largest scale (in samples). Defaults to ``sqrt(n_samples)`` when
        not provided.
    n_scales:
        Number of scales in the grid. Must be >= 2.

    Returns
    -------
    np.ndarray
        1-D array of scale values, shape ``(n_scales,)``.

    Raises
    ------
    ValueError
        If ``n_samples < 8``, ``min_scale <= 0``, ``n_scales < 2``, or
        ``max_scale <= min_scale``.
    """
    if n_samples < 8:
        raise ValueError("CWT requires at least 8 samples.")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive.")
    if n_scales < 2:
        raise ValueError("n_scales must be >= 2.")

    upper = np.sqrt(n_samples) if max_scale is None else float(max_scale)
    if upper <= min_scale:
        raise ValueError("max_scale must be greater than min_scale.")

    return np.geomspace(float(min_scale), upper, int(n_scales))


def duration_matched_cwt_scales(
    duration: float,
    cadence: float,
    multipliers: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> np.ndarray:
    """Build a compact CWT scale grid tied to an expected transit duration.

    Converts the physical duration to a scale in samples (``duration /
    cadence``) and applies each multiplier to bracket the expected transit
    width. Useful when calling `cwt_diagnostics` for focused vetting of a
    specific candidate rather than a blind survey.

    Parameters
    ----------
    duration:
        Expected transit duration in the same time units as ``cadence``.
    cadence:
        Median cadence (time per sample).
    multipliers:
        Scale factors applied to ``duration / cadence``. The default
        ``(0.5, 1.0, 1.5)`` brackets the expected duration from half to
        one-and-a-half times.

    Returns
    -------
    np.ndarray
        Sorted, unique scale array (in samples). At least two elements are
        guaranteed: if all multipliers collapse to the same scale, a second
        scale at ``1.5×`` is appended.

    Raises
    ------
    ValueError
        If ``duration <= 0``, ``cadence <= 0``, ``multipliers`` is empty,
        or any multiplier is non-positive.
    """
    duration = float(duration)
    cadence = float(cadence)

    if duration <= 0:
        raise ValueError("duration must be positive.")
    if cadence <= 0:
        raise ValueError("cadence must be positive.")
    if len(multipliers) == 0:
        raise ValueError("multipliers must contain at least one value.")

    mult = np.asarray(multipliers, dtype=float)
    if np.any(mult <= 0):
        raise ValueError("all multipliers must be positive.")

    n_samples = duration / cadence
    scales = np.maximum(mult * n_samples, 1.0)
    scales = np.unique(np.sort(scales.astype(float)))

    if scales.size == 1:
        # CWT benefits from at least two scales for local scale context.
        scales = np.array([scales[0], scales[0] * 1.5], dtype=float)

    return scales


def cwt_diagnostics(
    time: np.ndarray,
    flux: np.ndarray,
    scales: np.ndarray | None = None,
    wavelet: str = "morl",
    center_flux: bool = True,
) -> CWTResult:
    """Compute CWT coefficients and salience diagnostics for a light curve.

    Wraps ``pywt.cwt`` with input validation, automatic scale selection,
    and an optional median-centering step. The resulting `CWTResult` exposes
    derived quantities (power, global spectrum, salience) as properties.

    Unlike the SWT-based functions, this routine does **not** interpolate
    NaN values — both ``time`` and ``flux`` must be finite.

    Parameters
    ----------
    time:
        Strictly increasing time array, length N. Must be finite and have
        positive cadence.
    flux:
        Flux array of length N. Must be finite (no NaNs).
    scales:
        CWT scale grid. If None, `default_cwt_scales` is called with
        defaults (32 log-spaced scales from 1 to ``sqrt(N)``).
    wavelet:
        PyWavelets continuous-wavelet name. Default ``"morl"`` (Morlet),
        which is well-suited to localised oscillatory features. For
        symmetric dip detection, ``"mexh"`` (Mexican hat) is a common
        alternative.
    center_flux:
        If True, subtract the median flux before computing the CWT. This
        removes a DC offset that would otherwise dominate the low-frequency
        scales.

    Returns
    -------
    CWTResult
        Structured result with coefficients, power, and salience.

    Raises
    ------
    ValueError
        If ``time`` and ``flux`` have different lengths, either contains
        NaN, or the cadence is non-positive or non-finite. Also raised if
        any provided scale is non-positive.
    """
    time = as_1d_array(time).astype(float)
    flux = as_1d_array(flux).astype(float)

    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length.")
    if np.any(np.isnan(time)):
        raise ValueError("time contains NaN values.")
    if np.any(np.isnan(flux)):
        raise ValueError("flux contains NaN values.")

    if center_flux:
        flux = flux - np.nanmedian(flux)

    dt = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time must be strictly increasing with finite cadence.")

    if scales is None:
        scales = default_cwt_scales(len(time))
    else:
        scales = as_1d_array(scales).astype(float)
        if np.any(scales <= 0):
            raise ValueError("scales must all be positive.")

    coeffs, freqs = pywt.cwt(flux, scales, wavelet, sampling_period=dt)

    return CWTResult(
        time=time,
        scales=np.asarray(scales, dtype=float),
        coefficients=np.asarray(coeffs),
        frequencies=np.asarray(freqs, dtype=float),
        wavelet=wavelet,
    )


def cwt_transit_vetting(
    time: np.ndarray,
    flux: np.ndarray,
    epoch: float,
    duration: float,
    wavelet: str = "mexh",
    scale_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5),
    local_window_duration: float = 1.5,
    edge_guard_duration: float = 1.0,
    center_flux: bool = True,
) -> CWTTransitVettingResult:
    """Run transit-focused CWT vetting with duration-matched scales.

    Computes the CWT using scales bracketing the expected transit duration
    (via `duration_matched_cwt_scales`) and evaluates a set of boolean
    vetting flags. The flags test whether the strongest local response is
    epoch-consistent, scale-consistent, away from the array edges, and of
    the correct (dip) sign.

    This is a targeted companion to `cwt_diagnostics`: use `cwt_diagnostics`
    for exploratory surveys and this function when a specific candidate
    epoch and duration are already known.

    Parameters
    ----------
    time:
        Strictly increasing time array, length N (no NaNs).
    flux:
        Flux array, length N (no NaNs).
    epoch:
        Expected transit midpoint time.
    duration:
        Expected transit duration (same time units as ``time``).
    wavelet:
        PyWavelets continuous-wavelet name. Default ``"mexh"`` (Mexican hat)
        is symmetric and well-matched to transit-like dips.
    scale_multipliers:
        Multipliers applied to ``duration / cadence`` to bracket the
        expected transit timescale. Passed to `duration_matched_cwt_scales`.
    local_window_duration:
        Half-width of the local epoch search window, in units of ``duration``.
    edge_guard_duration:
        Minimum separation (in units of ``duration``) required between the
        local peak and the array edges for ``edge_safe = True``.
    center_flux:
        If True, subtract the median flux before computing the CWT.

    Returns
    -------
    CWTTransitVettingResult
        Structured result with vetting flags and localised diagnostics.

    Raises
    ------
    ValueError
        If ``time`` and ``flux`` have different lengths, fewer than 8
        samples, or an invalid cadence.
    """
    time = as_1d_array(time).astype(float)
    flux = as_1d_array(flux).astype(float)

    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length.")
    if len(time) < 8:
        raise ValueError("CWT vetting requires at least 8 samples.")

    dt = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time must be strictly increasing with finite cadence.")

    scales = duration_matched_cwt_scales(
        duration=float(duration),
        cadence=dt,
        multipliers=scale_multipliers,
    )

    cwt = cwt_diagnostics(
        time=time,
        flux=flux,
        scales=scales,
        wavelet=wavelet,
        center_flux=center_flux,
    )

    idx_epoch = int(np.argmin(np.abs(time - float(epoch))))
    half_window = max(
        1,
        int(np.ceil(float(local_window_duration) * float(duration) / dt)),
    )
    lo = max(0, idx_epoch - half_window)
    hi = min(len(time), idx_epoch + half_window + 1)

    local_idx = lo + int(np.nanargmax(cwt.salience[lo:hi]))
    global_idx, global_sal = cwt.strongest_match()

    per_scale_local = cwt.power[:, local_idx]
    best_scale_idx = int(np.nanargmax(per_scale_local))
    best_scale = float(cwt.scales[best_scale_idx])
    best_timescale = float(best_scale * dt)

    local_sal = float(cwt.salience[local_idx])
    duration_low = float(np.min(scale_multipliers)) * float(duration)
    duration_high = float(np.max(scale_multipliers)) * float(duration)

    # For vetting, require local response near expected epoch and duration,
    # avoid edge-dominated events, and verify dip-like sign in local flux.
    epoch_consistent = bool(abs(time[local_idx] - float(epoch)) <= float(duration))
    scale_consistent = bool(duration_low <= best_timescale <= duration_high)

    edge_guard = max(1, int(np.ceil(float(edge_guard_duration) * float(duration) / dt)))
    edge_safe = bool(edge_guard <= local_idx < (len(time) - edge_guard))

    sign_half = max(1, int(np.ceil(0.5 * float(duration) / dt)))
    slo = max(0, local_idx - sign_half)
    shi = min(len(flux), local_idx + sign_half + 1)
    local_mean_flux = float(np.nanmean(flux[slo:shi]))
    dip_sign_consistent = bool(local_mean_flux < np.nanmedian(flux))

    return CWTTransitVettingResult(
        cwt=cwt,
        epoch=float(epoch),
        duration=float(duration),
        cadence=dt,
        local_idx=int(local_idx),
        global_idx=int(global_idx),
        local_salience=local_sal,
        global_salience=float(global_sal),
        best_scale=best_scale,
        best_timescale=best_timescale,
        epoch_consistent=epoch_consistent,
        scale_consistent=scale_consistent,
        edge_safe=edge_safe,
        dip_sign_consistent=dip_sign_consistent,
    )
