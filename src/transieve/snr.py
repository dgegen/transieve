import numpy as np
from dataclasses import dataclass
from typing import List, Sequence
from scipy.stats import median_abs_deviation

from .bin import bin_light_curve


def snr_delta_chi2(flux, injection, flux_err):
    r"""
    SNR computed from the delta-chi-squared formulation (Kipping 2023, Eq. 3).

    Returns $\sqrt{\Delta\chi^2}$ for the comparison between the null
    model (flat baseline at 1.0) and the model including ``injection``::

        SNR = sqrt( sum_i [ (f_i - 1 - s_i)^2 - (f_i - 1)^2 ] / sigma_i^2 )

    where ``f`` = ``flux``, ``s`` = ``injection``, and ``sigma`` = ``flux_err``.

    Parameters
    ----------
    flux : array_like, shape (N,)
        Observed normalized flux values (baseline ≈ 1.0).
    injection : array_like, shape (N,)
        Transit template (model) to be tested.
    flux_err : array_like, shape (N,)
        Per-sample standard deviations used in chi-squared sums.

    Returns
    -------
    float
        SNR computed as $\sqrt{\Delta\chi^2}$. If the summed argument is
        negative due to numerical issues, the returned value will be ``nan``
        or raise a runtime warning from ``np.sqrt``.

    References
    ----------
    Kipping (2023), "The SNR of a transit" (Eq. 3).
    """
    return np.sqrt(
        np.sum(((flux - 1 - injection) ** 2 - (flux - 1) ** 2) / (flux_err**2))
    )


def calculate_average_red_noise_coefficient(
    time: np.ndarray,
    residuals: np.ndarray,
    bin_times: np.ndarray = np.linspace(0.25, 0.41, 20),
    flux_median: bool = False,
):
    """
    Calculate the average red noise coefficient (beta) over different bin durations for a given time series.

    Parameters
    ----------
    time : array-like
        Time values of the time series.

    residuals : array-like
        Residuals of the time series data.

    bin_times : array-like, optional
        Array of bin durations in hours for which to calculate beta.
        Default is np.linspace(0.25, 0.41, 20).

    flux_median : bool, optional
        If True, calculate median instead of mean for bin values.
        Default is False.

    mid_bin_time : bool, optional
        If True, use mid-bin time for bin calculations.
        Default is False.

    n_min : int, optional
        Minimum number of data points per bin for calculation.
        Default is 4.

    Returns
    -------
    mean_beta : float
        The average red noise coefficient (beta) calculated for each bin duration.

    See Also
    –-------
    mixed_transit_analysis.utils.calculate_red_noise_coefficient
        The function used for calculating red noise coefficients for a single bin duration.
    """
    mean_beta = np.mean(
        calculate_multiple_red_noise_coefficients(
            time=time,
            residuals=residuals,
            bin_times=bin_times,
            flux_median=flux_median,
        )
    )
    return mean_beta


def calculate_multiple_red_noise_coefficients(
    time: np.ndarray,
    residuals: np.ndarray,
    bin_times: np.ndarray = np.linspace(0.25, 0.41, 20),
    flux_median: bool = False,
) -> List[float]:
    """
    Calculate red noise coefficient (beta) for different bin times for a given time series.

    Parameters
    ----------
    time : array-like
        Time values of the time series.

    residuals : array-like
        Residuals of the time series data.

    bin_times : array-like, optional
        Array of bin durations in hours for which to calculate beta.
        Default is np.linspace(0.25, 0.41, 20).

    flux_median : bool, optional
        If True, calculate median instead of mean for bin values.
        Default is False.

    mid_bin_time : bool, optional
        If True, use mid-bin time for bin calculations.
        Default is False.

    n_min : int, optional
        Minimum number of data points per bin for calculation.
        Default is 4.

    Returns
    -------
    list
        A list of red noise coefficients (betas) calculated for each bin duration.

    See Also
    –-------
    mixed_transit_analysis.utils.calculate_red_noise_coefficient
        The function used for calculating red noise coefficients for a single bin duration.
    """
    betas = [
        calculate_red_noise_coefficient(
            time,
            residuals,
            bin_duration_hours=bin_time,
            flux_median=flux_median,
        )
        for bin_time in bin_times
    ]
    return betas


def calculate_red_noise_coefficient(
    time: np.ndarray,
    residuals: np.ndarray,
    bin_duration_hours: float = 1 / 3,
    flux_median: bool = False,
) -> float:
    """
    Calculate the quality beta coefficient for red noise (beta) in time series data.

    Parameters
    ----------
    time : array-like
        Time values of the time series.

    residuals : array-like
        Residuals of the time series data.

    bin_duration_hours : float, optional
        Duration in hours for creating data bins. Default is 1/3 hours.

    flux_median : bool, optional
        If True, calculate the median instead of the mean for bin values. Default is False.

    mid_bin_time : bool, optional
        If True, use the mid-bin time for bin calculations. Default is False.

    n_min : int, optional
        Minimum number of data points required per bin for calculation. Default is 4.

    Returns
    -------
    float
        The quality beta coefficient, which quantifies the presence of red noise in the data.
        A higher beta value indicates a stronger presence of red noise.

    Notes
    -----
    The most relevant timescale is 20 minutes, i.e. the ingress or egress duration.

    This function does the following
    1. Calculate sigma_1, the standard deviation of the unbinned out-of-transit data.
    2. Average the out-of-transit data into bins of bin_duration_hours, with each bin consisting of
        N data points, depending on the cadence.
    3. Then calculate the standard deviation sigma_N.
    In the absence of red noise, sigma_N ~ sigma_1/sqrt(N), but in practice,
    sigma_N > sigma_1/sqrt(N) or equivalently sigma_N = beta * sigma_1/sqrt(N)
    with beta >= 1 as N as well as the length of the time series goes to infinity.

    Typically, beta is around 2.

    Source:
    J. N. Winn et. al. (2007): THE TRANSIT LIGHT CURVE PROJECT. VII. THE NOT-SO-BLOATED EXOPLANET HAT-P-1b
    """
    # Calculate the standard deviation of the unbinned out-of-transit data
    sigma_1 = residuals.std()

    # Average the out-of-transit data into bins using `bin_light_curve`.
    # `bin_light_curve` expects bin_time in days, so convert hours→days.
    method = "median" if flux_median else "mean"
    bin_time = bin_duration_hours / 24.0
    _, binned_residuals, _ = bin_light_curve(
        time=time,
        flux=residuals,
        flux_err=None,
        bin_time=bin_time,
        return_error=False,
        method=method,
    )
    # Calculate the standard deviation
    sigma_N = binned_residuals.std()

    # Average number of data_points per bin
    N = residuals.size / binned_residuals.size

    # As sigma_N = sigma_1/sqrt(N) * beta, solving for beta delivers
    beta = np.sqrt(N) * sigma_N / sigma_1
    return beta


def calculate_v_n(
    time: np.ndarray,
    residuals: np.ndarray,
    duration: float,
    step: float | None = None,
    min_count: int = 2,
) -> dict[int, float]:
    """Pont et al. (2006) sliding-window variance estimator V(n).

    Slides a window of width ``duration`` across ``time`` in steps smaller
    than the sampling interval. For each window position j, the mean
    residual F_j and the actual number of points n_j it contains are
    recorded. The F_j are then grouped by n_j and the variance within each
    group gives V(n).

    Parameters
    ----------
    time:
        Time array of the out-of-transit residuals (in-transit points and
        known systematics already removed by the caller).
    residuals:
        Out-of-transit flux residuals, same shape as ``time``.
    duration:
        Window duration l (typically the transit duration), in the same
        units as ``time``.
    step:
        Step size for sliding a uniform grid of window anchors. If given,
        windows are anchored on ``np.arange(time[0], time[-1], step)``.
        If ``None`` (the default), windows are instead anchored on every
        timestamp in ``time`` directly. A uniform grid finer than the
        cadence produces many windows whose ``(left_idx, right_idx)`` are
        identical to their neighbours (the window hasn't crossed a new data
        point yet), which artificially deflates the variance of F_j within
        an n-group. Anchoring on the data's own timestamps avoids this:
        each anchor is a real observation, so neighbouring windows differ
        whenever the data does.
    min_count:
        Minimum number of windows required to report V(n) for a given n.
        Groups below this threshold are omitted; callers fall back to the
        closest available n via ``_lookup_v_n``, or use
        ``calculate_v_n_for_mask`` for gap-aware estimation. Default 2.

    Returns
    -------
    dict[int, float]
        Mapping from n (number of points in a window) to V(n), the
        variance of the window means F_j among windows containing exactly
        n points. Groups that never reach ``min_count`` windows are omitted.

    References
    ----------
    Pont, Zucker & Queloz (2006), "The effect of red noise on planetary
    transit detection".
    """
    if step is None:
        grid = time
    else:
        grid = np.arange(time[0], time[-1], step)
    left_idx = np.searchsorted(time, grid - duration / 2.0, side="left")
    right_idx = np.searchsorted(time, grid + duration / 2.0, side="right")

    n_j = right_idx - left_idx
    valid_mask = n_j > 0
    n_j = n_j[valid_mask]
    left_idx = left_idx[valid_mask]
    right_idx = right_idx[valid_mask]

    cum_sum = np.concatenate(([0.0], np.cumsum(residuals)))
    F_j = (cum_sum[right_idx] - cum_sum[left_idx]) / n_j

    unique_n, counts = np.unique(n_j, return_counts=True)
    v_n = {}
    for un, count in zip(unique_n, counts):
        if count >= min_count:
            v_n[int(un)] = float(np.var(F_j[n_j == un], ddof=1))
    return v_n


def infer_gap_mask(
    time_in_window: np.ndarray, reference_cadence: float | None = None
) -> np.ndarray:
    """Infer which nominal cadence slots are missing within a window, from
    jumps in ``np.diff(time_in_window)`` relative to a reference cadence.

    Real instrumental/quality-flag gaps show up as a step larger than one
    cadence between two otherwise-present points. This reconstructs the
    boolean "present" pattern over the nominal (gap-free) grid the window
    would have had, without needing the raw, pre-extraction dataset.

    Parameters
    ----------
    time_in_window:
        Sorted timestamps of the actually-present points within one window
        (e.g. the in-transit cadences of a single transit).
    reference_cadence:
        Cadence to compare ``np.diff(time_in_window)`` against. If ``None``
        (the default), uses this window's own median cadence — fine in
        isolation, but if the resulting mask is meant to be reproduced on a
        much larger baseline (e.g. via `calculate_v_n_for_mask`), a per-window
        cadence estimated from only a handful of points can differ from the
        baseline's true cadence by enough to round differently, silently
        inflating the inferred slot count by one and making it impossible
        for any baseline window to ever qualify. Pass the baseline's own
        (much better sampled) cadence here to avoid that mismatch.

    Returns
    -------
    np.ndarray
        Boolean mask of length >= ``len(time_in_window)``, True where a
        cadence is present, False where it's inferred missing. ``mask.sum()
        == len(time_in_window)``; ``len(mask)`` is the nominal gap-free slot
        count.
    """
    if len(time_in_window) < 2:
        return np.ones(len(time_in_window), dtype=bool)
    dt = np.diff(time_in_window)
    cadence = reference_cadence if reference_cadence is not None else np.median(dt)
    n_skipped = np.maximum(np.round(dt / cadence).astype(int) - 1, 0)
    mask = [True]
    for ns in n_skipped:
        mask.extend([False] * int(ns))
        mask.append(True)
    return np.array(mask, dtype=bool)


def calculate_v_n_for_mask(
    time: np.ndarray,
    residuals: np.ndarray,
    duration: float,
    gap_mask: np.ndarray,
    step: float | None = None,
) -> tuple[float | None, int, int]:
    """Estimate V(n) for ``n = gap_mask.sum()`` by reproducing a specific
    missingness pattern on baseline windows, rather than matching on raw
    window-point-count alone.

    For every sliding window with at least ``len(gap_mask)`` points, take its
    first ``len(gap_mask)`` points and drop those at the positions
    ``~gap_mask`` marks missing, then average the rest. This exactly mimics
    a transit window whose own real cadences were missing at those same
    relative positions (see `infer_gap_mask`), which is more faithful than
    either pooling by raw count or assuming a contiguous/random drop pattern.

    If ``gap_mask`` is longer than the largest window this ``time``/``duration``
    combination ever actually produces (a boundary/edge-alignment artifact of
    reconstructing the nominal slot count from a handful of local points, even
    when the cadence estimate itself is correct), it's trimmed from the end
    down to that achievable size — at most a couple of points — rather than
    finding zero qualifying windows. The actual ``n`` solved for after any
    trimming is returned as the third element.

    Returns
    -------
    tuple[float | None, int, int]
        Variance of the masked window means (None if fewer than 2 windows
        have enough points to apply the mask), the number of windows that
        supported the estimate, and the actual n solved for.
    """
    if step is None:
        grid = time
    else:
        grid = np.arange(time[0], time[-1], step)
    left_idx = np.searchsorted(time, grid - duration / 2.0, side="left")
    right_idx = np.searchsorted(time, grid + duration / 2.0, side="right")

    n_j = right_idx - left_idx
    m = len(gap_mask)
    max_n_j = int(n_j.max()) if len(n_j) else 0
    if m > max_n_j:
        gap_mask = gap_mask[:max_n_j]
        m = max_n_j

    qualifying = np.where(n_j >= m)[0]
    n_used = int(gap_mask.sum())
    if len(qualifying) < 2 or m == 0:
        return None, len(qualifying), n_used

    idx_matrix = left_idx[qualifying][:, None] + np.arange(m)[None, :]
    values = residuals[idx_matrix]
    masked_values = values[:, gap_mask]
    f_j = masked_values.mean(axis=1)
    return float(np.var(f_j, ddof=1)), len(qualifying), n_used


def _lookup_v_n(v_n: dict[int, float], n: int, sigma_1: float) -> float:
    """Look up V(n) for a given n, falling back to the closest available n."""
    if n in v_n:
        return v_n[n]
    valid_keys = np.array(list(v_n.keys()))
    if len(valid_keys) > 0:
        closest_n = int(valid_keys[np.argmin(np.abs(valid_keys - n))])
        return v_n[closest_n]
    return (sigma_1**2) / n if n > 0 else 0.0


@dataclass
class RedNoiseSnrResult:
    snr_white: float
    snr_diff: float
    beta: float
    snr_red: float
    betas: tuple[float, ...]


def snr_red_noise(
    depth: float,
    time: np.ndarray,
    residuals: np.ndarray,
    n_in_transit: int | Sequence[int],
    bin_times: np.ndarray = np.linspace(0.25, 0.41, 20),
    flux_median: bool = False,
    duration: float | None = None,
    v_n: dict[int, float] | None = None,
) -> RedNoiseSnrResult:
    """SNR of a transit corrected for red noise (Winn et al. 2007 or Pont et al. 2006).

    Parameters
    ----------
    depth:
        Transit depth δ (dimensionless, e.g. ror²).
    time:
        Time array for the *out-of-transit* residuals.
    residuals:
        Out-of-transit flux residuals (in-transit points must be masked by
        the caller so that σ₁ is not inflated by the transit dip).
    n_in_transit:
        Number of in-transit cadences. When `duration` is given, this may be
        a sequence of n_k, the number of in-transit points of each
        individual transit k (e.g. for multiple transits with varying
        coverage); a scalar is treated as a single transit.
    bin_times:
        Bin durations in hours used to estimate β.
    flux_median:
        If True, use median binning when computing β.
    duration:
        Transit duration (in days) for Pont et al. 2006 sliding-window V(n) estimation.
        If provided, `bin_times` is ignored and the Pont et al. 2006 method is used.
    v_n:
        Optional precomputed V(n) mapping (see `calculate_v_n`) to use instead of
        estimating it from `time`/`residuals`. Useful when `residuals` only spans a
        short local chunk (e.g. a single sequence) but a longer baseline is available
        to estimate V(n) with less sampling noise; `sigma_1`/`snr_white` are still
        computed from the local `residuals`, only the V(n) lookup is overridden.

    Returns
    -------
    RedNoiseSnrResult
        snr_white = δ / σ₁ · √N_in  (white-noise limit, std of residuals)
        snr_diff  = δ / σ_diff · √N_in  (robust limit, MAD of consecutive diffs)
        beta      = effective red-noise coefficient, snr_white / snr_red
        snr_red   = S_r = δ·n / √(Σ_k n_k² V(n_k))  (Pont et al. 2006, Eq. 13)
        betas     = tuple of betas observed across the sliding-window n groups
    """
    n_k = np.atleast_1d(np.asarray(n_in_transit, dtype=float))
    n_total = float(np.sum(n_k))

    sigma_1 = float(np.std(residuals))
    snr_white = depth / sigma_1 * np.sqrt(n_total)

    diffs = np.diff(residuals)
    sigma_diff = float(median_abs_deviation(diffs, scale="normal") / np.sqrt(2))  # type: ignore[arg-type]
    snr_diff = depth / sigma_diff * np.sqrt(n_total)

    if duration is not None:
        # Pont et al. (2006) sliding-window V(n) estimation.
        if v_n is None:
            v_n = calculate_v_n(time, residuals, duration)

        v_n_k = np.array([_lookup_v_n(v_n, int(round(nk)), sigma_1) for nk in n_k])
        denom = float(np.sum((n_k**2) * np.maximum(0.0, v_n_k)))

        snr_red = depth * n_total / np.sqrt(denom) if denom > 0 else np.nan
        beta_val = snr_white / snr_red if snr_red and snr_red > 0 else np.nan
        betas_val = tuple(
            np.sqrt(un * max(0.0, v_n[un])) / sigma_1 for un in sorted(v_n.keys())
        )

        return RedNoiseSnrResult(
            snr_white=snr_white,
            snr_diff=snr_diff,
            beta=float(beta_val),
            snr_red=float(snr_red),
            betas=betas_val,
        )
    else:
        betas = calculate_multiple_red_noise_coefficients(
            time=time,
            residuals=residuals,
            bin_times=bin_times,
            flux_median=flux_median,
        )
        beta_val = np.mean(betas)
        betas_val = tuple(float(b) for b in betas)

    return RedNoiseSnrResult(
        snr_white=snr_white,
        snr_diff=snr_diff,
        beta=float(beta_val),
        snr_red=snr_white / beta_val if beta_val > 0 else np.nan,
        betas=betas_val,
    )
