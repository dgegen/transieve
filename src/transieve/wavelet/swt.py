"""Stationary wavelet transform (SWT) matched filtering for transit detection.

Implements an undecimated, shift-invariant wavelet decomposition in which
each band is locally whitened by its sliding variance before cross-correlation
with a transit template. Two variants are provided: a simple band-summed
z-score statistic and a Kepler-style inverse-variance-weighted single-event
statistic (SES). A candidate evaluation routine converts raw z-score peaks into
a suite of significance and coherence diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

from ..utils import as_1d_array


@dataclass
class SWTChannel:
    """Per-channel SWT diagnostics used by the matched filter.

    Each channel corresponds to one decomposition level (detail or
    approximation) produced by `pywt.swt`. The channel stores both the
    raw wavelet coefficients and their locally-whitened counterparts
    used when accumulating the matched-filter statistic.

    Attributes
    ----------
    level:
        SWT decomposition level (1-indexed). Approximation channels are
        assigned ``level = swt_levels + 1``.
    kind:
        Either ``"detail"`` or ``"approx"``.
    coeffs:
        Raw SWT coefficients for this channel. Length equals N (the input
        series length) because SWT is shift-invariant.
    var:
        Local sliding variance of ``coeffs`` at each sample, estimated
        with a window that grows with decomposition level.
    whitened:
        ``coeffs / sqrt(var + eps)`` — variance-normalised coefficients.
    template:
        Variance-normalised template coefficients for this channel,
        divided by the same local scale as ``whitened``.
    window:
        Window length (in samples) passed to `sliding_variance` for this
        channel.
    """

    level: int
    kind: str
    coeffs: np.ndarray
    var: np.ndarray
    whitened: np.ndarray
    template: np.ndarray
    window: int


@dataclass
class SWTMatchedFilterResult:
    """Output of the SWT-based matched filter statistic.

    Produced by `wavelet_matched_filter`. The z-score is a shift-varying
    scalar that ranks how well the template matches the flux at each epoch.
    The denominator is constant across time (it depends only on the template
    energy), so the statistic is a pure cross-correlation ranking, not a
    proper position-varying SNR.

    Attributes
    ----------
    z_score:
        Matched-filter statistic at each sample epoch (length N). Larger
        absolute values indicate stronger template agreement.
    numerator:
        Accumulated cross-correlation sum across all whitened channels.
    denominator:
        Scalar template-energy term broadcast to length N. Used to
        normalise ``z_score``.
    channels:
        Per-level `SWTChannel` diagnostics. Length is ``swt_levels``
        (detail channels) plus 1 if the approximation channel is included.
    swt_levels:
        Number of SWT decomposition levels actually used (may be less than
        the requested value when N is too short for the full decomposition).
    wavelet:
        PyWavelets wavelet name used for the decomposition.
    """

    z_score: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    channels: list[SWTChannel]
    swt_levels: int
    wavelet: str

    @property
    def Z(self) -> np.ndarray:
        """Compatibility alias for notebook key name."""
        return self.z_score

    @property
    def N(self) -> np.ndarray:
        """Compatibility alias for notebook key name."""
        return self.numerator

    @property
    def D(self) -> np.ndarray:
        """Compatibility alias for notebook key name."""
        return self.denominator

    def strongest_match(self, absolute: bool = True) -> tuple[int, float]:
        """Return the index and value of the peak z-score.

        Parameters
        ----------
        absolute:
            If True, find the peak of ``|z_score|`` (two-sided). If False,
            find the maximum of the signed ``z_score`` (positive tail only).

        Returns
        -------
        tuple[int, float]
            ``(index, z_score[index])`` of the strongest match.
        """
        if absolute:
            idx = int(np.nanargmax(np.abs(self.z_score)))
        else:
            idx = int(np.nanargmax(self.z_score))
        return idx, float(self.z_score[idx])


@dataclass
class OWTSESResult:
    """Adaptive OWT-style single-event statistic (SES) time series.

    Produced by `kepler_owt_ses_filter`. Unlike `SWTMatchedFilterResult`,
    the denominator here is position-varying: each sample accumulates an
    inverse-variance-weighted template energy that accounts for local noise
    fluctuations, following the Kepler pipeline's SES construction.

    Attributes
    ----------
    ses:
        Single-event statistic at each sample epoch (length N). Equivalent
        to an inverse-variance-weighted matched-filter z-score.
    numerator:
        Sum of per-band inverse-variance-weighted cross-correlations.
    denominator:
        Sum of per-band inverse-variance-weighted template energies (varies
        with sample position). Used to normalise ``ses``.
    channels:
        Per-level `SWTChannel` diagnostics.
    swt_levels:
        Number of SWT decomposition levels actually used.
    wavelet:
        PyWavelets wavelet name used for the decomposition.
    template_support_samples:
        Number of samples in the original template before zero-padding,
        used to set the default edge exclusion in `strongest_match`.
    """

    ses: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    channels: list[SWTChannel]
    swt_levels: int
    wavelet: str
    template_support_samples: int

    @property
    def z_score(self) -> np.ndarray:
        """Compatibility alias for tooling expecting z_score-like arrays."""
        return self.ses

    @property
    def Z(self) -> np.ndarray:
        """Compatibility alias for notebook key name."""
        return self.ses

    @property
    def N(self) -> np.ndarray:
        """Compatibility alias for notebook key name."""
        return self.numerator

    @property
    def D(self) -> np.ndarray:
        """Compatibility alias for notebook key name."""
        return self.denominator

    def strongest_match(
        self,
        absolute: bool = True,
        edge_exclusion_samples: int | None = None,
    ) -> tuple[int, float]:
        metric = np.abs(self.ses) if absolute else self.ses

        if edge_exclusion_samples is None:
            edge_exclusion_samples = (
                self.template_support_samples // 2 if absolute else 0
            )

        edge_exclusion_samples = max(0, int(edge_exclusion_samples))
        if edge_exclusion_samples * 2 >= metric.size:
            edge_exclusion_samples = 0

        if edge_exclusion_samples > 0:
            masked_metric = metric.copy()
            masked_metric[:edge_exclusion_samples] = np.nan
            masked_metric[-edge_exclusion_samples:] = np.nan
            if np.all(np.isnan(masked_metric)):
                idx = int(np.nanargmax(metric))
            else:
                idx = int(np.nanargmax(masked_metric))
        else:
            idx = int(np.nanargmax(metric))

        return idx, float(self.ses[idx])


def sliding_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a symmetric moving variance with the same length as the input.

    Uses a running-sum approach (O(N) convolutions) rather than a sliding
    loop. The input is reflect-padded at both ends so edge samples receive
    a full-width estimate. Odd windows are enforced: even values are
    incremented by 1.

    Parameters
    ----------
    x:
        1-D input array.
    window:
        Number of samples in the moving window. Values <= 1 return the
        global variance as a constant array.

    Returns
    -------
    np.ndarray
        Moving variance, shape ``(len(x),)``. Values below ``1e-16`` are
        clipped to that floor to prevent division-by-zero in callers.
    """
    x = as_1d_array(x)

    if window <= 1:
        var = np.nanmean((x - np.nanmean(x)) ** 2)
        return np.full_like(x, var, dtype=float)

    if window % 2 == 0:
        window += 1

    pad = window // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(window, dtype=float)

    s1 = np.convolve(xp, kernel, mode="valid")
    s2 = np.convolve(xp * xp, kernel, mode="valid")

    mean = s1 / window
    mean_sq = s2 / window
    var = mean_sq - mean * mean
    var[var <= 1e-16] = 1e-16
    return var


def _fft_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la = len(a)
    lb = len(b)
    full_len = la + lb - 1
    nfft = 1 << (full_len - 1).bit_length()

    a_fft = np.fft.rfft(a, nfft)
    b_fft = np.fft.rfft(b, nfft)
    return np.fft.irfft(a_fft * b_fft, nfft)[:full_len]


def _fill_nan_by_interp(x: np.ndarray) -> np.ndarray:
    if not np.any(np.isnan(x)):
        return x

    nans = np.isnan(x)
    if np.all(nans):
        raise ValueError("Input flux cannot be all-NaN.")

    idx = np.arange(len(x))
    x = x.copy()
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def _fit_template_length(template: np.ndarray, n: int) -> np.ndarray:
    if len(template) == n:
        return template

    if len(template) < n:
        pad_left = (n - len(template)) // 2
        pad_right = n - len(template) - pad_left
        return np.pad(template, (pad_left, pad_right))

    start = (len(template) - n) // 2
    return template[start : start + n]


def wavelet_matched_filter(
    flux: np.ndarray,
    template: np.ndarray,
    wavelet: str = "db12",
    swt_levels: int = 5,
    base_window_samples: int = 128,
    eps: float = 1e-8,
    include_approx: bool = True,
) -> SWTMatchedFilterResult:
    """Compute an SWT-based matched-filter statistic across all wavelet bands.

    The flux and template are each decomposed using the stationary wavelet
    transform (SWT, also known as the *à trous* transform) so that all
    output channels have the same length N as the input. Each channel is
    locally whitened by its sliding variance, the whitened template is
    median-centred to remove baseline offsets, and the per-channel
    cross-correlations are summed into a single z-score time series.

    NaN samples in ``flux`` are replaced by linear interpolation before
    decomposition. The template is zero-padded or centre-cropped to match N.

    Parameters
    ----------
    flux:
        Observed flux time series (length N, NaNs allowed).
    template:
        Transit template with arbitrary length. Will be resized to N by
        symmetric zero-padding or centre-cropping.
    wavelet:
        PyWavelets wavelet name. The default ``"db12"`` provides smooth
        decomposition suitable for transit-like signals.
    swt_levels:
        Number of SWT decomposition levels requested. Silently capped at
        ``pywt.swt_max_level(N)`` when the series is too short.
    base_window_samples:
        Sliding-variance window at level 1. Doubled at each subsequent
        level (``window_i = base_window_samples * 2**i``) to track noise
        at the characteristic scale of each wavelet band.
    eps:
        Small constant added to the local standard deviation before
        division, preventing amplification of near-zero-variance regions.
    include_approx:
        If True, also include the coarsest approximation channel. This
        captures slow trends that survive all detail levels and can
        improve detection of very long-duration transits.

    Returns
    -------
    SWTMatchedFilterResult
        Structured result containing the z-score series, numerator,
        denominator, and per-channel diagnostics.

    Raises
    ------
    ValueError
        If ``flux`` has fewer than 4 samples, ``swt_levels < 1``, or N
        is incompatible with the SWT.
    """
    flux = _fill_nan_by_interp(as_1d_array(flux).astype(float))
    template = as_1d_array(template).astype(float)

    n = len(flux)
    if n < 4:
        raise ValueError("wavelet_matched_filter requires at least 4 samples.")
    if swt_levels < 1:
        raise ValueError("swt_levels must be >= 1.")

    template = _fit_template_length(template, n)

    max_level = pywt.swt_max_level(n)
    if max_level < 1:
        raise ValueError(
            "Input length is incompatible with SWT. Use a longer input series."
        )
    level = min(int(swt_levels), int(max_level))

    swt_flux = pywt.swt(flux, wavelet, level=level, start_level=0, axis=-1)
    swt_template = pywt.swt(template, wavelet, level=level, start_level=0, axis=-1)

    channels: list[SWTChannel] = []

    for i, (_, c_d) in enumerate(swt_flux, start=1):
        channels.append(
            SWTChannel(
                level=i,
                kind="detail",
                coeffs=c_d.copy(),
                var=np.zeros_like(c_d),
                whitened=np.zeros_like(c_d),
                template=np.zeros_like(c_d),
                window=1,
            )
        )

    if include_approx:
        channels.append(
            SWTChannel(
                level=level + 1,
                kind="approx",
                coeffs=swt_flux[-1][0].copy(),
                var=np.zeros_like(swt_flux[-1][0]),
                whitened=np.zeros_like(swt_flux[-1][0]),
                template=np.zeros_like(swt_flux[-1][0]),
                window=1,
            )
        )

    template_channels = [c_d.copy() for _, c_d in swt_template]
    if include_approx:
        template_channels.append(swt_template[-1][0].copy())

    for i, ch in enumerate(channels):
        window = int(base_window_samples * (2**i))
        window = max(3, min(window, n))
        if window % 2 == 0 and window > 1:
            window -= 1

        var = sliding_variance(ch.coeffs, window)
        scale = np.sqrt(var) + float(eps)

        ch.var = var
        ch.whitened = ch.coeffs / scale
        ch.template = template_channels[i] / scale
        ch.window = window

    z_num = np.zeros(n, dtype=float)
    z_den_scalar = 0.0

    for ch in channels:
        # Remove channel offsets so the statistic is driven by morphology,
        # not by low-frequency baseline drifts in SWT bands.
        xw = ch.whitened - np.nanmedian(ch.whitened)
        tw = ch.template - np.nanmedian(ch.template)

        conv = np.convolve(xw, tw[::-1], mode="same")
        z_num += conv

        z_den_scalar += float(np.dot(tw, tw))

    z_den_scalar = max(z_den_scalar, 1e-16)
    z_den = np.full(n, z_den_scalar, dtype=float)
    z = z_num / np.sqrt(z_den_scalar)

    return SWTMatchedFilterResult(
        z_score=z,
        numerator=z_num,
        denominator=z_den,
        channels=channels,
        swt_levels=level,
        wavelet=wavelet,
    )


def kepler_owt_ses_filter(
    flux: np.ndarray,
    template: np.ndarray,
    wavelet: str = "db12",
    owt_levels: int = 5,
    base_window_samples: int = 128,
    eps: float = 1e-8,
    include_approx: bool = False,
) -> OWTSESResult:
    """Compute a Kepler-style adaptive OWT/SWT single-event statistic (SES).

    Implements the inverse-variance-weighted wavelet matched filter described
    in Jenkins (2002) and used in the Kepler TPS pipeline. Each SWT band
    contributes a numerator term ``∑ (d_i * h_i) / σ_i²`` and a denominator
    term ``∑ h_i² / σ_i²``, where ``d_i`` is the whitened data, ``h_i`` the
    template, and ``σ_i²`` the local variance — all at band ``i``. The SES is
    ``numerator / sqrt(denominator)``.

    Unlike `wavelet_matched_filter`, the denominator varies with sample
    position, giving a proper locally-normalised statistic. The approximation
    channel is excluded by default because its slow-varying content is usually
    dominated by stellar variability rather than transit signal.

    NaN samples in ``flux`` are replaced by linear interpolation before
    decomposition. The template is zero-padded or centre-cropped to match N.

    Parameters
    ----------
    flux:
        Observed flux time series (length N, NaNs allowed).
    template:
        Transit template with arbitrary length. Will be resized to N by
        symmetric zero-padding or centre-cropping.
    wavelet:
        PyWavelets wavelet name. Default ``"db12"``.
    owt_levels:
        Number of SWT decomposition levels. Silently capped at the maximum
        allowed by N.
    base_window_samples:
        Sliding-variance window at level 1, doubled at each subsequent level.
    eps:
        Regularisation added to variance before inversion.
    include_approx:
        If True, include the coarsest approximation channel in the statistic.

    Returns
    -------
    OWTSESResult
        Structured result with the SES time series, numerator, denominator,
        and per-channel diagnostics.

    Raises
    ------
    ValueError
        If ``flux`` has fewer than 4 samples, ``owt_levels < 1``, or N is
        incompatible with the SWT.
    """
    flux = _fill_nan_by_interp(as_1d_array(flux).astype(float))
    template = as_1d_array(template).astype(float)
    template_support_samples = int(len(template))

    n = len(flux)
    if n < 4:
        raise ValueError("kepler_owt_ses_filter requires at least 4 samples.")
    if owt_levels < 1:
        raise ValueError("owt_levels must be >= 1.")

    template = _fit_template_length(template, n)

    max_level = pywt.swt_max_level(n)
    if max_level < 1:
        raise ValueError(
            "Input length is incompatible with SWT/OWT. Use a longer input series."
        )
    level = min(int(owt_levels), int(max_level))

    swt_flux = pywt.swt(flux, wavelet, level=level, start_level=0, axis=-1)
    swt_template = pywt.swt(template, wavelet, level=level, start_level=0, axis=-1)

    channels: list[SWTChannel] = []

    for i, (_, c_d) in enumerate(swt_flux, start=1):
        channels.append(
            SWTChannel(
                level=i,
                kind="detail",
                coeffs=c_d.copy(),
                var=np.zeros_like(c_d),
                whitened=np.zeros_like(c_d),
                template=np.zeros_like(c_d),
                window=1,
            )
        )

    if include_approx:
        channels.append(
            SWTChannel(
                level=level + 1,
                kind="approx",
                coeffs=swt_flux[-1][0].copy(),
                var=np.zeros_like(swt_flux[-1][0]),
                whitened=np.zeros_like(swt_flux[-1][0]),
                template=np.zeros_like(swt_flux[-1][0]),
                window=1,
            )
        )

    template_channels = [c_d.copy() for _, c_d in swt_template]
    if include_approx:
        template_channels.append(swt_template[-1][0].copy())

    ses_num = np.zeros(n, dtype=float)
    ses_den = np.zeros(n, dtype=float)

    for i, ch in enumerate(channels):
        # Adaptive per-band local noise model (Kepler-style weighting).
        window = int(base_window_samples * (2**i))
        window = max(3, min(window, n))
        if window % 2 == 0 and window > 1:
            window -= 1

        var = sliding_variance(ch.coeffs, window)
        inv_var = 1.0 / (var + float(eps))

        band_data = ch.coeffs - np.nanmedian(ch.coeffs)
        band_template = template_channels[i] - np.nanmedian(template_channels[i])

        num_i = np.convolve(band_data * inv_var, band_template[::-1], mode="same")
        den_i = np.convolve(inv_var, (band_template * band_template)[::-1], mode="same")

        ses_num += num_i
        ses_den += den_i

        ch.var = var
        ch.whitened = band_data / np.sqrt(var + float(eps))
        ch.template = band_template / np.sqrt(var + float(eps))
        ch.window = window

    ses_den = np.maximum(ses_den, 1e-16)
    ses = ses_num / np.sqrt(ses_den)

    return OWTSESResult(
        ses=ses,
        numerator=ses_num,
        denominator=ses_den,
        channels=channels,
        swt_levels=level,
        wavelet=wavelet,
        template_support_samples=template_support_samples,
    )


def evaluate_monotransit_candidate(
    result: SWTMatchedFilterResult,
    idx: int,
    flux: np.ndarray | None = None,
    template: np.ndarray | None = None,
    exclude_halfwidth: int | None = None,
    n_shifts: int = 2000,
    peak_threshold_sigma: float | None = None,
    use_absolute: bool = True,
    apply_trials_correction: bool = True,
) -> dict[str, float | int | None | str]:
    """Evaluate a candidate peak in a wavelet matched-filter result.

    Computes a suite of diagnostics to help distinguish genuine transit-like
    signals from noise peaks:

    - **Empirical p-value**: fraction of out-of-candidate z-score samples
      (or a random subsample via ``n_shifts``) that are at least as extreme
      as the candidate.
    - **Effective-trials correction**: the empirical p-value is adjusted for
      multiple comparisons using the ACF-derived correlation length of the
      z-score series to estimate the number of independent trials.
    - **Channel coherence**: measures whether all wavelet bands contribute
      to the candidate in the same direction. High coherence (→1) is
      consistent with a real broadband signal; incoherent contributions
      suggest noise.
    - **Residual reduction** (optional, requires ``flux`` and ``template``):
      fraction of local variance explained by fitting the template in a
      window around the candidate.

    Parameters
    ----------
    result:
        Output of `wavelet_matched_filter` or `kepler_owt_ses_filter`.
    idx:
        Sample index of the candidate peak in ``result.z_score``.
    flux:
        Original flux array. Required to compute ``residual_reduction``.
    template:
        Transit template. Required to compute ``residual_reduction`` and
        used to infer ``exclude_halfwidth`` when not provided explicitly.
    exclude_halfwidth:
        Number of samples around ``idx`` to exclude from the background
        distribution. Defaults to ``len(template) // 2`` when ``template``
        is given, or the median channel window otherwise.
    n_shifts:
        Maximum number of background samples to draw when computing the
        empirical p-value. Set to 0 to use all available samples.
    peak_threshold_sigma:
        Threshold for counting ``n_global_similar`` peaks globally. Defaults
        to ``0.5 * abs_z_obs`` when not provided.
    use_absolute:
        If True, rank on ``|z|`` (two-sided, no sign assumption). If False,
        rank on signed ``z`` (one-sided positive tail), useful when the
        template polarity is known.
    apply_trials_correction:
        If True, correct the empirical p-value for multiple comparisons
        via the effective-trials factor derived from the z-score ACF.
        If False, ``adj_p_trials`` equals the raw empirical p-value.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``z_obs``
            Signed z-score at ``idx``.
        ``abs_z_obs``
            Absolute z-score at ``idx``.
        ``empirical_p``
            Empirical p-value: fraction of background samples >= candidate.
        ``N_similar``
            Number of local maxima in the background as extreme as the candidate.
        ``peak_density``
            ``N_similar / len(background)`` — how crowded the tail is.
        ``adj_p_trials``
            Empirical p-value corrected for multiple trials (or raw if
            ``apply_trials_correction=False``).
        ``N_eff_trials``
            Effective number of independent trials (N / ACF correlation length).
        ``coherence``
            Channel coherence score in [0, 1]. Near 1 means all bands agree.
        ``frac_channels_strong``
            Fraction of channels contributing above the median channel contribution.
        ``prominence``
            Candidate z-score minus the median |z| in the local neighbourhood.
        ``local_mean_Z``
            Mean z-score in the local neighbourhood around ``idx``.
        ``local_std_Z``
            Standard deviation of z-scores in the local neighbourhood.
        ``n_global_similar``
            Number of samples in the background exceeding ``peak_threshold_sigma``.
        ``residual_reduction``
            Fraction of local flux variance explained by the template fit,
            or ``None`` if ``flux`` and ``template`` were not provided.
        ``statistic_mode``
            ``"absolute"`` or ``"signed_positive"``, reflecting ``use_absolute``.
        ``trials_corrected``
            Whether the effective-trials correction was applied.
        ``notes``
            Semicolon-separated human-readable summary of the evaluation.

    Raises
    ------
    IndexError
        If ``idx`` is outside ``[0, len(z_score))``.
    """
    z = np.asarray(result.z_score, dtype=float)
    n = len(z)

    idx = int(idx)
    if idx < 0 or idx >= n:
        raise IndexError("idx is out of bounds for the z-score array.")

    z_obs = float(z[idx])
    abs_z_obs = abs(z_obs)
    z_obs_stat = abs_z_obs if use_absolute else z_obs

    if exclude_halfwidth is None:
        if template is not None:
            halfw = len(as_1d_array(template)) // 2
        else:
            windows = [ch.window for ch in result.channels]
            halfw = int(np.median(windows)) if windows else max(3, n // 1000)
    else:
        halfw = int(exclude_halfwidth)

    mask = np.ones(n, dtype=bool)
    lo = max(0, idx - halfw)
    hi = min(n, idx + halfw + 1)
    mask[lo:hi] = False

    other_z = np.abs(z[mask]) if use_absolute else z[mask]
    if other_z.size == 0:
        empirical_p = 1.0
    else:
        empirical_p = (np.sum(other_z >= z_obs_stat) + 1) / (other_z.size + 1)

    if 0 < int(n_shifts) < other_z.size:
        rng = np.random.default_rng(0)
        sample = rng.choice(other_z, size=int(n_shifts), replace=False)
        empirical_p = (np.sum(sample >= z_obs_stat) + 1) / (sample.size + 1)

    z_metric = np.abs(z) if use_absolute else z
    maxima = np.zeros(n, dtype=bool)
    maxima[1:-1] = (z_metric[1:-1] > z_metric[:-2]) & (z_metric[1:-1] > z_metric[2:])
    maxima[lo:hi] = False

    n_similar = int(np.sum(maxima & (z_metric >= z_obs_stat)))
    search_length = int(np.sum(mask))
    peak_density = n_similar / search_length if search_length > 0 else np.nan

    loc_w = max(3, halfw * 2)
    lw = min(n, loc_w)
    llo = max(0, idx - lw // 2)
    lhi = min(n, llo + lw)

    local_z = z[llo:lhi]
    local_mean = float(np.mean(local_z))
    local_std = float(np.std(local_z, ddof=1)) if local_z.size > 1 else np.nan
    if use_absolute:
        prominence = float(abs_z_obs - np.median(np.abs(local_z)))
    else:
        prominence = float(z_obs - np.median(local_z))

    z_centered = z_metric - np.mean(z_metric)
    acf_full = np.correlate(z_centered, z_centered, mode="full")
    acf = acf_full[acf_full.size // 2 :]

    if acf[0] <= 0 or np.all(np.isclose(acf, 0)):
        corr_len = 1.0
    else:
        acf = acf / acf[0]
        below = np.where(acf <= (1.0 / np.e))[0]
        corr_len = float(below[0]) if below.size else float(len(acf))
        corr_len = max(1.0, corr_len)

    n_eff_trials = max(1.0, n / corr_len)
    if apply_trials_correction:
        adj_p_trials = 1.0 - (1.0 - empirical_p) ** n_eff_trials
    else:
        adj_p_trials = empirical_p
    adj_p_trials = float(min(max(adj_p_trials, 0.0), 1.0))

    per_ch_contrib = []
    for ch in result.channels:
        xw = ch.whitened
        tw = ch.template[::-1]

        if len(xw) != n:
            xw = np.resize(xw, n)
        if len(tw) == 0:
            per_ch_contrib.append(0.0)
            continue

        conv = _fft_conv(xw, tw)
        center = len(conv) // 2
        start = center - (n // 2)
        pos = start + idx
        if 0 <= pos < len(conv):
            val = float(conv[pos])
        else:
            val = 0.0
        per_ch_contrib.append(val)

    per_ch_contrib = np.asarray(per_ch_contrib, dtype=float)
    sum_abs = np.sum(np.abs(per_ch_contrib)) + 1e-30
    coherence = float(np.abs(np.sum(per_ch_contrib)) / sum_abs) if sum_abs > 0 else 0.0

    if per_ch_contrib.size > 0:
        threshold = np.median(np.abs(per_ch_contrib))
        frac_channels_strong = float(
            np.sum(np.abs(per_ch_contrib) >= threshold) / per_ch_contrib.size
        )
    else:
        frac_channels_strong = 0.0

    residual_reduction = None
    notes: list[str] = []

    if flux is not None and template is not None:
        x = as_1d_array(flux)
        s = _fit_template_length(as_1d_array(template).astype(float), len(x))

        wlo = max(0, idx - halfw)
        whi = min(len(x), idx + halfw + 1)
        xwin = s[wlo:whi]
        ywin = x[wlo:whi]

        if xwin.size >= 3:
            design = np.vstack([xwin, np.ones_like(xwin)]).T
            sol, *_ = np.linalg.lstsq(design, ywin, rcond=None)
            a_hat, c_hat = float(sol[0]), float(sol[1])

            resid_before = np.sum((ywin - np.mean(ywin)) ** 2)
            y_fit = a_hat * xwin + c_hat
            resid_after = np.sum((ywin - y_fit) ** 2)

            residual_reduction = float(
                max(0.0, (resid_before - resid_after) / (resid_before + 1e-30))
            )
            notes.append(f"residual_reduction_window_frac={residual_reduction:.3f}")
        else:
            notes.append("residual test skipped: insufficient local samples")

    if peak_threshold_sigma is None:
        count_threshold = z_obs_stat * 0.5
    else:
        count_threshold = float(peak_threshold_sigma)

    n_global_similar = int(np.sum((z_metric >= count_threshold) & mask))

    if (
        empirical_p < 1e-4
        and coherence > 0.6
        and residual_reduction is not None
        and residual_reduction > 0.2
    ):
        notes.append(
            "candidate looks promising: low empirical_p, coherent channel contributions, residual reduced."
        )
    elif empirical_p < 1e-3 and coherence > 0.5:
        notes.append(
            "candidate potentially interesting but inspect lightcurve and nearby systematics."
        )
    else:
        notes.append(
            "candidate likely noise-like (high empirical_p and/or low coherence)."
        )

    return {
        "z_obs": z_obs,
        "abs_z_obs": abs_z_obs,
        "empirical_p": float(empirical_p),
        "N_similar": n_similar,
        "peak_density": float(peak_density),
        "adj_p_trials": adj_p_trials,
        "N_eff_trials": float(n_eff_trials),
        "coherence": float(coherence),
        "frac_channels_strong": float(frac_channels_strong),
        "prominence": float(prominence),
        "local_mean_Z": float(local_mean),
        "local_std_Z": float(local_std),
        "n_global_similar": n_global_similar,
        "residual_reduction": float(residual_reduction)
        if residual_reduction is not None
        else None,
        "statistic_mode": "absolute" if use_absolute else "signed_positive",
        "trials_corrected": bool(apply_trials_correction),
        "notes": "; ".join(notes),
    }
