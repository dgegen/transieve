import numpy as np
from dataclasses import dataclass
from typing import List

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


@dataclass
class RedNoiseSnrResult:
    snr_white: float
    beta: float
    snr_red: float


def snr_red_noise(
    depth: float,
    time: np.ndarray,
    residuals: np.ndarray,
    n_in_transit: int,
    bin_times: np.ndarray = np.linspace(0.25, 0.41, 20),
    flux_median: bool = False,
) -> RedNoiseSnrResult:
    """SNR of a transit corrected for red noise (Winn et al. 2007).

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
        Number of in-transit cadences.
    bin_times:
        Bin durations in hours used to estimate β.
    flux_median:
        If True, use median binning when computing β.

    Returns
    -------
    RedNoiseSnrResult
        snr_white = δ / σ₁ · √N_in  (white-noise limit)
        beta      = average red-noise coefficient over `bin_times`
        snr_red   = snr_white / beta
    """
    sigma_1 = float(np.std(residuals))
    snr_white = depth / sigma_1 * np.sqrt(n_in_transit)
    beta = calculate_average_red_noise_coefficient(
        time=time,
        residuals=residuals,
        bin_times=bin_times,
        flux_median=flux_median,
    )
    return RedNoiseSnrResult(
        snr_white=snr_white,
        beta=float(beta),
        snr_red=snr_white / beta,
    )
