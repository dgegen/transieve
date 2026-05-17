import logging
from typing import Optional, Tuple

import numpy as np
from scipy.stats import binned_statistic, median_abs_deviation

logger = logging.getLogger("lightkite")


def _bin_weighted_mean(
    flux,
    flux_err,
    bin_indices,
    non_empty_indices,
    points_per_bin,
    return_variance,
    return_error,
):
    """Internal helper for weighted mean and variance."""
    weights = 1.0 if flux_err is None else 1 / flux_err**2
    sum_w = np.bincount(bin_indices, weights=None if flux_err is None else weights)
    w_sum = np.bincount(bin_indices, weights=flux * weights)

    mean = w_sum[non_empty_indices] / sum_w[non_empty_indices]

    if return_error is False:
        return mean, None

    # Variance calculation
    mean_rep = np.repeat(mean, points_per_bin[non_empty_indices])
    var_sum = np.bincount(bin_indices, weights=(flux - mean_rep) ** 2 * weights)
    variance = var_sum[non_empty_indices] / sum_w[non_empty_indices]

    if return_variance:
        return mean, variance

    std_err = np.where(points_per_bin[non_empty_indices] == 1, 0, np.sqrt(variance))
    return mean, std_err / np.sqrt(points_per_bin[non_empty_indices])


def _bin_median(
    time, flux, bins, non_empty_indices, points_per_bin, return_variance, return_error
):
    """Internal helper for binned median and standard error."""
    median_flux, _, _ = binned_statistic(time, flux, statistic="median", bins=bins)
    median = median_flux[non_empty_indices]

    if return_error is False:
        return median, None

    robust_mad_stats, _, _ = binned_statistic(
        time,
        flux,
        statistic=lambda x: median_abs_deviation(x, scale="normal"),  # type: ignore
        bins=bins,
    )
    robust_mad_stats = np.nan_to_num(robust_mad_stats[non_empty_indices])

    if return_variance:
        return median, robust_mad_stats**2

    # efficiency_factor = sqrt(pi / 2)
    std_err = (
        1.2533141373 * robust_mad_stats / np.sqrt(points_per_bin[non_empty_indices])
    )
    return median, std_err


def bin_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    bin_time: float = 1 / 24,
    return_error: bool = True,
    return_variance: bool = False,
    method: str = "mean",  # Options: "mean" or "median"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bins a light curve into time intervals and calculates the weighted mean
    and optionally the standard deviation of the flux in each bin.

    Parameters
    ----------
    time : np.ndarray
        The 1D array containing the time points of the light curve.
    flux : np.ndarray
        The 1D array containing the flux values of the light curve at the
        corresponding time points.
    flux_err : Optional[np.ndarray], optional
        The 1D array containing the error (uncertainty) on the flux values.
        If not provided, a weight of 1 is used for each data point.
    bin_time : float, optional
        The width of the bins in units of time. Defaults to 1/24 (days).
    return_error : bool, optional
        If True (default), calculate and return the weighted standard deviation
        of the flux in each bin. If False, only the bin centers and weighted
        mean flux are returned.

    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        A tuple containing the bin centers and the weighted mean flux in each bin.
        If `return_error` is True, the tuple also includes the weighted standard
        error of the flux in each bin.

    Notes
    -----
    This function removes any NaN values from the input arrays before binning.
    For bins with only one data point, the standard deviation is set to zero.

    See Also
    --------
    bin_light_curve_slow
    """
    # Remove NaN values
    mask = ~np.isnan(time) & ~np.isnan(flux)
    if flux_err is not None:
        mask &= ~np.isnan(flux_err)

    time, flux = time[mask], flux[mask]
    flux_err = flux_err[mask] if flux_err is not None else None

    # Create bins and digitize
    bins = np.arange(np.nanmin(time), np.nanmax(time) + 2 * bin_time, bin_time)
    bin_indices = np.digitize(time, bins) - 1

    # Find non-empty bins
    points_per_bin = np.bincount(bin_indices)
    non_empty_mask = points_per_bin > 0
    non_empty_indices = np.where(non_empty_mask)[0]
    bin_centers = bins[non_empty_indices] + bin_time / 2

    if method == "mean":
        final_flux, final_err = _bin_weighted_mean(
            flux,
            flux_err,
            bin_indices,
            non_empty_indices,
            points_per_bin,
            return_variance,
            return_error,
        )
    elif method == "median":
        final_flux, final_err = _bin_median(
            time,
            flux,
            bins,
            non_empty_indices,
            points_per_bin,
            return_variance,
            return_error,
        )
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'mean' or 'median'.")

    return bin_centers, final_flux, final_err


def plot_binned_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    bin_time: float = 1 / 24,
    method: str = "mean",
    ax=None,
    **plot_kwargs,
):
    """Plot a binned light curve with error bars.

    Parameters
    ----------
    time : np.ndarray
        The 1D array containing the time points of the light curve.
    flux : np.ndarray
        The 1D array containing the flux values of the light curve at the
        corresponding time points.
    flux_err : Optional[np.ndarray], optional
        The 1D array containing the error (uncertainty) on the flux values.
        If not provided, a weight of 1 is used for each data point.
    bin_time : float, optional
        The width of the bins in units of time. Defaults to 1/24 (days).
    method : str, optional
        The method to use for calculating the central tendency in each bin.
        Options are "mean" (default) or "median".
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plotted binned light curve.

    Notes
    -----
    This function uses `bin_light_curve` to compute the binned light curve and then
    plots it using Matplotlib. Error bars are included if `flux_err` is provided.
    """
    import matplotlib.pyplot as plt

    binned_time, binned_flux, binned_flux_err = bin_light_curve(
        time, flux, flux_err, bin_time, return_error=True, method=method
    )

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")

    plot_kwargs = {"fmt": ".", "ls": ""} | plot_kwargs
    ax.errorbar(binned_time, binned_flux, yerr=binned_flux_err, **plot_kwargs)

    return ax
