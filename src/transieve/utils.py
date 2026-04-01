from __future__ import annotations

import numpy as np


def make_out_of_transit_mask(
    time: np.ndarray,
    epoch: float,
    duration: float,
    margin: float = 0.0,
) -> np.ndarray:
    """Build an out-of-transit mask for GP fitting.

    Parameters
    ----------
    time : np.ndarray
            Observation times in days.
    epoch : float
            Transit center in days.
    duration : float
            Transit duration in days.
    margin : float, optional
            Extra exclusion margin on each side in days.
    """
    half_window = 0.5 * duration + margin
    in_transit = np.abs(time - epoch) <= half_window
    return ~in_transit


def as_1d_array(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return arr


def center_flux(flux: np.ndarray, center: bool) -> np.ndarray:
    if not center:
        return flux
    return flux - np.nanmedian(flux)
