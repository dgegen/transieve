import numpy as np

"""
SNR utilities following Kipping (2023).

Functions expect 1D arrays of equal length:
- `flux`: observed (normalized) flux, typically centered around 1.0
- `injection`: transit model (same sign convention as subtraction from `flux`)
- `flux_err`: per-sample standard deviations (white-noise uncertainties)

All functions return a scalar SNR (float).
"""


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
