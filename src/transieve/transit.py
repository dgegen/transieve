import numpy as np


def transit(time, epoch, duration, period, c=12):
    """Empirical transit model from Protopapas et al. 2005.

    Parameters
    ----------
    time : np.ndarray
        array of times
    epoch : float
        signal epoch
    duration : float
        signal duration
    period : float
        signal period
    c : int, optional
        controls the 'roundness' of transit shape, by default 12

    Returns
    -------
    np.ndarray
        array of transit model values
    """
    _t = period * np.sin(np.pi * (time - epoch) / period) / (np.pi * duration)
    return -0.5 * np.tanh(c * (_t + 1 / 2)) + 0.5 * np.tanh(c * (_t - 1 / 2))


def get_monotransit_model(
    epoch, depth=1.0, duration=0.2, period=100.0, mean=1.0, **kwargs
):
    def transit_function(time):
        return (
            depth * transit(time, epoch=epoch, duration=duration, period=period) + mean
        )

    return transit_function


def get_monotransit_from_epoch(
    time, depth=1.0, duration=0.2, period=100.0, mean=1.0, **kwargs
):
    default_epoch = (time[-1] + time[0]) / 2

    def transit_function(epoch=default_epoch):
        return (
            depth * transit(time, epoch=epoch, duration=duration, period=period) + mean
        )

    return transit_function


def _batman_params(epoch, duration, depth, period, impact_param, u, limb_dark):
    """Build a batman.TransitParams from physical transit parameters.

    Converts duration + impact parameter to semi-major axis and inclination
    using the circular-orbit transit duration formula:
        sin(pi * T / P) = sqrt((1 + k)^2 - b^2) / a
    """
    import batman

    rp = np.sqrt(depth)
    b = impact_param
    a = np.sqrt((1.0 + rp) ** 2 - b**2) / np.sin(np.pi * duration / period)
    inc = np.degrees(np.arccos(b / a))

    p = batman.TransitParams()
    p.t0 = epoch
    p.per = period
    p.rp = rp
    p.a = a
    p.inc = inc
    p.ecc = 0.0
    p.w = 90.0
    p.u = u
    p.limb_dark = limb_dark
    return p


def get_limb_dark_monotransit(
    epoch,
    depth=1.0,
    duration=0.2,
    period=100.0,
    mean=1.0,
    u=(0.3, 0.1),
    impact_param=0.0,
    limb_dark="quadratic",
    **kwargs,
):
    import batman

    # Try pre-build the model once
    params = _batman_params(epoch, duration, depth, period, impact_param, u, limb_dark)
    _m = None

    def transit_function(t):
        nonlocal _m
        if _m is None:
            _m = batman.TransitModel(params, t)
        elif np.all(_m.t != t):
            _m = batman.TransitModel(params, t)
        return _m.light_curve(params) - 1.0 + mean

    return transit_function


def get_limb_dark_monotransit_from_epoch(
    time,
    depth=1.0,
    duration=0.2,
    period=100.0,
    mean=1.0,
    u=(0.3, 0.1),
    impact_param=0.0,
    limb_dark="quadratic",
    **kwargs,
):
    import batman

    default_epoch = (time[-1] + time[0]) / 2

    def transit_function(epoch=default_epoch):
        params = _batman_params(
            epoch, duration, depth, period, impact_param, u, limb_dark
        )
        m = batman.TransitModel(params, time)
        return m.light_curve(params) - 1.0 + mean

    return transit_function
