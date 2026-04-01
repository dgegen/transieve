from __future__ import annotations

import numpy as np


def plot_swt_statistic(result, time: np.ndarray | None = None, ax=None, **kwargs):
    """Plot SWT matched-filter z-score as a function of time or index."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    z = result.z_score
    x_label = "Index"
    if time is None:
        time = np.arange(len(z))
    else:
        x_label = "Time"

    ax.plot(time, z, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel("SWT matched-filter Z")
    return ax


def plot_cwt_power(result, ax=None, **kwargs):
    """Plot the CWT power map."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    power = result.power
    extent = (
        float(result.time[0]),
        float(result.time[-1]),
        float(result.scales[-1]),
        float(result.scales[0]),
    )

    img = ax.imshow(power, aspect="auto", extent=extent, origin="upper", **kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    plt.colorbar(img, ax=ax, label="Power")
    return ax
