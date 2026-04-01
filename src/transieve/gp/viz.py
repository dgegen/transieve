import numpy as np


def plot_covariance_matrix(gp, time=None, ax=None, thinning=1, inverse=False, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    plot_time = gp._t[::thinning] if time is None else time[::thinning]

    if inverse:
        identity = np.eye(len(gp._t))
        C_inv = gp.apply_inverse(identity)[::thinning, ::thinning]
        matrix = np.abs(C_inv) / np.abs(C_inv).max()
        label = "Inverse covariance / max value"
    else:
        matrix = gp.kernel.get_value(plot_time[:, None] - plot_time[None, :])
        matrix = matrix / matrix.max()
        label = "Covariance / $\\sigma^2$"

    extent = (plot_time[0], plot_time[-1], plot_time[-1], plot_time[0])

    img = ax.imshow(
        matrix,
        extent=extent,
        origin="upper",
        cmap="binary",
        **kwargs,
    )

    cbar = plt.colorbar(img, orientation="vertical", pad=0.05, ax=ax, fraction=0.046)
    cbar.set_label(label)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Time [days]")
