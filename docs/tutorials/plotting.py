"""Lightweight plotting utilities for transieve notebooks.

Provides paper-quality matplotlib configuration and a subplots wrapper
that sizes figures relative to a standard text width.

Usage:
    import plotting as plm

    fig, ax = plm.subplots()
    fig, axes = plm.subplots(2, 1, rescale_height=0.7, sharex=True)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg")
except Exception:
    pass

__all__ = ["subplots", "TEXT_WIDTH"]


INCHES_PER_PT = 1 / 72.27
TEXT_WIDTH = 483.69687 * INCHES_PER_PT
GOLDEN_RATIO = (5**0.5 - 1) / 2

mpl.rcParams.update(
    {
        # Colors
        "xtick.color": "#323034",
        "ytick.color": "#323034",
        "text.color": "#323034",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "grid.color": "#b1afb5",
        "patch.facecolor": "#bc80bd",
        "patch.force_edgecolor": True,
        "patch.linewidth": 0.8,
        "scatter.edgecolors": "black",
        # Color cycle
        "axes.prop_cycle": mpl.cycler(
            "color",
            [
                "bc80bd",
                "fb8072",
                "b3de69",
                "fdb462",
                "fccde5",
                "8dd3c7",
                "ffed6f",
                "bebada",
                "80b1d3",
                "ccebc5",
                "d9d9d9",
            ],
        ),
        # Font
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "figure.titlesize": 12,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        # "text.usetex": True,
        # Lines
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.7,
        # Legend
        "legend.fancybox": False,
        "legend.frameon": False,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.9",
        "legend.borderpad": 0.2,
        "legend.columnspacing": 1.5,
        "legend.labelspacing": 0.4,
        # Axes
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": False,
        "axes.titlelocation": "center",
        "axes.formatter.use_mathtext": True,
        "axes.formatter.limits": (-4, 4),
        "axes.labelpad": 3,
        # Figure / saving
        "figure.dpi": 300,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "image.cmap": "magma",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)


def subplots(nrows=1, ncols=1, rescale_height=1.0, **kwargs):
    """Create a figure sized to TEXT_WIDTH with proportional height.

    Parameters
    ----------
    nrows, ncols : int
        Passed directly to plt.subplots.
    rescale_height : float
        Height as a fraction of TEXT_WIDTH. Default 1.0.
    **kwargs
        Forwarded to plt.subplots.
    """

    ratio = nrows / ncols * GOLDEN_RATIO
    height = TEXT_WIDTH * ratio * rescale_height
    return plt.subplots(nrows, ncols, figsize=(TEXT_WIDTH, height), **kwargs)
