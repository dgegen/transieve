# ---
# title: "GP Noise Modelling"
# format:
#   html:
#     html-math-method: mathjax
#     default-image-extension: svg
# jupyter: python3
# fig-format: svg
# ---

# %% [markdown]
# # Retrievability Demo on a Synthetic Light Curve
#
# This notebook demonstrates the new retrievability APIs:
#
# - `assess_retrievability`: fit a GP and evaluate matched-filter retrievability diagnostics
# - `scan_gp_stability_map`: map detectability across GP hyperparameters (sigma, timescale)
#
# The example uses a synthetic monotransit generated with `SimulatedLightCurve`.

# %%


import numpy as np
import matplotlib.pyplot as plt

from transieve import SimulatedLightCurve
from transieve.gp import (
    SHOGPFamily,
    assess_retrievability,
    check_retrievability,
    scan_gp_stability_map,
)
from transieve.transit import get_monotransit_from_epoch

import plotting as plm


# %% [markdown]
# ## Build a synthetic light curve
#
# Note: GP parameters in this package use log-parameterization (`log_omega`, `log_sigma`, `log_quality`, `log_jitter`).

# %%


planet_params = {
    "depth": (3 / (1 * 109)) ** 2,
    "epoch": 0.0,
    "duration": 0.2,
    "period": 100.0,
}

gp_params = {
    "log_omega": np.log(2 * np.pi / 4),
    "log_sigma": np.log(5e-4),
    "log_quality": np.log(1 / np.sqrt(2)),
    "log_jitter": np.log(1e-3),
}

lc = SimulatedLightCurve.from_transit(
    **planet_params,
    gp_params=gp_params,
    baseline=15,
    seed=0,
    multiply_signal=False,
    gap_windows=[[1, 2]],
)

time, flux = lc.time, lc.flux
true_signal = lc.deterministic_component


# %%


_, axes = plm.subplots(3, 1, sharex=True, rescale_height=0.4)
_ = lc.plot(axes)


# %% [markdown]
# ## Baseline retrievability with fitted GP
#
# We fit an SHO GP and evaluate retrievability over epoch shifts using `get_monotransit_from_epoch`.

# %%


gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))

template_generator = get_monotransit_from_epoch(
    time,
    depth=planet_params["depth"],
    duration=planet_params["duration"],
    period=planet_params["period"],
    mean=0.0,
)

initial_theta = np.array(
    [
        gp_params["log_omega"],
        gp_params["log_sigma"],
        gp_params["log_quality"],
        gp_params["log_jitter"],
    ]
)

retr = assess_retrievability(
    time=time,
    flux=flux,
    template_bank=template_generator,
    gp_family=gp_family,
    fit_mean=1.0,
    fit_method="L-BFGS-B",
    fit_max_retries=1,
    fit_kwargs={"initial_theta": initial_theta},
    threshold=5,
)

peak_idx, peak_z = retr.strongest_match()
peak_time = retr.time[peak_idx]
print(f"Peak Z-score: {peak_z:.3f} at t = {peak_time:.3f} d")
print(f"Peak recovery fraction: {retr.peak_recovery_fraction:.3f}")
print(f"Peak relative capacity: {retr.peak_relative_capacity:.3f}")
print("Recovered GP parameters:", retr.gp_physical_params)


# %%


# Quick sanity checks for the white-noise baseline diagnostics
print("Z-score range:", float(np.nanmin(retr.z_score)), float(np.nanmax(retr.z_score)))
print(
    "Z_ideal range:",
    float(np.nanmin(retr.z_white_noise)),
    float(np.nanmax(retr.z_white_noise)),
)
print(
    "Recovery fraction range:",
    float(np.nanmin(retr.recovery_fraction)),
    float(np.nanmax(retr.recovery_fraction)),
)
print(
    "Relative capacity range:",
    float(np.nanmin(retr.relative_capacity)),
    float(np.nanmax(retr.relative_capacity)),
)


# %%


fig, axes = plm.subplots(4, 1, sharex=True, rescale_height=0.4)

axes[0].plot(retr.time, retr.z_score)
axes[0].axvline(peak_time, color="black", ls="--")
axes[0].set_ylabel("Z-score")
axes[0].axhline(0, color="black", ls=":")

axes[1].plot(retr.time, retr.recovery_fraction)
axes[1].axhline(1.0, color="black", ls=":")
axes[1].set_ylabel("Recovery fraction")

axes[2].plot(retr.time, retr.white_template_norm)
axes[2].set_ylabel("White noise capacity")

axes[3].plot(retr.time, retr.relative_capacity)
axes[3].axhline(1.0, color="black", ls=":")
axes[3].set_ylabel("Relative capacity")
axes[3].set_xlabel("Template epoch [days]")

plt.tight_layout()

# %% [markdown]
# ## 2D GP stability landscape
#
# Now we hold the transit template fixed and scan GP `sigma` and timescale (`period`) to see where detection is both strong and stable.

# %%


sigma_grid = np.geomspace(2e-4, 3e-3, 20)
period_grid = np.geomspace(0.2, 40.0, 24)

template_fixed = true_signal - 1.0

stab = scan_gp_stability_map(
    time=time,
    flux=flux,
    template=template_fixed,
    gp_family=gp_family,
    sigma_grid=sigma_grid,
    timescale_grid=period_grid,
    fit_mean=1.0,
    fit_method="L-BFGS-B",
    fit_max_retries=1,
    fit_kwargs={"initial_theta": initial_theta},
)

best = stab.strongest_point()
summary = stab.summary(5)
print("Best grid point:", best)
print("Summary (including log-space plateau area):", summary)

features = stab.robust_features(z_threshold=5.0, max_fragility=0.1)
print("\nRobust features (grid-invariant):")
for k, v in features.items():
    print(f"  {k}: {v:.4g}")

z_marg = stab.marginalized_z()
print(f"\nMarginalized Z-score: {z_marg:.3f}")


# %%
# | code-fold: true
# | code-summary: "Show plotting code"

X, Y = np.meshgrid(stab.timescale_grid, stab.sigma_grid)
plateau_contour = np.where(stab.get_plateau_mask(5), 1.0, 0.0)

fig, axes = plm.subplots(1, 4, sharex=True, sharey=True)

im0 = axes[0].pcolormesh(X, Y, stab.z_score, shading="auto", cmap="viridis")
axes[0].set_title("Peak Z-score")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(X, Y, stab.recovery_fraction, shading="auto", cmap="magma")
axes[1].set_title("Recovery fraction")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(X, Y, stab.local_drop, shading="auto", cmap="cividis")
axes[2].contour(X, Y, plateau_contour, levels=[0.5], colors="white", linewidths=1.5)
axes[2].set_title("Local drop (plateau: white)")
fig.colorbar(im2, ax=axes[2])

# Log-marginal likelihood — shows where data support the GP hyperparameters.
# Posterior weights for marginalized_z() are proportional to exp(log_L).
log_L = stab.log_marginal_likelihood
im3 = axes[3].pcolormesh(X, Y, log_L, shading="auto", cmap="plasma")
axes[3].contour(X, Y, plateau_contour, levels=[0.5], colors="white", linewidths=1.5)
axes[3].set_title("Log-marginal likelihood")
fig.colorbar(im3, ax=axes[3])

for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{stab.timescale_name} [days]")
axes[0].set_ylabel("sigma")

plt.show()

# %% [markdown]
# The plateau is a diagonal "stripe" rather than a circle, confirming the covariance
# degeneracy: increasing the GP noise amplitude requires a longer period to maintain the
# same degree of signal filtering.
#
# ### Reading the four panels
#
# - **Peak Z-score** — where the matched-filter response is strongest.
# - **Recovery fraction** — what fraction of the transit signal survives the GP filter.
#   Values below ~0.8 mean the model is absorbing the transit into the noise term.
# - **Local drop (plateau: white contour)** — brittleness of each grid cell relative to
#   its four neighbours.  The white contour marks the stable plateau
#   (`z_threshold=5`, `max_fragility=0.1`).
# - **Log-marginal likelihood** — how well each `(sigma, period)` combination explains
#   the data *without* the transit template.  This is the posterior weight used by
#   `marginalized_z()`: high-Z regions that sit in a low-likelihood part of parameter
#   space are naturally down-weighted, correcting for the look-elsewhere effect.
#
# ### Grid-invariant summary scalars
#
# | Metric | Description |
# |--------|-------------|
# | `log_space_plateau_area` | Area of the stable plateau in log(σ)×log(τ) space — invariant to grid bounds, unlike `plateau_fraction`. |
# | `capacity_bounded_peak_z` | Max raw Z inside the physically valid envelope (recovery > 0.8, 0.8 < relative_capacity < 1.2).  Retains standard-normal properties. |
# | `peak_brittleness` | `local_drop` at the global Z-score maximum — use as a veto: high values flag spurious stellar-activity ridges. |
# | `is_safe_harbor` | 1 if the plateau is non-empty and the peak is not brittle. |
# | `marginalized_z()` | Posterior-weighted average Z-score — corrects for look-elsewhere inflation of the peak. |
#
# ## Interpretation checklist
#
# - Prefer regions with high `Z-score` and high `Recovery Fraction`.
# - Use `capacity_bounded_peak_z` instead of the raw grid maximum as the primary
#   detection statistic; it avoids the degenerate signal-absorbing regime.
# - Use `log_space_plateau_area` instead of `plateau_fraction` when comparing maps
#   computed on different grids.
# - Check `peak_brittleness` as a veto: a high value suggests the peak is a narrow ridge
#   that vanishes under small hyperparameter shifts.
# - Use `marginalized_z()` when you need a single scalar that accounts for the full
#   posterior over GP hyperparameters rather than a point estimate.

# %%


# Final retrievability verdict for the injected transit

verdict, diagnostics = check_retrievability(
    retr,
    injected_time=planet_params["epoch"],
    duration=planet_params["duration"],
    recovery_cutoff=0.5,
    capacity_bounds=(0.5, 1.5),
    stability_map=stab,
)
print("Retrievable:", verdict)
for k, v in diagnostics.items():
    print(f"  {k}: {v}")
