# ---
# title: "Quickstart"
# jupyter: python3
# ---

# %% [markdown]
# This page walks through a minimal end-to-end run of the `transieve` pipeline:
# simulate a light curve with an injected transit, build a template bank,
# fit a GP noise model, and assess how well the transit can be recovered.

# %%
import matplotlib.pyplot as plt
from transieve import SimulatedLightCurve
from transieve.gp import SHOGPFamily, assess_retrievability
from transieve.transit import get_monotransit_from_epoch

# %%
# | echo: false
# | output: false
import plotting as plt  # noqa: F811

# %% [markdown]
# ## Simulate a light curve
#
# `SimulatedLightCurve.from_transit` draws a realisation of SHO GP noise and
# multiplies it by (or adds) a deterministic transit model.
# Setting `multiply_signal=False` uses additive composition, which keeps the
# signal zero-centred — the preferred mode for matched filtering.

# %%
lc = SimulatedLightCurve.from_transit(
    epoch=0.0,
    depth=0.002,
    duration=0.2,
    period=100.0,
    baseline=15,
    cadence=10,
    multiply_signal=False,
    seed=42,
)
_ = lc.plot()

# %% [markdown]
# ## Build a template bank
#
# `get_monotransit_from_epoch` returns a callable `f(epoch) -> flux array`
# that evaluates the empirical transit model centred at any requested epoch.
# Passing it directly to `assess_retrievability` causes the filter to slide
# the template across every cadence in `lc.time`.

# %%
template_bank = get_monotransit_from_epoch(
    lc.time,
    depth=0.002,
    duration=0.2,
    mean=0.0,
)

# %% [markdown]
# ## Fit a GP and run the matched filter
#
# `assess_retrievability` fits the chosen GP family to the light curve,
# then computes the exact GP-inverse matched-filter Z-score at each epoch.
# `SHOGPFamily(jitter_range=...)` adds a white-noise jitter term to the fit,
# which is required when no per-cadence `flux_err` is provided.

# %%
result = assess_retrievability(
    time=lc.time,
    flux=lc.flux,
    template_bank=template_bank,
    gp_family=SHOGPFamily(jitter_range=(1e-4, 1e-2)),
    center_flux=True,
)

# %% [markdown]
# ## Inspect the results
#
# `z_score` is the matched-filter detection statistic at each candidate epoch.
# A peak near the injected epoch (0.0 d) indicates a successful recovery.
# `recovery_fraction` measures how much of the transit signal survives the
# GP noise model relative to a white-noise baseline.

# %%
fig, ax = plt.subplots()
ax.plot(result.time, result.z_score, lw=1)
ax.axvline(0.0, ls="--", label="injected epoch")
ax.set_ylabel("Z-score")
ax.set_xlabel("Time [days]")
ax.legend()

# %%
idx, z_peak = result.strongest_match()
print(f"Peak Z-score:              {z_peak:.2f}")
print(f"Epoch at peak:             {result.time[idx]:.3f} days")
print(f"Recovery fraction at peak: {result.peak_recovery_fraction:.2f}")
