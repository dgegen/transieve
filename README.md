# transieve

Transit-signal analysis in correlated noise.

`transieve` is a research package for detecting single-transit (monotransit) events in stellar light curves. It combines empirical and limb-darkened transit models, Gaussian-process noise modelling with `celerite2`, and matched-filter diagnostics to assess transit retrievability in red and white noise.

## Installation

```bash
git clone https://github.com/dgegen/transieve.git
cd transieve
uv sync
```

## Quickstart

```python
import numpy as np
from transieve import SimulatedLightCurve
from transieve.gp import SHOGPFamily, assess_retrievability
from transieve.transit import get_monotransit_from_epoch

lc = SimulatedLightCurve.from_transit(
    epoch=0.0, depth=0.002, duration=0.2, period=100.0,
    baseline=15, cadence=10, multiply_signal=False, seed=42,
)

result = assess_retrievability(
    time=lc.time,
    flux=lc.flux,
    template_bank=get_monotransit_from_epoch(lc.time, depth=0.002, duration=0.2),
    gp_family=SHOGPFamily(jitter_range=(1e-4, 1e-2)),
)

idx, z_peak = result.strongest_match()
print(f"Peak Z-score: {z_peak:.2f} at t = {lc.time[idx]:.3f} d")
```

## Features

- **Transit models** — empirical (Protopapas et al. 2005) and limb-darkened (`batman`) transit shapes
- **GP noise modelling** — SHO and exponential families via `celerite2`, with optimizer wrappers
- **Matched-filter detection** — exact GP-inverse Z-scores with white-noise baselines
- **Retrievability assessment** — recovery fractions, detectability depth, and sensitivity over template banks
- **GP stability maps** — 2D scans over amplitude and timescale hyperparameters

## Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/dgegen/transieve.git
cd transieve
uv sync --all-groups
```

Install the pre-commit hooks and run the test suite before submitting a pull request:

```bash
uv run pre-commit install
uv run pytest
```

### Building the docs

The documentation is built with [Quarto](https://quarto.org) and [quartodoc](https://machow.github.io/quartodoc). Install Quarto, then run:

```bash
# Move to docs directory
cd docs

# Generate the API reference pages
uv run quartodoc build

# Render and preview the site
uv run quarto preview
```

To build a static copy of the site:

```bash
uv run quarto render
```

The rendered site is written to `docs/_site/`.
