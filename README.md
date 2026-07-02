# transieve

[![CI](https://github.com/dgegen/transieve/actions/workflows/ci.yml/badge.svg)](https://github.com/dgegen/transieve/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dgegen.github.io%2Ftransieve-blue)](https://dgegen.github.io/transieve/)

Transit-signal analysis in correlated noise.

`transieve` is a research package for quantifying how detectable a single-transit is in a stellar light curve. It combines empirical and limb-darkened transit models with Gaussian-process noise modelling (`celerite2`) to compute GP-whitened matched-filter Z-scores — an algorithmic proxy for detectability.

We built `transieve` to find the "edge of detectability" for long-period, single-transit exoplanets in TESS data: the regime that's too rare for humans to vet at scale, but exactly where machine-learning search pipelines need well-targeted training examples. By injecting synthetic transits into real TESS light curves and comparing the resulting Z-scores against a human-consensus baseline from a citizen-science-style validation study (in the spirit of [Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess)), we showed that a contrast-adjusted, sector-rigid Z-score closely tracks human detectability (AUROC ≈ 0.98). That gives a scalable formula — $P(D_H=1 \mid Z_\text{contrast})$ — for curating ML training curricula: injecting transits right at the ambiguous $P=0.5$ boundary instead of at random depths, and mining high-Z, no-signal light curves as hard negatives against instrumental false alarms.

Documentation, including the API reference and tutorials, is available at [dgegen.github.io/transieve](https://dgegen.github.io/transieve/).

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
from transieve.gp import SHOGPFamily, evaluate_frequentist_detection
from transieve.transit import get_monotransit_from_epoch

lc = SimulatedLightCurve.from_transit(
    epoch=0.0, depth=0.002, duration=0.2, period=100.0,
    baseline=15, cadence=10, multiply_signal=False, seed=42,
)

result = evaluate_frequentist_detection(
    time=lc.time,
    flux=lc.flux,
    template_bank=get_monotransit_from_epoch(lc.time, depth=0.002, duration=0.2),
    gp_family=SHOGPFamily(jitter_range=(1e-4, 1e-2)),
)

idx, z_peak = result.strongest_match()
print(f"Peak Z-score: {z_peak:.2f} at t = {lc.time[idx]:.3f} d")
```

## Features

- **Transit models** — empirical (Protopapas et al. 2005) and limb-darkened (`batman`) transit shapes for generating and injecting synthetic monotransits
- **GP noise modelling** — SHO and exponential families via `celerite2`, fit either sector-rigid (one global model per TESS sector) or sequence-adaptive (local windows), with optimizer wrappers
- **Matched-filter detection** — exact GP-inverse Z-scores and contrast-adjusted variants, with white-noise SNR baselines for comparison
- **Retrievability assessment** — recovery fractions, detectability depth, and sensitivity over template banks, used to benchmark algorithmic metrics against human-vetted detection labels
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
```

To build a static copy of the site:

```bash
uv run quarto render
```

The rendered site is written to `docs/_site/`.
