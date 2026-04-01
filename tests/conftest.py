"""Shared pytest fixtures for transieve tests."""

from __future__ import annotations

import numpy as np
import pytest

from transieve.gp.fit import SHOGPFamily
from transieve.lightcurve import LightCurve


_THETA = np.array([np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)])


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_lc():
    """Small, uniform light curve with a known GP draw."""
    np.random.seed(42)
    n = 200
    time = np.linspace(0.0, 5.0, n)
    flux_err = np.full(n, 1e-4)
    family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    lc0 = LightCurve.from_arrays(time=time, flux=np.zeros(n), flux_err=flux_err)
    gp = family.build_gp_from_theta(_THETA, lc0)
    flux = gp.sample()
    return LightCurve.from_arrays(time=time, flux=flux, flux_err=flux_err)


@pytest.fixture
def fitted_gp(simple_lc):
    """A GaussianProcess pre-computed on simple_lc at the true theta."""
    family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    return family.build_gp_from_theta(_THETA, simple_lc)
