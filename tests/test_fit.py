"""Tests for gp/fit.py."""

from __future__ import annotations

import numpy as np
import pytest
from celerite2 import GaussianProcess

from transieve.gp.fit import (
    ExpGPFamily,
    GPFamily,
    SHOGPFamily,
    make_neg_log_like,
)
from transieve.lightcurve import LightCurve


class TestSHOGPFamily:
    def test_bounds_length(self):
        fam = SHOGPFamily()
        # 3 params: log_omega, log_sigma, log_quality
        assert len(fam.bounds) == 3

    def test_bounds_length_with_jitter(self):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        assert len(fam.bounds) == 4
        assert "log_jitter" in fam.theta_names

    def test_theta_to_dict_keys(self):
        fam = SHOGPFamily()
        theta = np.array([1.0, 2.0, 3.0])
        d = fam.theta_to_dict(theta)
        assert set(d.keys()) == {"log_omega", "log_sigma", "log_quality"}

    def test_theta_to_physical_keys(self):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        theta = np.array(
            [np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)]
        )
        phys = fam.theta_to_physical(theta)
        assert set(phys.keys()) == {"period", "sigma", "quality", "jitter"}

    def test_theta_to_physical_values(self):
        fam = SHOGPFamily()
        omega = 2 * np.pi / 3.0  # period = 3
        sigma = 1e-3
        quality = 4.0
        theta = np.array([np.log(omega), np.log(sigma), np.log(quality)])
        phys = fam.theta_to_physical(theta)
        assert abs(phys["period"] - 3.0) < 1e-10
        assert abs(phys["sigma"] - sigma) < 1e-10
        assert abs(phys["quality"] - quality) < 1e-10

    def test_build_returns_gp(self, simple_lc):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        theta = np.array(
            [np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)]
        )
        gp = fam.build_gp_from_theta(theta, simple_lc)
        assert isinstance(gp, GaussianProcess)

    def test_build_requires_time(self):
        fam = SHOGPFamily()
        params = {"log_omega": 0.5, "log_sigma": -7.0, "log_quality": 0.7}
        with pytest.raises(ValueError, match="time must be provided"):
            fam.build(params, time=None)

    def test_build_gp_from_theta_wrong_type(self):
        fam = SHOGPFamily()
        theta = np.array([0.5, -7.0, 0.7])
        with pytest.raises(TypeError, match="LightCurve"):
            fam.build_gp_from_theta(theta, object())

    def test_sample_theta_in_bounds(self, rng):
        fam = SHOGPFamily()
        theta = fam.sample_theta(rng=rng)
        assert theta.shape == (3,)
        for val, (lo, hi) in zip(theta, fam.bounds):
            assert lo <= val <= hi

    def test_physical_bounds_keys(self):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        pb = fam.physical_bounds
        assert "period" in pb and "sigma" in pb and "quality" in pb and "jitter" in pb

    def test_from_string_sho(self):
        fam = GPFamily.from_string("sho")
        assert isinstance(fam, SHOGPFamily)

    def test_from_string_unknown(self):
        with pytest.raises(ValueError, match="Unknown GP family"):
            GPFamily.from_string("foobar")

    def test_fit_light_curve_returns_result(self, simple_lc):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        theta0 = np.array(
            [np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)]
        )
        result = fam.fit_light_curve(
            simple_lc, method="L-BFGS-B", initial_theta=theta0, max_retries=1
        )
        assert hasattr(result, "x")
        assert len(result.x) == 4
        assert np.isfinite(result.fun)

    def test_fit_light_curve_wrong_type(self):
        fam = SHOGPFamily()
        with pytest.raises(TypeError, match="LightCurve"):
            fam.fit_light_curve(object())

    def test_validate_white_noise_baseline_no_jitter_no_err(self):
        fam = SHOGPFamily()  # no jitter
        with pytest.raises(ValueError, match="flux_err"):
            fam.validate_white_noise_baseline(flux_err=None)

    def test_validate_white_noise_baseline_with_err(self):
        fam = SHOGPFamily()
        fam.validate_white_noise_baseline(flux_err=np.ones(10) * 1e-4)

    def test_validate_white_noise_baseline_with_jitter(self):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        fam.validate_white_noise_baseline(flux_err=None)

    def test_sample_light_curve(self, simple_lc):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        theta = np.array(
            [np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)]
        )
        sample = fam.sample_light_curve(theta, simple_lc)
        assert sample.shape == simple_lc.flux.shape


class TestExpGPFamily:
    def test_bounds_length(self):
        fam = ExpGPFamily()
        assert len(fam.bounds) == 2

    def test_bounds_length_with_jitter(self):
        fam = ExpGPFamily(jitter_range=(1e-6, 1e-2))
        assert len(fam.bounds) == 3
        assert "log_jitter" in fam.theta_names

    def test_theta_to_physical_keys(self):
        fam = ExpGPFamily(jitter_range=(1e-6, 1e-2))
        theta = np.array([np.log(5.0), np.log(1e-3), np.log(1e-4)])
        phys = fam.theta_to_physical(theta)
        assert set(phys.keys()) == {"scale", "sigma", "jitter"}

    def test_build_returns_gp(self):
        fam = ExpGPFamily(jitter_range=(1e-6, 1e-2))
        time = np.linspace(0, 5, 50)
        flux_err = np.full(50, 1e-4)
        lc = LightCurve.from_arrays(time, np.zeros(50), flux_err=flux_err)
        theta = np.array([np.log(5.0), np.log(1e-3), np.log(1e-4)])
        gp = fam.build_gp_from_theta(theta, lc)
        assert isinstance(gp, GaussianProcess)

    def test_from_string_exp(self):
        fam = GPFamily.from_string("exp")
        assert isinstance(fam, ExpGPFamily)

    def test_fit_light_curve(self):
        np.random.seed(99)
        n = 100
        time = np.linspace(0, 5, n)
        flux_err = np.full(n, 1e-4)
        fam = ExpGPFamily(jitter_range=(1e-6, 1e-2))
        lc0 = LightCurve.from_arrays(time, np.zeros(n), flux_err=flux_err)
        theta0 = np.array([np.log(2.0), np.log(5e-4), np.log(5e-5)])
        gp = fam.build_gp_from_theta(theta0, lc0)
        flux = gp.sample()
        lc = LightCurve.from_arrays(time, flux, flux_err=flux_err)
        result = fam.fit_light_curve(lc, method="L-BFGS-B", initial_theta=theta0)
        assert hasattr(result, "x")
        assert np.isfinite(result.fun)


class TestMakeNegLogLike:
    def test_finite_at_valid_theta(self, simple_lc):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        nll = make_neg_log_like(fam, simple_lc)
        theta = np.array(
            [np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)]
        )
        val = nll(theta)
        assert np.isfinite(val)

    def test_returns_inf_on_bad_theta(self, simple_lc):
        fam = SHOGPFamily(jitter_range=(1e-6, 1e-2))
        nll = make_neg_log_like(fam, simple_lc)
        # Extreme theta that will likely cause a numerical issue
        val = nll(np.array([1e10, 1e10, 1e10, 1e10]))
        assert not np.isneginf(val)  # Must be finite or +inf, not -inf
