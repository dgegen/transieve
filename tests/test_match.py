"""Tests for gp/match.py."""

from __future__ import annotations

import numpy as np
import pytest

from transieve.gp.fit import SHOGPFamily
from transieve.gp.match import MatchedFilter, MatchedFilterStatistics, TemplateBank
from transieve.lightcurve import LightCurve


_THETA = np.array([np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)])


def _build_matched_filter(lc: LightCurve, family=None) -> MatchedFilter:
    if family is None:
        family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    gp = family.build_gp_from_theta(_THETA, lc)
    centered = lc.flux - np.median(lc.flux)
    return MatchedFilter(gp=gp, flux=centered, check_zero_centered=False)


def _make_stats(n=50, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    norm = np.abs(rng.standard_normal(n)) + 0.1
    return MatchedFilterStatistics(z_score=z, template_norm=norm)


class TestTemplateBank:
    def test_len(self):
        bank = TemplateBank(make=lambda t: np.zeros(10), epochs=np.linspace(0, 4, 5))
        assert len(bank) == 5

    def test_iter_length(self):
        bank = TemplateBank(
            make=lambda t: np.full(8, t), epochs=np.array([0.0, 1.0, 2.0])
        )
        assert len(list(bank)) == 3

    def test_iter_values(self):
        bank = TemplateBank(make=lambda t: np.full(4, t), epochs=np.array([1.0, 2.0]))
        templates = list(bank)
        np.testing.assert_array_equal(templates[0], np.full(4, 1.0))
        np.testing.assert_array_equal(templates[1], np.full(4, 2.0))

    def test_coerce_callable(self):
        epochs = np.array([0.0, 1.0])
        bank = TemplateBank._coerce(lambda t: np.zeros(5), default_epochs=epochs)
        assert isinstance(bank, TemplateBank)
        assert len(bank) == 2

    def test_coerce_passthrough(self):
        original = TemplateBank(make=lambda t: np.zeros(5), epochs=np.array([0.0, 1.0]))
        coerced = TemplateBank._coerce(original, default_epochs=np.array([]))
        assert coerced is original

    def test_epochs_coerced_to_float(self):
        bank = TemplateBank(make=lambda t: np.zeros(3), epochs=np.array([0, 1, 2]))
        assert bank.epochs.dtype == float


class TestMatchedFilterConstructor:
    def test_raises_on_nan_flux(self, fitted_gp, simple_lc):
        flux_with_nan = simple_lc.flux.copy()
        flux_with_nan[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            MatchedFilter(gp=fitted_gp, flux=flux_with_nan)

    def test_raises_on_non_zero_centered(self, fitted_gp, simple_lc):
        flux_near_one = np.ones_like(simple_lc.flux)
        with pytest.raises(ValueError, match="zero-centered"):
            MatchedFilter(gp=fitted_gp, flux=flux_near_one)

    def test_no_check_zero_centered(self, fitted_gp, simple_lc):
        flux_near_one = np.ones_like(simple_lc.flux)
        mf = MatchedFilter(gp=fitted_gp, flux=flux_near_one, check_zero_centered=False)
        assert mf.flux is flux_near_one


class TestMatchedFilterOperations:
    def test_apply_inverse_variance_shape(self, simple_lc, fitted_gp):
        centered = simple_lc.flux - np.median(simple_lc.flux)
        mf = MatchedFilter(gp=fitted_gp, flux=centered, check_zero_centered=False)
        result = mf.apply_inverse_variance(centered)
        assert result.shape == centered.shape

    def test_whiten_shape(self, simple_lc, fitted_gp):
        centered = simple_lc.flux - np.median(simple_lc.flux)
        mf = MatchedFilter(gp=fitted_gp, flux=centered, check_zero_centered=False)
        w = mf.whiten(centered)
        assert w.shape == centered.shape
        assert np.all(np.isfinite(w))

    def test_template_norm_and_projection_shapes(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        template = np.zeros(len(simple_lc.time))
        template[90:110] = -1e-3
        norm, proj = mf.template_norm_and_projection(template)
        assert np.isscalar(norm) or norm.ndim == 0
        assert proj.shape == template.shape
        assert norm > 0

    def test_z_score_returns_two_floats(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        template = np.zeros(len(simple_lc.time))
        template[90:110] = -1e-3
        z, norm = mf.z_score(template)
        assert np.isfinite(z)
        assert norm > 0

    def test_z_score_inject_increases_z(self, simple_lc):
        """Injecting the template into the data should increase z-score."""
        mf = _build_matched_filter(simple_lc)
        template = np.zeros(len(simple_lc.time))
        template[90:110] = -5e-3
        z_no_inject, _ = mf.z_score(template, inject_template=False)
        z_inject, _ = mf.z_score(template, inject_template=True)
        assert z_inject > z_no_inject

    def test_white_noise_z_score_finite(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        template = np.zeros(len(simple_lc.time))
        template[80:100] = -1e-3
        z, norm = mf.white_noise_z_score(template)
        assert np.isfinite(z)
        assert norm > 0

    def test_white_noise_template_norm_positive(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        template = np.zeros(len(simple_lc.time))
        template[80:100] = -1e-3
        norm = mf.white_noise_template_norm(template)
        assert norm > 0

    def test_template_metrics_keys(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        template = np.zeros(len(simple_lc.time))
        template[80:100] = -1e-3
        metrics = mf.template_metrics(template)
        expected_keys = {
            "z_score",
            "template_norm",
            "z_white_noise",
            "white_template_norm",
            "recovery_fraction",
            "relative_capacity",
        }
        assert set(metrics.keys()) == expected_keys
        assert all(np.isfinite(v) for v in metrics.values())

    def test_z_score_map_shape(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        n = len(simple_lc.time)
        template_zero = np.zeros(n)
        template_zero[90:100] = -1e-3
        epochs = simple_lc.time[::20]
        bank = TemplateBank(make=lambda t: template_zero, epochs=epochs)
        stats = mf.z_score_map(bank)
        assert isinstance(stats, MatchedFilterStatistics)
        assert stats.z_score.shape == (len(epochs),)
        assert stats.template_norm.shape == (len(epochs),)

    def test_z_score_map_callable_uses_gp_times(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        n = len(simple_lc.time)
        stats = mf.z_score_map(lambda t: np.zeros(n))
        assert stats.z_score.shape == (n,)

    def test_z_score_map_fft_shape(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        n = len(simple_lc.time)
        template = np.zeros(n)
        template[90:110] = -1e-3
        stats = mf.z_score_map_fft(template)
        assert isinstance(stats, MatchedFilterStatistics)
        assert stats.z_score.shape == (n,)
        assert stats.template_norm.shape == (n,)


class TestMatchedFilterStatistics:
    def test_best_fit_scale(self):
        stats = MatchedFilterStatistics(
            z_score=np.array([2.0, 4.0]), template_norm=np.array([1.0, 2.0])
        )
        np.testing.assert_allclose(stats.best_fit_scale, np.array([2.0, 2.0]))

    def test_log_likelihood_ratio_non_negative(self):
        assert np.all(_make_stats().log_likelihood_ratio >= 0)

    def test_log_likelihood_ratio_zero_for_negative_z(self):
        stats = MatchedFilterStatistics(
            z_score=np.array([-3.0, -1.0]), template_norm=np.ones(2)
        )
        np.testing.assert_array_equal(stats.log_likelihood_ratio, np.zeros(2))

    def test_get_detectability_shapes(self):
        dtt, sens, thresh = _make_stats(n=30).get_detectability(threshold=7.0)
        assert dtt.shape == (30,)
        assert sens.shape == (30,)
        assert np.isfinite(thresh)

    def test_get_detectability_robust_false(self):
        _, _, thresh = _make_stats(n=20).get_detectability(threshold=5.0, robust=False)
        assert thresh == pytest.approx(5.0)

    def test_get_detectability_zero_noise_floor_raises(self):
        stats = MatchedFilterStatistics(z_score=np.ones(10), template_norm=np.ones(10))
        with pytest.raises(ValueError, match="Noise floor"):
            stats.get_detectability(threshold=7.0, robust=True)

    def test_theoretical_fap_in_range(self):
        fap = _make_stats(n=50, seed=7).theoretical_fap()
        assert 0.0 <= fap <= 1.0

    def test_empirical_significance_keys(self):
        result = _make_stats(n=50).empirical_significance()
        assert "p_value" in result and "empirical_z" in result

    def test_empirical_significance_with_window(self):
        result = _make_stats(n=50).empirical_significance(window_size=5)
        assert "p_value" in result

    def test_strongest_match_valid_index(self):
        stats = _make_stats(n=30)
        idx, val = stats.strongest_match()
        assert 0 <= idx < 30
        assert val == stats.z_score[idx]
        assert val == np.nanmax(stats.z_score)

    def test_repr(self):
        r = repr(_make_stats(n=10))
        assert "MatchedFilterStatistics" in r and "max_z_score" in r
