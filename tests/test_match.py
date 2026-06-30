"""Tests for gp/match.py."""

from __future__ import annotations

import numpy as np
import pytest

from transieve.gp.fit import SHOGPFamily
from transieve.gp.match import MatchedFilter, SearchProfile, TemplateBank
from transieve.lightcurve import LightCurve

_THETA = np.array([np.log(2 * np.pi / 2.0), np.log(5e-4), np.log(2.0), np.log(5e-5)])


def _build_matched_filter(lc: LightCurve, family=None) -> MatchedFilter:
    if family is None:
        family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    gp = family.build_gp_from_theta(_THETA, lc)
    centered = lc.flux - np.median(lc.flux)
    return MatchedFilter(gp=gp, flux=centered, check_zero_centered=False)


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

    def test_log_bayes_factor_gaussian_matches_projection(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        template = np.zeros(len(simple_lc.time))
        template[90:110] = -1e-3
        projection, template_norm = mf.template_projection_and_norm(template)
        sigma_a = 2e-3
        expected = MatchedFilter._log_bayes_factor_from_projection(
            projection, template_norm, sigma_a
        )
        log_bf = mf.log_bayes_factor(template, sigma_a)
        assert log_bf == pytest.approx(float(expected), rel=1e-9, abs=1e-12)

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
        stats = mf.z_score_map(bank, epochs=epochs)
        assert isinstance(stats, SearchProfile)
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
        assert isinstance(stats, SearchProfile)
        assert stats.z_score.shape == (n,)
        assert stats.template_norm.shape == (n,)


class TestMatchedFilterBatchMethods:
    """Tests for matrix_metrics, log_bayes_factor_matrix, compute_sensitivity_limits."""

    def _mf_and_matrix(self, simple_lc, m=3):
        mf = _build_matched_filter(simple_lc)
        n = len(simple_lc.time)
        template = np.zeros(n)
        template[90:110] = -1e-3
        mat = np.column_stack([np.roll(template, k * 5) for k in range(m)])
        return mf, mat, template

    def test_matrix_metrics_shapes(self, simple_lc):
        mf, mat, _ = self._mf_and_matrix(simple_lc, m=3)
        result = mf.matrix_metrics(mat)
        for key in (
            "z_score",
            "template_norm",
            "z_white_noise",
            "white_template_norm",
            "recovery_fraction",
            "relative_capacity",
        ):
            assert result[key].shape == (3,), f"{key} shape mismatch"
            assert np.all(np.isfinite(result[key])), f"{key} contains non-finite values"

    def test_matrix_metrics_agrees_with_template_metrics(self, simple_lc):
        mf, _, template = self._mf_and_matrix(simple_lc)
        scalar = mf.template_metrics(template)
        batch = mf.matrix_metrics(template[:, None])
        for key in scalar:
            assert batch[key][0] == pytest.approx(scalar[key], rel=1e-9, abs=1e-14), key

    def test_matrix_metrics_bad_shape_raises(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        with pytest.raises(ValueError, match="shape"):
            mf.matrix_metrics(np.zeros((5, 3)))

    def test_log_bayes_factor_matrix_shape(self, simple_lc):
        mf, mat, _ = self._mf_and_matrix(simple_lc, m=4)
        result = mf.log_bayes_factor_matrix(mat, sigma_a=2e-3)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_log_bayes_factor_matrix_agrees_with_scalar(self, simple_lc):
        mf, _, template = self._mf_and_matrix(simple_lc)
        scalar = mf.log_bayes_factor(template, sigma_a=2e-3)
        batch = mf.log_bayes_factor_matrix(template[:, None], sigma_a=2e-3)
        assert batch[0] == pytest.approx(scalar, rel=1e-9, abs=1e-14)

    def test_compute_sensitivity_limits_scalar(self, simple_lc):
        mf, _, template = self._mf_and_matrix(simple_lc)
        result = mf.compute_sensitivity_limits(template, z_threshold=7.0)
        assert np.isfinite(result["ideal"]) and result["ideal"] > 0
        assert np.isfinite(result["realization_adjusted"])

    def test_compute_sensitivity_limits_matrix_shapes(self, simple_lc):
        mf, mat, _ = self._mf_and_matrix(simple_lc, m=5)
        result = mf.compute_sensitivity_limits(mat, z_threshold=7.0)
        assert result["ideal"].shape == (5,)
        assert result["realization_adjusted"].shape == (5,)
        assert np.all(np.isfinite(result["ideal"]))
        assert np.all(np.isfinite(result["realization_adjusted"]))


class TestMatrixMetricsND:
    """matrix_metrics and log_bayes_factor_matrix with N-D template arrays."""

    def test_3d_output_shape(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        N, M, T = len(simple_lc.time), 3, 5
        tm = np.random.default_rng(0).standard_normal((N, M, T)) * 1e-4
        result = mf.matrix_metrics(tm)
        for key in result:
            assert result[key].shape == (M, T), f"{key}: got {result[key].shape}"

    def test_2d_output_shape_unchanged(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        N, K = len(simple_lc.time), 7
        tm = np.random.default_rng(1).standard_normal((N, K)) * 1e-4
        result = mf.matrix_metrics(tm)
        assert result["z_score"].shape == (K,)

    def test_3d_values_match_flat_2d(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        N, M, T = len(simple_lc.time), 2, 4
        tm3d = np.random.default_rng(2).standard_normal((N, M, T)) * 1e-4
        r3d = mf.matrix_metrics(tm3d)
        r2d = mf.matrix_metrics(tm3d.reshape(N, M * T))
        np.testing.assert_allclose(r3d["z_score"].ravel(), r2d["z_score"])
        np.testing.assert_allclose(r3d["template_norm"].ravel(), r2d["template_norm"])

    def test_log_bayes_factor_matrix_3d_shape(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        N, M, T = len(simple_lc.time), 2, 6
        tm = np.random.default_rng(3).standard_normal((N, M, T)) * 1e-4
        lbf = mf.log_bayes_factor_matrix(tm, sigma_a=1e-3)
        assert lbf.shape == (M, T)

    def test_log_bayes_factor_matrix_3d_values_match_flat_2d(self, simple_lc):
        mf = _build_matched_filter(simple_lc)
        N, M, T = len(simple_lc.time), 2, 4
        tm3d = np.random.default_rng(4).standard_normal((N, M, T)) * 1e-4
        lbf3d = mf.log_bayes_factor_matrix(tm3d, sigma_a=1e-3)
        lbf2d = mf.log_bayes_factor_matrix(tm3d.reshape(N, M * T), sigma_a=1e-3)
        np.testing.assert_allclose(lbf3d.ravel(), lbf2d)
