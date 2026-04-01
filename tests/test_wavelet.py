import numpy as np
import pytest

import transieve
from transieve.wavelet import (
    CWTResult,
    CWTTransitVettingResult,
    OWTSESResult,
    SWTMatchedFilterResult,
    cwt_diagnostics,
    cwt_transit_vetting,
    default_cwt_scales,
    duration_matched_cwt_scales,
    evaluate_monotransit_candidate,
    kepler_owt_ses_filter,
    sliding_variance,
    wavelet_matched_filter,
)


class TestWaveletExports:
    def test_lazy_submodule_import(self):
        assert hasattr(transieve, "wavelet")

    def test_result_types_exported(self):
        assert CWTResult is not None
        assert SWTMatchedFilterResult is not None
        assert OWTSESResult is not None


class TestSlidingVariance:
    def test_output_shape(self):
        x = np.random.default_rng(0).normal(size=256)
        out = sliding_variance(x, window=17)
        assert out.shape == x.shape

    def test_variance_floor(self):
        x = np.ones(128)
        out = sliding_variance(x, window=9)
        assert np.all(out >= 1e-16)


class TestSWTMatchedFilter:
    @pytest.fixture
    def arrays(self):
        rng = np.random.default_rng(1)
        n = 512
        time = np.linspace(-3.0, 3.0, n)

        template = np.exp(-0.5 * (time / 0.15) ** 2)
        template = template - np.mean(template)

        flux = 2e-3 * rng.normal(size=n)
        flux += -0.02 * template
        return flux, template

    def test_output_lengths(self, arrays):
        flux, template = arrays
        result = wavelet_matched_filter(
            flux,
            template,
            swt_levels=5,
            base_window_samples=32,
        )

        assert isinstance(result, SWTMatchedFilterResult)
        assert len(result.z_score) == len(flux)
        assert len(result.numerator) == len(flux)
        assert len(result.denominator) == len(flux)
        assert len(result.channels) == result.swt_levels + 1

    def test_candidate_evaluation_shape(self, arrays):
        flux, template = arrays
        result = wavelet_matched_filter(flux, template, swt_levels=4)
        idx, _ = result.strongest_match()

        metrics = evaluate_monotransit_candidate(
            result,
            idx=idx,
            flux=flux,
            template=template,
            n_shifts=200,
        )

        assert "empirical_p" in metrics
        assert "coherence" in metrics
        assert "residual_reduction" in metrics
        assert 0.0 <= metrics["empirical_p"] <= 1.0
        assert 0.0 <= metrics["adj_p_trials"] <= 1.0


class TestKeplerOWTSES:
    @pytest.fixture
    def arrays(self):
        rng = np.random.default_rng(2)
        n = 512
        time = np.linspace(-3.0, 3.0, n)

        template = np.exp(-0.5 * (time / 0.12) ** 2)
        template = template - np.mean(template)

        flux = 2e-3 * rng.normal(size=n)
        flux += -0.03 * template
        return flux, template

    def test_output_lengths(self, arrays):
        flux, template = arrays
        result = kepler_owt_ses_filter(
            flux,
            template,
            owt_levels=4,
            base_window_samples=32,
        )

        assert isinstance(result, OWTSESResult)
        assert len(result.ses) == len(flux)
        assert len(result.numerator) == len(flux)
        assert len(result.denominator) == len(flux)
        assert len(result.channels) == result.swt_levels

    def test_detects_injected_feature(self, arrays):
        flux, template = arrays
        result = kepler_owt_ses_filter(flux, template, owt_levels=4)
        idx, ses = result.strongest_match(absolute=True)

        assert 0 <= idx < len(flux)
        assert np.isfinite(ses)
        assert np.nanmax(np.abs(result.ses)) > 0.0

    def test_blind_strongest_match_auto_excludes_edges(self):
        n = 256
        rng = np.random.default_rng(123)
        flux = 1e-3 * rng.normal(size=n)
        template = np.hanning(21)
        template = template - np.mean(template)

        result = kepler_owt_ses_filter(
            flux,
            template,
            owt_levels=3,
            base_window_samples=32,
        )

        # Inject synthetic edge spikes to emulate boundary artifacts.
        result.ses[0] = 1e6
        result.ses[-1] = 1e6
        result.ses[n // 2] = 100.0

        idx_auto, _ = result.strongest_match(absolute=True)
        idx_raw, _ = result.strongest_match(absolute=True, edge_exclusion_samples=0)

        assert idx_raw in (0, n - 1)
        assert idx_auto == n // 2


class TestCWTDiagnostics:
    def test_default_scales(self):
        scales = default_cwt_scales(n_samples=512, n_scales=24)
        assert scales.shape == (24,)
        assert np.all(np.diff(scales) > 0)

    def test_cwt_result_shapes(self):
        n = 512
        time = np.linspace(0.0, 10.0, n)
        flux = np.sin(2 * np.pi * time / 1.5)

        result = cwt_diagnostics(time, flux)

        assert isinstance(result, CWTResult)
        assert result.coefficients.shape[0] == len(result.scales)
        assert result.coefficients.shape[1] == len(time)
        assert result.power.shape == result.coefficients.shape
        assert result.global_spectrum.shape == (len(result.scales),)
        assert result.salience.shape == (len(time),)

    def test_rejects_nan_flux(self):
        time = np.linspace(0.0, 2.0, 64)
        flux = np.sin(time)
        flux[10] = np.nan

        with pytest.raises(ValueError, match="flux contains NaN"):
            cwt_diagnostics(time, flux)

    def test_duration_matched_scales(self):
        scales = duration_matched_cwt_scales(
            duration=0.2,
            cadence=0.02,
            multipliers=(0.5, 1.0, 1.5),
        )
        assert np.all(scales > 0)
        assert np.all(np.diff(scales) > 0)
        assert scales.shape[0] >= 2

    def test_cwt_transit_vetting_tracks_epoch(self):
        rng = np.random.default_rng(42)
        time = np.linspace(-5.0, 5.0, 1024)
        duration = 0.25
        epoch = 0.3

        dip = -0.02 * np.exp(-0.5 * ((time - epoch) / (duration / 2.35)) ** 2)
        flux = dip + 0.002 * rng.normal(size=time.size)

        vet = cwt_transit_vetting(
            time=time,
            flux=flux,
            epoch=epoch,
            duration=duration,
            wavelet="mexh",
        )

        assert isinstance(vet, CWTTransitVettingResult)
        assert abs(vet.local_time - epoch) <= duration
        assert vet.local_salience <= vet.global_salience + 1e-12
        assert vet.best_timescale > 0
