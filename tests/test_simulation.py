import numpy as np
import pytest

from transieve.simulation import (
    generate_time,
    generate_gap_windows,
    inject_gaps,
    construct_antisymmetric_template,
    SimulatedLightCurve,
)


class TestGenerateTime:
    def test_length(self):
        t = generate_time(baseline=30.0, cadence=10.0)
        expected = int(np.ceil(30.0 * 24 * 60 / 10.0))
        assert len(t) == expected

    def test_symmetric_around_zero(self):
        t = generate_time(baseline=30.0, cadence=10.0)
        assert pytest.approx(t[0], abs=1e-6) == -15.0
        assert pytest.approx(t[-1], abs=1e-6) == 15.0

    def test_cadence_affects_length(self):
        t5 = generate_time(baseline=10.0, cadence=5.0)
        t10 = generate_time(baseline=10.0, cadence=10.0)
        assert len(t5) > len(t10)


class TestConstructAntisymmetricTemplate:
    def test_output_shape(self):
        time = np.linspace(-1, 1, 100)
        signal = np.ones(100)
        result = construct_antisymmetric_template(time, signal, center=0.0)
        assert result.shape == signal.shape

    def test_left_side_negated(self):
        time = np.linspace(-1, 1, 101)
        signal = np.ones(101)
        result = construct_antisymmetric_template(time, signal, center=0.0)
        mid_idx = np.argmin(np.abs(time - 0.0))
        assert np.all(result[:mid_idx] == -1.0)
        assert np.all(result[mid_idx:] == 1.0)

    def test_does_not_modify_original(self):
        time = np.linspace(-1, 1, 100)
        signal = np.ones(100)
        original = signal.copy()
        construct_antisymmetric_template(time, signal, center=0.0)
        np.testing.assert_array_equal(signal, original)

    def test_off_center(self):
        time = np.linspace(0, 2, 200)
        signal = np.ones(200)
        center = 1.0
        result = construct_antisymmetric_template(time, signal, center=center)
        mid_idx = np.argmin(np.abs(time - center))
        assert np.all(result[:mid_idx] == -1.0)
        assert np.all(result[mid_idx:] == 1.0)


class TestSimulatedLightCurve:
    @pytest.fixture
    def slc(self):
        return SimulatedLightCurve.from_transit(epoch=0.0, seed=42)

    def test_from_transit_returns_instance(self, slc):
        assert isinstance(slc, SimulatedLightCurve)

    def test_arrays_same_length(self, slc):
        n = len(slc.time)
        assert len(slc.flux) == n
        assert len(slc.deterministic_component) == n
        assert len(slc.gp_component) == n

    def test_flux_is_finite(self, slc):
        assert np.all(np.isfinite(slc.flux))

    def test_signal_params_stored(self, slc):
        assert slc.signal_params["epoch"] == 0.0

    def test_gp_params_stored(self, slc):
        assert "log_sigma" in slc.gp_params

    def test_seed_reproducibility(self):
        a = SimulatedLightCurve.from_transit(epoch=0.0, seed=0)
        b = SimulatedLightCurve.from_transit(epoch=0.0, seed=0)
        np.testing.assert_array_equal(a.flux, b.flux)

    def test_different_seeds_differ(self):
        a = SimulatedLightCurve.from_transit(epoch=0.0, seed=1)
        b = SimulatedLightCurve.from_transit(epoch=0.0, seed=2)
        assert not np.allclose(a.flux, b.flux)

    def test_custom_time(self):
        time = np.linspace(-5, 5, 200)
        slc = SimulatedLightCurve.from_transit(epoch=0.0, time=time, seed=0)
        np.testing.assert_array_equal(slc.time, time)

    def test_deterministic_component_has_dip(self, slc):
        # The deterministic component should dip below 1.0 (transit signal)
        assert slc.deterministic_component.min() < 1.0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model must be one of"):
            SimulatedLightCurve.from_transit(epoch=0.0, model="invalid")

    def test_multiply_vs_additive(self):
        # Verify the two composition modes apply different formulas.
        # multiply: flux = gp * det
        # additive: flux = gp + det - 1
        # With a non-trivial GP component the difference is (gp-1)*(det-1),
        # which we check directly via the stored components.
        slc = SimulatedLightCurve.from_transit(
            epoch=0.0, seed=42, baseline=5, cadence=10
        )
        gp = slc.gp_component
        det = slc.deterministic_component
        np.testing.assert_allclose(
            slc.flux, gp * det
        )  # default is multiply_signal=True

        slc_add = SimulatedLightCurve.from_transit(
            epoch=0.0, seed=42, baseline=5, cadence=10, multiply_signal=False
        )
        gp2 = slc_add.gp_component
        det2 = slc_add.deterministic_component
        np.testing.assert_allclose(slc_add.flux, gp2 + det2 - 1.0)

    def test_from_model_with_custom_signal(self):
        from transieve.transit import get_monotransit_model

        time = np.linspace(-5, 5, 300)
        slc = SimulatedLightCurve.from_model(
            signal_params={"epoch": 0.0, "depth": 0.01, "duration": 0.2},
            gp_params=SimulatedLightCurve.DEFAULT_GP_PARAMS.copy(),
            signal_factory=get_monotransit_model,
            time=time,
            seed=7,
        )
        assert len(slc.flux) == len(time)
        assert np.all(np.isfinite(slc.flux))


class TestGenerateGapWindows:
    def test_returns_requested_number_of_windows(self):
        time = np.linspace(-15, 15, 4320)
        windows = generate_gap_windows(time, n_gaps=3, seed=0)
        assert len(windows) == 3

    def test_windows_within_baseline(self):
        time = np.linspace(-15, 15, 4320)
        windows = generate_gap_windows(
            time, n_gaps=5, gap_duration_range=(0.5, 2.0), seed=1
        )
        for t_start, t_end in windows:
            assert t_start >= time[0]
            assert t_end <= time[-1]

    def test_window_duration_within_range(self):
        time = np.linspace(-15, 15, 4320)
        min_dur, max_dur = 0.5, 2.0
        windows = generate_gap_windows(
            time, n_gaps=10, gap_duration_range=(min_dur, max_dur), seed=2
        )
        for t_start, t_end in windows:
            assert t_end - t_start >= min_dur
            assert t_end - t_start <= max_dur

    def test_seed_reproducibility(self):
        time = np.linspace(-15, 15, 4320)
        a = generate_gap_windows(time, n_gaps=3, seed=42)
        b = generate_gap_windows(time, n_gaps=3, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        time = np.linspace(-15, 15, 4320)
        a = generate_gap_windows(time, n_gaps=3, seed=0)
        b = generate_gap_windows(time, n_gaps=3, seed=1)
        assert a != b


class TestInjectGaps:
    @pytest.fixture
    def arrays(self):
        time = np.linspace(-5, 5, 1000)
        flux = np.ones(1000)
        return time, flux

    def test_remove_reduces_length(self, arrays):
        time, flux = arrays
        t_out, f_out = inject_gaps(time, flux, [(-2.0, -1.0)], mode="remove")
        assert len(t_out) < len(time)
        assert len(t_out) == len(f_out)

    def test_remove_drops_correct_cadences(self, arrays):
        time, flux = arrays
        t_out, _ = inject_gaps(time, flux, [(-2.0, -1.0)], mode="remove")
        assert not np.any((t_out >= -2.0) & (t_out <= -1.0))

    def test_nan_preserves_length(self, arrays):
        time, flux = arrays
        t_out, f_out = inject_gaps(time, flux, [(-2.0, -1.0)], mode="nan")
        assert len(t_out) == len(time)
        assert len(f_out) == len(flux)

    def test_nan_masks_correct_cadences(self, arrays):
        time, flux = arrays
        _, f_out = inject_gaps(time, flux, [(-2.0, -1.0)], mode="nan")
        in_gap = (time >= -2.0) & (time <= -1.0)
        assert np.all(np.isnan(f_out[in_gap]))
        assert np.all(np.isfinite(f_out[~in_gap]))

    def test_invalid_mode_raises(self, arrays):
        time, flux = arrays
        with pytest.raises(ValueError, match="mode must be"):
            inject_gaps(time, flux, [], mode="drop")

    def test_no_gaps_is_identity_remove(self, arrays):
        time, flux = arrays
        t_out, f_out = inject_gaps(time, flux, [], mode="remove")
        np.testing.assert_array_equal(t_out, time)
        np.testing.assert_array_equal(f_out, flux)


class TestSimulatedLightCurveWithGaps:
    @pytest.fixture
    def slc(self):
        return SimulatedLightCurve.from_transit(
            epoch=0.0, baseline=10, cadence=10, seed=42
        )

    def test_remove_mode_reduces_length(self, slc):
        gapped = slc.with_gaps([(-3.0, -1.0)], mode="remove")
        assert len(gapped.time) < len(slc.time)

    def test_remove_mode_all_arrays_consistent(self, slc):
        gapped = slc.with_gaps([(-3.0, -1.0)], mode="remove")
        n = len(gapped.time)
        assert len(gapped.flux) == n
        assert len(gapped.deterministic_component) == n
        assert len(gapped.gp_component) == n

    def test_remove_mode_no_cadences_in_gap(self, slc):
        gapped = slc.with_gaps([(-3.0, -1.0)], mode="remove")
        assert not np.any((gapped.time >= -3.0) & (gapped.time <= -1.0))

    def test_nan_mode_preserves_length(self, slc):
        gapped = slc.with_gaps([(-3.0, -1.0)], mode="nan")
        assert len(gapped.time) == len(slc.time)
        assert len(gapped.flux) == len(slc.flux)

    def test_nan_mode_masks_flux_in_gap(self, slc):
        gapped = slc.with_gaps([(-3.0, -1.0)], mode="nan")
        in_gap = (gapped.time >= -3.0) & (gapped.time <= -1.0)
        assert np.all(np.isnan(gapped.flux[in_gap]))
        assert np.all(np.isfinite(gapped.flux[~in_gap]))

    def test_nan_mode_does_not_modify_original(self, slc):
        original_flux = slc.flux.copy()
        slc.with_gaps([(-3.0, -1.0)], mode="nan")
        np.testing.assert_array_equal(slc.flux, original_flux)

    def test_invalid_mode_raises(self, slc):
        with pytest.raises(ValueError, match="mode must be"):
            slc.with_gaps([(-1.0, 0.0)], mode="drop")

    def test_metadata_preserved(self, slc):
        gapped = slc.with_gaps([(-3.0, -1.0)], mode="remove")
        assert gapped.signal_params == slc.signal_params
        assert gapped.gp_params == slc.gp_params

    def test_generate_gap_windows_roundtrip(self, slc):
        windows = generate_gap_windows(slc.time, n_gaps=2, seed=99)
        gapped = slc.with_gaps(windows, mode="remove")
        assert len(gapped.time) < len(slc.time)
        assert np.all(np.isfinite(gapped.flux))
