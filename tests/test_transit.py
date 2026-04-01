import numpy as np
import pytest

from transieve.transit import (
    transit,
    get_monotransit_model,
    get_monotransit_from_epoch,
    _batman_params,
)


@pytest.fixture
def time():
    return np.linspace(-1, 1, 500)


class TestTransit:
    def test_output_shape(self, time):
        result = transit(time, epoch=0.0, duration=0.2, period=100.0)
        assert result.shape == time.shape

    def test_minimum_at_epoch(self, time):
        epoch = 0.0
        result = transit(time, epoch=epoch, duration=0.2, period=100.0)
        min_idx = np.argmin(result)
        assert abs(time[min_idx] - epoch) < 0.05

    def test_negative_dip(self, time):
        result = transit(time, epoch=0.0, duration=0.2, period=100.0)
        assert result.min() < 0.0

    def test_out_of_transit_near_zero(self, time):
        # Far from transit the value should be close to 0
        result = transit(time, epoch=0.0, duration=0.2, period=100.0)
        assert abs(result[0]) < 0.01
        assert abs(result[-1]) < 0.01

    def test_depth_scales_with_depth_param(self, time):
        r1 = transit(time, epoch=0.0, duration=0.2, period=100.0, c=12)
        r2 = transit(time, epoch=0.0, duration=0.2, period=100.0, c=24)
        # Higher c → sharper / deeper minimum
        assert r2.min() <= r1.min()

    def test_epoch_shift(self, time):
        r0 = transit(time, epoch=0.0, duration=0.2, period=100.0)
        r1 = transit(time, epoch=0.3, duration=0.2, period=100.0)
        assert not np.allclose(r0, r1)
        assert abs(time[np.argmin(r1)] - 0.3) < 0.05


class TestGetMonotransitModel:
    def test_returns_callable(self):
        model = get_monotransit_model(epoch=0.0)
        assert callable(model)

    def test_output_shape(self, time):
        model = get_monotransit_model(epoch=0.0, depth=0.01, duration=0.2)
        flux = model(time)
        assert flux.shape == time.shape

    def test_mean_baseline(self, time):
        mean = 1.0
        model = get_monotransit_model(epoch=0.0, depth=0.01, duration=0.2, mean=mean)
        flux = model(time)
        # Out-of-transit flux should be close to mean
        assert abs(flux[0] - mean) < 1e-3

    def test_depth(self, time):
        depth = 0.01
        model = get_monotransit_model(epoch=0.0, depth=depth, duration=0.2, mean=1.0)
        flux = model(time)
        # At epoch the flux dips below mean by approximately depth
        assert flux.min() < 1.0 - depth * 0.9

    def test_zero_depth_is_flat(self, time):
        model = get_monotransit_model(epoch=0.0, depth=0.0, mean=1.0)
        flux = model(time)
        assert np.allclose(flux, 1.0)


class TestGetMonotransitFromEpoch:
    def test_returns_callable(self, time):
        fn = get_monotransit_from_epoch(time)
        assert callable(fn)

    def test_default_epoch_at_midpoint(self, time):
        fn = get_monotransit_from_epoch(time, depth=0.01, mean=1.0)
        flux = fn()
        mid = (time[-1] + time[0]) / 2
        min_idx = np.argmin(flux)
        assert abs(time[min_idx] - mid) < 0.1

    def test_custom_epoch(self, time):
        fn = get_monotransit_from_epoch(time, depth=0.01, mean=1.0)
        flux = fn(epoch=0.5)
        min_idx = np.argmin(flux)
        assert abs(time[min_idx] - 0.5) < 0.1


class TestBatmanParams:
    def test_batman_params(self):
        batman = pytest.importorskip("batman", exc_type=ImportError)
        params = _batman_params(
            epoch=0.0,
            duration=0.1,
            depth=0.01,
            period=10.0,
            impact_param=0.0,
            u=[0.3, 0.1],
            limb_dark="quadratic",
        )
        assert isinstance(params, batman.TransitParams)
        assert params.t0 == 0.0
        assert params.per == 10.0
        assert pytest.approx(params.rp, rel=1e-4) == np.sqrt(0.01)
        assert params.inc == pytest.approx(90.0, abs=1.0)  # b=0 → near 90 deg
