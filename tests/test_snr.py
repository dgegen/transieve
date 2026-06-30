import numpy as np
from transieve.snr import snr_red_noise, calculate_v_n, RedNoiseSnrResult


def test_snr_red_noise_without_duration():
    # Simple test to verify backwards compatibility when duration is None.
    np.random.seed(42)
    time = np.linspace(0.0, 10.0, 1000)
    # Add a bit of correlation to noise
    noise = np.random.normal(0.0, 0.01, 1000)
    for i in range(1, 1000):
        noise[i] += 0.5 * noise[i - 1]

    depth = 0.005
    n_in_transit = 15

    result = snr_red_noise(
        depth=depth,
        time=time,
        residuals=noise,
        n_in_transit=n_in_transit,
        duration=None,
    )

    assert isinstance(result, RedNoiseSnrResult)
    assert result.snr_white > 0
    assert result.snr_diff > 0
    assert result.beta >= 1.0
    assert result.snr_red > 0
    assert len(result.betas) == 20  # default bin_times size


def test_snr_red_noise_with_duration():
    np.random.seed(42)
    time = np.linspace(0.0, 10.0, 1000)
    # White noise + red noise trend
    noise = np.random.normal(0.0, 0.01, 1000) + 0.02 * np.sin(2 * np.pi * time / 2.0)

    depth = 0.01
    duration = 0.1  # in days
    n_in_transit = 10

    result = snr_red_noise(
        depth=depth,
        time=time,
        residuals=noise,
        n_in_transit=n_in_transit,
        duration=duration,
    )

    assert isinstance(result, RedNoiseSnrResult)
    assert result.snr_white > 0
    assert result.snr_diff > 0
    assert result.beta >= 0.0
    assert result.snr_red > 0
    assert len(result.betas) > 0


def test_snr_red_noise_with_duration_grouping():
    # Test grouping by nj and calculating variance in each bin.
    # Regular grid implies nj will be highly uniform.
    time = np.linspace(0.0, 1.0, 100)  # dt = 0.01
    residuals = np.random.normal(0.0, 0.01, 100)

    # Choose duration such that it spans exactly 10 points:
    # duration = 0.10, so window is [g - 0.05, g + 0.05] which spans roughly 11 points (including endpoints).
    duration = 0.10

    result = snr_red_noise(
        depth=0.01,
        time=time,
        residuals=residuals,
        n_in_transit=10,
        duration=duration,
    )

    # We should have calculated some betas
    assert len(result.betas) > 0
    assert result.beta > 0


def test_calculate_v_n_white_noise_scaling():
    # Pure Gaussian white noise: V(n) should scale as 1/n, i.e. V(n)*n ~ sigma_w^2.
    # Use several window durations so each run is dominated by a different
    # bulk (interior) n, well-sampled by many windows, avoiding the noisy
    # boundary n-groups that only contain a handful of edge windows.
    np.random.seed(0)
    sigma_w = 0.01
    time = np.linspace(0.0, 20.0, 4000)
    residuals = np.random.normal(0.0, sigma_w, time.size)

    for duration in (0.1, 0.2, 0.4, 0.8):
        v_n = calculate_v_n(time, residuals, duration)
        assert len(v_n) > 0
        # The dominant (interior) n is the one backed by the most windows.
        unique_n = np.array(sorted(v_n.keys()))
        residuals_dt = (time[-1] - time[0]) / (time.size - 1)
        bulk_n = unique_n[np.argmin(np.abs(unique_n - round(duration / residuals_dt)))]
        np.testing.assert_allclose(v_n[bulk_n] * bulk_n, sigma_w**2, rtol=0.3)


def test_calculate_v_n_data_gaps():
    # High-cadence region plus a region with significant gaps (TESS-style).
    np.random.seed(1)
    sigma_w = 0.01
    dense_time = np.arange(0.0, 5.0, 0.001)
    # Gappy region: bursts of points separated by large gaps.
    gappy_chunks = [
        np.arange(start, start + 0.05, 0.001) for start in np.arange(5.0, 10.0, 0.5)
    ]
    gappy_time = np.concatenate(gappy_chunks)
    time = np.concatenate([dense_time, gappy_time])
    residuals = np.random.normal(0.0, sigma_w, time.size)

    duration = 0.05
    v_n = calculate_v_n(time, residuals, duration)

    assert len(v_n) > 0
    # Windows with fewer points (from the gappy region) should have higher variance
    # than windows with many points (from the dense region).
    n_values = np.array(sorted(v_n.keys()))
    low_n = n_values[: max(1, len(n_values) // 4)]
    high_n = n_values[-max(1, len(n_values) // 4) :]
    mean_var_low = np.mean([v_n[n] for n in low_n])
    mean_var_high = np.mean([v_n[n] for n in high_n])
    assert mean_var_low > mean_var_high


def test_calculate_v_n_red_noise_saturation():
    # Strong time-correlated systematics: V(n) should saturate at large n
    # instead of continuing to fall off as 1/n.
    np.random.seed(2)
    sigma_r = 0.02
    time = np.linspace(0.0, 20.0, 4000)
    # Slowly varying correlated signal (red noise) with a long correlation timescale.
    residuals = sigma_r * np.sin(2 * np.pi * time / 3.0)
    residuals += np.random.normal(0.0, sigma_r * 0.05, time.size)  # tiny white jitter

    residuals_dt = (time[-1] - time[0]) / (time.size - 1)

    # Small duration (approx 10 points)
    dur_small = 0.05
    v_n_small = calculate_v_n(time, residuals, dur_small)
    n_small_keys = np.array(sorted(v_n_small.keys()))
    bulk_n_small = n_small_keys[
        np.argmin(np.abs(n_small_keys - round(dur_small / residuals_dt)))
    ]
    var_small = v_n_small[bulk_n_small]

    # Large duration (approx 80 points)
    dur_large = 0.4
    v_n_large = calculate_v_n(time, residuals, dur_large)
    n_large_keys = np.array(sorted(v_n_large.keys()))
    bulk_n_large = n_large_keys[
        np.argmin(np.abs(n_large_keys - round(dur_large / residuals_dt)))
    ]
    var_large = v_n_large[bulk_n_large]

    # V(n) should not drop nearly as fast as 1/n: ratio of variances much
    # smaller than the ratio of n's that 1/n scaling would predict.
    n_ratio = bulk_n_large / bulk_n_small
    var_ratio = var_small / var_large
    assert var_ratio < n_ratio / 2

    # For large n, V(n) should be close to the red-noise variance sigma_r^2 / 2
    # (variance of a sinusoid), i.e. it has saturated rather than vanished.
    assert var_large > 0.1 * (sigma_r**2)


def test_snr_red_noise_multiple_transits():
    # Multiple transits with varying point counts: n_in_transit is a sequence.
    np.random.seed(3)
    time = np.linspace(0.0, 20.0, 2000)
    residuals = np.random.normal(0.0, 0.01, time.size)

    duration = 0.1
    n_k = [8, 12, 10]  # three transits with different in-transit point counts

    result = snr_red_noise(
        depth=0.01,
        time=time,
        residuals=residuals,
        n_in_transit=n_k,
        duration=duration,
    )

    assert isinstance(result, RedNoiseSnrResult)
    assert result.snr_red > 0
    assert result.beta > 0

    # Total n used for snr_white/snr_diff should be the sum of n_k.
    sigma_1 = float(np.std(residuals))
    expected_snr_white = 0.01 / sigma_1 * np.sqrt(sum(n_k))
    np.testing.assert_allclose(result.snr_white, expected_snr_white, rtol=1e-6)
