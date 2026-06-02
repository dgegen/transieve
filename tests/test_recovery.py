import numpy as np
import pytest

from transieve import SimulatedLightCurve
from transieve.gp import (
    SHOGPFamily,
    assess_retrievability,
    scan_gp_stability_map,
)
from transieve.simulation import generate_gap_windows
from transieve.transit import get_monotransit_from_epoch


def test_assess_retrievability_with_callable_template_bank():
    lc = SimulatedLightCurve.from_transit(
        epoch=0.0,
        depth=0.002,
        duration=0.25,
        period=120.0,
        baseline=5.0,
        cadence=20.0,
        multiply_signal=False,
        seed=7,
    )

    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    template_generator = get_monotransit_from_epoch(
        lc.time,
        depth=0.002,
        duration=0.25,
        period=120.0,
        mean=0.0,
    )

    initial_theta = np.array(
        [
            lc.gp_params["log_omega"],
            lc.gp_params["log_sigma"],
            lc.gp_params["log_quality"],
            lc.gp_params["log_jitter"],
        ]
    )

    result = assess_retrievability(
        time=lc.time,
        flux=lc.flux,
        template_bank=template_generator,
        gp_family=gp_family,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
        center_flux=True,
        robust_threshold=True,
    )

    assert result.z_score.shape == lc.time.shape
    assert result.z_white_noise.shape == lc.time.shape
    assert np.all(np.isfinite(result.recovery_fraction))

    peak_idx, peak_z = result.strongest_match()
    assert 0 <= peak_idx < len(lc.time)
    assert np.isfinite(peak_z)


def test_scan_gp_stability_map_shapes_and_plateau_mask():
    lc = SimulatedLightCurve.from_transit(
        epoch=0.0,
        depth=0.002,
        duration=0.2,
        period=100.0,
        baseline=4.0,
        cadence=20.0,
        multiply_signal=False,
        seed=11,
    )

    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    template = lc.deterministic_component - 1.0

    initial_theta = np.array(
        [
            lc.gp_params["log_omega"],
            lc.gp_params["log_sigma"],
            lc.gp_params["log_quality"],
            lc.gp_params["log_jitter"],
        ]
    )

    sigma_grid = np.geomspace(2e-4, 2e-3, 4)
    period_grid = np.geomspace(0.5, 20.0, 5)

    stability = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        template=template,
        gp_family=gp_family,
        sigma_grid=sigma_grid,
        timescale_grid=period_grid,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
    )

    assert stability.z_score.shape == (len(sigma_grid), len(period_grid))
    assert stability.recovery_fraction.shape == stability.z_score.shape
    assert stability.get_plateau_mask(z_threshold=3).shape == stability.z_score.shape
    assert stability.timescale_name == "period"

    best = stability.strongest_point()
    assert "sigma" in best
    assert "period" in best
    assert "z_score" in best


def test_robust_features_values():
    lc = SimulatedLightCurve.from_transit(
        epoch=0.0,
        depth=0.002,
        duration=0.2,
        period=100.0,
        baseline=4.0,
        cadence=20.0,
        multiply_signal=False,
        seed=11,
    )

    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    template = lc.deterministic_component - 1.0

    initial_theta = np.array(
        [
            lc.gp_params["log_omega"],
            lc.gp_params["log_sigma"],
            lc.gp_params["log_quality"],
            lc.gp_params["log_jitter"],
        ]
    )

    stability = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        template=template,
        gp_family=gp_family,
        sigma_grid=np.geomspace(2e-4, 2e-3, 4),
        timescale_grid=np.geomspace(0.5, 20.0, 5),
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
    )

    features = stability.robust_features(z_threshold=3.0, max_fragility=0.1)

    assert set(features) == {
        "log_space_plateau_area",
        "capacity_bounded_peak_z",
        "max_z_plateau",
        "peak_brittleness",
        "is_safe_harbor",
    }
    assert features["log_space_plateau_area"] >= 0.0
    assert features["capacity_bounded_peak_z"] >= 0.0
    assert features["peak_brittleness"] >= 0.0
    assert features["is_safe_harbor"] in (0.0, 1.0)

    summary = stability.summary(z_threshold=3.0)
    assert "log_space_plateau_area" in summary


def test_marginalized_z():
    lc = SimulatedLightCurve.from_transit(
        epoch=0.0,
        depth=0.002,
        duration=0.2,
        period=100.0,
        baseline=4.0,
        cadence=20.0,
        multiply_signal=False,
        seed=11,
    )

    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    template = lc.deterministic_component - 1.0

    initial_theta = np.array(
        [
            lc.gp_params["log_omega"],
            lc.gp_params["log_sigma"],
            lc.gp_params["log_quality"],
            lc.gp_params["log_jitter"],
        ]
    )

    stability = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        template=template,
        gp_family=gp_family,
        sigma_grid=np.geomspace(2e-4, 2e-3, 4),
        timescale_grid=np.geomspace(0.5, 20.0, 5),
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
    )

    assert stability.log_marginal_likelihood is not None
    assert stability.log_marginal_likelihood.shape == stability.z_score.shape
    assert np.all(np.isfinite(stability.log_marginal_likelihood))

    z_marg = stability.marginalized_z()
    assert np.isfinite(z_marg)
    # Marginalized Z should not exceed the grid maximum (it's a weighted average)
    assert z_marg <= float(np.nanmax(stability.z_score)) + 1e-9


def test_marginalized_z_gate_excludes_broken_cells():
    """Broken cells with inflated Z far outside the likelihood gate must not bias the integral."""
    from transieve.gp.stability import GPStabilityMap

    rng = np.random.default_rng(42)
    sigma_grid = np.geomspace(1e-4, 1e-2, 10)
    timescale_grid = np.geomspace(0.5, 20.0, 10)

    # One valid cell (top-left) with a healthy signal; rest are numerically broken.
    z = np.full((10, 10), 11.5)  # inflated, unphysical Z for broken cells
    z[0, 0] = 3.5  # the one physically valid detection

    log_L = np.full((10, 10), -1000.0)  # floor: broken GP solver output
    log_L[0, 0] = 0.0  # best-fit cell

    stab = GPStabilityMap(
        sigma_grid=sigma_grid,
        timescale_grid=timescale_grid,
        timescale_name="period",
        z_score=z,
        z_white_noise=np.ones((10, 10)),
        recovery_fraction=np.ones((10, 10)),
        relative_capacity=np.ones((10, 10)),
        local_drop=np.zeros((10, 10)),
        reference_gp_params={},
        log_marginal_likelihood=log_L,
    )

    # With the gate (default delta_log_L=16), only the valid cell contributes.
    z_gated = stab.marginalized_z(delta_log_L=16.0)
    assert abs(z_gated - 3.5) < 1e-9, f"Expected 3.5, got {z_gated}"

    # Without the gate, broken cells at Δlog_L=1000 are still negligible (e^-1000 ≈ 0),
    # but using a tiny gate of 0.5 should also isolate just the valid cell.
    z_tight = stab.marginalized_z(delta_log_L=0.5)
    assert abs(z_tight - 3.5) < 1e-9


# ---------------------------------------------------------------------------
# Pipeline behaviour with observational gaps
# ---------------------------------------------------------------------------


@pytest.fixture
def base_lc():
    return SimulatedLightCurve.from_transit(
        epoch=0.0,
        depth=0.002,
        duration=0.25,
        period=120.0,
        baseline=5.0,
        cadence=20.0,
        multiply_signal=False,
        seed=7,
    )


def _run_pipeline(lc):
    """Run GP fitting + matched filtering on a light curve and return the result."""
    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    template_generator = get_monotransit_from_epoch(
        lc.time,
        depth=0.002,
        duration=0.25,
        period=120.0,
        mean=0.0,
    )
    initial_theta = np.array(
        [
            lc.gp_params["log_omega"],
            lc.gp_params["log_sigma"],
            lc.gp_params["log_quality"],
            lc.gp_params["log_jitter"],
        ]
    )
    return assess_retrievability(
        time=lc.time,
        flux=lc.flux,
        template_bank=template_generator,
        gp_family=gp_family,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
        center_flux=True,
        robust_threshold=True,
    )


def test_pipeline_time_gaps_runs_and_produces_finite_results(base_lc):
    """GP fitting and matched filtering succeed on non-uniform (gapped) time grids.

    celerite2 handles non-uniform cadence natively, so removing cadences should
    leave the pipeline fully functional.
    """
    windows = generate_gap_windows(
        base_lc.time, n_gaps=2, gap_duration_range=(0.3, 0.8), seed=5
    )
    gapped = base_lc.with_gaps(windows, mode="remove")

    result = _run_pipeline(gapped)

    assert np.all(np.isfinite(result.z_score))
    assert np.all(np.isfinite(result.z_white_noise))


def test_pipeline_nan_flux_raises(base_lc):
    """assess_retrievability raises ValueError when flux contains NaN values."""
    windows = generate_gap_windows(
        base_lc.time, n_gaps=1, gap_duration_range=(0.3, 0.5), seed=5
    )
    gapped = base_lc.with_gaps(windows, mode="nan")

    with pytest.raises(ValueError, match="NaN"):
        _run_pipeline(gapped)
