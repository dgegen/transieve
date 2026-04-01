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
