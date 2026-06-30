import numpy as np
import pytest

from transieve import SimulatedLightCurve
from transieve.gp import (
    SHOGPFamily,
    evaluate_frequentist_detection,
    evaluate_bayesian_detection,
    scan_gp_stability_map,
)
from transieve.gp.match import TemplateBank
from transieve.simulation import generate_gap_windows
from transieve.transit import get_monotransit_from_epoch


def test_evaluate_frequentist_detection_with_callable_template_bank():
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

    result = evaluate_frequentist_detection(
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


def test_evaluate_bayesian_detection_with_callable_template_bank():
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

    result = evaluate_bayesian_detection(
        time=lc.time,
        flux=lc.flux,
        template_bank=template_generator,
        sigma_a=2e-3,
        gp_family=gp_family,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
        center_flux=True,
    )

    assert result.log_bayes_factor.shape == lc.time.shape
    assert np.isfinite(result.log_bayes_factor_global)
    peak_idx, peak_log_bf = result.strongest_match()
    assert 0 <= peak_idx < len(lc.time)
    assert np.isfinite(peak_log_bf)


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
    assert stability.timescale_name == "period"

    best = stability.strongest_point()
    assert "sigma" in best
    assert "period" in best
    assert "z_score" in best


def test_scan_gp_stability_map_global_mode():
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
    template_generator = get_monotransit_from_epoch(
        lc.time,
        depth=0.002,
        duration=0.2,
        period=100.0,
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

    sigma_grid = np.geomspace(2e-4, 2e-3, 3)
    period_grid = np.geomspace(0.5, 20.0, 4)
    epochs = lc.time[::25]
    bank = TemplateBank(make=template_generator, epochs=epochs)
    template_matrix = np.column_stack(list(bank))

    stability = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        context="global",
        template_matrix=template_matrix,
        sigma_a=2e-3,
        gp_family=gp_family,
        sigma_grid=sigma_grid,
        timescale_grid=period_grid,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
    )

    assert stability.log_bf_global is not None
    assert stability.log_bf_global.shape == (len(sigma_grid), len(period_grid))
    assert np.isfinite(stability.marginalized_bayes_factor())


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
    return evaluate_frequentist_detection(
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
    """evaluate_frequentist_detection raises ValueError when flux contains NaN values."""
    windows = generate_gap_windows(
        base_lc.time, n_gaps=1, gap_duration_range=(0.3, 0.5), seed=5
    )
    gapped = base_lc.with_gaps(windows, mode="nan")

    with pytest.raises(ValueError, match="NaN"):
        _run_pipeline(gapped)


# ---------------------------------------------------------------------------
# GPStabilityMap tensor fields and reduction properties
# ---------------------------------------------------------------------------


class TestGPStabilityMapTensorProperties:
    S, T_g, M, T_t = 3, 4, 2, 5

    def _stub(self):
        from transieve.gp.stability import GPStabilityMap

        rng = np.random.default_rng(0)
        zt = rng.standard_normal((self.S, self.T_g, self.M, self.T_t))
        lbt = rng.standard_normal((self.S, self.T_g, self.M, self.T_t))
        stub = GPStabilityMap(
            sigma_grid=np.ones(self.S),
            timescale_grid=np.ones(self.T_g),
            timescale_name="period",
            z_score=zt.max(axis=(2, 3)),
            z_white_noise=np.zeros((self.S, self.T_g)),
            recovery_fraction=np.zeros((self.S, self.T_g)),
            relative_capacity=np.zeros((self.S, self.T_g)),
            local_drop=np.zeros((self.S, self.T_g)),
            reference_gp_params={},
            z_score_tensor=zt,
            log_bf_tensor=lbt,
        )
        return stub, zt, lbt

    def test_profiled_z_score_map_shape(self):
        stub, _, _ = self._stub()
        assert stub.profiled_z_score_map.shape == (self.S, self.T_g, self.T_t)

    def test_marginalized_log_bf_map_shape(self):
        stub, _, _ = self._stub()
        assert stub.marginalized_log_bf_map.shape == (self.S, self.T_g, self.T_t)

    def test_profiled_z_score_map_is_nanmax_over_duration(self):
        stub, zt, _ = self._stub()
        np.testing.assert_allclose(stub.profiled_z_score_map, np.nanmax(zt, axis=2))

    def test_raises_without_tensor(self):
        from transieve.gp.stability import GPStabilityMap

        stub = GPStabilityMap(
            sigma_grid=np.ones(2),
            timescale_grid=np.ones(2),
            timescale_name="period",
            z_score=np.zeros((2, 2)),
            z_white_noise=np.zeros((2, 2)),
            recovery_fraction=np.zeros((2, 2)),
            relative_capacity=np.zeros((2, 2)),
            local_drop=np.zeros((2, 2)),
            reference_gp_params={},
        )
        with pytest.raises(ValueError):
            _ = stub.profiled_z_score_map
        with pytest.raises(ValueError):
            _ = stub.marginalized_log_bf_map


# ---------------------------------------------------------------------------
# scan_gp_stability_map with 3-D template_matrix
# ---------------------------------------------------------------------------


def test_scan_gp_stability_map_3d_template_matrix_populates_tensor(base_lc):
    lc = base_lc
    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    initial_theta = np.array(
        [
            lc.gp_params["log_omega"],
            lc.gp_params["log_sigma"],
            lc.gp_params["log_quality"],
            lc.gp_params["log_jitter"],
        ]
    )
    sigma_grid = np.geomspace(2e-4, 2e-3, 2)
    period_grid = np.geomspace(0.5, 20.0, 3)
    epochs = lc.time[::50]

    # Build two duration banks sharing the same epoch grid.
    bank_short = TemplateBank(
        make=get_monotransit_from_epoch(
            lc.time, depth=0.002, duration=0.15, period=120.0, mean=0.0
        ),
        epochs=epochs,
    )
    bank_long = TemplateBank(
        make=get_monotransit_from_epoch(
            lc.time, depth=0.002, duration=0.35, period=120.0, mean=0.0
        ),
        epochs=epochs,
    )
    template_matrix_3d = np.stack(
        [np.column_stack(list(bank_short)), np.column_stack(list(bank_long))], axis=1
    )  # (N, 2, t_transit)

    stability = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        context="global",
        template_matrix=template_matrix_3d,
        sigma_a=2e-3,
        gp_family=gp_family,
        sigma_grid=sigma_grid,
        timescale_grid=period_grid,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": initial_theta},
    )

    S, T_g, M, T_t = len(sigma_grid), len(period_grid), 2, len(epochs)
    assert stability.z_score.shape == (S, T_g)
    assert stability.z_score_tensor is not None
    assert stability.z_score_tensor.shape == (S, T_g, M, T_t)
    assert stability.log_bf_tensor is not None
    assert stability.log_bf_tensor.shape == (S, T_g, M, T_t)
    assert stability.profiled_z_score_map.shape == (S, T_g, T_t)
    assert stability.marginalized_log_bf_map.shape == (S, T_g, T_t)
    assert np.all(np.isfinite(stability.z_score))


# ---------------------------------------------------------------------------
# Multi-duration detection
# ---------------------------------------------------------------------------


def _initial_theta(lc):
    return np.array(
        [
            lc.gp_params["log_omega"],
            lc.gp_params["log_sigma"],
            lc.gp_params["log_quality"],
            lc.gp_params["log_jitter"],
        ]
    )


def test_evaluate_frequentist_detection_multi_bank(base_lc):
    lc = base_lc
    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    epochs = lc.time[::30]
    banks = [
        TemplateBank(
            make=get_monotransit_from_epoch(
                lc.time, depth=0.002, duration=d, period=120.0, mean=0.0
            ),
            epochs=epochs,
        )
        for d in (0.15, 0.25, 0.35)
    ]

    result = evaluate_frequentist_detection(
        time=lc.time,
        flux=lc.flux,
        template_bank=banks,
        gp_family=gp_family,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": _initial_theta(lc)},
        center_flux=True,
    )

    from transieve.gp.detection import MultiProfileFrequentistDetectionResult

    T_t = len(epochs)
    M = len(banks)
    assert isinstance(result, MultiProfileFrequentistDetectionResult)
    assert result.z_score.shape == (T_t,)
    assert result.z_score_tensor.shape == (M, T_t)
    assert result.template_norm_tensor.shape == (M, T_t)
    assert result.profiled_z_score.shape == (T_t,)
    np.testing.assert_allclose(result.profiled_z_score, result.z_score)
    assert np.all(result.z_score <= np.nanmax(result.z_score_tensor) + 1e-10)


def test_evaluate_bayesian_detection_multi_bank(base_lc):
    lc = base_lc
    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    epochs = lc.time[::30]
    banks = [
        TemplateBank(
            make=get_monotransit_from_epoch(
                lc.time, depth=0.002, duration=d, period=120.0, mean=0.0
            ),
            epochs=epochs,
        )
        for d in (0.15, 0.25, 0.35)
    ]

    result = evaluate_bayesian_detection(
        time=lc.time,
        flux=lc.flux,
        template_bank=banks,
        sigma_a=2e-3,
        gp_family=gp_family,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": _initial_theta(lc)},
        center_flux=True,
    )

    from transieve.gp.detection import MultiProfileBayesianDetectionResult

    T_t = len(epochs)
    M = len(banks)
    assert isinstance(result, MultiProfileBayesianDetectionResult)
    assert result.log_bayes_factor.shape == (T_t,)
    assert result.log_bayes_factor_tensor.shape == (M, T_t)
    assert result.marginalized_log_bf.shape == (T_t,)
    assert np.isfinite(result.log_bayes_factor_global)


def test_evaluate_multi_bank_mismatched_epochs_raises(base_lc):
    lc = base_lc
    banks = [
        TemplateBank(make=lambda t: np.zeros_like(lc.time), epochs=lc.time[::30]),
        TemplateBank(make=lambda t: np.zeros_like(lc.time), epochs=lc.time[::40]),
    ]
    with pytest.raises(ValueError, match="epoch grid"):
        evaluate_frequentist_detection(
            time=lc.time,
            flux=lc.flux,
            template_bank=banks,
            fit_method="L-BFGS-B",
            fit_max_retries=1,
        )


# ---------------------------------------------------------------------------
# Sensitivity mode (operational_mode="sensitivity")
# ---------------------------------------------------------------------------


def test_scan_gp_stability_map_local_sensitivity(base_lc):
    """Local context + sensitivity mode populates ideal_sensitivity maps."""
    lc = base_lc
    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    template = lc.deterministic_component - 1.0

    sigma_grid = np.geomspace(2e-4, 2e-3, 3)
    period_grid = np.geomspace(0.5, 20.0, 4)

    stab = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        template=template,
        gp_family=gp_family,
        sigma_grid=sigma_grid,
        timescale_grid=period_grid,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": _initial_theta(lc)},
        operational_mode="sensitivity",
        z_threshold=5.0,
    )

    assert stab.ideal_sensitivity is not None
    assert stab.ideal_sensitivity.shape == (len(sigma_grid), len(period_grid))
    assert stab.realization_adjusted_sensitivity is not None
    assert stab.realization_adjusted_sensitivity.shape == stab.ideal_sensitivity.shape
    assert np.all(np.isfinite(stab.ideal_sensitivity))
    # Ideal sensitivity is data-independent: z_threshold / norm — always positive.
    assert np.all(stab.ideal_sensitivity > 0)
    # z_score is still populated as a byproduct.
    assert stab.z_score.shape == (len(sigma_grid), len(period_grid))


def test_scan_gp_stability_map_global_sensitivity_2d(base_lc):
    """Global context + sensitivity mode populates ideal_sensitivity for a 2-D template matrix."""
    lc = base_lc
    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    epochs = lc.time[::50]
    bank = TemplateBank(
        make=get_monotransit_from_epoch(
            lc.time, depth=0.002, duration=0.25, period=120.0, mean=0.0
        ),
        epochs=epochs,
    )
    template_matrix = np.column_stack(list(bank))  # (N, t_transit)

    sigma_grid = np.geomspace(2e-4, 2e-3, 2)
    period_grid = np.geomspace(0.5, 20.0, 3)

    stab = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        context="global",
        template_matrix=template_matrix,
        gp_family=gp_family,
        sigma_grid=sigma_grid,
        timescale_grid=period_grid,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": _initial_theta(lc)},
        operational_mode="sensitivity",
        z_threshold=5.0,
    )

    T_t = len(epochs)
    assert stab.ideal_sensitivity is not None
    assert stab.ideal_sensitivity.shape == (len(sigma_grid), len(period_grid), T_t)
    assert stab.realization_adjusted_sensitivity is not None
    assert np.all(stab.ideal_sensitivity > 0)
    # No Bayes factor maps in sensitivity mode.
    assert stab.log_bf_global is None
    assert stab.log_bf_conditional is None


def test_scan_gp_stability_map_global_sensitivity_3d(base_lc):
    """Global context + sensitivity mode populates ideal_sensitivity for a 3-D template matrix."""
    lc = base_lc
    gp_family = SHOGPFamily(jitter_range=(1e-6, 1e-2))
    epochs = lc.time[::50]

    bank_short = TemplateBank(
        make=get_monotransit_from_epoch(
            lc.time, depth=0.002, duration=0.15, period=120.0, mean=0.0
        ),
        epochs=epochs,
    )
    bank_long = TemplateBank(
        make=get_monotransit_from_epoch(
            lc.time, depth=0.002, duration=0.35, period=120.0, mean=0.0
        ),
        epochs=epochs,
    )
    template_matrix_3d = np.stack(
        [np.column_stack(list(bank_short)), np.column_stack(list(bank_long))], axis=1
    )  # (N, 2, t_transit)

    sigma_grid = np.geomspace(2e-4, 2e-3, 2)
    period_grid = np.geomspace(0.5, 20.0, 3)

    stab = scan_gp_stability_map(
        time=lc.time,
        flux=lc.flux,
        context="global",
        template_matrix=template_matrix_3d,
        gp_family=gp_family,
        sigma_grid=sigma_grid,
        timescale_grid=period_grid,
        fit_mean=1.0,
        fit_method="L-BFGS-B",
        fit_max_retries=1,
        fit_kwargs={"initial_theta": _initial_theta(lc)},
        operational_mode="sensitivity",
        z_threshold=5.0,
    )

    S, T_g, M, T_t = len(sigma_grid), len(period_grid), 2, len(epochs)
    assert stab.ideal_sensitivity is not None
    assert stab.ideal_sensitivity.shape == (S, T_g, M, T_t)
    assert stab.z_score_tensor is not None
    assert stab.z_score_tensor.shape == (S, T_g, M, T_t)
    # BF tensor is not computed in sensitivity mode.
    assert stab.log_bf_tensor is None
    assert np.all(stab.ideal_sensitivity > 0)


def test_scan_gp_stability_map_sensitivity_requires_z_threshold(base_lc):
    """operational_mode='sensitivity' without z_threshold must raise ValueError."""
    lc = base_lc
    template = lc.deterministic_component - 1.0
    with pytest.raises(ValueError, match="z_threshold"):
        scan_gp_stability_map(
            time=lc.time,
            flux=lc.flux,
            template=template,
            fit_method="L-BFGS-B",
            fit_max_retries=1,
            operational_mode="sensitivity",
        )
