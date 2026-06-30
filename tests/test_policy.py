"""Tests for gp/policy.py."""

from __future__ import annotations

import numpy as np
import pytest

from transieve.gp.policy import TransitVettingPolicy
from transieve.gp.detection import FrequentistDetectionResult
from transieve.gp.stability import GPStabilityMap


def _make_retrievability(z_peak: float, recovery_peak: float, n: int = 20):
    """Build a minimal FrequentistDetectionResult with a known peak at index n//2."""
    time = np.linspace(0.0, 5.0, n)
    z = np.full(n, 1.0)
    z[n // 2] = z_peak
    recovery = np.full(n, recovery_peak)
    return FrequentistDetectionResult(
        time=time,
        z_score=z,
        template_norm=np.ones(n),
        z_white_noise=np.ones(n),
        recovery_fraction=recovery,
        relative_capacity=np.ones(n),
        depth_to_threshold=np.zeros(n),
        sensitivity=np.ones(n),
        detection_threshold=7.0,
        gp_theta=np.zeros(4),
        gp_physical_params={},
        gp_optimization_result=None,
    )


def _make_stability_map(marg_z: float):
    """Build a minimal GPStabilityMap whose marginalized_z() returns approx marg_z."""
    shape = (4, 4)
    sigma_grid = np.logspace(-1, 0, shape[0])
    ts_grid = np.logspace(-1, 0, shape[1])
    z_score = np.full(shape, marg_z)
    log_ml = np.zeros(shape)
    return GPStabilityMap(
        sigma_grid=sigma_grid,
        timescale_grid=ts_grid,
        timescale_name="period",
        z_score=z_score,
        z_white_noise=np.ones(shape),
        recovery_fraction=np.ones(shape),
        relative_capacity=np.ones(shape),
        local_drop=np.zeros(shape),
        reference_gp_params={"sigma": sigma_grid[2], "period": ts_grid[2]},
        log_marginal_likelihood=log_ml,
    )


def test_policy_passes_on_strong_detection():
    retr = _make_retrievability(z_peak=10.0, recovery_peak=0.8)
    policy = TransitVettingPolicy(z_threshold=7.0, recovery_fraction_cutoff=0.5)
    result = policy.validate(retr)
    assert result.passed is True
    assert result.report["significance_ok"] is True
    assert result.report["recovery_ok"] is True


def test_policy_fails_on_weak_z():
    retr = _make_retrievability(z_peak=5.0, recovery_peak=0.8)
    policy = TransitVettingPolicy(z_threshold=7.0, recovery_fraction_cutoff=0.5)
    result = policy.validate(retr)
    assert result.passed is False
    assert result.report["significance_ok"] is False


def test_policy_fails_on_low_recovery():
    retr = _make_retrievability(z_peak=10.0, recovery_peak=0.2)
    policy = TransitVettingPolicy(z_threshold=7.0, recovery_fraction_cutoff=0.5)
    result = policy.validate(retr)
    assert result.passed is False
    assert result.report["recovery_ok"] is False


def test_policy_validate_stability_map():
    smap = _make_stability_map(marg_z=8.0)
    policy = TransitVettingPolicy(z_threshold=7.0)
    result = policy.validate(smap)
    assert result.passed is True
    assert result.report["profiled_max_z"] == pytest.approx(8.0, abs=0.1)


def test_vetting_result_is_namedtuple():
    retr = _make_retrievability(z_peak=10.0, recovery_peak=0.8)
    policy = TransitVettingPolicy()
    result = policy.validate(retr)
    assert isinstance(result, tuple)
    assert result.passed is True
    assert "z" in result.report


def test_policy_wrong_type_raises():
    policy = TransitVettingPolicy()
    with pytest.raises(TypeError, match="FrequentistDetectionResult or GPStabilityMap"):
        policy.validate("not a result")


def test_profiled_max_z_calculation():
    shape = (3, 3)
    sigma_grid = np.logspace(-1, 0, shape[0])
    ts_grid = np.logspace(-1, 0, shape[1])
    z_score = np.array([[1.0, 2.0, 3.0], [4.0, 8.5, 6.0], [7.0, 8.0, 9.0]])
    log_ml = np.array(
        [
            [-100.0, -100.0, -100.0],
            [-100.0, 0.0, -100.0],  # MLE at (1, 1)
            [-100.0, -100.0, -100.0],
        ]
    )
    smap = GPStabilityMap(
        sigma_grid=sigma_grid,
        timescale_grid=ts_grid,
        timescale_name="period",
        z_score=z_score,
        z_white_noise=np.ones(shape),
        recovery_fraction=np.ones(shape),
        relative_capacity=np.ones(shape),
        local_drop=np.zeros(shape),
        reference_gp_params={},
        log_marginal_likelihood=log_ml,
    )
    assert smap.profiled_max_z() == 8.5


def test_policy_validate_stability_map_profiled():
    shape = (3, 3)
    sigma_grid = np.logspace(-1, 0, shape[0])
    ts_grid = np.logspace(-1, 0, shape[1])
    z_score = np.array([[1.0, 2.0, 3.0], [4.0, 8.5, 6.0], [7.0, 8.0, 9.0]])
    log_ml = np.array(
        [
            [-100.0, -100.0, -100.0],
            [-100.0, 0.0, -100.0],  # MLE at (1, 1) → profiled z = 8.5
            [-100.0, -100.0, -100.0],
        ]
    )
    smap = GPStabilityMap(
        sigma_grid=sigma_grid,
        timescale_grid=ts_grid,
        timescale_name="period",
        z_score=z_score,
        z_white_noise=np.ones(shape),
        recovery_fraction=np.ones(shape),
        relative_capacity=np.ones(shape),
        local_drop=np.zeros(shape),
        reference_gp_params={},
        log_marginal_likelihood=log_ml,
    )

    policy = TransitVettingPolicy(z_threshold=7.0)
    result = policy.validate(smap)
    assert result.report["profiled_max_z"] == pytest.approx(8.5, abs=1e-9)
    assert result.passed is True
