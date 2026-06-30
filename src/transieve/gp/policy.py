from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .detection import FrequentistDetectionResult
from .stability import GPStabilityMap

__all__ = ["TransitVettingPolicy", "VettingResult"]


class VettingResult(NamedTuple):
    passed: bool
    report: dict


@dataclass
class TransitVettingPolicy:
    """User-defined threshold bundle for transit vetting decisions.

    Parameters
    ----------
    z_threshold : float
        Minimum matched-filter Z-score to pass the significance gate.
    recovery_fraction_cutoff : float
        Minimum recovery fraction at the target epoch.
    ln_b_threshold : float
        Minimum log Bayes factor when vetting via a GPStabilityMap with
        populated log_bf_global.
    """

    z_threshold: float = 7.0
    recovery_fraction_cutoff: float = 0.5
    ln_b_threshold: float = 0.0

    def validate(
        self,
        result: FrequentistDetectionResult | GPStabilityMap,
        injected_time: float | None = None,
    ) -> VettingResult:
        """Apply threshold gates and return a structured verdict.

        Parameters
        ----------
        result :
            Either a FrequentistDetectionResult (Z-score path) or a GPStabilityMap
            (posterior-marginalised path).
        injected_time :
            If provided, evaluate at the closest epoch; otherwise evaluate at
            the strongest-match index.

        Returns
        -------
        VettingResult
            Named tuple with ``passed: bool`` and ``report: dict``.
        """
        if isinstance(result, FrequentistDetectionResult):
            return self._validate_retrievability(result, injected_time)
        if isinstance(result, GPStabilityMap):
            return self._validate_stability_map(result)
        raise TypeError(
            f"result must be FrequentistDetectionResult or GPStabilityMap, got {type(result)}"
        )

    def _validate_retrievability(
        self,
        result: FrequentistDetectionResult,
        injected_time: float | None,
    ) -> VettingResult:
        if injected_time is None:
            idx = int(np.nanargmax(result.z_score))
        else:
            t = np.asarray(result.time, dtype=float)
            idx = int(np.argmin(np.abs(t - float(injected_time))))

        z = float(result.z_score[idx])
        recovery = float(result.recovery_fraction[idx])

        sig_ok = z >= self.z_threshold
        rec_ok = recovery >= self.recovery_fraction_cutoff
        passed = sig_ok and rec_ok

        report = dict(
            index=idx,
            z=z,
            z_threshold=self.z_threshold,
            recovery=recovery,
            recovery_fraction_cutoff=self.recovery_fraction_cutoff,
            significance_ok=sig_ok,
            recovery_ok=rec_ok,
        )
        return VettingResult(passed=passed, report=report)

    def _validate_stability_map(self, result: GPStabilityMap) -> VettingResult:
        z = result.profiled_max_z()
        sig_ok = z >= self.z_threshold

        bf_ok = True
        marg_ln_b = None
        if result.log_bf_global is not None:
            marg_ln_b = result.marginalized_bayes_factor()
            bf_ok = marg_ln_b >= self.ln_b_threshold

        passed = sig_ok and bf_ok

        report = dict(
            profiled_max_z=z,
            z_threshold=self.z_threshold,
            significance_ok=sig_ok,
            marginalized_bayes_factor=marg_ln_b,
            ln_b_threshold=self.ln_b_threshold,
            bayes_factor_ok=bf_ok,
        )
        return VettingResult(passed=passed, report=report)
