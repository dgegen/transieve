from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import as_1d_array


@dataclass
class LightCurve:
    """Validated light-curve container used by low-level GP routines."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None
    mean: float | None = None

    def __post_init__(self) -> None:
        self.time = as_1d_array(self.time)
        self.flux = as_1d_array(self.flux)

        if len(self.time) != len(self.flux):
            raise ValueError("time and flux must have the same length.")

        if self.flux_err is not None:
            self.flux_err = as_1d_array(self.flux_err)
            if len(self.flux_err) != len(self.time):
                raise ValueError("flux_err must have the same length as time.")

        if np.any(np.isnan(self.time)):
            raise ValueError("time contains NaN values.")
        if np.any(np.isnan(self.flux)):
            raise ValueError("flux contains NaN values.")
        if self.flux_err is not None and np.any(np.isnan(self.flux_err)):
            raise ValueError("flux_err contains NaN values.")

        if self.mean is None:
            self.mean = float(np.nanmedian(self.flux))
        else:
            self.mean = float(self.mean)

    @classmethod
    def from_arrays(
        cls,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray | None = None,
        mean: float | None = None,
    ) -> LightCurve:
        return cls(time=time, flux=flux, flux_err=flux_err, mean=mean)

    def centered_flux(self, center: bool = True) -> np.ndarray:
        """Return flux optionally centered by its median."""
        if not center:
            return self.flux
        return self.flux - np.nanmedian(self.flux)
