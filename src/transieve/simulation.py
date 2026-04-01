import numpy as np


def generate_time(baseline=30.0, cadence=10.0):
    return np.linspace(
        -baseline / 2,
        baseline / 2,
        int(np.ceil(baseline * 24 * 60 / cadence)),
    )


def generate_gap_windows(time, n_gaps=2, gap_duration_range=(0.5, 2.0), seed=None):
    """Generate random observational gap windows within the time baseline.

    Parameters
    ----------
    time : ndarray
        Time array defining the observation baseline.
    n_gaps : int
        Number of gaps to generate.
    gap_duration_range : tuple of float
        (min_duration, max_duration) of each gap in the same units as time.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list of (float, float)
        Gap windows as (t_start, t_end) pairs.
    """
    rng = np.random.default_rng(seed)
    t_min, t_max = float(time[0]), float(time[-1])
    min_dur, max_dur = gap_duration_range
    windows = []
    for _ in range(n_gaps):
        t_start = rng.uniform(t_min, t_max - max_dur)
        duration = rng.uniform(min_dur, max_dur)
        windows.append((float(t_start), float(t_start + duration)))
    return windows


def inject_gaps(time, flux, gap_windows, mode="remove"):
    """Apply observational gaps to time and flux arrays.

    Parameters
    ----------
    time : ndarray
        Sorted time array.
    flux : ndarray
        Flux array of the same length as time.
    gap_windows : list of (float, float)
        Each entry is a (t_start, t_end) interval to mask.
    mode : {"remove", "nan"}
        "remove" drops cadences within gaps (non-uniform grid);
        "nan" keeps cadences but sets flux to NaN at gap positions.

    Returns
    -------
    time : ndarray
    flux : ndarray
    """
    if mode not in ("remove", "nan"):
        raise ValueError(f"mode must be 'remove' or 'nan', got {mode!r}")
    in_gap = np.zeros(len(time), dtype=bool)
    for t_start, t_end in gap_windows:
        in_gap |= (time >= t_start) & (time <= t_end)
    if mode == "nan":
        flux = flux.copy()
        flux[in_gap] = np.nan
        return time, flux
    keep = ~in_gap
    return time[keep], flux[keep]


def construct_antisymmetric_template(time, signal, center):
    """Construct an antisymmetric template by flipping the signal around the center."""
    mid_index = np.argmin(np.abs(time - center))
    s_odd = np.copy(signal)
    s_odd[:mid_index] *= -1
    return s_odd


class SimulatedLightCurve:
    DEFAULT_GP_PARAMS = {
        "log_jitter": np.log(1e-3),
        "log_omega": np.log(2 * np.pi / 4),
        "log_sigma": np.log(5 * 1e-4),
        "log_quality": np.log(1 / np.sqrt(2)),
    }

    def __init__(
        self,
        time,
        flux,
        flux_err,
        deterministic_component,
        gp_component,
        gp_model,
        deterministic_model,
        signal_params,
        gp_params,
    ):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.deterministic_component = deterministic_component
        self.gp_component = gp_component
        self.gp_model = gp_model
        self.deterministic_model = deterministic_model
        self.signal_params = signal_params
        self.gp_params = gp_params

    @classmethod
    def from_model(
        cls,
        signal_params,
        gp_params,
        signal_factory,
        time=None,
        baseline=15,
        cadence=10,
        gap_windows=None,
        multiply_signal=True,
        gp_family=None,
        seed=None,
    ):
        """Generic simulation factory. Combine any signal_factory with SHO GP noise.

        Parameters
        ----------
        signal_params : dict
            Parameters forwarded to signal_factory.
        gp_params : dict
            Parameters forwarded to gp_family.build().
        signal_factory : callable
            Function(**signal_params) -> callable(time) -> flux array.
        """
        if seed is not None:
            np.random.seed(seed)

        if time is None:
            time = generate_time(baseline=baseline, cadence=cadence)
            if gap_windows is not None:
                time, _ = inject_gaps(
                    time, np.zeros_like(time), gap_windows, mode="remove"
                )

        deterministic_model = signal_factory(**signal_params)
        deterministic_component = deterministic_model(time)

        if gp_family is None:
            from transieve.gp.fit import SHOGPFamily

            gp_family = SHOGPFamily()

        gp_model = gp_family.build(gp_params, time, mean=1.0)
        gp_component = gp_model.sample()

        if multiply_signal:
            flux = gp_component * deterministic_component
        else:
            flux = gp_component + deterministic_component - 1.0

        return cls(
            time=time,
            flux=flux,
            flux_err=None,
            deterministic_component=deterministic_component,
            gp_component=gp_component,
            gp_model=gp_model,
            deterministic_model=deterministic_model,
            signal_params=signal_params,
            gp_params=gp_params,
        )

    @classmethod
    def from_transit(
        cls,
        epoch=0,
        depth=(3 / 109) ** 2,
        duration=0.2,
        period=100.0,
        model="empirical",
        u=(0.3, 0.1),
        impact_param=0.0,
        limb_dark="quadratic",
        gp_params={},
        **kwargs,
    ):
        """Convenience factory for monotransit simulations.

        Parameters
        ----------
        epoch : float
            Transit midpoint time.
        depth : float
            Transit depth (fractional flux decrement).
        duration : float
            Transit duration in days.
        period : float
            Orbital period in days.
        model : {"empirical", "limb_dark"}
            Transit model. "empirical" uses the Protopapas et al. 2005 tanh shape;
            "limb_dark" uses a batman quadratic limb-darkened model.
        u : tuple
            Limb-darkening coefficients (only used when model="limb_dark").
        impact_param : float
            Impact parameter (only used when model="limb_dark").
        limb_dark : str
            batman limb-darkening law (only used when model="limb_dark").
        gp_params : dict, optional
            GP noise parameters. Defaults to DEFAULT_GP_PARAMS.
        **kwargs
            Passed to from_model (time, baseline, cadence, multiply_signal, gp_family, seed).
        """
        from .transit import get_monotransit_model, get_limb_dark_monotransit

        _factories = {
            "empirical": get_monotransit_model,
            "limb_dark": get_limb_dark_monotransit,
        }
        if model not in _factories:
            raise ValueError(f"model must be one of {list(_factories)}, got {model!r}")

        signal_params = {
            "epoch": epoch,
            "depth": depth,
            "duration": duration,
            "period": period,
        }
        if model == "limb_dark":
            signal_params.update(
                {"u": u, "impact_param": impact_param, "limb_dark": limb_dark}
            )

        return cls.from_model(
            signal_params=signal_params,
            gp_params=cls.DEFAULT_GP_PARAMS.copy() | gp_params,
            signal_factory=_factories[model],
            **kwargs,
        )

    def with_gaps(self, gap_windows, mode="remove"):
        """Return a new SimulatedLightCurve with observational gaps applied.

        Parameters
        ----------
        gap_windows : list of (float, float)
            Each entry is a (t_start, t_end) interval to mask.
            Use generate_gap_windows() to produce random windows.
        mode : {"remove", "nan"}
            "remove" drops cadences within gaps (non-uniform time grid);
            "nan" keeps all cadences but sets flux to NaN at gap positions.

        Returns
        -------
        SimulatedLightCurve
        """
        if mode not in ("remove", "nan"):
            raise ValueError(f"mode must be 'remove' or 'nan', got {mode!r}")
        in_gap = np.zeros(len(self.time), dtype=bool)
        for t_start, t_end in gap_windows:
            in_gap |= (self.time >= t_start) & (self.time <= t_end)

        if mode == "nan":
            flux = self.flux.copy()
            flux[in_gap] = np.nan
            return SimulatedLightCurve(
                time=self.time,
                flux=flux,
                flux_err=self.flux_err,
                deterministic_component=self.deterministic_component,
                gp_component=self.gp_component,
                gp_model=self.gp_model,
                deterministic_model=self.deterministic_model,
                signal_params=self.signal_params,
                gp_params=self.gp_params,
            )

        keep = ~in_gap
        return SimulatedLightCurve(
            time=self.time[keep],
            flux=self.flux[keep],
            flux_err=self.flux_err[keep] if self.flux_err is not None else None,
            deterministic_component=self.deterministic_component[keep],
            gp_component=self.gp_component[keep],
            gp_model=self.gp_model,
            deterministic_model=self.deterministic_model,
            signal_params=self.signal_params,
            gp_params=self.gp_params,
        )

    def plot(self, axes=None):
        """Plot the simulated light curve and its components."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(nrows=3, sharex=True)

        # Stochastic plot
        axes[0].plot(self.time, self.gp_component, ".", mec="None")
        axes[0].set_ylabel("GP component")

        # Deterministic plot
        axes[1].plot(self.time, self.deterministic_component, c="black", lw=1.0)
        axes[1].set_ylabel("Deterministic component")

        # Observed plot
        axes[2].plot(self.time, self.flux, ".", mec="None")
        axes[2].set_ylabel("Observed flux")
        axes[2].set_xlabel("Time")

        plt.tight_layout()
        return axes
