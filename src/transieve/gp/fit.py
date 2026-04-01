from abc import ABC, abstractmethod

import numpy as np
from celerite2 import GaussianProcess, terms
from scipy.optimize import minimize

from ..lightcurve import LightCurve

__all__ = ["GPFamily", "SHOGPFamily", "ExpGPFamily"]


class GPFamily(ABC):
    def __init__(self, name, theta_names, ranges, jitter_range=None):
        self.name = name
        theta_names = list(theta_names)
        ranges = list(ranges)
        if jitter_range is not None:
            theta_names.append("log_jitter")
            ranges.append(jitter_range)
        self.theta_names = tuple(theta_names)
        self._bounds = [(np.log(low), np.log(high)) for (low, high) in ranges]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self._bounds

    def theta_to_dict(self, theta) -> dict[str, float]:
        return dict(zip(self.theta_names, theta))

    def sample_theta(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return np.array(
            [rng.uniform(low, high) for (low, high) in self.bounds], dtype=float
        )

    def fit_light_curve(
        self,
        light_curve: LightCurve,
        method: str = "L-BFGS-B",
        initial_theta=None,
        rng=None,
        max_retries: int = 1,
        de_options: dict | None = None,
        minimize_options: dict | None = None,
    ):
        """Fit this GP family to a light curve.

        method: either 'L-BFGS-B' (uses ``scipy.optimize.minimize``) or
        'differential_evolution' (uses ``scipy.optimize.differential_evolution``).
        max_retries: when >1, retries the chosen optimizer with different
        initializations (for DE it repeats runs and picks the best solution).
        """
        if not isinstance(light_curve, LightCurve):
            raise TypeError("fit_light_curve expects a LightCurve instance.")

        neg_log_like = make_neg_log_like(self, light_curve)

        if method == "differential_evolution":
            from scipy.optimize import differential_evolution

            de_defaults = dict(
                strategy="best1bin",
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
            )
            if de_options:
                de_defaults.update(de_options)

            best_result = differential_evolution(
                neg_log_like,
                self.bounds,
                **de_defaults,
            )
            for _ in range(max_retries - 1):
                result = differential_evolution(
                    neg_log_like,
                    self.bounds,
                    **de_defaults,
                )
                if result.fun < best_result.fun:
                    best_result = result
            return best_result

        # Fallback to local optimization (minimize)
        if rng is None:
            rng = np.random.default_rng()

        min_opts = dict(method=method)
        if minimize_options:
            min_opts.update(minimize_options)

        def one_minimize(start_theta):
            return minimize(
                neg_log_like,
                start_theta,
                bounds=self.bounds,
                **min_opts,
            )

        if max_retries <= 1:
            if initial_theta is None:
                initial_theta = self.sample_theta(rng)
            return one_minimize(initial_theta)

        best_result = None
        for i in range(max_retries):
            if i == 0 and initial_theta is not None:
                start = initial_theta
            else:
                start = self.sample_theta(rng)
            res = one_minimize(start)
            if best_result is None or res.fun < best_result.fun:
                best_result = res
        return best_result

    @property
    def physical_bounds(self):
        lower_bounds = self.theta_to_physical(
            np.array([bound[0] for bound in self.bounds])
        )
        upper_bounds = self.theta_to_physical(
            np.array([bound[1] for bound in self.bounds])
        )
        return {name: (lower_bounds[name], upper_bounds[name]) for name in lower_bounds}

    def params_to_theta(self, **kwargs):
        """Convert physical parameters to theta space.

        By default, this just looks up the corresponding theta for each parameter name,
        but subclasses can override this if needed.
        """
        return tuple(kwargs.get(name, np.nan) for name in self.theta_names)

    @abstractmethod
    def theta_to_physical(self, theta, prefix=""):
        """Return a dict of physically meaningful parameters."""
        raise NotImplementedError

    @abstractmethod
    def build(self, params, time=None, flux_err=None, mean=1.0):
        """Build a `GaussianProcess` for this family from physical parameters."""
        raise NotImplementedError

    def validate_white_noise_baseline(self, flux_err: np.ndarray | None) -> None:
        """Raise if white-noise baseline diagnostics cannot be computed."""
        if flux_err is None and "log_jitter" not in self.theta_names:
            raise ValueError(
                "White-noise baseline diagnostics require finite diagonal noise. "
                "Provide `flux_err` or use a GP family with `jitter_range` so "
                "`log_jitter` is fitted."
            )

    def build_gp_from_theta(
        self,
        theta,
        light_curve: LightCurve,
    ):
        """Construct a `GaussianProcess` for this family from `theta`."""
        if not isinstance(light_curve, LightCurve):
            raise TypeError("build_gp_from_theta expects a LightCurve instance.")

        params = self.theta_to_dict(theta)
        return self.build(
            params,
            light_curve.time,
            flux_err=light_curve.flux_err,
            mean=light_curve.mean,
        )

    def sample_light_curve(self, theta, light_curve: LightCurve):
        if isinstance(theta, dict):
            theta = self.params_to_theta(**theta)
        gp = self.build_gp_from_theta(theta, light_curve)
        return gp.sample()

    def plot_fit(
        self,
        theta,
        time,
        flux,
        flux_err=None,
        mean=1.0,
        t_pred=None,
        ax=None,
        nsigma=2,
        plot_data_kwargs=None,
        plot_gp_kwargs=None,
        shade_kwargs=None,
        show=True,
    ):
        """Plot observations and GP fit for a given ``theta``.

        Parameters
        - theta: array-like GP parameters in the family's theta space (e.g. result.x)
        - time, flux: data to plot
        - flux_err: optional per-point errors
        - mean: GP mean level used when building the GP
        - t_pred: times at which to evaluate the predictive mean/variance. If
          ``None`` a dense grid across ``time`` is used.
        - ax: matplotlib Axes to draw on (created if None)
        - nsigma: number of standard deviations to shade around the predictive mean
        - plot_data_kwargs: dict forwarded to the data scatter plot call
        - plot_gp_kwargs: dict forwarded to the predictive mean line plot call
        - shade_kwargs: dict forwarded to ``fill_between`` for the predictive interval
        - show: whether to call ``plt.show()`` before returning the axes

        Returns the matplotlib Axes instance.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if t_pred is None:
            t_pred = np.linspace(np.min(time), np.max(time), max(200, len(time) * 4))

        light_curve = LightCurve.from_arrays(
            time=time,
            flux=flux,
            flux_err=flux_err,
            mean=mean,
        )
        gp = self.build_gp_from_theta(theta, light_curve)

        # Predictive mean and variance on t_pred
        var = None
        try:
            mu, var = gp.predict(flux, t=t_pred, return_var=True)
        except TypeError:
            mu = gp.predict(flux, t=t_pred)

        # Plot data
        pd_kw = dict(marker=".", linestyle="none", color="k", alpha=0.6)
        if plot_data_kwargs:
            pd_kw.update(plot_data_kwargs)
        ax.plot(time, flux, **pd_kw)

        # Plot GP predictive mean
        gp_kw = dict(color="C1", lw=1.5)
        if plot_gp_kwargs:
            gp_kw.update(plot_gp_kwargs)
        ax.plot(t_pred, mu, **gp_kw)

        # Shade predictive interval
        if var is not None:
            std = np.sqrt(var)
            sh_kw = dict(color="C1", alpha=0.2)
            if shade_kwargs:
                sh_kw.update(shade_kwargs)
            ax.fill_between(t_pred, mu - nsigma * std, mu + nsigma * std, **sh_kw)

        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Relative flux")

        if show:
            plt.show()
        return ax

    @classmethod
    def from_string(cls, name, **kwargs):
        if name == "sho":
            return SHOGPFamily(**kwargs)
        elif name == "exp":
            return ExpGPFamily(**kwargs)
        else:
            raise ValueError(f"Unknown GP family name: {name}")


class SHOGPFamily(GPFamily):
    def __init__(
        self,
        period_range=(0.1, 30.0),
        sigma_range=(1e-5, 1e-3),
        quality_range=(0.5, 10.0),
        omega_range=None,
        jitter_range=None,
    ):
        if omega_range is None:
            omega_range = (2 * np.pi / period_range[1], 2 * np.pi / period_range[0])

        super().__init__(
            "sho",
            ("log_omega", "log_sigma", "log_quality"),
            [omega_range, sigma_range, quality_range],
            jitter_range=jitter_range,
        )

    def build(self, params, time=None, flux_err=None, mean=1.0):
        if time is None:
            raise ValueError("time must be provided to build a GaussianProcess")

        kernel = terms.SHOTerm(
            w0=np.exp(params["log_omega"]),
            Q=np.exp(params.get("log_quality", np.log(1 / np.sqrt(2)))),
            sigma=np.exp(params["log_sigma"]),
        )  # type: ignore
        yerr, diag = _resolve_noise(params, flux_err, time)
        return GaussianProcess(kernel, time, yerr=yerr, diag=diag, mean=mean)

    def theta_to_physical(self, theta=None, prefix=""):
        if theta is None:
            theta = np.full(len(self.theta_names), np.nan)
        d = self.theta_to_dict(theta)
        prefix = f"{prefix}_" if prefix else ""
        result = {
            prefix + "period": 2 * np.pi / np.exp(d["log_omega"]),
            prefix + "sigma": np.exp(d["log_sigma"]),
            prefix + "quality": np.exp(d["log_quality"]),
        }
        if "log_jitter" in d:
            result[prefix + "jitter"] = np.exp(d["log_jitter"])
        return result


class ExpGPFamily(GPFamily):
    def __init__(
        self,
        scale_range=(0.1, 100.0),
        sigma_range=(1e-5, 1e-3),
        jitter_range=None,
    ):
        super().__init__(
            "exp",
            ("log_scale", "log_sigma"),
            [scale_range, sigma_range],
            jitter_range=jitter_range,
        )

    def build(self, params, time=None, flux_err=None, mean=1.0):
        if time is None:
            raise ValueError("time must be provided to build a GaussianProcess")

        kernel = terms.RealTerm(
            a=np.exp(2 * params["log_sigma"]),
            c=np.exp(-params["log_scale"]),
        )
        yerr, diag = _resolve_noise(params, flux_err, time)
        return GaussianProcess(kernel, time, yerr=yerr, diag=diag, mean=mean)

    def theta_to_physical(self, theta=None, prefix=""):
        if theta is None:
            theta = np.full(len(self.theta_names), np.nan)
        d = self.theta_to_dict(theta)
        prefix = f"{prefix}_" if prefix else ""
        result = {
            prefix + "scale": np.exp(d["log_scale"]),
            prefix + "sigma": np.exp(d["log_sigma"]),
        }
        if "log_jitter" in d:
            result[prefix + "jitter"] = np.exp(d["log_jitter"])
        return result


def _resolve_noise(params, flux_err, time):
    """Return (yerr, diag) for GaussianProcess construction."""
    has_jitter = "log_jitter" in params
    jitter_var = np.exp(2 * params["log_jitter"]) if has_jitter else 0.0

    if flux_err is not None:
        # If we have per-point errors, we must return a vector in diag
        return None, flux_err**2 + jitter_var

    return None, jitter_var


def make_neg_log_like(
    family: GPFamily,
    light_curve: LightCurve,
):
    def neg_log_like_impl(theta):
        try:
            gp = family.build_gp_from_theta(theta, light_curve)
            return -gp.log_likelihood(light_curve.flux)
        except Exception:
            return np.inf

    return neg_log_like_impl
