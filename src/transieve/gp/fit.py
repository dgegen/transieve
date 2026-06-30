from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from celerite2 import GaussianProcess, terms
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..lightcurve import LightCurve

__all__ = [
    "GPFamily",
    "SHOGPFamily",
    "ExpGPFamily",
    "robust_jitter_seed",
    "extract_hess_inv_diag",
    "numeric_hess_inv_diag",
    "hessian_zoom_log_evidence",
    "fit_and_evidence",
]

# A3_light: compressed differential-evolution configuration validated by the
# 50-target benchmark (htd/scripts/benchmark_gp_strategies.py). These minimal
# population/iteration settings clear the flat degenerate valleys that stall
# L-BFGS-B while running in an ~85 ms window.
A3_LIGHT_DE_OPTIONS = dict(
    strategy="best1bin",
    maxiter=100,
    popsize=4,
    tol=1e-5,
    mutation=(0.5, 1.0),
    recombination=0.7,
    seed=42,
)


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

    def _seeded_x0(self, light_curve: LightCurve):
        """Build an initial guess with the jitter dimension robustly seeded.

        Returns ``None`` when this family has no ``log_jitter`` parameter, so the
        optimizer keeps its default (unseeded) initialization. Otherwise returns
        the bounds midpoint with ``log_jitter`` set from
        :func:`robust_jitter_seed` (clipped into its bound), as in benchmark A3.
        """
        if "log_jitter" not in self.theta_names:
            return None
        x0 = np.array([(lo + hi) / 2.0 for (lo, hi) in self.bounds], dtype=float)
        j_idx = self.theta_names.index("log_jitter")
        seed = robust_jitter_seed(light_curve)
        if np.isfinite(seed) and seed > 0:
            lo, hi = self.bounds[j_idx]
            x0[j_idx] = float(np.clip(np.log(seed), lo, hi))
        return x0

    def fit_light_curve(
        self,
        light_curve: LightCurve,
        method: str = "differential_evolution",
        initial_theta=None,
        rng=None,
        max_retries: int = 1,
        de_options: dict | None = None,
        minimize_options: dict | None = None,
        use_regularization: bool = True,
    ):
        """Fit this GP family to a light curve.

        method: either 'differential_evolution' (default; uses the validated
        A3_light configuration of ``scipy.optimize.differential_evolution``) or
        'L-BFGS-B' (uses ``scipy.optimize.minimize``).
        max_retries: when >1, retries the chosen optimizer with different
        initializations (for DE it repeats runs and picks the best solution).

        The 'differential_evolution' path uses :data:`A3_LIGHT_DE_OPTIONS` by
        default (popsize=4, maxiter=100, ...); pass ``de_options`` to override.
        """
        if not isinstance(light_curve, LightCurve):
            raise TypeError("fit_light_curve expects a LightCurve instance.")

        neg_log_like = make_neg_log_like(
            self, light_curve, use_regularization=use_regularization
        )

        if method == "differential_evolution":
            from scipy.optimize import differential_evolution

            de_defaults = dict(A3_LIGHT_DE_OPTIONS)
            if de_options:
                de_defaults.update(de_options)

            # Seed the jitter dimension with a robust white-noise estimate so DE
            # starts from a physically sensible noise floor (benchmark A2/A3).
            x0 = self._seeded_x0(light_curve)
            de_kwargs = dict(de_defaults)
            if x0 is not None and "x0" not in de_kwargs:
                de_kwargs["x0"] = x0

            best_result = differential_evolution(
                neg_log_like,
                self.bounds,
                **de_kwargs,
            )
            for _ in range(max_retries - 1):
                result = differential_evolution(
                    neg_log_like,
                    self.bounds,
                    **de_kwargs,
                )
                if result.fun < best_result.fun:
                    best_result = result
            return best_result

        if method == "tri_seeded":
            if "log_quality" not in self.theta_names:
                return self.fit_light_curve(
                    light_curve,
                    method="L-BFGS-B",
                    initial_theta=initial_theta,
                    rng=rng,
                    max_retries=max_retries,
                    minimize_options=minimize_options,
                    use_regularization=use_regularization,
                )

            q_idx = self.theta_names.index("log_quality")
            q_seeds = [0.5, 1.0 / np.sqrt(2), 10.0]
            best_res = None
            if rng is None:
                rng = np.random.default_rng()

            min_opts = dict(method="L-BFGS-B")
            if minimize_options:
                min_opts.update(minimize_options)

            for q_val in q_seeds:
                log_q = np.log(q_val)
                log_q = np.clip(log_q, self.bounds[q_idx][0], self.bounds[q_idx][1])

                if initial_theta is not None:
                    theta0 = np.asarray(initial_theta).copy()
                else:
                    theta0 = self.sample_theta(rng)
                theta0[q_idx] = log_q

                res = minimize(neg_log_like, theta0, bounds=self.bounds, **min_opts)
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            return best_res

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
    use_regularization: bool = False,
    sigma_prior_mu: float = -5.0,
    sigma_prior_omega: float = 2.0,
):
    try:
        sigma_idx = family.theta_names.index("log_sigma")
    except ValueError:
        sigma_idx = None

    def neg_log_like_impl(theta):
        try:
            gp = family.build_gp_from_theta(theta, light_curve)
            nll = -gp.log_likelihood(light_curve.flux)
        except Exception:
            return np.inf

        if use_regularization and sigma_idx is not None:
            log_sigma = theta[sigma_idx]
            penalty = 0.5 * ((log_sigma - sigma_prior_mu) / sigma_prior_omega) ** 2
            return nll + penalty
        return nll

    return neg_log_like_impl


# ---------------------------------------------------------------------------
# A3_light / B1 backend — validated by htd/scripts/benchmark_gp_strategies.py
# ---------------------------------------------------------------------------


def robust_jitter_seed(light_curve: LightCurve) -> float:
    """Robust white-noise estimate used to seed ``log_jitter`` (benchmark A2/A3).

    Returns ``max(median(flux_err), median(|diff(flux)|))`` so the estimate
    falls back to the point-to-point scatter when per-point errors are absent.
    """
    err = (
        np.median(light_curve.flux_err) if light_curve.flux_err is not None else -np.inf
    )
    return float(np.maximum(err, np.median(np.abs(np.diff(light_curve.flux)))))


def extract_hess_inv_diag(
    lbfgsb_result, n_params: int, fallback: np.ndarray | None = None
) -> np.ndarray:
    """Per-parameter log-space std from a scipy L-BFGS-B inverse Hessian.

    Falls back to ``fallback`` (e.g. DE population std) then to ``0.5`` when the
    Hessian is unavailable or singular, so a numerical fault never crashes a
    batch (benchmark ``_extract_hess_inv_diag``).
    """
    try:
        raw = lbfgsb_result.hess_inv
        if hasattr(raw, "todense"):
            H_inv = np.asarray(raw.todense())
        else:
            H_inv = raw @ np.eye(n_params)
        diag = np.maximum(np.diag(H_inv), 1e-12)
        return np.sqrt(diag)
    except Exception:
        if fallback is not None:
            return np.asarray(fallback, dtype=float)
        return np.ones(n_params) * 0.5


def _get_2d_sigma_omega_indices(family: GPFamily) -> tuple[int, int]:
    """Return the (log_sigma, log_omega) theta indices for the zoom grid."""
    names = list(family.theta_names)
    return names.index("log_sigma"), names.index("log_omega")


def numeric_hess_inv_diag(
    family: GPFamily,
    light_curve: LightCurve,
    theta: np.ndarray,
    indices,
    step: float = 1e-2,
    use_regularization: bool = True,
):
    """Finite-difference log-space std (curvature width) for selected theta dims.

    Computes the diagonal second derivative of the negative log-likelihood at
    ``theta`` for each index via central differences and returns
    ``1/sqrt(d2L/dx2)``. Used to width the B1 zoom grid: it is far more reliable
    than scipy's L-BFGS-B ``hess_inv`` when the polish converges in ~0 steps and
    leaves the inverse Hessian at the identity. Non-positive curvature yields
    ``nan`` for that index (caller should fall back).
    """
    nll = make_neg_log_like(family, light_curve, use_regularization=use_regularization)
    theta = np.asarray(theta, dtype=float)
    f0 = nll(theta)
    widths = {}
    for idx in indices:
        th = theta.copy()
        th[idx] = theta[idx] + step
        fp = nll(th)
        th[idx] = theta[idx] - step
        fm = nll(th)
        d2 = (fp - 2.0 * f0 + fm) / step**2
        widths[idx] = float(1.0 / np.sqrt(d2)) if np.isfinite(d2) and d2 > 0 else np.nan
    return widths


def _zoom_width_diag(family, light_curve, theta, lbfgsb_result, fallback=None):
    """Robust per-parameter log-space std for the B1 zoom grid.

    Starts from the L-BFGS-B inverse-Hessian diagonal (benchmark port), then
    overrides the marginalized (log_sigma, log_omega) dims with a numerical
    finite-difference width whenever that is finite.
    """
    n_params = len(family.theta_names)
    diag = extract_hess_inv_diag(lbfgsb_result, n_params, fallback=fallback)
    s_idx, w_idx = _get_2d_sigma_omega_indices(family)
    fd = numeric_hess_inv_diag(family, light_curve, theta, (s_idx, w_idx))
    for idx, width in fd.items():
        if np.isfinite(width) and width > 0:
            diag[idx] = width
    return diag


def _compute_log_bf_at_theta(
    theta: np.ndarray,
    family: GPFamily,
    light_curve: LightCurve,
    template: np.ndarray,
    sigma_a: float,
    prior: str,
) -> tuple[float, float]:
    """Return ``(log_bf, log_lml)`` for a single theta (benchmark helper).

    Guards every matrix operation so a singular node degrades to
    ``(0.0, -inf)`` rather than aborting the whole grid.
    """
    from .match import MatchedFilter

    try:
        gp = family.build_gp_from_theta(theta, light_curve)
        centered = light_curve.flux - float(np.nanmedian(light_curve.flux))
        mf = MatchedFilter(gp, centered, check_zero_centered=False)
        z, norm = mf.z_score(template)
        proj = z * norm
        log_bf = float(
            MatchedFilter._log_bayes_factor_from_projection(
                np.array([proj]),
                np.array([norm]),
                sigma_a=sigma_a,
                prior=prior,
            )[0]
        )
        log_lml = float(gp.log_likelihood(light_curve.flux))
        return log_bf, log_lml
    except Exception:
        return 0.0, -np.inf


def _marginalize_log_bf(
    log_lml: np.ndarray, log_bf: np.ndarray, log_w: np.ndarray | None = None
) -> float:
    """log [ Σ p(θ) B(θ) / Σ p(θ) ] via log-sum-exp (benchmark ``_marginalize``).

    ``log_w`` are additional log-weights (here the log-space Voronoi cell areas).
    Mirrors the weighting convention of
    ``GPStabilityMap._marginalise_bayes_factor`` in stability.py.
    """
    if log_w is None:
        log_w = np.zeros(len(log_lml))
    finite = np.isfinite(log_lml) & np.isfinite(log_bf) & np.isfinite(log_w)
    if not np.any(finite):
        return float("nan")
    ll = log_lml[finite]
    lb = log_bf[finite]
    lw = log_w[finite]
    max_ll = np.nanmax(ll)
    numer = logsumexp(ll - max_ll + lb + lw)
    denom = logsumexp(ll - max_ll + lw)
    return float(numer - denom)


def hessian_zoom_log_evidence(
    family: GPFamily,
    light_curve: LightCurve,
    mle_theta: np.ndarray,
    hess_inv_diag: np.ndarray,
    template: np.ndarray,
    grid_size: int = 20,
    n_sigma: float = 3.0,
    sigma_a: float = 1.0,
    prior: str = "half-normal",
) -> float:
    """Marginalized log Bayes factor via a Hessian-informed zoom grid (B1).

    Builds a ``grid_size``×``grid_size`` grid spanning ``mle_theta ± n_sigma *
    hess_inv_diag`` (clamped to ``family.bounds``) over the (log_sigma,
    log_omega) axes, with the remaining parameters fixed at the MLE, then
    marginalizes with log-space Voronoi weights. Matches a 64×64 brute grid to
    Δln B ≈ 0 at ~10% of the cost (benchmark ``run_b1_hessian_zoom``).
    """
    mle_theta = np.asarray(mle_theta, dtype=float)
    s_idx, w_idx = _get_2d_sigma_omega_indices(family)

    def _grid(idx: int) -> np.ndarray:
        lo = max(mle_theta[idx] - n_sigma * hess_inv_diag[idx], family.bounds[idx][0])
        hi = min(mle_theta[idx] + n_sigma * hess_inv_diag[idx], family.bounds[idx][1])
        return np.linspace(lo, hi, grid_size)

    sigma_pts = _grid(s_idx)
    omega_pts = _grid(w_idx)

    # Log-space Voronoi cell areas: grid points already live in log space.
    log_ds = np.log(np.gradient(sigma_pts))
    log_dw = np.log(np.gradient(omega_pts))
    W_s, W_w = np.meshgrid(log_ds, log_dw, indexing="ij")

    log_lml_vals, log_bf_vals = [], []
    for ls in sigma_pts:
        for lw in omega_pts:
            theta = mle_theta.copy()
            theta[s_idx] = ls
            theta[w_idx] = lw
            lb, ll = _compute_log_bf_at_theta(
                theta, family, light_curve, template, sigma_a, prior
            )
            log_lml_vals.append(ll)
            log_bf_vals.append(lb)

    log_w = (W_s + W_w).ravel()
    return _marginalize_log_bf(np.array(log_lml_vals), np.array(log_bf_vals), log_w)


def hessian_zoom_log_evidence_map(
    family: GPFamily,
    light_curve: LightCurve,
    mle_theta: np.ndarray,
    hess_inv_diag: np.ndarray,
    template_bank,
    grid_size: int = 20,
    n_sigma: float = 3.0,
    sigma_a: float = 1.0,
    prior: str = "half-normal",
) -> np.ndarray:
    """Vectorized :func:`hessian_zoom_log_evidence` over an entire template bank.

    Identical (log_sigma, log_omega) zoom grid to :func:`hessian_zoom_log_evidence`,
    but at each grid node the GP is built/factorized once and then solved against
    every template in ``template_bank`` simultaneously (via
    :meth:`MatchedFilter._bank_stats`), exactly as :meth:`MatchedFilter.log_bayes_factor_map`
    reuses one factorization across epochs for a fixed theta. The marginalization
    weights (``log_lml`` and the Voronoi ``log_w``) are theta-only and therefore
    shared across templates, so they're computed once and broadcast.

    Net effect: scoring every epoch in the bank costs the same ``grid_size**2``
    GP builds as scoring a single epoch with :func:`hessian_zoom_log_evidence` —
    only the (cheap, already-factorized) linear solve grows with the number of
    templates.

    Returns
    -------
    np.ndarray
        Marginal log evidence, one value per template in ``template_bank``
        (in ``template_bank.epochs`` order).
    """
    from .match import MatchedFilter, TemplateBank

    mle_theta = np.asarray(mle_theta, dtype=float)
    s_idx, w_idx = _get_2d_sigma_omega_indices(family)

    def _grid(idx: int) -> np.ndarray:
        lo = max(mle_theta[idx] - n_sigma * hess_inv_diag[idx], family.bounds[idx][0])
        hi = min(mle_theta[idx] + n_sigma * hess_inv_diag[idx], family.bounds[idx][1])
        return np.linspace(lo, hi, grid_size)

    sigma_pts = _grid(s_idx)
    omega_pts = _grid(w_idx)

    log_ds = np.log(np.gradient(sigma_pts))
    log_dw = np.log(np.gradient(omega_pts))
    W_s, W_w = np.meshgrid(log_ds, log_dw, indexing="ij")
    log_w = (W_s + W_w).ravel()

    bank = TemplateBank._coerce(template_bank, default_epochs=light_curve.time)
    # Templates don't depend on theta — materialize the [N, M] matrix once and
    # reuse it at every grid node instead of re-evaluating `bank.make` 400x.
    template_matrix = np.column_stack(list(bank))
    centered_flux = light_curve.flux - float(np.nanmedian(light_curve.flux))

    n_grid = grid_size * grid_size
    n_epochs = template_matrix.shape[1]
    log_lml_grid = np.full(n_grid, -np.inf)
    log_bf_grid = np.full((n_grid, n_epochs), -np.inf)

    g = 0
    for ls in sigma_pts:
        for lw in omega_pts:
            theta = mle_theta.copy()
            theta[s_idx] = ls
            theta[w_idx] = lw
            try:
                gp = family.build_gp_from_theta(theta, light_curve)
                mf = MatchedFilter(gp, centered_flux, check_zero_centered=False)
                projections, template_norms = mf._bank_stats(template_matrix)
                log_bf_grid[g] = MatchedFilter._log_bayes_factor_from_projection(
                    projections, template_norms, sigma_a=sigma_a, prior=prior
                )
                log_lml_grid[g] = float(gp.log_likelihood(light_curve.flux))
            except Exception:
                pass  # leaves -inf defaults for this grid node
            g += 1

    finite_grid = np.isfinite(log_lml_grid) & np.isfinite(log_w)
    if not np.any(finite_grid):
        return np.full(n_epochs, np.nan)

    ll = np.where(finite_grid, log_lml_grid, -np.inf)
    lb = np.where(np.isfinite(log_bf_grid), log_bf_grid, -np.inf)
    max_ll = np.max(ll[finite_grid])

    base = (ll - max_ll + log_w)[:, None]  # [n_grid, 1], shared across templates
    numer = logsumexp(base + lb, axis=0)  # [n_epochs]
    denom = logsumexp(base[:, 0])  # scalar — doesn't depend on the template
    return numer - denom


@dataclass
class FitEvidenceResult:
    """Bundle returned by :func:`fit_and_evidence`."""

    theta: np.ndarray
    params: dict
    hess_inv_diag: np.ndarray
    log_bf_marg: float
    log_likelihood: float
    samples: np.ndarray | None = None


def fit_and_evidence(
    family: GPFamily,
    light_curve: LightCurve,
    template: np.ndarray,
    de_options: dict | None = None,
    grid_size: int = 20,
    n_sigma: float = 3.0,
    sigma_a: float = 1.0,
    prior: str = "half-normal",
    use_regularization: bool = True,
    fit_method: str = "differential_evolution",
    integration_method: str = "importance_sampling",
    integration_samples: int = 400,
) -> FitEvidenceResult:
    """Single entry point: GP optimization fit → L-BFGS-B polish → evidence calculation.

    Chains the validated optimizer and integrator: optimize with the chosen fit method
    (default: differential_evolution / A3_light), run one L-BFGS-B polish from the
    optimum to obtain an inverse Hessian, extract its diagonal (with DE population std fallback),
    then integrate the log Bayes factor over the parameter space using the chosen
    integration method (default: importance_sampling).
    """
    de_result = family.fit_light_curve(
        light_curve,
        method=fit_method,
        de_options=de_options,
        use_regularization=use_regularization,
    )
    mle_theta = np.asarray(de_result.x, dtype=float)

    # DE population std as a fallback width when the Hessian is unavailable.
    try:
        pop_std = np.maximum(np.std(de_result.population, axis=0), 1e-12)
    except Exception:
        pop_std = None

    neg_log_like = make_neg_log_like(
        family, light_curve, use_regularization=use_regularization
    )
    try:
        polish = minimize(
            neg_log_like, mle_theta, bounds=family.bounds, method="L-BFGS-B"
        )
        if np.isfinite(polish.fun) and polish.fun <= de_result.fun:
            mle_theta = np.asarray(polish.x, dtype=float)
    except Exception:
        polish = None

    hess_inv_diag = _zoom_width_diag(
        family, light_curve, mle_theta, polish, fallback=pop_std
    )

    from .integration import integrate_gp_evidence

    integration_kwargs = {}
    if integration_method in ("grid", "hessian_zoom"):
        integration_kwargs["grid_size"] = grid_size
        integration_kwargs["n_sigma"] = n_sigma
    elif integration_method == "importance_sampling":
        integration_kwargs["n_samples"] = integration_samples

    result = integrate_gp_evidence(
        family=family,
        light_curve=light_curve,
        template=template,
        method=integration_method,
        mle_theta=mle_theta,
        lbfgsb_result=polish,
        fallback_std=hess_inv_diag,
        sigma_a=sigma_a,
        prior=prior,
        use_regularization=use_regularization,
        **integration_kwargs,
    )

    try:
        gp = family.build_gp_from_theta(mle_theta, light_curve)
        log_likelihood = float(gp.log_likelihood(light_curve.flux))
    except Exception:
        log_likelihood = float("nan")

    return FitEvidenceResult(
        theta=mle_theta,
        params=family.theta_to_physical(mle_theta),
        hess_inv_diag=hess_inv_diag,
        log_bf_marg=result.log_bf_marg
        if result.log_bf_marg is not None
        else float("nan"),
        log_likelihood=log_likelihood,
        samples=result.samples,
    )
