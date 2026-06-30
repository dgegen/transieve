from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

try:
    from scipy.stats import multivariate_t

    _HAS_MULTIVARIATE_T = True
except ImportError:
    _HAS_MULTIVARIATE_T = False

from ..lightcurve import LightCurve
from .fit import (
    GPFamily,
    make_neg_log_like,
    extract_hess_inv_diag,
    _get_2d_sigma_omega_indices,
)
from .match import MatchedFilter


@dataclass
class IntegrationResult:
    log_bf_marg: float | None = None
    log_lml_marg: float | None = None
    n_samples: int = 0
    n_inversions: int = 0
    samples: np.ndarray | None = None
    weights: np.ndarray | None = None


def _compute_bf_lml_at_theta(
    theta: np.ndarray,
    family: GPFamily,
    light_curve: LightCurve,
    template: np.ndarray | None,
    sigma_a: float = 1.0,
    prior: str = "half-normal",
) -> tuple[float | None, float]:
    """Evaluate log likelihood (and optionally log Bayes factor) at a given theta."""
    try:
        gp = family.build_gp_from_theta(theta, light_curve)
        log_lml = float(gp.log_likelihood(light_curve.flux))
        if template is not None:
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
            return log_bf, log_lml
        return None, log_lml
    except Exception:
        return (0.0 if template is not None else None), -np.inf


def _marginalize_lml(log_lml: np.ndarray, log_w: np.ndarray | None = None) -> float:
    """Compute log E_p[p(y|θ)] = logsumexp(lml + w) - logsumexp(w)."""
    if log_w is None:
        log_w = np.zeros(len(log_lml))
    finite = np.isfinite(log_lml) & np.isfinite(log_w)
    if not np.any(finite):
        return float("nan")
    ll = log_lml[finite]
    lw = log_w[finite]
    return float(logsumexp(ll + lw) - logsumexp(lw))


def _marginalize_log_bf(
    log_lml: np.ndarray, log_bf: np.ndarray, log_w: np.ndarray | None = None
) -> float:
    """log [ Σ p(θ) B(θ) / Σ p(θ) ] via log-sum-exp."""
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


def importance_sampling_evidence(
    family: GPFamily,
    light_curve: LightCurve,
    mle_theta: np.ndarray,
    lbfgsb_result=None,
    template: np.ndarray | None = None,
    n_samples: int = 400,
    df: float = 3.0,
    fallback_std: np.ndarray | None = None,
    sigma_a: float = 1.0,
    prior: str = "half-normal",
    use_regularization: bool = True,
) -> IntegrationResult:
    """Multivariate Student-t importance sampling centered at the MLE/MAP
    with scale from the approximate inverse Hessian.
    """
    n_params = len(family.theta_names)
    try:
        raw = lbfgsb_result.hess_inv if lbfgsb_result is not None else None
        if raw is not None:
            if hasattr(raw, "todense"):
                H_inv = np.asarray(raw.todense())
            else:
                H_inv = raw @ np.eye(n_params)
            H_inv = (H_inv + H_inv.T) / 2.0  # symmetrize
            H_inv += np.eye(n_params) * 1e-6  # numerical stability
        else:
            raise ValueError("No L-BFGS-B result provided")
    except Exception:
        if fallback_std is not None:
            H_inv = np.diag(np.asarray(fallback_std) ** 2)
        else:
            H_inv = np.eye(n_params) * 0.25

    lo = np.array([b[0] for b in family.bounds])
    hi = np.array([b[1] for b in family.bounds])

    # Sample from multivariate Student-t (or normal)
    if _HAS_MULTIVARIATE_T:
        dist = multivariate_t(loc=mle_theta, shape=H_inv, df=df)
        samples = dist.rvs(n_samples)
        if n_samples == 1:
            samples = samples[np.newaxis, :]
        log_q = dist.logpdf(samples)
    else:
        from scipy.stats import multivariate_normal

        dist = multivariate_normal(mean=mle_theta, cov=H_inv)
        samples = dist.rvs(n_samples)
        if n_samples == 1:
            samples = samples[np.newaxis, :]
        log_q = dist.logpdf(samples)

    in_bounds = np.all((samples >= lo) & (samples <= hi), axis=1)

    target_nll = make_neg_log_like(
        family, light_curve, use_regularization=use_regularization
    )

    log_lml_vals = []
    log_target_posterior = []
    log_bf_vals = []

    for i, theta in enumerate(samples):
        if not in_bounds[i]:
            log_lml_vals.append(-np.inf)
            log_target_posterior.append(-np.inf)
            if template is not None:
                log_bf_vals.append(0.0)
        else:
            log_bf, log_lml = _compute_bf_lml_at_theta(
                theta, family, light_curve, template, sigma_a=sigma_a, prior=prior
            )
            log_lml_vals.append(log_lml)
            log_target_posterior.append(-target_nll(theta))
            if template is not None:
                log_bf_vals.append(log_bf)

    ll_arr = np.array(log_lml_vals)
    lp_arr = np.array(log_target_posterior)
    log_w = lp_arr - log_q

    # Calculate normalized weights for return
    try:
        norm_w = np.exp(log_w - logsumexp(log_w))
    except Exception:
        norm_w = np.ones(len(log_w)) / len(log_w)

    log_lml_marg = _marginalize_lml(ll_arr, log_w)
    log_bf_marg = None
    if template is not None:
        log_bf_marg = _marginalize_log_bf(ll_arr, np.array(log_bf_vals), log_w)

    return IntegrationResult(
        log_bf_marg=log_bf_marg,
        log_lml_marg=log_lml_marg,
        n_samples=n_samples,
        n_inversions=n_samples,
        samples=samples,
        weights=norm_w,
    )


def nested_sampling_evidence(
    family: GPFamily,
    light_curve: LightCurve,
    template: np.ndarray | None = None,
    nlive: int = 100,
    dlogz: float = 0.1,
    sigma_a: float = 1.0,
    prior: str = "half-normal",
    use_regularization: bool = True,
) -> IntegrationResult:
    """Nested sampling evidence via dynesty."""
    import dynesty

    ndim = len(family.theta_names)
    nll = make_neg_log_like(family, light_curve, use_regularization=use_regularization)

    class _TrackedLikelihood:
        def __init__(self):
            self.n_inversions = 0

        def __call__(self, theta):
            self.n_inversions += 1
            return -nll(theta)

    tracker = _TrackedLikelihood()

    def prior_transform(u):
        x = np.array(u)
        for i in range(ndim):
            lo, hi = family.bounds[i]
            x[i] = lo + u[i] * (hi - lo)
        return x

    sampler = dynesty.NestedSampler(
        tracker,
        prior_transform,
        ndim=ndim,
        nlive=nlive,
        bound="multi",
        sample="rslice",
        bootstrap=0,
    )
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    results = sampler.results
    log_lml_marg = float(results.logz[-1])

    # Equal weighted samples for corner plotting
    from dynesty import utils as dyutils

    try:
        weights = np.exp(results.logwt - results.logz[-1])
        equal_samples = dyutils.resample_equal(results.samples, weights)
    except Exception:
        equal_samples = results.samples

    log_bf_marg = None
    if template is not None:
        log_lml_vals = results.logl
        log_bf_vals = []
        for theta in results.samples:
            log_bf, _ = _compute_bf_lml_at_theta(
                theta, family, light_curve, template, sigma_a=sigma_a, prior=prior
            )
            log_bf_vals.append(log_bf)

        log_prior_w = results.logwt - results.logl
        log_bf_marg = _marginalize_log_bf(
            np.array(log_lml_vals), np.array(log_bf_vals), log_w=log_prior_w
        )

    return IntegrationResult(
        log_bf_marg=log_bf_marg,
        log_lml_marg=log_lml_marg,
        n_samples=len(results.samples),
        n_inversions=tracker.n_inversions,
        samples=equal_samples,
        weights=None,
    )


def grid_evidence(
    family: GPFamily,
    light_curve: LightCurve,
    mle_theta: np.ndarray,
    hess_inv_diag: np.ndarray,
    template: np.ndarray,
    grid_size: int = 20,
    n_sigma: float = 3.0,
    sigma_a: float = 1.0,
    prior: str = "half-normal",
) -> IntegrationResult:
    """Marginalized log Bayes factor via a Hessian-informed zoom grid (B1).

    Builds a ``grid_size``×``grid_size`` grid spanning ``mle_theta ± n_sigma *
    hess_inv_diag`` (clamped to ``family.bounds``) over the (log_sigma,
    log_omega) axes, with the remaining parameters fixed at the MLE, then
    marginalizes with log-space Voronoi weights.
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
            lb, ll = _compute_bf_lml_at_theta(
                theta, family, light_curve, template, sigma_a=sigma_a, prior=prior
            )
            log_lml_vals.append(ll)
            log_bf_vals.append(lb)

    log_w = (W_s + W_w).ravel()
    log_bf_marg = _marginalize_log_bf(
        np.array(log_lml_vals), np.array(log_bf_vals), log_w
    )
    log_lml_marg = _marginalize_lml(np.array(log_lml_vals), log_w)

    return IntegrationResult(
        log_bf_marg=log_bf_marg,
        log_lml_marg=log_lml_marg,
        n_samples=grid_size * grid_size,
        n_inversions=grid_size * grid_size,
    )


def integrate_gp_evidence(
    family: GPFamily,
    light_curve: LightCurve,
    template: np.ndarray,
    method: str = "importance_sampling",
    mle_theta: np.ndarray | None = None,
    lbfgsb_result=None,
    fallback_std: np.ndarray | None = None,
    **kwargs,
) -> IntegrationResult:
    """Compute the marginalized log evidence (log Bayes factor and log marginal likelihood)
    integrated over the GP hyperparameter space.

    Supported methods:
      - 'importance_sampling': Multivariate Student-t Importance Sampling (df=3)
        centered at the MLE, using the inverse Hessian covariance scale.
      - 'nested_sampling': Nested sampling using dynesty.
      - 'grid': Grid integration over selected dimensions.
    """
    if method == "importance_sampling":
        if mle_theta is None:
            raise ValueError(
                "mle_theta must be provided for importance_sampling method."
            )
        return importance_sampling_evidence(
            family=family,
            light_curve=light_curve,
            mle_theta=mle_theta,
            lbfgsb_result=lbfgsb_result,
            template=template,
            fallback_std=fallback_std,
            **kwargs,
        )
    elif method == "nested_sampling":
        return nested_sampling_evidence(
            family=family,
            light_curve=light_curve,
            template=template,
            **kwargs,
        )
    elif method in ("grid", "hessian_zoom"):
        if mle_theta is None:
            raise ValueError("mle_theta must be provided for grid/hessian_zoom method.")
        if fallback_std is None:
            fallback_std = extract_hess_inv_diag(lbfgsb_result, len(family.theta_names))
        return grid_evidence(
            family=family,
            light_curve=light_curve,
            mle_theta=mle_theta,
            hess_inv_diag=fallback_std,
            template=template,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown integration method: {method}")
