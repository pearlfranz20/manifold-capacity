"""Mean-field (replica) capacity formulas from Chung, Lee & Sompolinsky (2018).

All formulas here are for margin kappa >= 0 and the large-N limit.
"""
import numpy as np
from scipy import integrate, stats


def _gauss_quad_moment(a, b, c):
    """Integral over t in [a, b] of (c - t)^2 * phi(t) dt, phi = standard normal pdf.

    Closed form, works with a = -inf and/or b = +inf.
    """
    Phi, phi = stats.norm.cdf, stats.norm.pdf

    def tphi(x):
        return 0.0 if np.isinf(x) else x * phi(x)

    dPhi = Phi(b) - Phi(a)
    return (c * c + 1.0) * dPhi - 2.0 * c * (phi(a) - phi(b)) + tphi(a) - tphi(b)


def _sphere_inner(r, R, D, kappa):
    """Inner Gaussian integral over t0 for a fixed axis-norm r = |t_vec|.

    Three regimes for the projection of (t0, t_vec) onto the cone
    {v0 - R |v_vec| >= kappa}:
      interior   t0 > kappa + R r         cost 0
      touching   kappa - r/R < t0 < kappa + R r   cost (kappa + R r - t0)^2 / (1 + R^2)
      embedded   t0 < kappa - r/R         cost (kappa - t0)^2 + r^2
    """
    if R == 0:
        return _gauss_quad_moment(-np.inf, kappa, kappa)
    lo = kappa - r / R
    hi = kappa + R * r
    touching = _gauss_quad_moment(lo, hi, kappa + R * r) / (1.0 + R * R)
    embedded = _gauss_quad_moment(-np.inf, lo, kappa) + r * r * stats.norm.cdf(lo)
    return touching + embedded


def alpha_sphere(R, D, kappa=0.0):
    """Capacity alpha_c = P/N of D-dimensional L2 spheres (or balls) of radius R.

    alpha^{-1} = int dr chi_D(r) int Dt0 F(r, t0)   (Chung et al. 2018, Eq. for L2 balls)
    R is in units of the center norm. D = 0 or R = 0 gives points (alpha = 2 at kappa = 0).
    """
    if D == 0 or R == 0:
        inv = _gauss_quad_moment(-np.inf, kappa, kappa)
        return 1.0 / inv
    chi = stats.chi(D)
    upper = chi.ppf(1 - 1e-12)
    inv, _ = integrate.quad(lambda r: chi.pdf(r) * _sphere_inner(r, R, D, kappa),
                            0.0, upper, limit=200)
    return 1.0 / inv


def alpha_sphere_large_D(R, D):
    """Large-D approximation alpha ~ (1 + R^2) / (R^2 D) (valid when R sqrt(D) >> 1)."""
    return (1.0 + R ** 2) / (R ** 2 * D)


def cover_prob_separable(P, N):
    """Exact Cover (1965) probability that P random points in N dims are
    linearly separable (homogeneous / through the origin) under random labels."""
    from scipy.special import comb
    k = np.arange(N)
    return 2.0 ** (1 - P) * np.sum(comb(P - 1, k, exact=False))
