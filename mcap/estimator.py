"""Mean-field capacity estimator for general (point-cloud) manifolds.

Implements the anchor-point / quadratic-program estimator of Chung, Lee &
Sompolinsky (2018) and Cohen, Chung, Lee & Sompolinsky (2020) at kappa = 0.

For one manifold with points S (columns) the replica theory gives

    alpha_M^{-1} = E_t [ min_{V : V . s >= 0 for all s in S} ||V - t||^2 ],
    t ~ N(0, I) in the span of the manifold.

The feasible set is the polyhedral cone K = {V : S^T V >= 0}. By Moreau's
decomposition, t - Proj_K(t) = Proj_{K°}(t), and the polar cone is
K° = {-S lam : lam >= 0}. So the QP reduces to a non-negative least squares
problem   min_{lam >= 0} || S lam + t ||^2, and the cost is ||S lam*||^2.
The anchor point is s~(t) = S lam / sum(lam) (a point of the convex hull),
or argmin_s t.s when t is already in the cone (interior regime).
"""
import numpy as np
from scipy.optimize import nnls


def manifold_coords(X, tol=1e-9):
    """Express manifold points X (N x M) in an orthonormal basis of their span
    whose first axis is the center direction. Returns (S, center_norm)."""
    c = X.mean(axis=1)
    cn = np.linalg.norm(c)
    chat = c / cn
    Y = X - np.outer(chat, chat @ X)
    U, sv, _ = np.linalg.svd(Y, full_matrices=False)
    r = int(np.sum(sv > tol * max(sv.max(), 1e-300)))
    Q = np.column_stack([chat, U[:, :r]])
    return Q.T @ X, cn


def _project_cost_nnls(S, t):
    """Return (cost, lam) for min_{lam>=0} ||S lam + t||^2."""
    if np.min(S.T @ t) >= 0:  # t already inside the cone: zero cost
        return 0.0, None
    lam, _ = nnls(S, -t)
    # ||S lam + t|| is the norm of Proj_K(t). The cost we want is the distance
    # from t to K, i.e. the norm of the polar part ||S lam||.
    return float(np.sum((S @ lam) ** 2)), lam


def _project_cost_cvxpy(S, t):
    """Reference implementation: solve the primal QP with cvxpy (slow)."""
    import cvxpy as cp
    v = cp.Variable(S.shape[0])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(v - t)), [S.T @ v >= 0])
    prob.solve(solver=cp.CLARABEL)
    return float(np.sum((v.value - t) ** 2)), None


def analyze_manifold(X, n_t=200, rng=None, solver="nnls"):
    """Estimate capacity, radius and dimension of a single point-cloud manifold.

    X: (N x M) array, columns are points (assumed already globally centered).
    Returns dict(alpha, R, D).
    """
    rng = np.random.default_rng(rng)
    S, cn = manifold_coords(X)
    d1 = S.shape[0]
    proj = _project_cost_nnls if solver == "nnls" else _project_cost_cvxpy
    costs, R2, Dvals = [], [], []
    for _ in range(n_t):
        t = rng.standard_normal(d1)
        cost, lam = proj(S, t)
        costs.append(cost)
        if lam is None or lam.sum() <= 0:
            s = S[:, np.argmin(S.T @ t)]
        else:
            s = S @ lam / lam.sum()
        s_ax = s[1:] / cn  # component orthogonal to the center, in units of |center|
        nrm = np.linalg.norm(s_ax)
        R2.append(nrm ** 2)
        if nrm > 0:
            Dvals.append((t[1:] @ (s_ax / nrm)) ** 2)
        else:
            Dvals.append(0.0)
    mean_cost = np.mean(costs)
    return dict(alpha=1.0 / mean_cost if mean_cost > 0 else np.inf,
                R=float(np.sqrt(np.mean(R2))), D=float(np.mean(Dvals)))


def manifold_analysis(manifolds, n_t=200, rng=0, center=True, solver="nnls"):
    """Analyze a list of manifolds (each N x M). Returns the population capacity
    alpha_M = 1 / mean(1/alpha_mu), and mean R_M, D_M.

    center=True subtracts the global mean across all points first (as in
    Cohen et al. 2020)."""
    rng = np.random.default_rng(rng)
    if center:
        gm = np.concatenate(manifolds, axis=1).mean(axis=1, keepdims=True)
        manifolds = [X - gm for X in manifolds]
    res = [analyze_manifold(X, n_t=n_t, rng=rng, solver=solver) for X in manifolds]
    a = np.array([r["alpha"] for r in res])
    return dict(alpha=1.0 / np.mean(1.0 / a),
                R=float(np.mean([r["R"] for r in res])),
                D=float(np.mean([r["D"] for r in res])),
                per_manifold=res)
