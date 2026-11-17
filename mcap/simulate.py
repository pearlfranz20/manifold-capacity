"""Direct simulation of manifold classification capacity.

For each trial we draw P_max manifolds and random +/-1 labels once, then
bisect on P using the first P manifolds. Separability of nested prefixes is
monotone in P (a separating plane for P manifolds also separates any subset),
so integer bisection finds the exact largest separable P* for that draw.
The critical load is reported as alpha_c = (mean P* + 0.5) / N, i.e. the
midpoint between the last separable and first non-separable P.

All classifiers are homogeneous (no bias), matching the theory.
"""
import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------- feasibility
def separable_points(X, y):
    """LP feasibility: exists w with y_i w.x_i >= 1 for all rows x_i of X."""
    A = -(y[:, None] * X)
    res = linprog(np.zeros(X.shape[1]), A_ub=A, b_ub=-np.ones(len(y)),
                  bounds=[(None, None)] * X.shape[1], method="highs")
    return res.status == 0


def separable_spheres(C, U, y, R):
    """SOCP feasibility for L2 spheres: y_mu w.c_mu - R ||U_mu^T w|| >= 1.

    C: (P x N) centers, U: (P x N x D) axes, y: (P,) labels."""
    import cvxpy as cp
    P, N = C.shape
    w = cp.Variable(N)
    cons = [y[m] * (C[m] @ w) - R * cp.norm(U[m].T @ w) >= 1 for m in range(P)]
    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        prob.solve(solver=cp.SCS)
    return prob.status in ("optimal", "optimal_inaccurate")


# ---------------------------------------------------------------- generators
def gen_points(P, N, rng):
    return rng.standard_normal((P, N)) / np.sqrt(N)


def gen_spheres(P, N, D, rng):
    C = rng.standard_normal((P, N)) / np.sqrt(N)
    U = rng.standard_normal((P, N, D)) / np.sqrt(N)
    return C, U


def gen_point_clouds(P, N, D, M, R, rng, centers=None):
    """P manifolds, each M points c_mu + R * U_mu s, s ~ N(0, I_D / D) so that the
    expected squared radius is R^2 (centers have norm ~1).

    Returns array (P, M, N)."""
    C = rng.standard_normal((P, N)) / np.sqrt(N) if centers is None else centers
    U = rng.standard_normal((P, N, D)) / np.sqrt(N)
    s = rng.standard_normal((P, M, D)) / np.sqrt(D)
    return C[:, None, :] + R * np.einsum("pnd,pmd->pmn", U, s)


# ---------------------------------------------------------------- bisection
def _bisect(is_sep, P_max):
    """Largest P in [0, P_max] with is_sep(P) True, assuming monotonicity."""
    if is_sep(P_max):
        return P_max
    lo, hi = 0, P_max  # is_sep(lo) True (empty set), is_sep(hi) False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if is_sep(mid):
            lo = mid
        else:
            hi = mid
    return lo


def capacity_points(N, n_trials=20, rng=0, P_max=None):
    rng = np.random.default_rng(rng)
    P_max = P_max or 4 * N
    Ps = []
    for _ in range(n_trials):
        X = gen_points(P_max, N, rng)
        y = rng.choice([-1.0, 1.0], P_max)
        Ps.append(_bisect(lambda P: separable_points(X[:P], y[:P]), P_max))
    Ps = np.array(Ps)
    return (Ps.mean() + 0.5) / N, Ps.std() / N / np.sqrt(n_trials)


def capacity_spheres(N, D, R, n_trials=10, rng=0, P_max=None):
    rng = np.random.default_rng(rng)
    P_max = P_max or 3 * N
    Ps = []
    for _ in range(n_trials):
        C, U = gen_spheres(P_max, N, D, rng)
        y = rng.choice([-1.0, 1.0], P_max)
        Ps.append(_bisect(lambda P: separable_spheres(C[:P], U[:P], y[:P], R), P_max))
    Ps = np.array(Ps)
    return (Ps.mean() + 0.5) / N, Ps.std() / N / np.sqrt(n_trials)


def capacity_point_clouds(N, D, M, R, n_trials=10, rng=0, P_max=None,
                          center_fn=None, return_manifolds=False):
    """Simulated capacity for point-cloud manifolds.

    center_fn(P, N, rng) optionally generates correlated centers."""
    rng = np.random.default_rng(rng)
    P_max = P_max or 3 * N
    Ps, examples = [], []
    for _ in range(n_trials):
        C = center_fn(P_max, N, rng) if center_fn else None
        Xm = gen_point_clouds(P_max, N, D, M, R, rng, centers=C)
        y = rng.choice([-1.0, 1.0], P_max)

        def is_sep(P):
            X = Xm[:P].reshape(-1, N)
            return separable_points(X, np.repeat(y[:P], M))

        Ps.append(_bisect(is_sep, P_max))
        examples.append(Xm)
    Ps = np.array(Ps)
    out = ((Ps.mean() + 0.5) / N, Ps.std() / N / np.sqrt(n_trials))
    if return_manifolds:
        return out + (examples,)
    return out
