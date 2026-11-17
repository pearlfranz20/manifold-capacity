import numpy as np
import pytest

from mcap import estimator, simulate, theory


def test_theory_points_is_two():
    assert theory.alpha_sphere(0.0, 5) == pytest.approx(2.0, abs=1e-9)


def test_theory_sphere_R1_closed_form():
    # At R = 1 and kappa = 0 the sphere capacity is 2 / (D + 1).
    for D in [1, 2, 5, 10]:
        assert theory.alpha_sphere(1.0, D) == pytest.approx(2.0 / (D + 1), rel=1e-6)


def test_cover_formula_half_at_2N():
    assert theory.cover_prob_separable(100, 50) == pytest.approx(0.5, abs=1e-12)


def test_simulated_cover_capacity_near_two():
    a, se = simulate.capacity_points(60, n_trials=15, rng=0)
    assert abs(a - 2.0) < 0.15


def test_estimator_points_gives_two():
    rng = np.random.default_rng(0)
    mans = [rng.standard_normal((40, 1)) for _ in range(20)]
    r = estimator.manifold_analysis(mans, n_t=3000, center=False, rng=1)
    assert r["alpha"] == pytest.approx(2.0, rel=0.08)


def test_nnls_matches_cvxpy():
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(1)
    X = simulate.gen_point_clouds(1, 60, 4, 10, 1.0, rng)[0].T
    a1 = estimator.analyze_manifold(X, n_t=20, rng=0, solver="nnls")["alpha"]
    a2 = estimator.analyze_manifold(X, n_t=20, rng=0, solver="cvxpy")["alpha"]
    assert a1 == pytest.approx(a2, rel=1e-4)


def test_estimator_matches_simulation_point_clouds():
    rng = np.random.default_rng(2)
    Xm = simulate.gen_point_clouds(40, 80, 3, 10, 1.0, rng)
    est = estimator.manifold_analysis([x.T for x in Xm], n_t=150, rng=0)["alpha"]
    sim, _ = simulate.capacity_point_clouds(80, 3, 10, 1.0, n_trials=3, rng=3,
                                            P_max=int(2 * est * 80) + 5)
    assert sim == pytest.approx(est, rel=0.2)
