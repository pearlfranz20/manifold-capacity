"""Regenerate every figure and results/results.json.

Usage:  python run_all.py            (full run, a few minutes on CPU)
        python run_all.py --quick    (fewer trials, for smoke testing)
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcap import estimator, simulate, theory, untangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

COLORS = ["#2a6fdb", "#e0612b", "#2b9e6b", "#8a4fc7"]
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. Cover
def fig_cover(q):
    N = 50
    trials = 20 if q else 60
    rng = np.random.default_rng(0)
    Pstars = []
    for _ in range(trials):
        X = simulate.gen_points(4 * N, N, rng)
        y = rng.choice([-1.0, 1.0], 4 * N)
        Pstars.append(simulate._bisect(lambda P: simulate.separable_points(X[:P], y[:P]), 4 * N))
    Pstars = np.array(Pstars)
    Ps = np.arange(1, 4 * N + 1)
    p_sim = np.array([(Pstars >= P).mean() for P in Ps])
    p_th = np.array([theory.cover_prob_separable(P, N) for P in Ps])
    alphas_N = {}
    for n in [25, 50, 100, 200]:
        a, se = simulate.capacity_points(n, n_trials=10 if q else 20, rng=n)
        alphas_N[n] = (a, se)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(Ps / N, p_th, color="k", lw=1.5, label="Cover formula (exact)")
    ax[0].plot(Ps / N, p_sim, "o", ms=3, color=COLORS[0], label=f"simulation ({trials} trials)")
    ax[0].axvline(2, color="gray", ls=":", lw=1)
    ax[0].set(xlabel="P / N", ylabel="P(separable)", title=f"Points, N = {N}")
    ax[0].legend(frameon=False)
    ns = sorted(alphas_N)
    ax[1].errorbar(ns, [alphas_N[n][0] for n in ns], yerr=[alphas_N[n][1] for n in ns],
                   fmt="o", color=COLORS[0], capsize=3, label="simulation (bisection)")
    ax[1].axhline(2, color="k", lw=1.5, label="theory  alpha_c = 2")
    ax[1].set(xscale="log", xlabel="N", ylabel="alpha_c", ylim=(1.6, 2.4),
              title="Critical capacity of points vs N")
    ax[1].minorticks_off()
    ax[1].set_xticks(ns, [str(n) for n in ns])
    ax[1].legend(frameon=False)
    save(fig, "fig1_cover_points.png")
    return {"N_curve": N, "alpha_c_vs_N": {str(k): v for k, v in alphas_N.items()}}


# ------------------------------------------------------------------ 2. Spheres
def fig_spheres(q):
    N = 50
    Ds = [2, 5, 20]
    Rs_th = np.logspace(-1.3, 1.0, 60)
    Rs_sim = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    trials = 3 if q else 6
    out = {}
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for i, D in enumerate(Ds):
        th = np.array([theory.alpha_sphere(R, D) for R in Rs_th])
        ax.plot(Rs_th, th, color=COLORS[i], lw=1.6, label=f"theory, D = {D}")
        if D == 20:
            ax.plot(Rs_th, theory.alpha_sphere_large_D(Rs_th, D), color=COLORS[i], ls="--",
                    lw=1, label="large-D approx (1+R²)/(R²D), D = 20")
        sims = []
        for R in Rs_sim:
            a_th = theory.alpha_sphere(R, D)
            P_max = int(np.ceil(min(2.6, 2.2 * a_th) * N)) + 5
            a, se = simulate.capacity_spheres(N, D, R, n_trials=trials, rng=int(100 * R) + D,
                                              P_max=P_max)
            sims.append((R, a, se, a_th))
        sims = np.array(sims)
        ax.errorbar(sims[:, 0], sims[:, 1], yerr=sims[:, 2], fmt="o", color=COLORS[i],
                    ms=5, capsize=2, mec="white")
        out[str(D)] = sims.tolist()
    ax.axhline(2, color="gray", ls=":", lw=1)
    ax.set_ylim(0.03, 3)
    ax.set(xscale="log", yscale="log", xlabel="radius R (units of center norm)",
           ylabel="alpha_c = P / N",
           title=f"L2 spheres: theory (lines) vs simulation (dots, N = {N})")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig2_spheres_capacity_vs_R.png")
    return out


# ------------------------------------------------------------------ 3. Point clouds
def fig_point_clouds(q):
    N, M = 100, 20
    Ds = [3, 10]
    Rs = [0.25, 0.5, 1.0, 1.5, 2.5]
    trials = 3 if q else 5
    out = {}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for i, D in enumerate(Ds):
        rows = []
        for R in Rs:
            rng = np.random.default_rng(int(1000 * R) + D)
            # theory: estimator on one sample of P0 manifolds
            Xm = simulate.gen_point_clouds(60, N, D, M, R, rng)
            est = estimator.manifold_analysis([x.T for x in Xm], n_t=200 if not q else 80, rng=1)
            ball = theory.alpha_sphere(est["R"], est["D"])
            P_max = int(np.ceil(min(2.6, 2.0 * est["alpha"]) * N)) + 5
            a, se = simulate.capacity_point_clouds(N, D, M, R, n_trials=trials,
                                                   rng=int(10 * R) + D, P_max=P_max)
            rows.append(dict(R=R, sim=a, sim_se=se, est=est["alpha"], R_M=est["R"],
                             D_M=est["D"], ball_RM_DM=ball))
        out[str(D)] = rows
        r = [x["R"] for x in rows]
        ax[0].plot(r, [x["est"] for x in rows], "-", color=COLORS[i], lw=1.6,
                   label=f"theory estimator, D = {D}")
        ax[0].plot(r, [x["ball_RM_DM"] for x in rows], ":", color=COLORS[i], lw=1.4,
                   label=f"ball formula at (R_M, D_M), D = {D}")
        ax[0].errorbar(r, [x["sim"] for x in rows], yerr=[x["sim_se"] for x in rows],
                       fmt="o", color=COLORS[i], capsize=2, mec="white",
                       label=f"simulation, D = {D}")
        ax[1].plot(r, [x["R_M"] for x in rows], "o-", color=COLORS[i], label=f"D = {D}")
        ax[2].plot(r, [x["D_M"] for x in rows], "o-", color=COLORS[i], label=f"D = {D}")
    ax[0].set(xlabel="spread R", ylabel="alpha_c", yscale="log",
              title=f"Point clouds (M = {M} pts, N = {N})")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(Rs, Rs, color="gray", ls=":", lw=1, label="R_M = R")
    ax[1].set(xlabel="spread R", ylabel="R_M", title="Manifold radius R_M")
    ax[1].legend(frameon=False)
    ax[2].set(xlabel="spread R", ylabel="D_M", title="Manifold dimension D_M")
    ax[2].legend(frameon=False)
    save(fig, "fig3_point_clouds.png")

    # scatter: theory vs simulation, all conditions
    fig, ax = plt.subplots(figsize=(4, 4))
    allrows = [x for v in out.values() for x in v]
    s = np.array([x["sim"] for x in allrows])
    e = np.array([x["est"] for x in allrows])
    lim = [0.8 * min(s.min(), e.min()), 1.2 * max(s.max(), e.max())]
    ax.plot(lim, lim, color="gray", ls=":")
    for i, D in enumerate(Ds):
        ax.errorbar([x["est"] for x in out[str(D)]], [x["sim"] for x in out[str(D)]],
                    yerr=[x["sim_se"] for x in out[str(D)]], fmt="o", color=COLORS[i],
                    capsize=2, label=f"D = {D}")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="theory alpha_M (estimator)",
           ylabel="simulated alpha_c", title="Point clouds: theory vs simulation")
    ax.legend(frameon=False)
    save(fig, "fig4_point_clouds_theory_vs_sim.png")
    return out


# ------------------------------------------------------------------ 4. Extension: speed
def fig_speed(q):
    rng = np.random.default_rng(3)
    Ms = [5, 10, 20, 50] if q else [5, 10, 20, 50, 100]
    n_t = 20 if q else 40
    rows = []
    for M in Ms:
        X = simulate.gen_point_clouds(1, 200, min(M, 20), M, 1.0, rng)[0].T
        t0 = time.perf_counter()
        a_n = estimator.analyze_manifold(X, n_t=n_t, rng=0, solver="nnls")["alpha"]
        t1 = time.perf_counter()
        a_c = estimator.analyze_manifold(X, n_t=n_t, rng=0, solver="cvxpy")["alpha"]
        t2 = time.perf_counter()
        rows.append(dict(M=M, t_nnls=(t1 - t0) / n_t, t_cvxpy=(t2 - t1) / n_t,
                         alpha_nnls=a_n, alpha_cvxpy=a_c))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(Ms, [r["t_cvxpy"] * 1e3 for r in rows], "o-", color=COLORS[1], label="cvxpy QP (Clarabel)")
    ax.plot(Ms, [r["t_nnls"] * 1e3 for r in rows], "o-", color=COLORS[0], label="NNLS on polar cone")
    ax.set(xscale="log", yscale="log", xlabel="points per manifold M",
           ylabel="ms per Gaussian sample", title="Estimator inner solve: QP vs NNLS")
    ax.set_xticks(Ms, [str(m) for m in Ms])
    ax.legend(frameon=False)
    save(fig, "fig5_ext_speedup.png")
    return rows


# ------------------------------------------------------------------ 5. Extension: correlated centers
def fig_correlated(q):
    N, D, M, R, K = 100, 5, 20, 0.5, 3
    rhos = [0.0, 0.5, 0.8, 0.9, 0.95]
    trials = 3 if q else 5
    rows = []
    for rho in rhos:
        rng_v = np.random.default_rng(7)
        V = np.linalg.qr(rng_v.standard_normal((N, K)))[0]

        def center_fn(P, N_, rng, rho=rho, V=V):
            g = rng.standard_normal((P, N_)) / np.sqrt(N_)
            a = rng.standard_normal((P, K)) / np.sqrt(K)
            return np.sqrt(1 - rho) * g + np.sqrt(rho) * a @ V.T

        rng = np.random.default_rng(11)
        Xm = simulate.gen_point_clouds(60, N, D, M, R, rng, centers=center_fn(60, N, rng))
        mans = [x.T for x in Xm]
        est_raw = estimator.manifold_analysis(mans, n_t=150, rng=1)
        # correction: project out the top-K principal axes of the manifold centers
        C = np.stack([x.mean(axis=1) for x in mans])
        C = C - C.mean(axis=0)
        Vc = np.linalg.svd(C, full_matrices=False)[2][:K].T
        Pr = np.eye(N) - Vc @ Vc.T
        est_corr = estimator.manifold_analysis([Pr @ x for x in mans], n_t=150, rng=1)
        P_max = int(np.ceil(2.0 * max(est_raw["alpha"], est_corr["alpha"]) * N)) + 5
        a, se = simulate.capacity_point_clouds(N, D, M, R, n_trials=trials, rng=5,
                                               P_max=P_max, center_fn=center_fn)
        rows.append(dict(rho=rho, sim=a, sim_se=se, est_raw=est_raw["alpha"],
                         est_corrected=est_corr["alpha"]))
    fig, ax = plt.subplots(figsize=(5, 3.8))
    r = [x["rho"] for x in rows]
    ax.errorbar(r, [x["sim"] for x in rows], yerr=[x["sim_se"] for x in rows], fmt="o",
                color="k", capsize=2, label="simulation")
    ax.plot(r, [x["est_raw"] for x in rows], "s-", color=COLORS[1],
            label="theory, assumes random centers")
    ax.plot(r, [x["est_corrected"] for x in rows], "^-", color=COLORS[0],
            label=f"theory after removing top-{K} center axes")
    ax.set(xlabel="fraction of center variance in shared rank-3 subspace (rho)",
           ylabel="alpha_c", title=f"Correlated centers (N={N}, D={D}, M={M}, R={R})")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig6_ext_correlated_centers.png")
    return rows


# ------------------------------------------------------------------ 6. Extension: untangling
def fig_untangle(q):
    P, N, M = 30, 200, 40
    widths = [200, 200, 200]
    X, ids = untangle.make_tangled_manifolds(P=P, N=N, R=2.0, n_per=240, rng=0)
    rng = np.random.default_rng(0)
    test = np.zeros(len(ids), bool)
    for p in range(P):
        idx = np.where(ids == p)[0]
        test[rng.choice(idx, M, replace=False)] = True
    Wr, br = untangle.random_network(N, widths, rng=1)
    Wt, bt, clf = untangle.trained_network(X[~test], ids[~test], widths, rng=0,
                                           max_iter=60 if q else 200)
    acc = clf.score(X[test], ids[test])
    out = {"heldout_accuracy": acc}
    n_t = 80 if q else 200
    for name, (W, b) in {"random": (Wr, br), "trained": (Wt, bt)}.items():
        acts = untangle.layer_activations(W, b, X[test])
        res = []
        for A in acts:
            mans = [A[ids[test] == p].T for p in range(P)]
            r = estimator.manifold_analysis(mans, n_t=n_t, rng=2)
            res.append(dict(alpha=r["alpha"], R=r["R"], D=r["D"]))
        out[name] = res
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    layers = ["input", "h1", "h2", "h3"]
    for j, key in enumerate(["alpha", "R", "D"]):
        for i, name in enumerate(["random", "trained"]):
            ax[j].plot(layers, [r[key] for r in out[name]], "o-", color=COLORS[i],
                       label=f"{name} ReLU net")
        ax[j].set(title={"alpha": "capacity alpha_M", "R": "radius R_M",
                         "D": "dimension D_M"}[key], xlabel="layer")
    ax[0].legend(frameon=False)
    fig.suptitle(f"Toy untangling: {P} curved object manifolds, {M} held-out points each "
                 f"(trained net held-out accuracy {acc:.2f})", fontsize=10)
    save(fig, "fig7_ext_untangling_depth.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    results, t0 = {}, time.time()
    for name, fn in [("cover", fig_cover), ("spheres", fig_spheres),
                     ("point_clouds", fig_point_clouds), ("speed", fig_speed),
                     ("correlated_centers", fig_correlated), ("untangle", fig_untangle)]:
        t = time.time()
        results[name] = fn(q)
        print(f"[{name}] done in {time.time() - t:.1f}s", flush=True)
    results["total_seconds"] = time.time() - t0
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"total {results['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
