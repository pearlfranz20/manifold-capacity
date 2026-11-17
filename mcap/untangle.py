"""Extension (iii): toy 'untangling' across network depth.

A stand-in for a visual hierarchy (V1 -> higher areas). Input manifolds are
curved, low-dimensional object manifolds with a large radius relative to the
spread of their centers (tangled). We pass them through (a) a random ReLU
network and (b) the same architecture trained to classify the objects, and
measure manifold capacity, radius and dimension at every layer.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier


def make_tangled_manifolds(P=30, N=200, D_latent=3, R=2.0, n_per=240, rng=0):
    """Each object: x = c + R * normalize(W2 cos(W1 z + b)), z ~ N(0, I_latent).

    Returns X (P*n_per x N) and labels (object ids)."""
    rng = np.random.default_rng(rng)
    H = 20  # number of random Fourier features per object
    X, ids = [], []
    for p in range(P):
        c = rng.standard_normal(N) / np.sqrt(N)
        W1 = rng.standard_normal((H, D_latent))
        b = rng.uniform(0, 2 * np.pi, H)
        W2 = rng.standard_normal((N, H)) / np.sqrt(N * H / 2)
        z = rng.standard_normal((n_per, D_latent))
        f = np.cos(z @ W1.T + b) @ W2.T  # (n_per, N), norm ~ 1
        X.append(c + R * f)
        ids.append(np.full(n_per, p))
    return np.vstack(X), np.concatenate(ids)


def layer_activations(weights, biases, X):
    """Return [input, h1, h2, ...] for a ReLU MLP (hidden layers only)."""
    acts, h = [X], X
    for W, b in zip(weights, biases):
        h = np.maximum(h @ W + b, 0.0)
        acts.append(h)
    return acts


def random_network(N, widths, rng=0):
    rng = np.random.default_rng(rng)
    Ws, bs, d = [], [], N
    for w in widths:
        Ws.append(rng.standard_normal((d, w)) * np.sqrt(2.0 / d))
        bs.append(np.zeros(w))
        d = w
    return Ws, bs


def trained_network(X, y, widths, rng=0, max_iter=300):
    clf = MLPClassifier(hidden_layer_sizes=tuple(widths), max_iter=max_iter,
                        random_state=rng, alpha=1e-4)
    clf.fit(X, y)
    nh = len(widths)
    return clf.coefs_[:nh], clf.intercepts_[:nh], clf
