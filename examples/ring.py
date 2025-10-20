#!/usr/bin/env python3
# Detect local structure preservation on a noisy geodesic ring (S^{d-1}).
import numpy as np
from numpy.linalg import norm
from psmds import ParametricSphericalMDS
from pathlib import Path
import matplotlib.pyplot as plt

from visualize import scatter_sphere3d, scatter_euclid3d, scatter_sphere2d, scatter_euclid2d


img_path =  Path(__file__).parents[1] / 'images'


# ---------- dataset ----------
def _normalize_rows(X, eps=1e-12):
    return X / np.clip(norm(X, axis=1, keepdims=True), eps, None)

def _tangent_noise(x, sigma, rng):
    z = rng.normal(size=x.shape)
    z -= np.dot(z, x) * x
    y = x + sigma * z
    return y / max(1e-12, norm(y))

def make_geodesic_ring(d=8, n=2500, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2*np.pi, size=n)
    X = np.zeros((n, d))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    for i in range(n):
        X[i] = _tangent_noise(X[i], noise, rng)
    return _normalize_rows(X)

# ---------- metrics ----------
def angle_rmse(X, Y, sample_k=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    i = rng.integers(0, n, size=sample_k)
    j = rng.integers(0, n, size=sample_k)
    m = i != j
    i, j = i[m], j[m]
    cX = np.sum(X[i]*X[j], axis=1).clip(-1,1)
    cY = np.sum(Y[i]*Y[j], axis=1).clip(-1,1)
    aX = np.arccos(cX)
    aY = np.arccos(cY)
    return float(np.sqrt(np.mean((aX - aY)**2)))

def knn_overlap(X, Y, k=10, sample_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    cosX = np.clip(X @ X.T, -1, 1)
    if Y.shape[1] == 3 and np.allclose(norm(Y, axis=1), 1, atol=1e-3):
        cosY = np.clip(Y @ Y.T, -1, 1)
    else:
        # Work on normalized copy for angle-based kNN
        Yn = Y / np.clip(norm(Y, axis=1, keepdims=True), 1e-12, None)
        cosY = np.clip(Yn @ Yn.T, -1, 1)
    acc = 0.0
    for i in idx:
        Nx = np.argsort(-cosX[i])[1:k+1]
        Ny = np.argsort(-cosY[i])[1:k+1]
        acc += len(set(Nx.tolist()).intersection(Ny.tolist())) / k
    return float(acc / len(idx))

def triplet_accuracy(X, Y, m=40000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    k = rng.integers(0, n, size=m)
    mask = (i!=j)&(i!=k)&(j!=k)
    i,j,k = i[mask], j[mask], k[mask]
    cXij = np.sum(X[i]*X[j], axis=1)
    cXik = np.sum(X[i]*X[k], axis=1)
    Yn = Y / np.clip(norm(Y, axis=1, keepdims=True), 1e-12, None)
    cYij = np.sum(Yn[i]*Yn[j], axis=1)
    cYik = np.sum(Yn[i]*Yn[k], axis=1)
    return float(( (cXij > cXik) == (cYij > cYik) ).mean())

# ---------- runner ----------
def run(target, n_components, seed=0):
    X = make_geodesic_ring(d=8, n=2500, noise=0.03, seed=seed)
    est = ParametricSphericalMDS(n_components=n_components, init="pca", target=target,
                                 max_iter=300, lr=0.05, n_pairs=40000,
                                 batch_size=4000, random_state=seed, verbose=0)
    Y = est.fit_transform(X)
    rmse = angle_rmse(X, Y, sample_k=20000, seed=seed)
    knn = knn_overlap(X, Y, k=10, sample_n=1000, seed=seed)
    tri = triplet_accuracy(X, Y, m=40000, seed=seed)
    print(f"[ring_local] target={target} k={n_components} angle_RMSE(rad)={rmse:.4f}  kNN@10={knn:.4f}  triplet_acc={tri:.4f}")

    if n_components == 3:
        if t == 'sphere':
            fig = scatter_sphere3d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'ring_S^2.png')
            else:
                fig.write_html(img_path / 'ring_S^2.html')
        else:
            fig = scatter_euclid3d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'ring_R^3.png')
            else:
                fig.write_html(img_path / 'ring_R^3.html')
    else:
        if t == 'sphere':
            fig = scatter_sphere2d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'ring_S^1.png')
            else:
                fig.write_html(img_path / 'ring_S^1.html')
        else:
            fig = scatter_euclid2d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'ring_R^2.png')
            else:
                fig.write_html(img_path / 'ring_R^2.html')

if __name__ == "__main__":
    for t in ("sphere", "euclid"):
        for k in (2, 3):
            run(t, k, seed=0)
