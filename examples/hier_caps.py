#!/usr/bin/env python3
# Test hierarchical structure: super-clusters on a great circle, sub-clusters as spherical caps.
import numpy as np
from numpy.linalg import norm
from psmds import ParametricSphericalMDS
from pathlib import Path
import matplotlib.pyplot as plt

from visualize import scatter_sphere3d, scatter_euclid3d, scatter_sphere2d, scatter_euclid2d


img_path =  Path(__file__).parents[1] / 'images'


def _normalize_rows(X, eps=1e-12):
    return X / np.clip(norm(X, axis=1, keepdims=True), eps, None)

def _tangent_noise(x, sigma, rng):
    z = rng.normal(size=x.shape); z -= np.dot(z, x) * x
    y = x + sigma * z
    return y / max(1e-12, norm(y))

def make_hierarchical_caps(d=8, K_super=5, K_sub=4, n_per_sub=250, super_spread=0.35, sub_spread=0.12, seed=13):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2*np.pi, K_super, endpoint=False)
    supers = []
    for th in angles:
        c = np.zeros(d); c[0] = np.cos(th); c[1] = np.sin(th)
        supers.append(c)
    X = []
    for c in supers:
        subs = [ _tangent_noise(c, super_spread, rng) for _ in range(K_sub) ]
        for sub in subs:
            for _ in range(n_per_sub):
                X.append(_tangent_noise(sub, sub_spread, rng))
    return _normalize_rows(np.asarray(X))

# metrics
def angle_rmse(X, Y, sample_k=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    i = rng.integers(0, n, size=sample_k); j = rng.integers(0, n, size=sample_k)
    m = i != j; i, j = i[m], j[m]
    Yn = Y / np.clip(norm(Y, axis=1, keepdims=True), 1e-12, None)
    aX = np.arccos(np.sum(X[i]*X[j], axis=1).clip(-1,1))
    aY = np.arccos(np.sum(Yn[i]*Yn[j], axis=1).clip(-1,1))
    return float(np.sqrt(np.mean((aX - aY)**2)))

def knn_overlap(X, Y, k=10, sample_n=1000, seed=0):
    Yn = Y / np.clip(norm(Y, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    cosX = np.clip(X @ X.T, -1, 1); cosY = np.clip(Yn @ Yn.T, -1, 1)
    acc = 0.0
    for i in idx:
        Nx = np.argsort(-cosX[i])[1:k+1]; Ny = np.argsort(-cosY[i])[1:k+1]
        acc += len(set(Nx.tolist()).intersection(Ny.tolist())) / k
    return float(acc / len(idx))

def triplet_accuracy(X, Y, m=40000, seed=0):
    rng = np.random.default_rng(seed); n = len(X)
    i = rng.integers(0, n, size=m); j = rng.integers(0, n, size=m); k = rng.integers(0, n, size=m)
    mask = (i!=j)&(i!=k)&(j!=k); i,j,k = i[mask], j[mask], k[mask]
    Yn = Y / np.clip(norm(Y, axis=1, keepdims=True), 1e-12, None)
    return float(((np.sum(X[i]*X[j],1) > np.sum(X[i]*X[k],1)) ==
                  (np.sum(Yn[i]*Yn[j],1) > np.sum(Yn[i]*Yn[k],1))).mean())

def run(target, n_components, seed=13):
    X = make_hierarchical_caps(d=8, K_super=5, K_sub=4, n_per_sub=250,
                               super_spread=0.35, sub_spread=0.1, seed=seed)
    est = ParametricSphericalMDS(n_components=n_components, init="pca", target=target,
                                 max_iter=300, lr=0.05, n_pairs=40000,
                                 batch_size=4000, random_state=seed, verbose=0)
    Y = est.fit_transform(X)
    print(f"[hier_caps_local_global] target={target} k={n_components} "
          f"angle_RMSE(rad)={angle_rmse(X,Y):.4f}  kNN@10={knn_overlap(X,Y):.4f}  triplet_acc={triplet_accuracy(X,Y):.4f}")
    
    if n_components == 3:
        if t == 'sphere':
            fig = scatter_sphere3d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'hier_caps_S^2.png')
            else:
                fig.write_html(img_path / 'hier_caps_S^2.html')
        else:
            fig = scatter_euclid3d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'hier_caps_R^3.png')
            else:
                fig.write_html(img_path / 'hier_caps_R^3.html')
    else:
        if t == 'sphere':
            fig = scatter_sphere2d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'hier_caps_S^1.png')
            else:
                fig.write_html(img_path / 'hier_caps_S^1.html')
        else:
            fig = scatter_euclid2d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'hier_caps_R^2.png')
            else:
                fig.write_html(img_path / 'hier_caps_R^2.html')


if __name__ == "__main__":
    for t in ("sphere", "euclid"):
        for k in (2, 3):
            run(t, k, seed=0)
