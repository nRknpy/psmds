#!/usr/bin/env python3
# Test global geodesic network: hubs connected by great-circle segments.
import numpy as np
from numpy.linalg import norm
from psmds import ParametricSphericalMDS
from pathlib import Path

from visualize import scatter_sphere3d, scatter_euclid3d, scatter_sphere2d, scatter_euclid2d


img_path =  Path(__file__).parents[1] / 'images'


def _normalize_rows(X, eps=1e-12):
    return X / np.clip(norm(X, axis=1, keepdims=True), eps, None)

def _tangent_noise(x, sigma, rng):
    z = rng.normal(size=x.shape); z -= np.dot(z, x) * x
    y = x + sigma * z
    return y / max(1e-12, norm(y))

def _slerp(a, b, t):
    a = a / norm(a); b = b / norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-8:
        return np.outer(np.ones_like(t), a)
    s = np.sin(omega)
    return (np.sin((1-t)*omega)[:,None]*a + np.sin(t*omega)[:,None]*b) / s

def make_spherical_graph(d=8, n_hubs=32, edges_per_hub=3, samples_per_edge=40, noise=0.01, seed=14):
    rng = np.random.default_rng(seed)
    hubs = _normalize_rows(rng.normal(size=(n_hubs, d)))
    cosHH = np.clip(hubs @ hubs.T, -1, 1)
    X = []
    for i in range(n_hubs):
        nbrs = np.argsort(-cosHH[i])
        nbrs = [j for j in nbrs if j != i][:edges_per_hub]
        for j in nbrs:
            ts = np.linspace(0, 1, samples_per_edge)
            seg = _slerp(hubs[i], hubs[j], ts)
            for p in seg:
                X.append(_tangent_noise(p, noise, rng))
    return _normalize_rows(np.asarray(X))

# metrics
def angle_rmse(X, Y, sample_k=20000, seed=0):
    rng = np.random.default_rng(seed); n = len(X)
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

def run(target, n_components, seed=0):
    X = make_spherical_graph(d=8, n_hubs=32, edges_per_hub=3, samples_per_edge=40, noise=0.01, seed=seed)
    est = ParametricSphericalMDS(n_components=n_components, init="pca", target=target,
                                 max_iter=300, lr=0.05, n_pairs=40000,
                                 batch_size=4000, random_state=seed, verbose=0)
    Y = est.fit_transform(X)
    print(f"[sph_graph_global] target={target} k={n_components} "
          f"angle_RMSE(rad)={angle_rmse(X,Y):.4f}  kNN@10={knn_overlap(X,Y):.4f}  triplet_acc={triplet_accuracy(X,Y):{'.4f'}}")
    
    if n_components == 3:
        if t == 'sphere':
            fig = scatter_sphere3d(Y)
            fig.savefig(img_path / 'sph_graph_S^2.png')
        else:
            fig = scatter_euclid3d(Y)
            fig.savefig(img_path / 'sph_graph_R^3.png')
    else:
        if t == 'sphere':
            fig = scatter_sphere2d(Y)
            fig.savefig(img_path / 'sph_graph_S^1.png')
        else:
            fig = scatter_euclid2d(Y)
            fig.savefig(img_path / 'sph_graph_R^2.png')

if __name__ == "__main__":
    for t in ("sphere", "euclid"):
        for k in (2, 3):
            run(t, k, seed=0)
