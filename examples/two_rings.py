#!/usr/bin/env python3
# Detect global separation on two parallel geodesic rings with an angular gap.
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

def make_two_rings(d=8, n_per_ring=1500, gap_deg=35.0, noise=0.03, seed=1):
    rng = np.random.default_rng(seed)
    def ring(seed2):
        r = np.random.default_rng(seed2)
        t = r.uniform(0, 2*np.pi, size=n_per_ring)
        X = np.zeros((n_per_ring, d))
        X[:, 0] = np.cos(t); X[:, 1] = np.sin(t)
        for i in range(n_per_ring):
            X[i] = _tangent_noise(X[i], noise, r)
        return _normalize_rows(X)
    A = ring(rng.integers(1<<31))
    B = ring(rng.integers(1<<31))
    gap = np.deg2rad(gap_deg)
    R = np.eye(d)
    c, s = np.cos(gap), np.sin(gap)
    R[[0,2],[0,2]] = c; R[0,2] = -s; R[2,0] = s
    R[[1,3],[1,3]] = c; R[1,3] = -s; R[3,1] = s
    B = B @ R.T
    X = np.vstack([A, B])
    return _normalize_rows(X)

# metrics (same as ring_local)
def angle_rmse(X, Y, sample_k=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    i = rng.integers(0, n, size=sample_k); j = rng.integers(0, n, size=sample_k)
    m = i != j; i, j = i[m], j[m]
    cX = np.sum(X[i]*X[j], axis=1).clip(-1,1)
    cY = np.sum((Y/np.clip(norm(Y,axis=1,keepdims=True),1e-12,None))[i]*
                (Y/np.clip(norm(Y,axis=1,keepdims=True),1e-12,None))[j], axis=1).clip(-1,1)
    aX = np.arccos(cX); aY = np.arccos(cY)
    return float(np.sqrt(np.mean((aX - aY)**2)))

def knn_overlap(X, Y, k=10, sample_n=1000, seed=0):
    Yn = Y / np.clip(norm(Y, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    cosX = np.clip(X @ X.T, -1, 1)
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
    i = rng.integers(0, n, size=m); j = rng.integers(0, n, size=m); k = rng.integers(0, n, size=m)
    mask = (i!=j)&(i!=k)&(j!=k); i,j,k = i[mask], j[mask], k[mask]
    Yn = Y / np.clip(norm(Y, axis=1, keepdims=True), 1e-12, None)
    cXij = np.sum(X[i]*X[j], axis=1); cXik = np.sum(X[i]*X[k], axis=1)
    cYij = np.sum(Yn[i]*Yn[j], axis=1); cYik = np.sum(Yn[i]*Yn[k], axis=1)
    return float(((cXij > cXik) == (cYij > cYik)).mean())

def run(target, n_components, seed=1):
    X = make_two_rings(d=8, n_per_ring=1500, gap_deg=35.0, noise=0.03, seed=seed)
    est = ParametricSphericalMDS(n_components=n_components, init="pca", target=target,
                                 max_iter=300, lr=0.05, n_pairs=40000,
                                 batch_size=4000, random_state=seed, verbose=0)
    Y = est.fit_transform(X)
    print(f"[two_rings_global] target={target} k={n_components} "
          f"angle_RMSE(rad)={angle_rmse(X,Y):.4f}  kNN@10={knn_overlap(X,Y):.4f}  triplet_acc={triplet_accuracy(X,Y):.4f}")
    
    if n_components == 3:
        if t == 'sphere':
            fig = scatter_sphere3d(Y)
            fig.savefig(img_path / 'two_rings_S^2.png')
        else:
            fig = scatter_euclid3d(Y)
            fig.savefig(img_path / 'two_rings_R^3.png')
    else:
        if t == 'sphere':
            fig = scatter_sphere2d(Y)
            fig.savefig(img_path / 'two_rings_S^1.png')
        else:
            fig = scatter_euclid2d(Y)
            fig.savefig(img_path / 'two_rings_R^2.png')

if __name__ == "__main__":
    for t in ("sphere", "euclid"):
        for k in (2, 3):
            run(t, k, seed=0)
