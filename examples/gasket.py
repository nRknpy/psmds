#!/usr/bin/env python3
# Detect global separation on two parallel geodesic rings with an angular gap.
import numpy as np
from numpy.linalg import norm
from psmds import ParametricSphericalMDS
from pathlib import Path
import matplotlib.pyplot as plt

from visualize import scatter_sphere3d, scatter_euclid3d, scatter_sphere2d, scatter_euclid2d


img_path =  Path(__file__).parents[1] / 'images'


import numpy as np

# ---------- Utils ----------

def _normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def _rand_unit_vectors(d, k, rng):
    X = rng.normal(size=(k, d))
    return _normalize_rows(X)

def _tangent_basis(mu):
    """
    Build an orthonormal basis U for the tangent space at mu on S^{d-1}.
    Returns U in R^{d x (d-1)} with columns orthonormal and U.T @ mu = 0.
    """
    d = mu.shape[0]
    # Start with random matrix, then Gram-Schmidt against mu
    v = np.eye(d)
    # Remove component along mu from each column, then QR
    V = v - np.outer(mu, mu) @ v
    Q, _ = np.linalg.qr(V)
    # One column of Q will align with mu; drop it
    # Find the column with the largest |dot(mu, q)|
    dots = Q.T @ mu
    keep = [i for i in range(d) if abs(dots[i]) < 1e-6]
    U = Q[:, keep]
    # Ensure shape d x (d-1)
    if U.shape[1] == d:
        U = U[:, :-1]
    return U

# ---------- Sphere log/exp maps ----------

def sphere_log(mu, p):
    """
    Log map on unit sphere: v in T_mu S^{d-1} such that exp_mu(v) = p.
    mu, p: shape (d,)
    """
    dot = float(np.clip(np.dot(mu, p), -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-8:
        return np.zeros_like(mu)
    v = p - dot * mu
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros_like(mu)
    return (theta / v_norm) * v

def sphere_exp(mu, v):
    """
    Exp map on unit sphere at mu for tangent vector v in R^d with v·mu=0.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        # First-order step, then renormalize for stability
        x = mu + v
        return x / (np.linalg.norm(x) + 1e-12)
    return np.cos(norm) * mu + np.sin(norm) * (v / norm)

# ---------- Random orthogonal in R^{m} ----------

def random_orthogonal(m, rng):
    """Haar-random orthogonal matrix via QR."""
    A = rng.normal(size=(m, m))
    Q, R = np.linalg.qr(A)
    # Fix signs to make Q uniform
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s

# ---------- Spherical IFS Fractal Generator ----------

def make_spherical_ifs_fractal(
    d=8,
    n_points=10000,
    k=5,
    contractions=0.55,
    centers="random",     # "random" or array shape (k,d) on sphere
    rotations="random",   # "random" or None or list of (d-1)x(d-1) matrices
    burn_in=500,
    steps_per_sample=1,   # do multiple IFS steps between recorded samples
    noise=0.0,            # small tangent noise magnitude
    seed=1,
):
    """
    Generate a self-similar, fractal-like point cloud supported on S^{d-1}.

    IFS maps: f_i(p) = exp_{mu_i}( a_i * R_i( log_{mu_i}(p) ) )
    where:
      - mu_i are centers on the sphere (k of them),
      - 0 < a_i < 1 are contraction factors,
      - R_i are orthogonal rotations acting in the tangent space at mu_i.
    """
    assert d >= 3, "d must be >= 3 for a nontrivial sphere."
    rng = np.random.default_rng(seed)

    # Centers
    if isinstance(centers, str) and centers == "random":
        MU = _rand_unit_vectors(d, k, rng)  # shape (k,d)
    else:
        MU = np.asarray(centers, dtype=float)
        assert MU.shape == (k, d)
        MU = _normalize_rows(MU)

    # Contractions
    if np.isscalar(contractions):
        A = np.full(k, float(contractions))
    else:
        A = np.asarray(contractions, dtype=float)
        assert A.shape == (k,)
    assert np.all((A > 0) & (A < 1)), "All contraction factors must be in (0,1)."

    # Tangent bases and rotations per center
    bases = [ _tangent_basis(MU[i]) for i in range(k) ]  # each d x (d-1)
    m = d - 1
    if rotations == "random":
        Rm = [ random_orthogonal(m, rng) for _ in range(k) ]
    elif rotations is None:
        Rm = [ np.eye(m) for _ in range(k) ]
    else:
        Rm = [ np.asarray(M, dtype=float) for M in rotations ]
        assert len(Rm) == k and all(M.shape == (m, m) for M in Rm)

    # Start from a random point on the sphere
    x = _rand_unit_vectors(d, 1, rng)[0]
    pts = []

    # Pre-choose which map to apply at each step (uniform or weighted)
    idx_stream = rng.integers(0, k, size=burn_in + n_points * steps_per_sample)

    # Chaos game
    # Burn-in (throw away)
    for t in range(burn_in):
        i = int(idx_stream[t])
        mu = MU[i]
        U = bases[i]          # d x (d-1)
        R = Rm[i]             # (d-1) x (d-1)
        a = A[i]

        v = sphere_log(mu, x)           # in T_mu S^{d-1}, ambient coords
        # Project to coordinates in tangent basis, rotate, contract
        coords = U.T @ v                 # (d-1,)
        coords = a * (R @ coords)        # (d-1,)

        # Optional small tangent noise
        if noise > 0:
            coords = coords + rng.normal(scale=noise, size=coords.shape)

        v_new = U @ coords               # back to ambient tangent vector
        x = sphere_exp(mu, v_new)

    # Collect samples
    t0 = burn_in
    T = burn_in + n_points * steps_per_sample
    for t in range(t0, T):
        i = int(idx_stream[t])
        mu = MU[i]
        U = bases[i]
        R = Rm[i]
        a = A[i]

        v = sphere_log(mu, x)
        coords = U.T @ v
        coords = a * (R @ coords)
        if noise > 0:
            coords = coords + rng.normal(scale=noise, size=coords.shape)
        v_new = U @ coords
        x = sphere_exp(mu, v_new)

        # Record every steps_per_sample steps
        if (t - t0 + 1) % steps_per_sample == 0:
            pts.append(x.copy())

    X = np.vstack(pts)
    return _normalize_rows(X)  # shape (n_points, d)


def make_spherical_gasket(
    d=4, n_points=20000, seed=7, angle_spread=0.65, contraction=0.5, noise=0.01
):
    """
    A Sierpinski-gasket-like pattern on S^{d-1} using k=4 centers (tetrahedral-ish).
    """
    rng = np.random.default_rng(seed)
    # Four spread-out centers
    MU = simplex_centers(d, 4, rng)
    # Small random rotations in each tangent space
    m = d - 1
    Rm = []
    for _ in range(4):
        # Interpolate between I and a random orthogonal to control "twist"
        Q = random_orthogonal(m, rng)
        # Cayley-like blend: exp(t * skew) ≈ I + t*(Q-I) (simple linear blend for simplicity)
        Rm.append((1 - angle_spread) * np.eye(m) + angle_spread * Q)
    X = make_spherical_ifs_fractal(
        d=d,
        n_points=n_points,
        k=4,
        contractions=contraction,
        centers=MU,
        rotations=Rm,
        burn_in=1000,
        steps_per_sample=1,
        noise=noise,
        seed=seed,
    )
    return X

# ---------- Convenient presets ----------

def simplex_centers(d, k, rng):
    """
    Approximate 'spread-out' centers via random QR, then pick k rows.
    """
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    C = Q[:k, :]
    return _normalize_rows(C)


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
    X = make_spherical_gasket(d=4, n_points=10000, seed=7, angle_spread=0.7, contraction=0.55, noise=0.008)
    est = ParametricSphericalMDS(n_components=n_components, init="pca", target=target,
                                 max_iter=1000, lr=0.005, n_pairs=40000,
                                 batch_size=4000, random_state=seed, verbose=0)
    Y = est.fit_transform(X)
    print(f"[two_rings_global] target={target} k={n_components} "
          f"angle_RMSE(rad)={angle_rmse(X,Y):.4f}  kNN@10={knn_overlap(X,Y):.4f}  triplet_acc={triplet_accuracy(X,Y):.4f}")
    
    if n_components == 3:
        if t == 'sphere':
            fig = scatter_sphere3d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'gasket_S^2.png')
            else:
                fig.write_html(img_path / 'gasket_S^2.html')
        else:
            fig = scatter_euclid3d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'gasket_R^3.png')
            else:
                fig.write_html(img_path / 'gasket_R^3.html')
    else:
        if t == 'sphere':
            fig = scatter_sphere2d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'gasket_S^1.png')
            else:
                fig.write_html(img_path / 'gasket_S^1.html')
        else:
            fig = scatter_euclid2d(Y, backend='plotly')
            if isinstance(fig, plt.Figure):
                fig.savefig(img_path / 'gasket_R^2.png')
            else:
                fig.write_html(img_path / 'gasket_R^2.html')

if __name__ == "__main__":
    for t in ("sphere", "euclid"):
        for k in (2, 3):
            run(t, k, seed=0)
