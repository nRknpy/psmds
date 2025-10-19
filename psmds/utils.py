import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA


def l2_normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = norm(A, axis=1, keepdims=True)
    return A / np.clip(n, eps, None)


def check_unit_rows(X: np.ndarray, tol: float = 1e-4) -> None:
    # Warn if rows are not (approximately) unit norm
    lens = norm(X, axis=1)
    if not np.allclose(lens, 1.0, atol=tol):
        print("[Warning] Input rows do not appear to be unit vectors. They will be normalized for internal use.")


def random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def pca_orthobasis(X: np.ndarray, k: int) -> np.ndarray:
    if PCA is None:
        # Fall back to random if sklearn not available
        rng = np.random.default_rng(0)
        return random_orthonormal(X.shape[1], k, rng)
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=k, svd_solver="full")
    pca.fit(Xc)
    W = pca.components_.T  # (d x k)
    # Ensure orthonormal columns via QR (safety)
    W, _ = np.linalg.qr(W)
    return W[:, :k]


def angle_rmse(X: np.ndarray, Y: np.ndarray, sample_k: int = 20000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(X)
    if n < 2:
        return 0.0
    i = rng.integers(0, n, size=sample_k)
    j = rng.integers(0, n, size=sample_k)
    m = i != j
    i, j = i[m], j[m]
    cX = np.sum(X[i] * X[j], axis=1).clip(-1, 1)
    cY = np.sum(Y[i] * Y[j], axis=1).clip(-1, 1)
    aX = np.arccos(cX)
    aY = np.arccos(cY)
    return float(np.sqrt(np.mean((aX - aY) ** 2)))
