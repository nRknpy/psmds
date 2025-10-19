from typing import Optional, List, Tuple, Literal
import numpy as np
from numpy.linalg import norm

from .utils import *


class ParametricSphericalMDS:
    """
    Parametric Spherical MDS with a Stiefel-constrained linear map.

    Learns W ∈ R^{d×k} with orthonormal columns (W ∈ Stiefel(d,k)) to embed points X on S^{d−1}
    either onto the unit sphere S^{k−1} (“sphere” target) or into Euclidean space R^k (“euclid”
    target) while preserving the angular geometry of X.

    Targets and objectives
    ----------------------
    - target="sphere" (default):
        Map y_i = normalize(x_i^T W) ∈ S^{k−1} and minimize angular distortion
            minimize  Σ_{(i,j)∈P} [ arccos(<x_i, x_j>) − arccos(<y_i, y_j>) ]^2,
        where P is a sampled set of index pairs.

    - target="euclid":
        Map y_i = α · (x_i^T W) ∈ R^k (no per-row normalization) and preserve spherical chordal
        distances D_ij = 2 sin(½ arccos(<x_i, x_j>)) by minimizing
            minimize  Σ_{(i,j)∈P} [ α · ||(x_i−x_j)^T W||_2 − D_ij ]^2.
        The global scale α > 0 is updated in closed form each iteration as a least-squares fit.

    Optimization
    ------------
    Stochastic gradient descent on sampled pairs with a QR retraction to keep W on Stiefel(d,k).
    Mini-batches are used for scalability. When target="sphere", gradients back-propagate through
    the per-row normalization. When target="euclid", distances are taken directly in R^k.

    Parameters
    ----------
    n_components : int
        Target dimension k (>=2 for "sphere"; 1 <= k <= d for "euclid"). Typically 3.
    target : {"sphere", "euclid"}, default="sphere"
        Output space. "sphere" produces points on S^{k−1} by row normalization and minimizes
        angular error. "euclid" produces points in R^k without normalization and preserves
        chordal (Euclidean) distances; a global scale `scale_` is learned.
    init : {"pca", "random"}, default="pca"
        Initialization for W (top-k PCA axes or random orthonormal basis).
    max_iter : int, default=300
        Number of optimization iterations.
    lr : float, default=0.05
        Learning rate for gradient updates.
    n_pairs : int, default=40000
        Total number of distinct (i,j) pairs to pre-sample for the loss.
    batch_size : int or None, default=4000
        Mini-batch size of pairs per iteration (for SGD). If None, use all sampled pairs.
    random_state : int or None, default=None
        RNG seed.
    verbose : int, default=0
        0: silent; 1: print progress every 50 iterations (includes a sampled angle-RMSE).

    Attributes
    ----------
    components_ : ndarray of shape (d, k)
        Learned orthonormal basis W.
    embedding_ : ndarray of shape (n, k)
        Last computed embedding Y. On target="sphere", rows lie on S^{k−1}. On target="euclid",
        rows lie in R^k (not normalized).
    scale_ : float
        Learned global scale α used when target="euclid". Equal to 1.0 for target="sphere".
    losses_ : list of float
        Per-iteration mini-batch RMSE (radians for "sphere", chordal distance for "euclid").

    Notes
    -----
    - Inputs are internally L2-normalized per row (expected to be on S^{d−1}).
    - `score_angle_rmse` always evaluates angular RMSE after normalizing outputs to S^{k−1},
    so it is comparable across both targets.
    """
    def __init__(self,
                 n_components: int = 3,
                 target: Literal['sphere', 'euclid'] = 'sphere',
                 init: Literal['random', 'pca'] = 'pca',
                 max_iter: int = 300,
                 lr: float = 0.05,
                 n_pairs: int = 40000,
                 batch_size: Optional[int] = 4000,
                 random_state: Optional[int] = None,
                 verbose: int = 0) -> None:
        self.n_components = n_components
        self.target = target
        self.init = init
        self.max_iter = max_iter
        self.lr = lr
        self.n_pairs = n_pairs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted attributes
        self.components_: Optional[np.ndarray] = None
        self.embedding_: Optional[np.ndarray] = None
        self.losses_: List[float] = []
        self.scale_: float = 1.0 # only used for target='euclid'
    
    def fit(self, X: np.ndarray, y=None) -> "ParametricSphericalMDS":
        """
        Fit the estimator on data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Input row-wise unit vectors (points on S^{d-1}). Non-unit inputs are
            L2-normalized internally.
        y : Ignored

        Returns
        -------
        self : ParametricSphericalMDS
            Fitted estimator with learned `components_` and `embedding_`.
        """
        Y, W = self._optimize(X)
        self.components_ = W
        self.embedding_ = Y
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project new data using the learned mapping.

        Behavior
        --------
        target="sphere":
            Returns points on S^{k-1} via row normalization of X @ components_.
        target="euclid":
            Returns points in R^k without normalization as `scale_ * (X @ components_)`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Input row-wise vectors. They are L2-normalized internally.

        Returns
        -------
        Y : ndarray of shape (n_samples, k)
            Embedded points on S^{k-1} (sphere) or in R^k (euclid).

        Raises
        ------
        ValueError
            If the estimator is not fitted.
        """
        if self.components_ is None:
            raise ValueError("Estimator is not fitted. Call fit() first.")
        
        Xn = l2_normalize_rows(X)
        Z = Xn @ self.components_
        if self.target == 'sphere':
            return l2_normalize_rows(Z)
        elif self.target == 'euclid':
            return self.scale_ * Z
        else:
            raise ValueError("Unknown target")
    
    def fit_transform(self, X: np.ndarray, y=None):
        """
        Fit the estimator on X and return the embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Input row-wise unit vectors (normalized internally if needed).
        y : Ignored

        Returns
        -------
        Y : ndarray of shape (n_samples, k)
            Embedded points. On `target="sphere"`, rows lie on S^{k-1}; on
            `target="euclid"`, rows lie in R^k with no per-row normalization.
        """
        Y, W = self._optimize(X)
        self.components_ = W
        self.embedding_ = Y
        return Y
    
    def get_params(self, deep: bool = True):
        """Return constructor parameters for sklearn-style cloning."""
        return {
            "n_components": self.n_components,
            "init": self.init,
            "max_iter": self.max_iter,
            "lr": self.lr,
            "n_pairs": self.n_pairs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "target": self.target,
        }

    def set_params(self, **params):
        """Set constructor parameters and return self (sklearn-style)."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
    
    def _optimize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        Xn = l2_normalize_rows(X)
        check_unit_rows(X)
        n, d = Xn.shape
        k = self.n_components
        if k < 2 or k > d:
            raise ValueError("n_components must satisfy 2 <= k <= d")
        
        # Init W on Stiefel(d, k)
        if self.init == "pca":
            W = pca_orthobasis(Xn, k)
        elif self.init == "random":
            W = random_orthonormal(d, k, rng)
        else:
            raise ValueError("init must be 'pca' or 'random'")
        
        # Pre-sample pair indices for objective
        i = rng.integers(0, n, size=self.n_pairs)
        j = rng.integers(0, n, size=self.n_pairs)
        m = i != j
        i, j = i[m], j[m]
        if len(i) == 0:
            raise ValueError("Not enough distinct pairs; increase n_pairs or n")
        
        xi = Xn[i]  # (M, d)
        xj = Xn[j]
        cX = np.sum(xi * xj, axis=1).clip(-1, 1)
        aX = np.arccos(cX)  # target angles, shape (M,)
        
        self.losses_ = []
        
        # Mini-batch schedule
        M = len(i)
        batch = self.batch_size or M
        steps_per_epoch = max(1, M // batch)
        
        # scale for euclid target
        alpha = 1.0
        
        for it in range(self.max_iter):
            # Draw a mini-batch of pairs
            start = (it % steps_per_epoch) * batch
            end = min(start + batch, M)
            if end - start < batch:
                # Shuffle for a new epoch
                perm = rng.permutation(M)
                xi, xj, aX = xi[perm], xj[perm], aX[perm]
                start, end = 0, batch

            xib = xi[start:end]
            xjb = xj[start:end]
            aXb = aX[start:end]
            
            if self.target == 'sphere':
                Zi = xib @ W  # (B, k)
                Zj = xjb @ W
                ni = np.clip(norm(Zi, axis=1, keepdims=True), 1e-12, None)
                nj = np.clip(norm(Zj, axis=1, keepdims=True), 1e-12, None)
                yi = Zi / ni
                yj = Zj / nj

                cy = np.sum(yi * yj, axis=1).clip(-1, 1)
                aY = np.arccos(cy)
                diff = aY - aXb  # (B,)

                # Loss surrogate: RMSE over this batch (record only)
                batch_rmse = float(np.sqrt(np.mean(diff ** 2)))
                self.losses_.append(batch_rmse)

                # Gradient of arccos wrt cosine: d arccos(u) / du = -1/sqrt(1-u^2)
                w = -diff / np.sqrt(np.clip(1.0 - cy ** 2, 1e-12, None))  # (B,)

                # d loss / d yi and d yj via derivative of cosine
                dyi = (w[:, None] * (yj - cy[:, None] * yi))           # (B, k)
                dyj = (w[:, None] * (yi - cy[:, None] * yj))

                # Back through normalization yi = Zi / ||Zi||
                dZi = (dyi / ni) - ((np.sum(dyi * Zi, axis=1, keepdims=True) / (ni ** 3)) * Zi)
                dZj = (dyj / nj) - ((np.sum(dyj * Zj, axis=1, keepdims=True) / (nj ** 3)) * Zj)

                # Accumulate gradient on W
                G = xib.T @ dZi + xjb.T @ dZj  # (d, k)

                # Gradient step + QR retraction to Stiefel(d, k)
                W = W - self.lr * G
                W, _ = np.linalg.qr(W)
            elif self.target == 'euclid':
                # chordal distance preservation in R^k
                Zi = xib @ W
                Zj = xjb @ W
                # target chordal distances D = 2 sin(theta/2)
                D = 2.0 * np.sin(aXb / 2.0)
                diff_vec = Zi - Zj
                dist = np.sqrt(np.sum(diff_vec**2, axis=1) + 1e-12)
                # optimal alpha (per-batch) for LS on distances
                denom = float(np.sum(dist**2) + 1e-12)
                alpha = float(np.sum(dist * D) / denom)
                e = alpha * dist - D
                batch_rmse = float(np.sqrt(np.mean(e**2)))
                self.losses_.append(batch_rmse)
                # gradient wrt Zi, Zj
                scale = (e / (dist + 1e-12)) * alpha
                dZi = (scale[:,None] * (Zi - Zj))
                dZj = (scale[:,None] * (Zj - Zi))
                G = xib.T @ dZi + xjb.T @ dZj
                W = W - self.lr * G
                W, _ = np.linalg.qr(W)
            else:
                raise ValueError("Unknown target")
            
            if self.verbose and ((it + 1) % 50 == 0 or it == 0):
                if self.target == "sphere":
                    Y_tmp = l2_normalize_rows(Xn @ W)
                else:
                    Y_tmp = alpha * (Xn @ W)
                rmse = angle_rmse(Xn, l2_normalize_rows(Y_tmp), sample_k=min(20000, n * 20), seed=it)
                print(f"[iter {it+1:4d}] batch RMSE={batch_rmse:.4f}  sample angle-RMSE={rmse:.4f}")
        
        # Final embedding
        if self.target == "sphere":
            Y = l2_normalize_rows(Xn @ W)
        else:
            self.scale_ = float(alpha)
            Y = self.scale_ * (Xn @ W)
        return Y, W
