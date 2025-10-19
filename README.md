# psmds: Parametric Spherical MDS

A lightweight, scikit-learn–style library for **parametric spherical multidimensional scaling**.  
It learns a linear map with **Stiefel constraint** to embed points from $\mathbb{S}^{d-1}$ into:

- **Sphere** $\mathbb{S}^{k-1}$: preserve **angles** (geodesic structure).
- **Euclidean** $\mathbb{R}^k$: preserve **spherical chordal distances**.

---

## Features

- **Single estimator**: `ParametricSphericalMDS(n_components, target=...)`
- Targets:
  - `target="sphere"` → $\mathbb{S}^{k-1}$, minimizes angular distortion
  - `target="euclid"` → $\mathbb{R}^k$, matches chordal distances with a learned global scale $\alpha$
- **Stiefel-constrained** linear map with **QR retraction**
- **Pair sampling** + mini-batch **SGD**
- Ready-to-use **example scripts** (`examples/`) and **plotting helpers** (Matplotlib / Plotly)

---

## Installation

~~~bash
pip install -e .
~~~

---

## Quickstart

~~~python
import numpy as np
from psmds import ParametricSphericalMDS

# Example: random unit vectors on S^{7}
rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 8))
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# (A) Sphere target: S^2 visualization (angle preserving)
model_sph = ParametricSphericalMDS(n_components=3, target="sphere",
                                   init="pca", max_iter=300, lr=0.05,
                                   n_pairs=40000, batch_size=4000,
                                   random_state=0, verbose=1)
Y_sph = model_sph.fit_transform(X)   # Y on S^2

# (B) Euclidean target: R^3 (chordal distance preserving)
model_euc = ParametricSphericalMDS(n_components=3, target="euclid",
                                   init="pca", max_iter=300, lr=0.05,
                                   n_pairs=40000, batch_size=4000,
                                   random_state=0, verbose=1)
Y_euc = model_euc.fit_transform(X)   # Y in R^3 (no row normalization)
print("alpha (scale):", model_euc.scale_)
~~~

---

## Plotting (Matplotlib / Plotly)

The library exposes both **functional** plotting helpers and **backend-agnostic dispatchers**.

~~~python
from psmds import (
    plot_sphere3d, plot_euclid3d,         # Matplotlib helpers (return fig, ax)
    plotly_sphere3d, plotly_euclid3d,     # Plotly helpers (return go.Figure)
    scatter_sphere3d, scatter_euclid3d    # Dispatchers: backend="matplotlib"|"plotly"
)

# Matplotlib
fig, ax = plot_sphere3d(Y_sph, normalize=True, title="S^2 (angles)")

# Plotly (interactive)
fig = plotly_euclid3d(Y_euc, title="R^3 (chordal)")

# Or via dispatchers
fig = scatter_sphere3d(Y_sph, backend="plotly", title="S^2 interactive")
~~~

> The sphere plots draw a unit circle/sphere as a reference surface when applicable.

---

## Example scripts (`examples/`)

Each script builds a **self-contained** synthetic dataset and evaluates both targets without visualization:

- `ring_local.py` — **local** structure: noisy geodesic ring on $\mathbb{S}^{d-1}$
- `two_rings_global.py` — **global** structure: two parallel rings with angular gap
- `hier_caps_local_global.py` — **hierarchical** caps (local + global)
- `sph_graph_global.py` — **geodesic network** (graph of great-circle segments)

Run:

~~~bash
python examples/ring_local.py
python examples/two_rings_global.py
python examples/hier_caps_local_global.py
python examples/sph_graph_global.py
~~~

Each script prints:
- Angle RMSE (radians; always evaluated after row-normalizing outputs)
- k-NN overlap@10
- Triplet accuracy

---

## API

### `ParametricSphericalMDS`

~~~python
ParametricSphericalMDS(
    n_components: int = 3,
    init: {"pca","random"} = "pca",
    max_iter: int = 300,
    lr: float = 0.05,
    n_pairs: int = 40000,
    batch_size: int | None = 4000,
    random_state: int | None = None,
    verbose: int = 0,
    target: {"sphere","euclid"} = "sphere",
    eval_sample_k: int = 20000,
)
~~~

- **`fit(X)`**: learns `components_` and stores `embedding_`
- **`transform(X)`**:
  - `sphere`: returns points on $\mathbb{S}^{k-1}$ (row-normalized)
  - `euclid`: returns points in $\mathbb{R}^k$ (no normalization), scaled by learned `scale_`
- **`fit_transform(X)`**: convenience

**Attributes**
- `components_` $(d,k)$: learned orthonormal basis $W$
- `embedding_` $(n,k)$: last embedding
- `scale_` (float): learned global scale $\alpha$ for `target="euclid"`
- `losses_` (list[float]): mini-batch RMSE history

---

## How it works

- **Objective (sphere)**  
  Minimize $\sum ( \arccos\langle x_i,x_j\rangle - \arccos\langle y_i,y_j\rangle )^2$  
  with $y_i=\mathrm{normalize}(x_i^\top W)$.
- **Objective (euclid)**  
  Minimize $\sum (\alpha\| (x_i-x_j)^\top W \|_2 - D_{ij})^2$,  
  where $D_{ij}=2\sin(\tfrac{1}{2}\arccos\langle x_i,x_j\rangle)$ (chordal distance). $\alpha$ is updated in closed form.
- **Optimization**  
  Mini-batch SGD on sampled pairs; **QR retraction** to keep $W\in\mathrm{St}(d,k)$.

---

## Tips

- Inputs should be (approximately) **unit-normalized rows** (the estimator also normalizes internally).
- Start with `init="pca"` for stability.
- Larger `n_pairs` improves fidelity; adjust `batch_size` for throughput.
- Use `target="sphere"` for spherical plots / angle-based neighborhoods;  
  `target="euclid"` for Euclidean workflows that should reflect spherical distances.
