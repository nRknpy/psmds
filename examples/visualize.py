import numpy as np
from typing import Literal

_Backend = Literal["matplotlib", "mpl", "plotly", "ply"]


def _normalize_backend(backend: _Backend) -> str:
    b = backend.lower()
    if b in ("matplotlib", "mpl"):
        return "mpl"
    if b in ("plotly", "ply"):
        return "plotly"
    raise ValueError("backend must be 'matplotlib'/'mpl' or 'plotly'/'ply'")


def scatter_sphere2d(Y, *, backend: _Backend = "matplotlib", **kwargs):
    b = _normalize_backend(backend)
    if b == "mpl":
        # Expects existing matplotlib impl: plot_sphere2d(...)
        return plot_sphere2d(Y, **kwargs)
    else:
        # Expects existing plotly impl: plotly_sphere2d(...)
        return plotly_sphere2d(Y, **kwargs)


def scatter_sphere3d(Y, *, backend: _Backend = "matplotlib", **kwargs):
    b = _normalize_backend(backend)
    if b == "mpl":
        return plot_sphere3d(Y, **kwargs)
    else:
        return plotly_sphere3d(Y, **kwargs)


def scatter_euclid2d(Y, *, backend: _Backend = "matplotlib", **kwargs):
    b = _normalize_backend(backend)
    if b == "mpl":
        return plot_euclid2d(Y, **kwargs)
    else:
        return plotly_euclid2d(Y, **kwargs)


def scatter_euclid3d(Y, *, backend: _Backend = "matplotlib", **kwargs):
    b = _normalize_backend(backend)
    if b == "mpl":
        return plot_euclid3d(Y, **kwargs)
    else:
        return plotly_euclid3d(Y, **kwargs)


def plot_sphere2d(Y: np.ndarray, normalize: bool = True, s: int = 3, title: str | None = None):
    """Scatter points on S^1 in R^2. Optionally normalize rows."""
    import matplotlib.pyplot as plt
    Y2 = Y[:, :2]
    if normalize:
        n = np.linalg.norm(Y2, axis=1, keepdims=True)
        Y2 = Y2 / np.clip(n, 1e-12, None)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Y2[:, 0], Y2[:, 1], s=s)
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), linewidth=0.8, alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return fig


def plot_sphere3d(Y: np.ndarray, normalize: bool = True, s: int = 3, elev: int = 20, azim: int = 35, title: str | None = None):
    """Scatter points on S^2 in R^3 with a unit-sphere wireframe."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    Y3 = Y[:, :3]
    if normalize:
        n = np.linalg.norm(Y3, axis=1, keepdims=True)
        Y3 = Y3 / np.clip(n, 1e-12, None)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Y3[:, 0], Y3[:, 1], Y3[:, 2], s=s)
    # unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.5)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title:
        ax.set_title(title)
    return fig


def plot_euclid2d(Y: np.ndarray, s: int = 3, title: str | None = None):
    """Scatter in R^2 (no normalization)."""
    import matplotlib.pyplot as plt
    Y2 = Y[:, :2]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Y2[:, 0], Y2[:, 1], s=s)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return fig


def plot_euclid3d(Y: np.ndarray, s: int = 3, elev: int = 20, azim: int = 35, title: str | None = None):
    """Scatter in R^3 (no normalization)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    Y3 = Y[:, :3]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Y3[:, 0], Y3[:, 1], Y3[:, 2], s=s)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title:
        ax.set_title(title)
    return fig


def plotly_sphere2d(Y, normalize=True, marker_size=3, opacity=0.9, title=None):
    """Interactive scatter on S^1 using Plotly. Optionally normalizes rows and draws unit circle."""
    import numpy as np
    import plotly.graph_objects as go

    Y = np.asarray(Y)
    Y2 = Y[:, :2]
    if normalize:
        n = np.linalg.norm(Y2, axis=1, keepdims=True)
        Y2 = Y2 / np.clip(n, 1e-12, None)

    th = np.linspace(0, 2*np.pi, 400)
    circle_x, circle_y = np.cos(th), np.sin(th)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode="lines",
                             line=dict(width=1), opacity=0.6, name="unit circle"))
    fig.add_trace(go.Scatter(x=Y2[:, 0], y=Y2[:, 1], mode="markers",
                             marker=dict(size=marker_size), opacity=opacity, name="points"))

    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.1)
    )
    return fig


def plotly_sphere3d(Y, normalize=True, marker_size=3, opacity=0.9, title=None,
                    sphere_resolution=48, sphere_opacity=0.2):
    """Interactive scatter on S^2 using Plotly. Draws a unit-sphere surface."""
    import numpy as np
    import plotly.graph_objects as go

    Y = np.asarray(Y)
    Y3 = Y[:, :3]
    if normalize:
        n = np.linalg.norm(Y3, axis=1, keepdims=True)
        Y3 = Y3 / np.clip(n, 1e-12, None)

    u = np.linspace(0, 2*np.pi, sphere_resolution)
    v = np.linspace(0, np.pi, sphere_resolution//2)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, showscale=False, opacity=sphere_opacity,
                             colorscale="Greys", name="unit sphere"))
    fig.add_trace(go.Scatter3d(x=Y3[:, 0], y=Y3[:, 1], z=Y3[:, 2],
                               mode="markers",
                               marker=dict(size=marker_size),
                               opacity=opacity,
                               name="points"))
    fig.update_scenes(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        aspectmode="cube"
    )
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.05))
    return fig


def plotly_euclid2d(Y, marker_size=3, opacity=0.9, title=None):
    """Interactive scatter in R^2 using Plotly (no normalization)."""
    import numpy as np
    import plotly.graph_objects as go

    Y = np.asarray(Y)
    Y2 = Y[:, :2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Y2[:, 0], y=Y2[:, 1], mode="markers",
                             marker=dict(size=marker_size), opacity=opacity, name="points"))
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.1)
    )
    return fig


def plotly_euclid3d(Y, marker_size=3, opacity=0.9, title=None):
    """Interactive scatter in R^3 using Plotly (no normalization)."""
    import numpy as np
    import plotly.graph_objects as go

    Y = np.asarray(Y)
    Y3 = Y[:, :3]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=Y3[:, 0], y=Y3[:, 1], z=Y3[:, 2],
                               mode="markers",
                               marker=dict(size=marker_size),
                               opacity=opacity,
                               name="points"))
    fig.update_scenes(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        aspectmode="cube"
    )
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.05))
    return fig
