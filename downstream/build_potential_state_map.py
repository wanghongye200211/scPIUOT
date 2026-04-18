#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RBFInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT


DEFAULT_TIME_COLORS = {
    "Day0": "#CDAA4A",
    "Day1": "#9CCB7A",
    "Day1.5": "#F1D7CC",
    "Day2": "#E76F51",
    "Day2.5": "#BDD3E8",
    "Day3": "#5FA8D3",
    "Day4": "#9CA6D8",
    "Day5": "#B794F4",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a public Figure 2-style potential landscape state map.")
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--id-column", default="obs_name")
    parser.add_argument("--labels-id-column", default="obs_name")
    parser.add_argument("--x-column", default="pca_1")
    parser.add_argument("--y-column", default="pca_2")
    parser.add_argument("--z-column", default="surface_z")
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--state-column", default=None)
    parser.add_argument("--time-colors-json", type=Path, default=None)
    parser.add_argument("--state-colors-json", type=Path, default=None)
    parser.add_argument("--state-short-json", type=Path, default=None)
    parser.add_argument("--elev", type=float, default=34.0)
    parser.add_argument("--azim", type=float, default=146.0)
    parser.add_argument("--box-z-aspect", type=float, default=0.65)
    parser.add_argument("--grid-size", type=int, default=140)
    parser.add_argument("--rbf-smoothing", type=float, default=0.04)
    parser.add_argument("--rbf-neighbors", type=int, default=128)
    parser.add_argument("--output-prefix", default="potential_landscape")
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_OUTPUT_ROOT / "figure2_potential_state_map")
    return parser


def load_json_dict(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def load_points(args) -> pd.DataFrame:
    df = pd.read_csv(args.points_csv.expanduser().resolve())
    if args.labels_csv is not None:
        labels = pd.read_csv(args.labels_csv.expanduser().resolve())
        df = df.merge(labels, left_on=args.id_column, right_on=args.labels_id_column, how="left")
    return df


def fit_surface(df: pd.DataFrame, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = df[[args.x_column, args.y_column]].to_numpy(dtype=np.float64)
    z = df[args.z_column].to_numpy(dtype=np.float64)

    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = maxs - mins
    mins = mins - 0.08 * span
    maxs = maxs + 0.08 * span

    gx = np.linspace(mins[0], maxs[0], int(args.grid_size))
    gy = np.linspace(mins[1], maxs[1], int(args.grid_size))
    xx, yy = np.meshgrid(gx, gy)

    epsilon = np.median(np.linalg.norm(xy - np.median(xy, axis=0), axis=1)) * 0.7
    rbf = RBFInterpolator(
        xy,
        z,
        kernel="gaussian",
        smoothing=float(args.rbf_smoothing),
        neighbors=int(args.rbf_neighbors),
        epsilon=max(float(epsilon), 1e-6),
    )
    zz = rbf(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz.astype(np.float64)


def setup_axes(fig: plt.Figure, args) -> plt.Axes:
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_box_aspect((1.25, 1.0, float(args.box_z_aspect)))
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))
    ax.set_facecolor("white")
    return ax


def apply_bounds(ax: plt.Axes, df: pd.DataFrame, args, *, z_base: float | None = None, z_top: float | None = None) -> None:
    x = df[args.x_column].to_numpy(dtype=np.float64)
    y = df[args.y_column].to_numpy(dtype=np.float64)
    z = df[args.z_column].to_numpy(dtype=np.float64)
    x_pad = 0.12 * (x.max() - x.min())
    y_pad = 0.12 * (y.max() - y.min())
    z_pad_low = 0.06 * (z.max() - z.min())
    z_pad_high = 0.16 * (z.max() - z.min())
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    if z_base is None:
        z_base = float(z.min() - z_pad_low)
    if z_top is None:
        z_top = float(z.max() + z_pad_high)
    ax.set_zlim(z_base, z_top)


def draw_scene(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, scatter_xyz: np.ndarray, scatter_colors, *, point_size: float, point_alpha: float) -> tuple[float, float, float]:
    finite = np.isfinite(zz)
    z_floor = float(np.nanmin(zz[finite]))
    z_ceil = float(np.nanmax(zz[finite]))
    span = max(z_ceil - z_floor, 1e-6)
    z_base = z_floor - 0.28 * span
    floor = np.full_like(xx, z_base)
    ax.plot_surface(xx, yy, floor, color="#fbfbfb", linewidth=0.0, antialiased=False, shade=False, alpha=1.0, zorder=-3)
    surface = np.ma.masked_invalid(zz)
    ax.plot_surface(xx, yy, surface, color="#f7f7f7", linewidth=0.0, antialiased=True, shade=False, alpha=0.045, zorder=1)
    ax.plot_wireframe(xx, yy, zz, rstride=1, cstride=1, color=(0.63, 0.63, 0.63, 0.92), linewidth=0.42, zorder=2)
    ax.scatter(
        scatter_xyz[:, 0],
        scatter_xyz[:, 1],
        scatter_xyz[:, 2],
        s=point_size,
        c=scatter_colors,
        alpha=point_alpha,
        depthshade=False,
        edgecolors="white",
        linewidths=0.28,
        zorder=4,
    )
    return z_base, z_floor, z_ceil


def scatter_colors_from_time(df: pd.DataFrame, time_column: str | None, time_colors: dict[str, str]) -> np.ndarray:
    if time_column is None or time_column not in df.columns:
        return np.asarray(["#5FA8D3"] * len(df), dtype=object)
    values = df[time_column].astype(str).tolist()
    return np.asarray([time_colors.get(v, "#5FA8D3") for v in values], dtype=object)


def state_groups(df: pd.DataFrame, state_column: str) -> dict[str, pd.DataFrame]:
    groups = {}
    for state in sorted(df[state_column].dropna().astype(str).unique().tolist()):
        groups[state] = df[df[state_column].astype(str) == state].copy()
    return groups


def auto_state_colors(states: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {state: matplotlib.colors.to_hex(cmap(idx % 10)) for idx, state in enumerate(states)}


def scatter_states(ax: plt.Axes, df: pd.DataFrame, args, state_column: str, state_colors: dict[str, str]) -> dict[str, dict[str, float]]:
    ax.scatter(
        df[args.x_column],
        df[args.y_column],
        df[args.z_column],
        s=8,
        c="#b9b9b9",
        alpha=0.18,
        depthshade=False,
        edgecolors="none",
        zorder=2,
    )
    centroids: dict[str, dict[str, float]] = {}
    for state, sub in state_groups(df, state_column).items():
        centroids[state] = {
            "x": float(sub[args.x_column].median()),
            "y": float(sub[args.y_column].median()),
            "z": float(sub[args.z_column].median()),
            "count": int(sub.shape[0]),
        }
        ax.scatter(
            sub[args.x_column],
            sub[args.y_column],
            sub[args.z_column],
            s=24,
            c=state_colors[state],
            alpha=0.95,
            depthshade=False,
            edgecolors="white",
            linewidths=0.45,
            zorder=4,
        )
    return centroids


def add_state_labels(ax: plt.Axes, centroids: dict[str, dict[str, float]], state_colors: dict[str, str], state_short: dict[str, str]) -> None:
    default_offsets = [
        (0.18, -0.16, 1.05),
        (-0.12, -0.20, 0.85),
        (-0.36, -0.05, 0.90),
        (0.18, 0.10, 0.95),
    ]
    for idx, (state, center) in enumerate(centroids.items()):
        dx, dy, dz = default_offsets[idx % len(default_offsets)]
        x0, y0, z0 = center["x"], center["y"], center["z"]
        x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=state_colors[state], linewidth=2.0, alpha=0.95, zorder=5)
        txt = ax.text(
            x1,
            y1,
            z1,
            state_short.get(state, state),
            color=state_colors[state],
            fontsize=15,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=3.2, foreground="white")])


def add_state_legend(ax: plt.Axes, state_colors: dict[str, str]) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.5, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8, label=label)
        for label, color in state_colors.items()
    ]
    legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.97), frameon=False, handletextpad=0.4, borderaxespad=0.0, fontsize=9.4)
    legend.set_zorder(10)


def save_clean(df: pd.DataFrame, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, args, out_path: Path, time_colors: dict[str, str]) -> None:
    fig = plt.figure(figsize=(6.9, 5.8), dpi=320)
    ax = setup_axes(fig, args)
    scatter_xyz = df[[args.x_column, args.y_column, args.z_column]].to_numpy(dtype=np.float64)
    colors = scatter_colors_from_time(df, args.time_column, time_colors)
    z_base, z_floor, z_ceil = draw_scene(ax, xx, yy, zz, scatter_xyz, colors, point_size=15, point_alpha=0.96)
    apply_bounds(ax, df, args, z_base=z_base, z_top=z_ceil + 0.14 * max(z_ceil - z_floor, 1e-6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def save_states(df: pd.DataFrame, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, args, out_path: Path, state_column: str, state_colors: dict[str, str], state_short: dict[str, str]) -> dict[str, dict[str, float]]:
    fig = plt.figure(figsize=(6.9, 5.8), dpi=320)
    ax = setup_axes(fig, args)
    background = np.full((df.shape[0], 4), (0.73, 0.73, 0.73, 0.18))
    z_base, z_floor, z_ceil = draw_scene(ax, xx, yy, zz, df[[args.x_column, args.y_column, args.z_column]].to_numpy(dtype=np.float64), background, point_size=8, point_alpha=0.18)
    centroids = scatter_states(ax, df, args, state_column, state_colors)
    add_state_labels(ax, centroids, state_colors, state_short)
    add_state_legend(ax, state_colors)
    apply_bounds(ax, df, args, z_base=z_base, z_top=z_ceil + 0.14 * max(z_ceil - z_floor, 1e-6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return centroids


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_points(args)
    xx, yy, zz = fit_surface(df, args)

    time_colors = DEFAULT_TIME_COLORS | load_json_dict(args.time_colors_json)
    state_colors = load_json_dict(args.state_colors_json)
    state_short = load_json_dict(args.state_short_json)

    clean_png = out_dir / f"{args.output_prefix}_clean.png"
    save_clean(df, xx, yy, zz, args, clean_png, time_colors)

    outputs = {"clean_png": str(clean_png)}
    state_counts = {}
    state_centroids = {}
    if args.state_column and args.state_column in df.columns:
        states = sorted(df[args.state_column].dropna().astype(str).unique().tolist())
        if not state_colors:
            state_colors = auto_state_colors(states)
        else:
            for state in states:
                state_colors.setdefault(state, auto_state_colors(states)[state])
        states_png = out_dir / f"{args.output_prefix}_states.png"
        state_centroids = save_states(df, xx, yy, zz, args, states_png, args.state_column, state_colors, state_short)
        outputs["states_png"] = str(states_png)
        state_counts = {state: int(count) for state, count in df[args.state_column].astype(str).value_counts().to_dict().items()}

    manifest = {
        "points_csv": str(args.points_csv.expanduser().resolve()),
        "labels_csv": str(args.labels_csv.expanduser().resolve()) if args.labels_csv else None,
        "columns": {
            "id": args.id_column,
            "x": args.x_column,
            "y": args.y_column,
            "z": args.z_column,
            "time": args.time_column,
            "state": args.state_column,
        },
        "view": {"elev": args.elev, "azim": args.azim},
        "outputs": outputs,
        "state_counts": state_counts,
        "state_centroids": state_centroids,
    }
    (out_dir / f"{args.output_prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
