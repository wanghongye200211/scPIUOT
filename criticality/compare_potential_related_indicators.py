from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DEFAULT_CONFIG_PATH
from yaml_config import (
    checkpoint_epoch_from_config,
    dataset_label_from_config,
    dataset_slug_from_config,
    device_from_config,
    load_yaml_config,
)

from compute_original_qreshape_mass_indicator import (
    EPS,
    _checkpoint_path,
    _extract_time_grid,
    _load_runtime_modules,
    _move_time_series_to_device,
    _resolve_run_dir,
    _subsample_initial,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple potential-related indicators for a PIUOT/MIOPISDE run."
    )
    parser.add_argument("--yaml-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-timepoints", type=int, default=201)
    parser.add_argument("--max-cells", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    return parser


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=np.float64) * np.asarray(weights, dtype=np.float64)))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mean = float(np.sum(w * x))
    var = float(np.sum(w * (x - mean) ** 2))
    return float(np.sqrt(max(var, 0.0)))


def safe_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / max(float(values[0]), EPS)


def dataset_meta(config: dict, yaml_cfg: dict, run_name: str) -> tuple[str, str]:
    label = dataset_label_from_config(yaml_cfg, fallback=run_name)
    slug = dataset_slug_from_config(yaml_cfg, fallback="dataset")
    embedding_key = str(config.get("embedding_key", "")).strip()
    display_label = label if not embedding_key or embedding_key == "X" else f"{label} ({embedding_key})"
    clean_slug = re.sub(r"[^0-9A-Za-z._-]+", "_", slug).strip("._") or "dataset"
    return clean_slug, display_label


def main() -> None:
    args = build_parser().parse_args()
    yaml_cfg = load_yaml_config(args.yaml_config)
    run_name = args.run_name or str(yaml_cfg["experiment"].get("run_name", "piuot_run"))
    run_dir = _resolve_run_dir(run_name, args.seed)
    checkpoint_path = _checkpoint_path(
        run_dir,
        args.checkpoint or checkpoint_epoch_from_config(yaml_cfg, fallback="auto"),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_mod, model_mod, train_mod = _load_runtime_modules(run_name)
    config = torch.load(run_dir / "config.pt", map_location="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset_slug, dataset_label = dataset_meta(config, yaml_cfg, run_name)

    output_dir = args.output_dir or (run_dir.parent.parent.parent / "figs" / dataset_slug / "potential_indicator_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or device_from_config(yaml_cfg, "analysis", "cpu"))
    x, y, _ = config_mod.load_data(type("Cfg", (), config))
    x = _move_time_series_to_device(x, device)

    model = model_mod.ForwardSDE(type("Cfg", (), config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    x0 = _subsample_initial(x[0].clone(), max_cells=int(args.max_cells), seed=int(args.seed))
    use_growth = bool(config.get("use_growth", False))
    clip_value = float(config.get("mass_clip_value", 30.0))
    x_r0 = train_mod.build_initial_state(x0, use_growth, clip_value=clip_value)

    dense_times = np.asarray(_extract_time_grid(y, args.n_timepoints), dtype=np.float64)
    traj_state = model(dense_times.tolist(), x_r0).detach()

    x_dim = int(config["x_dim"])
    hjb_coeff = float(config.get("hjb_growth_coeff", 2.0))
    alpha_g = float(config.get("action_alpha_g", 1.0))
    alpha_sigma = float(config.get("action_alpha_sigma", 1e-4))

    names = [
        "Q_potential_shift_mass",
        "Q_drift_reshape_mass",
        "Q_potential_spread_mass",
        "Q_drift_norm_mass",
        "Q_growth_flux_mass",
        "Q_growth_hetero_mass",
        "Q_hjb_residual_mass",
        "Action",
    ]
    metrics = {name: np.zeros(len(dense_times), dtype=np.float64) for name in names}

    for idx, t_val in enumerate(dense_times):
        state_t = traj_state[idx]
        x_t, _, logw_t = train_mod.unpack_state(state_t, x_dim, use_growth)
        if logw_t is not None:
            weights = train_mod.normalized_mass_from_logw(logw_t, clip_value=clip_value).detach().cpu().numpy()
        else:
            weights = np.full(x_t.shape[0], 1.0 / max(int(x_t.shape[0]), 1), dtype=np.float64)

        t0 = x_t.new_full((x_t.shape[0], 1), float(t_val))
        xt0 = torch.cat([x_t, t0], dim=1).requires_grad_(True)
        pot0 = model._func.net(xt0).squeeze(-1)
        grad0 = torch.autograd.grad(pot0, xt0, torch.ones_like(pot0), create_graph=True)[0]
        drift0 = -grad0[:, :-1]
        drift_t0 = grad0[:, -1]
        growth0 = model._func._growth(xt0).squeeze(-1) if use_growth else x_t.new_zeros((x_t.shape[0],))
        sigma0 = model._func.g(float(t_val), state_t)[:, :x_dim]

        metrics["Q_potential_spread_mass"][idx] = weighted_std(pot0.detach().cpu().numpy(), weights)
        metrics["Q_drift_norm_mass"][idx] = weighted_mean(torch.norm(drift0, dim=1).detach().cpu().numpy(), weights)
        growth_np = growth0.detach().cpu().numpy()
        growth_mean = weighted_mean(growth_np, weights)
        metrics["Q_growth_flux_mass"][idx] = weighted_mean(np.abs(growth_np), weights)
        metrics["Q_growth_hetero_mass"][idx] = float(
            np.sqrt(np.sum(weights * (growth_np - growth_mean) ** 2))
        )
        hjb_residual = torch.abs(drift_t0 - 0.5 * torch.sum(drift0.pow(2), dim=1) + 0.5 * hjb_coeff * growth0.pow(2))
        metrics["Q_hjb_residual_mass"][idx] = weighted_mean(hjb_residual.detach().cpu().numpy(), weights)
        action_term = (
            torch.sum(drift0.pow(2), dim=1)
            + alpha_g * growth0.pow(2)
            + alpha_sigma * torch.sum(sigma0.pow(2), dim=1)
        )
        metrics["Action"][idx] = weighted_mean(action_term.detach().cpu().numpy(), weights)

        if idx == len(dense_times) - 1:
            metrics["Q_potential_shift_mass"][idx] = metrics["Q_potential_shift_mass"][idx - 1] if idx > 0 else 0.0
            metrics["Q_drift_reshape_mass"][idx] = metrics["Q_drift_reshape_mass"][idx - 1] if idx > 0 else 0.0
            continue

        dt = max(float(dense_times[idx + 1] - t_val), EPS)
        t1 = x_t.new_full((x_t.shape[0], 1), float(dense_times[idx + 1]))
        xt1 = torch.cat([x_t, t1], dim=1).requires_grad_(True)
        pot1 = model._func.net(xt1).squeeze(-1)
        grad1 = torch.autograd.grad(pot1, xt1, torch.ones_like(pot1), create_graph=True)[0]
        drift1 = -grad1[:, :-1]

        potential_shift = torch.abs(pot1 - pot0).detach().cpu().numpy() / dt
        drift_reshape = torch.norm(drift1 - drift0, dim=1).detach().cpu().numpy() / dt
        metrics["Q_potential_shift_mass"][idx] = weighted_mean(potential_shift, weights)
        metrics["Q_drift_reshape_mass"][idx] = weighted_mean(drift_reshape, weights)

    metrics["Q_potential_mass_coupled"] = np.sqrt(
        safe_norm(metrics["Q_drift_reshape_mass"]) * safe_norm(metrics["Q_growth_hetero_mass"])
    )

    observed_times = np.asarray(y, dtype=np.float64)

    panel_names = names + ["Q_potential_mass_coupled"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), sharex=True)
    axes = axes.reshape(-1)
    colors = [
        "tab:blue",
        "seagreen",
        "purple",
        "teal",
        "slateblue",
        "mediumvioletred",
        "brown",
        "darkorange",
        "black",
    ]
    summary_rows = []
    for ax, name, color in zip(axes, panel_names, colors):
        values = metrics[name]
        peak_idx = int(np.nanargmax(values))
        peak_time = float(dense_times[peak_idx])
        peak_value = float(values[peak_idx])
        ax.plot(dense_times, values, lw=2.0, color=color)
        ax.scatter([peak_time], [peak_value], color="crimson", s=34, zorder=3)
        ax.annotate(
            f"peak {peak_time:.2f}",
            xy=(peak_time, peak_value),
            xytext=(6, 6),
            textcoords="offset points",
            color="crimson",
            fontsize=8,
        )
        for t_obs in observed_times:
            ax.axvline(float(t_obs), color="0.85", lw=0.9, ls="--", zorder=0)
        ax.set_title(name)
        ax.grid(alpha=0.22)
        summary_rows.append(
            {
                "indicator": name,
                "peak_time": peak_time,
                "peak_value": peak_value,
                "peak_norm": float(safe_norm(values)[peak_idx]),
            }
        )

    for ax in axes[-3:]:
        ax.set_xlabel("time")
    fig.suptitle(f"{dataset_label} | potential-related indicators for PIUOT run: {run_name}")
    fig.tight_layout()
    figure_path = output_dir / f"{dataset_slug}_potential_indicator_comparison.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 6), dpi=200)
    for name, color in zip(panel_names, colors):
        ax_overlay.plot(dense_times, safe_norm(metrics[name]), lw=2.0, color=color, label=name)
    for t_obs in observed_times:
        ax_overlay.axvline(float(t_obs), color="0.88", lw=0.8, ls="--", zorder=0)
    ax_overlay.set_title(f"{dataset_label} | normalized potential-related indicators for PIUOT run: {run_name}")
    ax_overlay.set_xlabel("time")
    ax_overlay.set_ylabel("normalized / first point")
    ax_overlay.grid(alpha=0.22)
    ax_overlay.legend(frameon=False, ncol=2)
    overlay_path = output_dir / f"{dataset_slug}_potential_indicator_normalized_overlay.png"
    fig_overlay.tight_layout()
    fig_overlay.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig_overlay)

    per_time = pd.DataFrame({"time": dense_times})
    for name in panel_names:
        per_time[name] = metrics[name]
        per_time[f"{name}_norm"] = safe_norm(metrics[name])
    per_time_path = output_dir / f"{dataset_slug}_potential_indicator_comparison_per_time.csv"
    summary_path = output_dir / f"{dataset_slug}_potential_indicator_comparison_summary.csv"
    per_time.to_csv(per_time_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Resolved run: {run_name}")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved comparison figure to: {figure_path}")
    print(f"Saved normalized overlay to: {overlay_path}")
    print(f"Saved per-time CSV to: {per_time_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
