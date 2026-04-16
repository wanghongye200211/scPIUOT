from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DEFAULT_CONFIG_PATH, PIUOT_ROOT
from yaml_config import (
    checkpoint_epoch_from_config,
    dataset_label_from_config,
    dataset_slug_from_config,
    device_from_config,
    load_yaml_config,
)


METHOD_ROOT = PIUOT_ROOT
OUTPUT_ROOT = METHOD_ROOT / "output"
EPS = 1e-8

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the original mass-weighted Q_reshape indicator for a PIUOT/MIOPISDE run."
    )
    parser.add_argument("--yaml-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name under piuot/output. If omitted, use experiment.run_name from YAML.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="best, auto, or explicit epoch tag like epoch_000500")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-timepoints", type=int, default=401)
    parser.add_argument("--max-cells", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _resolve_src_package(run_name: str) -> str:
    return "src_mps_druot_ablation_suite"


def _load_runtime_modules(run_name: str):
    src_pkg = importlib.import_module(_resolve_src_package(run_name))
    sys.modules["src"] = src_pkg
    config_mod = importlib.import_module("src.config_model")
    model_mod = importlib.import_module("src.model")
    train_mod = importlib.import_module("src.train")
    return config_mod, model_mod, train_mod


def _resolve_run_dir(run_name: str, seed: int) -> Path:
    matches = sorted((OUTPUT_ROOT / run_name).glob(f"*/seed_{seed}/alltime"))
    if not matches:
        raise FileNotFoundError(f"Could not find run '{run_name}' for seed {seed}.")
    return matches[-1]


def _best_epoch_from_eval(run_dir: Path) -> str:
    eval_path = run_dir / "interpolate-mioemd2.log"
    eval_df = pd.read_csv(eval_path, sep="\t")
    mean_eval = eval_df.groupby("epoch", as_index=False)["loss"].mean()
    return str(mean_eval.loc[mean_eval["loss"].idxmin(), "epoch"])


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "best":
        return run_dir / "train.best.pt"
    if checkpoint == "auto":
        epoch_tag = _best_epoch_from_eval(run_dir)
        return run_dir / f"train.{epoch_tag}.pt"
    return run_dir / f"train.{checkpoint}.pt"


def _move_time_series_to_device(x, device: torch.device):
    return [x_i.to(device) if x_i.device != device else x_i for x_i in x]


def _subsample_initial(x0: torch.Tensor, max_cells: int, seed: int) -> torch.Tensor:
    if max_cells <= 0 or x0.shape[0] <= max_cells:
        return x0
    gen_device = x0.device if x0.device.type != "mps" else torch.device("cpu")
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(seed)
    perm = torch.randperm(x0.shape[0], generator=gen, device=gen_device)[:max_cells]
    return x0.index_select(0, perm.to(x0.device))


def _extract_time_grid(y, n_timepoints: int) -> list[np.float64]:
    return np.linspace(float(y[0]), float(y[-1]), int(n_timepoints)).astype(np.float64).tolist()


def _dataset_meta(config, yaml_cfg: dict, run_name: str) -> tuple[str, str]:
    label = dataset_label_from_config(yaml_cfg, fallback=run_name)
    slug = dataset_slug_from_config(yaml_cfg, fallback="dataset")
    embedding_key = str(getattr(config, "embedding_key", "")).strip()
    display_label = label if not embedding_key or embedding_key == "X" else f"{label} ({embedding_key})"
    clean_slug = re.sub(r"[^0-9A-Za-z._-]+", "_", slug).strip("._") or "dataset"
    return clean_slug, display_label


def _compute_mass_weighted_qreshape(
    model,
    train_mod,
    config,
    dense_times: np.ndarray,
    traj_state: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(dense_times), dtype=np.float64)
    x_dim = int(config.x_dim)
    use_growth = bool(getattr(config, "use_growth", False))
    clip_value = float(getattr(config, "mass_clip_value", 30.0))

    for idx, t_val in enumerate(dense_times):
        if idx == len(dense_times) - 1:
            values[idx] = values[idx - 1] if idx > 0 else 0.0
            continue

        state_t = traj_state[idx]
        x_t, _, logw_t = train_mod.unpack_state(state_t, x_dim, use_growth)
        if logw_t is not None:
            weights = train_mod.normalized_mass_from_logw(logw_t, clip_value=clip_value).detach().cpu().numpy()
        else:
            weights = np.full(x_t.shape[0], 1.0 / max(int(x_t.shape[0]), 1), dtype=np.float64)

        dt = max(float(dense_times[idx + 1] - t_val), EPS)
        t0 = x_t.new_full((x_t.shape[0], 1), float(t_val))
        t1 = x_t.new_full((x_t.shape[0], 1), float(dense_times[idx + 1]))
        xt0 = torch.cat([x_t, t0], dim=1)
        xt1 = torch.cat([x_t, t1], dim=1)

        with torch.no_grad():
            potential_0 = -model._func.net(xt0).squeeze(-1)
            potential_1 = -model._func.net(xt1).squeeze(-1)

        diff = torch.abs(potential_1 - potential_0).detach().cpu().numpy()
        values[idx] = float(np.sum(weights * diff) / dt)

    normalized = values / max(float(values[0]), EPS)
    return values, normalized


def _compute_mass_weighted_drift_reshape(
    model,
    train_mod,
    config,
    dense_times: np.ndarray,
    traj_state: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(dense_times), dtype=np.float64)
    x_dim = int(config.x_dim)
    use_growth = bool(getattr(config, "use_growth", False))
    clip_value = float(getattr(config, "mass_clip_value", 30.0))

    for idx, t_val in enumerate(dense_times):
        if idx == len(dense_times) - 1:
            values[idx] = values[idx - 1] if idx > 0 else 0.0
            continue

        state_t = traj_state[idx]
        x_t, _, logw_t = train_mod.unpack_state(state_t, x_dim, use_growth)
        if logw_t is not None:
            weights = train_mod.normalized_mass_from_logw(logw_t, clip_value=clip_value).detach().cpu().numpy()
        else:
            weights = np.full(x_t.shape[0], 1.0 / max(int(x_t.shape[0]), 1), dtype=np.float64)

        dt = max(float(dense_times[idx + 1] - t_val), EPS)
        t0 = x_t.new_full((x_t.shape[0], 1), float(t_val))
        t1 = x_t.new_full((x_t.shape[0], 1), float(dense_times[idx + 1]))
        xt0 = torch.cat([x_t, t0], dim=1).requires_grad_(True)
        xt1 = torch.cat([x_t, t1], dim=1).requires_grad_(True)

        drift_0 = model._func._drift(xt0)
        drift_1 = model._func._drift(xt1)
        diff = torch.norm(drift_1 - drift_0, dim=1).detach().cpu().numpy()
        values[idx] = float(np.sum(weights * diff) / dt)

    normalized = values / max(float(values[0]), EPS)
    return values, normalized


def _compute_action_curve(
    model,
    train_mod,
    config,
    dense_times: np.ndarray,
    traj_state: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(dense_times), dtype=np.float64)
    x_dim = int(config.x_dim)
    use_growth = bool(getattr(config, "use_growth", False))
    clip_value = float(getattr(config, "mass_clip_value", 30.0))
    alpha_g = float(getattr(config, "action_alpha_g", 1.0))
    alpha_sigma = float(getattr(config, "action_alpha_sigma", 1e-4))

    for idx, t_val in enumerate(dense_times):
        state_t = traj_state[idx]
        z_t, _, logw_t = train_mod.unpack_state(state_t, x_dim, use_growth)
        t_batch = z_t.new_full((z_t.shape[0], 1), float(t_val))
        xt = torch.cat([z_t, t_batch], dim=1).requires_grad_(True)

        drift_t = model._func._drift(xt)
        with torch.no_grad():
            growth_t = (
                model._func._growth(xt).squeeze(-1)
                if use_growth else z_t.new_zeros((z_t.shape[0],))
            )
            sigma_diag = model._func.g(float(t_val), state_t)[:, :x_dim]

        if logw_t is not None:
            weights = train_mod.normalized_mass_from_logw(logw_t, clip_value=clip_value).detach().cpu().numpy()
        else:
            weights = np.full(z_t.shape[0], 1.0 / max(int(z_t.shape[0]), 1), dtype=np.float64)

        transport = torch.sum(drift_t.pow(2), dim=-1).detach().cpu().numpy()
        growth = growth_t.pow(2).detach().cpu().numpy()
        sigma = torch.sum(sigma_diag.pow(2), dim=-1).detach().cpu().numpy()
        term = transport + alpha_g * growth + alpha_sigma * sigma
        values[idx] = float(np.sum(weights * term))

    normalized = values / max(float(values[0]), EPS)
    return values, normalized


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
    config = SimpleNamespace(**torch.load(run_dir / "config.pt", map_location="cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset_slug, dataset_label = _dataset_meta(config, yaml_cfg, run_name)

    output_dir = args.output_dir or (OUTPUT_ROOT / "figs" / dataset_slug / "criticality_original")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or device_from_config(yaml_cfg, "analysis", "cpu"))
    x, y, _ = config_mod.load_data(config)
    x = _move_time_series_to_device(x, device)

    model = model_mod.ForwardSDE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    use_growth = bool(getattr(config, "use_growth", False))
    clip_value = float(getattr(config, "mass_clip_value", 30.0))
    x0 = _subsample_initial(x[0].clone(), max_cells=int(args.max_cells), seed=int(args.seed))
    x_r0 = train_mod.build_initial_state(x0, use_growth, clip_value=clip_value)

    dense_times = _extract_time_grid(y, args.n_timepoints)
    traj_state = model(dense_times, x_r0).detach()
    dense_times_np = np.asarray(dense_times, dtype=np.float64)
    q_raw, q_norm = _compute_mass_weighted_qreshape(
        model=model,
        train_mod=train_mod,
        config=config,
        dense_times=dense_times_np,
        traj_state=traj_state,
    )
    drift_raw, drift_norm = _compute_mass_weighted_drift_reshape(
        model=model,
        train_mod=train_mod,
        config=config,
        dense_times=dense_times_np,
        traj_state=traj_state,
    )
    action_raw, action_norm = _compute_action_curve(
        model=model,
        train_mod=train_mod,
        config=config,
        dense_times=dense_times_np,
        traj_state=traj_state,
    )

    peak_idx = int(np.nanargmax(q_raw))
    peak_time = float(dense_times_np[peak_idx])
    peak_value = float(q_raw[peak_idx])
    drift_peak_idx = int(np.nanargmax(drift_raw))
    drift_peak_time = float(dense_times_np[drift_peak_idx])
    drift_peak_value = float(drift_raw[drift_peak_idx])
    action_peak_idx = int(np.nanargmax(action_raw))
    action_peak_time = float(dense_times_np[action_peak_idx])
    action_peak_value = float(action_raw[action_peak_idx])
    observed_times = np.asarray(y, dtype=np.float64)
    fig, axes = plt.subplots(4, 1, figsize=(11, 13.0), sharex=True)
    for ax in axes:
        for t_obs in observed_times:
            ax.axvline(float(t_obs), color="0.84", lw=0.9, ls="--", zorder=0)
        ax.grid(alpha=0.22)

    axes[0].plot(dense_times_np, q_raw, lw=2.2, color="tab:blue")
    axes[0].scatter([peak_time], [peak_value], color="tab:red", s=42, zorder=3)
    axes[0].annotate(
        f"peak {peak_time:.2f}",
        xy=(peak_time, peak_value),
        xytext=(8, 8),
        textcoords="offset points",
        color="tab:red",
        fontsize=9,
    )
    axes[0].set_ylabel("Q_reshape_mass")
    axes[0].set_title(f"{dataset_label} | original Q_reshape^mass from PIUOT run: {run_name}")

    axes[1].plot(dense_times_np, q_norm, lw=2.0, color="royalblue")
    axes[1].scatter([peak_time], [float(q_norm[peak_idx])], color="navy", s=38, zorder=3)
    axes[1].set_ylabel("Q_reshape_mass / Q0")
    axes[2].plot(dense_times_np, drift_raw, lw=2.0, color="seagreen")
    axes[2].scatter([drift_peak_time], [drift_peak_value], color="darkgreen", s=40, zorder=3)
    axes[2].annotate(
        f"drift peak {drift_peak_time:.2f}",
        xy=(drift_peak_time, drift_peak_value),
        xytext=(8, 8),
        textcoords="offset points",
        color="darkgreen",
        fontsize=9,
    )
    axes[2].set_ylabel("Q_drift_reshape_mass")
    axes[3].plot(dense_times_np, action_raw, lw=2.0, color="darkorange")
    axes[3].scatter([action_peak_time], [action_peak_value], color="orangered", s=40, zorder=3)
    axes[3].annotate(
        f"action peak {action_peak_time:.2f}",
        xy=(action_peak_time, action_peak_value),
        xytext=(8, 8),
        textcoords="offset points",
        color="orangered",
        fontsize=9,
    )
    axes[3].set_ylabel("Action")
    axes[3].set_xlabel("time")

    fig.tight_layout()

    figure_path = output_dir / f"{dataset_slug}_qreshape_mass_original_indicator.png"
    action_figure_path = output_dir / f"{dataset_slug}_action_original_indicator.png"
    drift_figure_path = output_dir / f"{dataset_slug}_qdrift_reshape_mass_indicator.png"
    curve_path = output_dir / f"{dataset_slug}_qreshape_mass_original_indicator_per_time.csv"
    summary_path = output_dir / f"{dataset_slug}_qreshape_mass_original_indicator_summary.csv"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_action, ax_action = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax in ax_action:
        for t_obs in observed_times:
            ax.axvline(float(t_obs), color="0.84", lw=0.9, ls="--", zorder=0)
        ax.grid(alpha=0.22)
    ax_action[0].plot(dense_times_np, action_raw, lw=2.1, color="darkorange")
    ax_action[0].scatter([action_peak_time], [action_peak_value], color="orangered", s=40, zorder=3)
    ax_action[0].set_ylabel("Action")
    ax_action[0].set_title(f"{dataset_label} | action from PIUOT run: {run_name}")
    ax_action[1].plot(dense_times_np, action_norm, lw=2.0, color="peru")
    ax_action[1].scatter([action_peak_time], [float(action_norm[action_peak_idx])], color="saddlebrown", s=38, zorder=3)
    ax_action[1].set_ylabel("Action / A0")
    ax_action[1].set_xlabel("time")
    fig_action.tight_layout()
    fig_action.savefig(action_figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig_action)

    fig_drift, ax_drift = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax in ax_drift:
        for t_obs in observed_times:
            ax.axvline(float(t_obs), color="0.84", lw=0.9, ls="--", zorder=0)
        ax.grid(alpha=0.22)
    ax_drift[0].plot(dense_times_np, drift_raw, lw=2.1, color="seagreen")
    ax_drift[0].scatter([drift_peak_time], [drift_peak_value], color="darkgreen", s=40, zorder=3)
    ax_drift[0].set_ylabel("Q_drift_reshape_mass")
    ax_drift[0].set_title(f"{dataset_label} | drift-field reshape from PIUOT run: {run_name}")
    ax_drift[1].plot(dense_times_np, drift_norm, lw=2.0, color="mediumseagreen")
    ax_drift[1].scatter([drift_peak_time], [float(drift_norm[drift_peak_idx])], color="forestgreen", s=38, zorder=3)
    ax_drift[1].set_ylabel("Q_drift_reshape_mass / D0")
    ax_drift[1].set_xlabel("time")
    fig_drift.tight_layout()
    fig_drift.savefig(drift_figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig_drift)

    pd.DataFrame(
        {
            "time": dense_times_np,
            "Q_reshape_mass": q_raw,
            "Q_reshape_mass_norm": q_norm,
            "Q_drift_reshape_mass": drift_raw,
            "Q_drift_reshape_mass_norm": drift_norm,
            "action": action_raw,
            "action_norm": action_norm,
        }
    ).to_csv(curve_path, index=False)

    pd.DataFrame(
        [
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "checkpoint_path": str(checkpoint_path),
                "peak_time": peak_time,
                "peak_qreshape_mass": peak_value,
                "peak_qreshape_mass_norm": float(q_norm[peak_idx]),
                "drift_peak_time": drift_peak_time,
                "drift_peak_value": drift_peak_value,
                "drift_peak_norm": float(drift_norm[drift_peak_idx]),
                "action_peak_time": action_peak_time,
                "action_peak_value": action_peak_value,
                "action_peak_norm": float(action_norm[action_peak_idx]),
                "n_timepoints": int(len(dense_times_np)),
                "max_cells": int(args.max_cells),
                "dataset_slug": dataset_slug,
                "dataset_label": dataset_label,
                "observed_times": ";".join(f"{float(t):g}" for t in observed_times),
            }
        ]
    ).to_csv(summary_path, index=False)

    print(f"Resolved run: {run_name}")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved figure to: {figure_path}")
    print(f"Saved action figure to: {action_figure_path}")
    print(f"Saved drift figure to: {drift_figure_path}")
    print(f"Saved per-time CSV to: {curve_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
