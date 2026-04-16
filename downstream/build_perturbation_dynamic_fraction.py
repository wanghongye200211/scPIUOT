from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT, PIUOT_OUTPUT_ROOT, PIUOT_ROOT


EPS = 1e-8
SCREEN_SEED = 42
MANUAL_RUN_NAME = "piuot_run"
MANUAL_SEED = 0
MANUAL_CHECKPOINT = "auto"
MANUAL_DEVICE = "cpu"
MANUAL_DATA_PATH = str(PIUOT_ROOT / "input" / "input.h5ad")
MANUAL_EMBEDDING_KEY = "X_gae15"
MANUAL_RAW_TIME_KEY = "t"
MANUAL_FATE_KEY = "phenotype_facs"
MANUAL_OUTPUT_LABEL = "dataset"
MANUAL_START_TIME = None
MANUAL_END_TIME = None
MANUAL_TARGET_LABEL = None
MANUAL_N_TIMEPOINTS = 25
MANUAL_N_REPEATS = 4
MANUAL_MAX_START_CELLS = 96
MANUAL_SCALE = 2.0
MANUAL_TOP_TERMINAL_FATES = 5


def _resolve_src_package(run_name: str) -> str:
    return "core"


def _load_runtime_modules(run_name: str):
    if str(PIUOT_ROOT) not in sys.path:
        sys.path.insert(0, str(PIUOT_ROOT))
    src_pkg = importlib.import_module(_resolve_src_package(run_name))
    sys.modules["src"] = src_pkg
    config_mod = importlib.import_module("src.config_model")
    model_mod = importlib.import_module("src.model")
    train_mod = importlib.import_module("src.train")
    return config_mod, model_mod, train_mod


def _resolve_run_dir(run_name: str, seed: int) -> Path:
    matches = sorted((PIUOT_OUTPUT_ROOT / run_name).glob(f"*/seed_{seed}/alltime"))
    if not matches:
        raise FileNotFoundError(f"Could not find run '{run_name}' for seed {seed}.")
    return matches[-1]


def _best_epoch_from_eval(run_dir: Path) -> str:
    eval_path = run_dir / "interpolate-mioemd2.log"
    eval_df = pd.read_csv(eval_path, sep="\t")
    mean_eval = eval_df.groupby("epoch", as_index=False)["loss"].mean()
    return str(mean_eval.loc[mean_eval["loss"].idxmin(), "epoch"])


def _latest_epoch_from_checkpoints(run_dir: Path) -> str:
    matches = sorted(run_dir.glob("train.epoch_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No epoch checkpoints found under {run_dir}")
    latest = max(matches, key=lambda path: int(path.stem.split("epoch_")[1]))
    return latest.stem.split("train.", 1)[1]


def _resolve_checkpoint_path(run_dir: Path, selector: str) -> Path:
    if selector == "auto":
        selector = _best_epoch_from_eval(run_dir) if (run_dir / "interpolate-mioemd2.log").exists() else _latest_epoch_from_checkpoints(run_dir)
    if selector == "best":
        return run_dir / "train.best.pt"
    return run_dir / f"train.{selector}.pt"


def _normalize_strings(values) -> np.ndarray:
    return np.asarray(pd.Series(values).astype(str).fillna("nan"), dtype=object)


def _nearest_observed_time(value: float, observed_times: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(observed_times - float(value))))
    return float(observed_times[idx])


def _build_labelers(latent: np.ndarray, time_values: np.ndarray, labels: np.ndarray, observed_times: np.ndarray):
    labelers = {}
    for obs_time in observed_times:
        mask = np.isclose(time_values, obs_time)
        points = latent[mask]
        point_labels = labels[mask]
        classes, label_indices = np.unique(point_labels, return_inverse=True)
        nn_model = NearestNeighbors(
            n_neighbors=max(1, min(15, int(points.shape[0]))),
            metric="euclidean",
        )
        nn_model.fit(points)
        labelers[float(obs_time)] = (nn_model, label_indices.astype(np.int64), classes.astype(object))
    return labelers


def _weighted_knn_labels(labeler, query: np.ndarray) -> np.ndarray:
    nn_model, label_indices, classes = labeler
    distances, indices = nn_model.kneighbors(np.asarray(query, dtype=np.float32))
    weights = 1.0 / np.maximum(np.asarray(distances, dtype=np.float64), EPS)
    probs = np.zeros((query.shape[0], len(classes)), dtype=np.float64)
    row_index = np.arange(query.shape[0], dtype=np.int64)
    for col_idx in range(indices.shape[1]):
        np.add.at(probs, (row_index, label_indices[indices[:, col_idx]]), weights[:, col_idx])
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), EPS)
    return classes[probs.argmax(axis=1)]


def _rollout_condition(
    *,
    start_latent: np.ndarray,
    condition: str,
    latent_dim: int | None,
    scale: float | None,
    model,
    train_mod,
    cfg,
    labelers,
    observed_times: np.ndarray,
    dense_times: np.ndarray,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    x0 = torch.tensor(start_latent.astype(np.float32), dtype=torch.float32, device=device)
    x_r0 = train_mod.build_initial_state(
        x0,
        bool(getattr(cfg, "use_growth", False)),
        clip_value=float(getattr(cfg, "mass_clip_value", 30.0)),
    )
    traj_state = model(dense_times.astype(np.float64).tolist(), x_r0).detach()

    x_dim = int(getattr(cfg, "x_dim"))
    use_growth = bool(getattr(cfg, "use_growth", False))
    clip_value = float(getattr(cfg, "mass_clip_value", 30.0))
    rows = []
    all_labels = sorted({label for _, _, classes in labelers.values() for label in classes})

    for time_idx, time_value in enumerate(dense_times):
        state_t = traj_state[time_idx]
        x_t, _, logw_t = train_mod.unpack_state(state_t, x_dim, use_growth)
        x_np = x_t.detach().cpu().numpy().astype(np.float32)
        weights = (
            train_mod.normalized_mass_from_logw(logw_t, clip_value=clip_value).detach().cpu().numpy()
            if logw_t is not None
            else np.full(x_np.shape[0], 1.0 / max(int(x_np.shape[0]), 1), dtype=np.float64)
        )
        match_time = _nearest_observed_time(float(time_value), observed_times)
        predicted = _weighted_knn_labels(labelers[match_time], x_np)
        total = max(float(np.sum(weights)), EPS)
        row = {
            "condition": condition,
            "latent_dim": latent_dim,
            "scale": scale,
            "time": float(time_value),
            "matched_observed_time": float(match_time),
        }
        for label in all_labels:
            row[f"mass_{label}"] = float(np.sum(weights[predicted == label]) / total)
        rows.append(row)

    fraction_df = pd.DataFrame(rows)
    terminal_row = fraction_df.loc[fraction_df["time"].idxmax()]
    terminal_fraction = {
        col.replace("mass_", ""): float(terminal_row[col])
        for col in fraction_df.columns
        if col.startswith("mass_")
    }
    return fraction_df, terminal_fraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build generic perturbation dynamic-fraction plots.")
    parser.add_argument("--run-name", default=MANUAL_RUN_NAME)
    parser.add_argument("--seed", type=int, default=MANUAL_SEED)
    parser.add_argument("--checkpoint", default=MANUAL_CHECKPOINT)
    parser.add_argument("--device", default=MANUAL_DEVICE)
    parser.add_argument("--data-path", type=Path, default=Path(MANUAL_DATA_PATH))
    parser.add_argument("--embedding-key", default=MANUAL_EMBEDDING_KEY)
    parser.add_argument("--raw-time-key", default=MANUAL_RAW_TIME_KEY)
    parser.add_argument("--fate-key", default=MANUAL_FATE_KEY)
    parser.add_argument("--output-label", default=MANUAL_OUTPUT_LABEL)
    parser.add_argument("--start-time", type=float, default=MANUAL_START_TIME)
    parser.add_argument("--end-time", type=float, default=MANUAL_END_TIME)
    parser.add_argument("--target-label", default=MANUAL_TARGET_LABEL)
    parser.add_argument("--n-timepoints", type=int, default=MANUAL_N_TIMEPOINTS)
    parser.add_argument("--n-repeats", type=int, default=MANUAL_N_REPEATS)
    parser.add_argument("--max-start-cells", type=int, default=MANUAL_MAX_START_CELLS)
    parser.add_argument("--scale", type=float, default=MANUAL_SCALE)
    parser.add_argument("--top-terminal-fates", type=int, default=MANUAL_TOP_TERMINAL_FATES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_name = str(args.run_name)
    label = str(args.output_label)
    seed = int(args.seed)
    data_path = Path(args.data_path)
    embedding_key = str(args.embedding_key)
    raw_time_key = str(args.raw_time_key)
    fate_key = str(args.fate_key)
    target_label = args.target_label

    run_dir = _resolve_run_dir(run_name, seed)
    checkpoint = _resolve_checkpoint_path(run_dir, str(args.checkpoint))
    device = torch.device(str(args.device))

    config_mod, model_mod, train_mod = _load_runtime_modules(run_name)
    cfg_dict = torch.load(run_dir / "config.pt", map_location="cpu", weights_only=False)
    model_cfg = SimpleNamespace(**cfg_dict)
    _, _, model_cfg = config_mod.load_data(model_cfg)
    model = model_mod.ForwardSDE(model_cfg)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model.to(device)
    model.eval()

    adata = ad.read_h5ad(data_path)
    latent = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    time_values = np.asarray(adata.obs[raw_time_key], dtype=np.float64)
    labels = _normalize_strings(adata.obs[fate_key])
    observed_times = np.sort(np.unique(time_values))

    start_time = _nearest_observed_time(
        observed_times[0] if args.start_time is None else float(args.start_time),
        observed_times,
    )
    end_time = _nearest_observed_time(
        observed_times[-1] if args.end_time is None else float(args.end_time),
        observed_times,
    )
    dense_times = np.linspace(start_time, end_time, int(args.n_timepoints), dtype=np.float64)
    max_start_cells = int(args.max_start_cells)
    n_repeats = int(args.n_repeats)
    scale = float(args.scale)

    start_indices = np.flatnonzero(np.isclose(time_values, start_time))
    rng = np.random.default_rng(SCREEN_SEED)
    if start_indices.size > max_start_cells:
        start_indices = np.sort(rng.choice(start_indices, size=max_start_cells, replace=False))
    start_latent = np.repeat(latent[start_indices].astype(np.float32), n_repeats, axis=0)

    labelers = _build_labelers(latent, time_values, labels, observed_times)

    records = []
    summaries = []
    control_df, control_terminal = _rollout_condition(
        start_latent=start_latent,
        condition="control",
        latent_dim=None,
        scale=None,
        model=model,
        train_mod=train_mod,
        cfg=model_cfg,
        labelers=labelers,
        observed_times=observed_times,
        dense_times=dense_times,
        device=device,
    )
    records.append(control_df)

    if target_label is None:
        target_label = max(control_terminal.items(), key=lambda item: item[1])[0]

    summaries.append(
        {
            "condition": "control",
            "latent_dim": None,
            "scale": None,
            "terminal_target_fraction": float(control_terminal.get(target_label, 0.0)),
            "delta_target_fraction": 0.0,
        }
    )

    for latent_dim in range(start_latent.shape[1]):
        perturbed = start_latent.copy()
        perturbed[:, latent_dim] *= scale
        fraction_df, terminal = _rollout_condition(
            start_latent=perturbed,
            condition=f"latent_{latent_dim}_x{scale:g}",
            latent_dim=latent_dim,
            scale=scale,
            model=model,
            train_mod=train_mod,
            cfg=model_cfg,
            labelers=labelers,
            observed_times=observed_times,
            dense_times=dense_times,
            device=device,
        )
        records.append(fraction_df)
        summaries.append(
            {
                "condition": f"latent_{latent_dim}_x{scale:g}",
                "latent_dim": latent_dim,
                "scale": scale,
                "terminal_target_fraction": float(terminal.get(target_label, 0.0)),
                "delta_target_fraction": float(terminal.get(target_label, 0.0) - control_terminal.get(target_label, 0.0)),
            }
        )

    summary_df = pd.DataFrame(summaries)
    non_control = summary_df.loc[summary_df["condition"] != "control"].copy()
    best_pos = non_control.loc[non_control["delta_target_fraction"].idxmax()].to_dict()
    best_neg = non_control.loc[non_control["delta_target_fraction"].idxmin()].to_dict()
    selected_conditions = ["control", str(best_pos["condition"]), str(best_neg["condition"])]

    fraction_df = pd.concat(records, ignore_index=True)
    selected_df = fraction_df.loc[fraction_df["condition"].isin(selected_conditions)].copy()

    mass_cols = [col for col in selected_df.columns if col.startswith("mass_")]
    terminal_selected = (
        selected_df.loc[selected_df.groupby("condition")["time"].idxmax(), ["condition", *mass_cols]]
        .set_index("condition")
        .mean(axis=0)
        .sort_values(ascending=False)
    )
    top_fates = int(args.top_terminal_fates)
    keep_cols = list(terminal_selected.head(top_fates).index)
    other_cols = [col for col in mass_cols if col not in keep_cols]
    if other_cols:
        selected_df["mass_other"] = selected_df[other_cols].sum(axis=1)
        keep_cols.append("mass_other")

    out_dir = DOWNSTREAM_OUTPUT_ROOT / f"{label}_perturbation_dynamic_fraction"
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions_csv = out_dir / "condition_time_fractions.csv"
    screen_csv = out_dir / "latent_screen_terminal_summary.csv"
    selected_df.to_csv(fractions_csv, index=False)
    summary_df.to_csv(screen_csv, index=False)

    stack_png = out_dir / "dynamic_fraction_stackplot.png"
    fig, axes = plt.subplots(1, len(selected_conditions), figsize=(5.2 * len(selected_conditions), 4.8), constrained_layout=True)
    if len(selected_conditions) == 1:
        axes = [axes]
    for ax, condition in zip(axes, selected_conditions):
        cond_df = selected_df.loc[selected_df["condition"] == condition].sort_values("time")
        values = [cond_df[col].to_numpy() for col in keep_cols]
        labels_plot = [col.replace("mass_", "") for col in keep_cols]
        ax.stackplot(cond_df["time"].to_numpy(), values, labels=labels_plot, alpha=0.92)
        ax.set_title(condition)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Model time")
    axes[0].set_ylabel("Mass fraction")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(stack_png, dpi=220)
    plt.close(fig)

    target_png = out_dir / "target_group_dynamic_fraction.png"
    target_col = f"mass_{target_label}"
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for condition in selected_conditions:
        cond_df = selected_df.loc[selected_df["condition"] == condition].sort_values("time")
        ax.plot(cond_df["time"], cond_df[target_col], lw=2.0, marker="o", label=condition)
    ax.set_xlabel("Model time")
    ax.set_ylabel(f"{target_label} fraction")
    ax.set_title(f"{label} | target fraction trajectories")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(target_png, dpi=220)
    plt.close(fig)

    summary = {
        "run_name": run_name,
        "label": label,
        "data_path": str(data_path),
        "embedding_key": embedding_key,
        "checkpoint": checkpoint.name,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "target_label": target_label,
        "selected_conditions": selected_conditions,
        "best_positive_shift": best_pos,
        "best_negative_shift": best_neg,
        "artifacts": {
            "dynamic_fraction_stackplot_png": str(stack_png),
            "target_group_dynamic_fraction_png": str(target_png),
            "condition_time_fractions_csv": str(fractions_csv),
            "latent_screen_terminal_summary_csv": str(screen_csv),
        },
    }
    (out_dir / "perturbation_dynamic_fraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
