from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import PIUOT_ROOT


METHOD_ROOT = PIUOT_ROOT
OUTPUT_ROOT = METHOD_ROOT / "output"
EPS = 1e-8

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


def _resolve_src_package(run_name: str) -> str:
    return "core"


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


def _latest_epoch_from_checkpoints(run_dir: Path) -> str:
    matches = sorted(run_dir.glob("train.epoch_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No epoch checkpoints found under {run_dir}")
    latest = max(matches, key=lambda path: int(path.stem.split("epoch_")[1]))
    return latest.stem.split("train.", 1)[1]


def _resolve_epoch_tag(run_dir: Path, epoch_selector: str) -> str:
    if epoch_selector == "auto":
        eval_path = run_dir / "interpolate-mioemd2.log"
        if eval_path.exists():
            return _best_epoch_from_eval(run_dir)
        if (run_dir / "train.best.pt").exists():
            return "best"
        return _latest_epoch_from_checkpoints(run_dir)
    if epoch_selector == "final":
        return _latest_epoch_from_checkpoints(run_dir)
    return epoch_selector


def _checkpoint_path(run_dir: Path, epoch_tag: str) -> Path:
    if epoch_tag == "best":
        return run_dir / "train.best.pt"
    return run_dir / f"train.{epoch_tag}.pt"


def _move_time_series_to_device(x, device: torch.device):
    return [x_i.to(device) if x_i.device != device else x_i for x_i in x]


def _dense_time_grid(observed_times: np.ndarray, n_timepoints: int) -> np.ndarray:
    return np.linspace(float(observed_times[0]), float(observed_times[-1]), int(n_timepoints), dtype=np.float64)


def _first_existing_key(keys: list[str], candidates: list[str]) -> str | None:
    key_set = set(keys)
    for key in candidates:
        if key in key_set:
            return key
    return None


def _resolve_label_keys(
    adata: ad.AnnData,
    state_key: str | None,
    fate_key: str | None,
) -> tuple[str | None, str | None]:
    obs_keys = list(adata.obs_keys())
    if state_key is not None and state_key not in obs_keys:
        raise KeyError(f"Requested state key '{state_key}' not found in adata.obs.")
    if fate_key is not None and fate_key not in obs_keys:
        raise KeyError(f"Requested fate key '{fate_key}' not found in adata.obs.")
    state_key = state_key or _first_existing_key(
        obs_keys,
        [
            "consensus_cluster",
            "Assigned_subcluster",
            "Assigned_cluster",
            "cell_type",
            "celltype",
            "cluster",
            "state",
            "annotation",
            "label",
        ],
    )
    fate_key = fate_key or _first_existing_key(
        obs_keys,
        [
            "phenotype_facs",
            "fate",
            "cell_fate",
            "terminal_fate",
            "lineage",
        ],
    )
    return state_key, fate_key


def _embedding_array(adata: ad.AnnData, embedding_key: str) -> np.ndarray:
    if embedding_key == "X":
        matrix = adata.X
    elif embedding_key in adata.obsm:
        matrix = adata.obsm[embedding_key]
    else:
        raise KeyError(f"Embedding key '{embedding_key}' not found in adata.obsm and is not 'X'.")
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _read_prior(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_critical_window(args, prior: dict[str, Any], observed_times: np.ndarray) -> tuple[float | None, float | None]:
    if args.critical_window_start is not None and args.critical_window_end is not None:
        return float(args.critical_window_start), float(args.critical_window_end)

    use_instead = prior.get("use_instead", {})
    critical_window = use_instead.get("critical_window")
    if isinstance(critical_window, list) and len(critical_window) == 2:
        return float(critical_window[0]), float(critical_window[1])
    return float(observed_times[0]), float(observed_times[-1])


def _resolve_anchor_min_time(args, prior: dict[str, Any], observed_times: np.ndarray) -> float:
    if args.anchor_min_time is not None:
        return float(args.anchor_min_time)
    use_instead = prior.get("use_instead", {})
    if "early_post_branch_day" in use_instead:
        return float(use_instead["early_post_branch_day"])
    if observed_times.size >= 2:
        return float(observed_times[-2])
    return float(observed_times[-1])


def _normalize_string_array(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(pd.Series(values).astype(str).fillna("nan"), dtype=object)


def _inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(np.asarray(distances, dtype=np.float64), EPS)


def _weighted_knn_probabilities(
    nn_model: NearestNeighbors,
    reference_label_indices: np.ndarray,
    classes: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances, indices = nn_model.kneighbors(query)
    weights = _inverse_distance_weights(distances)
    probs = np.zeros((query.shape[0], len(classes)), dtype=np.float64)
    row_index = np.arange(query.shape[0], dtype=np.int64)
    for col_idx in range(indices.shape[1]):
        np.add.at(probs, (row_index, reference_label_indices[indices[:, col_idx]]), weights[:, col_idx])
    probs_sum = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(probs_sum, EPS)
    label_idx = probs.argmax(axis=1)
    labels = classes[label_idx]
    confidence = probs[np.arange(probs.shape[0]), label_idx]
    return probs, labels, confidence


def _state_predictor(
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    n_neighbors: int,
) -> tuple[NearestNeighbors, np.ndarray, np.ndarray]:
    classes, label_indices = np.unique(reference_labels, return_inverse=True)
    nn_model = NearestNeighbors(
        n_neighbors=max(1, min(int(n_neighbors), reference_points.shape[0])),
        metric="euclidean",
    )
    nn_model.fit(reference_points)
    return nn_model, classes.astype(object), label_indices.astype(np.int64)


def _choose_start_time(start_time: float | None, observed_times: np.ndarray) -> float:
    if start_time is None:
        return float(observed_times[0])
    observed_times = np.asarray(observed_times, dtype=np.float64)
    idx = int(np.argmin(np.abs(observed_times - float(start_time))))
    return float(observed_times[idx])


def _select_start_cells(
    adata: ad.AnnData,
    *,
    embedding_key: str,
    time_key: str,
    start_time: float,
    max_start_cells: int,
    rng: np.random.Generator,
    state_key: str | None,
    start_state_values: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    time_values = np.asarray(adata.obs[time_key], dtype=np.float64)
    mask = np.isclose(time_values, float(start_time))
    if state_key is not None and start_state_values:
        state_values = _normalize_string_array(adata.obs[state_key])
        state_mask = np.isin(state_values, np.asarray(start_state_values, dtype=object))
        mask = mask & state_mask

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(
            f"No starting cells found at time={start_time:g}"
            + ("" if not start_state_values else f" with {state_key} in {start_state_values}.")
        )

    if max_start_cells > 0 and indices.size > int(max_start_cells):
        indices = np.sort(rng.choice(indices, size=int(max_start_cells), replace=False))

    embedding = _embedding_array(adata, embedding_key)[indices]
    return indices.astype(np.int64), embedding


def _valid_fate_labels(labels: np.ndarray, excluded: set[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    return np.asarray([label not in excluded for label in labels], dtype=bool)


def _resolve_fate_anchors(
    adata: ad.AnnData,
    *,
    embedding_key: str,
    time_key: str,
    fate_key: str | None,
    anchor_min_time: float,
    excluded_fates: set[str],
    pseudo_fate_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    time_values = np.asarray(adata.obs[time_key], dtype=np.float64)
    anchor_mask = time_values >= float(anchor_min_time) - 1e-8
    if not np.any(anchor_mask):
        anchor_mask = np.isclose(time_values, float(np.max(time_values)))

    anchor_points = _embedding_array(adata, embedding_key)[anchor_mask]
    if anchor_points.shape[0] == 0:
        raise ValueError("Could not find any late-time anchor cells for fate inference.")

    if fate_key is not None and fate_key in adata.obs:
        labels = _normalize_string_array(adata.obs.loc[anchor_mask, fate_key])
        keep_mask = _valid_fate_labels(labels, excluded_fates)
        usable_labels = np.unique(labels[keep_mask])
        if keep_mask.sum() >= max(8, len(usable_labels)) and usable_labels.size >= 2:
            return anchor_points[keep_mask], labels[keep_mask], "annotated"

    n_clusters = max(2, min(int(pseudo_fate_clusters), max(2, anchor_points.shape[0] // 20)))
    if anchor_points.shape[0] < n_clusters:
        n_clusters = max(2, min(anchor_points.shape[0], 2))
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10)
    cluster_idx = kmeans.fit_predict(anchor_points)
    labels = np.asarray([f"terminal_cluster_{idx}" for idx in cluster_idx], dtype=object)
    return anchor_points, labels, "pseudo_cluster"


def _prepare_reference_projection(embedding: np.ndarray) -> PCA:
    reducer = PCA(n_components=2, random_state=0)
    reducer.fit(embedding)
    return reducer


def _resolve_anchor_index(time: np.ndarray, anchor_time: float | None) -> int:
    time = np.asarray(time, dtype=np.float64)
    if anchor_time is None:
        return 0
    candidates = np.flatnonzero(time >= float(anchor_time) - 1e-8)
    if candidates.size == 0:
        raise ValueError(f"Could not find any dense time >= anchor_time={anchor_time}")
    return int(candidates[0])


def _safe_normalize(values: np.ndarray, anchor_idx: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = values / max(float(values[anchor_idx]), EPS)
    if anchor_idx > 0:
        out[:anchor_idx] = np.nan
    return out


def _entropy_from_probs(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _predict_terminal_fates(
    final_points: np.ndarray,
    *,
    anchor_points: np.ndarray,
    anchor_labels: np.ndarray,
    n_neighbors: int,
    min_confidence: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    fate_nn, fate_classes, fate_label_indices = _state_predictor(anchor_points, anchor_labels, n_neighbors)
    probs, labels, confidence = _weighted_knn_probabilities(
        fate_nn,
        fate_label_indices,
        fate_classes,
        final_points,
    )
    predicted = labels.astype(object)
    predicted = np.where(confidence >= float(min_confidence), predicted, "unresolved")
    payload = {"terminal_fate": predicted, "terminal_fate_confidence": confidence}
    for idx, class_name in enumerate(fate_classes):
        payload[f"prob_{class_name}"] = probs[:, idx]
    return pd.DataFrame(payload), fate_classes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Physics-informed manifold downstream analysis: dense trajectory rollout, critical-point detection, "
            "and cell-fate inference from an existing MIOPISDE checkpoint."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="auto", help="best, auto, final, or explicit epoch tag.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--n-timepoints", type=int, default=121)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--max-start-cells", type=int, default=96)
    parser.add_argument("--n-repeats", type=int, default=12)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--fate-key", default=None)
    parser.add_argument("--start-state", action="append", default=[])
    parser.add_argument("--state-neighbors", type=int, default=15)
    parser.add_argument("--fate-neighbors", type=int, default=25)
    parser.add_argument("--fate-min-confidence", type=float, default=0.45)
    parser.add_argument("--pseudo-fate-clusters", type=int, default=3)
    parser.add_argument("--anchor-min-time", type=float, default=None)
    parser.add_argument("--exclude-fate", action="append", default=["intermediate", "nan", "None", "unknown"])
    parser.add_argument("--critical-indicator", choices=["product", "qphi", "action"], default="product")
    parser.add_argument("--critical-window-start", type=float, default=None)
    parser.add_argument("--critical-window-end", type=float, default=None)
    parser.add_argument("--critical-window-prior", type=Path, default=None)
    parser.add_argument(
        "--normalize-start-time",
        type=float,
        default=None,
        help="Anchor normalized metrics at the first dense time >= this value and mask earlier times.",
    )
    parser.add_argument("--trajectory-csv", action="store_true", default=True)
    parser.add_argument("--no-trajectory-csv", dest="trajectory_csv", action="store_false")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    run_dir = _resolve_run_dir(args.run_name, int(args.seed))
    epoch_tag = _resolve_epoch_tag(run_dir, str(args.checkpoint))
    checkpoint_path = _checkpoint_path(run_dir, epoch_tag)
    output_dir = args.output_dir or (run_dir.parent.parent.parent / "figs" / args.run_name / "downstream_fate")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_mod, model_mod, train_mod = _load_runtime_modules(args.run_name)
    config = torch.load(run_dir / "config.pt", map_location="cpu")
    if args.data_path is not None:
        config["data_path"] = str(args.data_path.resolve())
    cfg = type("Cfg", (), config)

    device = torch.device(str(args.device))
    x, y, _ = config_mod.load_data(cfg)
    x = _move_time_series_to_device(x, device)
    observed_times = np.asarray(y, dtype=np.float64)
    dense_times = _dense_time_grid(observed_times, int(args.n_timepoints))

    model = model_mod.ForwardSDE(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    adata = ad.read_h5ad(config["data_path"])
    embedding_key = str(config.get("embedding_key", "X"))
    time_key = str(config.get("raw_time_key", config.get("time_key", "t")))
    if time_key not in adata.obs:
        time_key = str(config.get("time_key", "time_bin"))
    state_key, fate_key = _resolve_label_keys(adata, args.state_key, args.fate_key)
    prior = _read_prior(args.critical_window_prior)
    critical_window_start, critical_window_end = _resolve_critical_window(args, prior, observed_times)
    anchor_min_time = _resolve_anchor_min_time(args, prior, observed_times)
    start_time = _choose_start_time(args.start_time, observed_times)

    start_indices, start_points_np = _select_start_cells(
        adata,
        embedding_key=embedding_key,
        time_key=time_key,
        start_time=start_time,
        max_start_cells=int(args.max_start_cells),
        rng=rng,
        state_key=state_key,
        start_state_values=list(args.start_state),
    )

    start_points = torch.tensor(start_points_np, dtype=torch.float32, device=device)
    n_start = int(start_points.shape[0])
    n_repeats = max(1, int(args.n_repeats))
    repeated_points = start_points.repeat_interleave(n_repeats, dim=0)
    seed_cell_index = np.repeat(np.arange(n_start, dtype=np.int64), n_repeats)
    repeat_index = np.tile(np.arange(n_repeats, dtype=np.int64), n_start)
    obs_index = np.repeat(start_indices, n_repeats)

    torch.manual_seed(int(args.seed))
    x_r0 = train_mod.build_initial_state(
        repeated_points,
        bool(config.get("use_growth", False)),
        clip_value=float(config.get("mass_clip_value", 30.0)),
    )
    traj_state = model(dense_times.tolist(), x_r0).detach()

    observed_embedding = _embedding_array(adata, embedding_key)
    reducer = _prepare_reference_projection(observed_embedding)
    observed_proj = reducer.transform(observed_embedding)

    state_predictor = None
    if state_key is not None:
        state_labels = _normalize_string_array(adata.obs[state_key])
        state_predictor = _state_predictor(
            observed_embedding,
            state_labels,
            int(args.state_neighbors),
        )

    excluded_fates = {str(value) for value in args.exclude_fate}
    anchor_points, anchor_labels, fate_mode = _resolve_fate_anchors(
        adata,
        embedding_key=embedding_key,
        time_key=time_key,
        fate_key=fate_key,
        anchor_min_time=float(anchor_min_time),
        excluded_fates=excluded_fates,
        pseudo_fate_clusters=int(args.pseudo_fate_clusters),
        seed=int(args.seed),
    )

    final_state = traj_state[-1]
    final_x, _, final_logw = train_mod.unpack_state(final_state, int(config["x_dim"]), bool(config.get("use_growth", False)))
    final_points_np = final_x.detach().cpu().numpy().astype(np.float32)
    final_probs_df, fate_classes = _predict_terminal_fates(
        final_points_np,
        anchor_points=anchor_points,
        anchor_labels=anchor_labels,
        n_neighbors=int(args.fate_neighbors),
        min_confidence=float(args.fate_min_confidence),
    )
    if final_logw is not None:
        final_mass = train_mod.normalized_mass_from_logw(
            final_logw,
            clip_value=float(config.get("mass_clip_value", 30.0)),
        ).detach().cpu().numpy()
    else:
        final_mass = np.full(final_points_np.shape[0], 1.0 / max(final_points_np.shape[0], 1), dtype=np.float64)

    terminal_df = pd.DataFrame(
        {
            "sim_index": np.arange(final_points_np.shape[0], dtype=np.int64),
            "seed_cell_index": seed_cell_index,
            "repeat_index": repeat_index,
            "obs_index": obs_index,
            "terminal_mass_norm": final_mass,
        }
    )
    terminal_df = pd.concat([terminal_df, final_probs_df], axis=1)

    per_seed_records = []
    prob_cols = [f"prob_{class_name}" for class_name in fate_classes]
    for local_seed in range(n_start):
        mask = terminal_df["seed_cell_index"].to_numpy() == local_seed
        subset = terminal_df.loc[mask]
        eq_mean = subset[prob_cols].mean(axis=0)
        mass_weights = subset["terminal_mass_norm"].to_numpy(dtype=np.float64)
        mass_weights = mass_weights / max(mass_weights.sum(), EPS)
        mass_mean = np.average(subset[prob_cols].to_numpy(dtype=np.float64), axis=0, weights=mass_weights)
        best_idx = int(np.argmax(eq_mean.to_numpy(dtype=np.float64)))
        per_seed_record: dict[str, Any] = {
            "seed_cell_index": int(local_seed),
            "obs_index": int(start_indices[local_seed]),
            "predicted_fate_equal_weight": str(fate_classes[best_idx]),
            "predicted_fate_equal_weight_prob": float(eq_mean.iloc[best_idx]),
            "predicted_fate_mass_weighted": str(fate_classes[int(np.argmax(mass_mean))]),
        }
        for col_name, value in eq_mean.items():
            per_seed_record[f"{col_name}_equal_weight"] = float(value)
        for col_name, value in zip(prob_cols, mass_mean):
            per_seed_record[f"{col_name}_mass_weighted"] = float(value)
        per_seed_records.append(per_seed_record)
    per_seed_df = pd.DataFrame(per_seed_records)

    per_time_frames = []
    fate_mass_rows = []
    x_dim = int(config["x_dim"])
    use_growth = bool(config.get("use_growth", False))
    clip_value = float(config.get("mass_clip_value", 30.0))
    alpha_g = float(config.get("action_alpha_g", 1.0))
    alpha_sigma = float(config.get("action_alpha_sigma", 1e-4))
    terminal_fate_labels = terminal_df["terminal_fate"].to_numpy(dtype=object)
    unique_terminal_fates = np.unique(terminal_fate_labels)
    terminal_fate_classes = np.asarray([str(item) for item in unique_terminal_fates], dtype=object)

    action_curve = np.zeros(len(dense_times), dtype=np.float64)
    qphi_curve = np.zeros(len(dense_times), dtype=np.float64)

    for time_idx, time_value in enumerate(dense_times):
        state_t = traj_state[time_idx]
        x_t, _, logw_t = train_mod.unpack_state(state_t, x_dim, use_growth)
        if logw_t is not None:
            weights = train_mod.normalized_mass_from_logw(logw_t, clip_value=clip_value).detach().cpu().numpy()
        else:
            weights = np.full(x_t.shape[0], 1.0 / max(int(x_t.shape[0]), 1), dtype=np.float64)

        x_np = x_t.detach().cpu().numpy().astype(np.float32)
        proj_np = reducer.transform(x_np)
        t_batch = x_t.new_full((x_t.shape[0], 1), float(time_value))
        xt = torch.cat([x_t, t_batch], dim=1).requires_grad_(True)
        potential_t = model._func.net(xt).squeeze(-1)
        grad_t = torch.autograd.grad(potential_t, xt, torch.ones_like(potential_t), create_graph=False)[0]
        drift_t = -grad_t[:, :-1]
        drift_norm = torch.norm(drift_t, dim=1).detach().cpu().numpy()
        growth_t = (
            model._func._growth(xt).squeeze(-1).detach().cpu().numpy()
            if use_growth
            else np.zeros(x_t.shape[0], dtype=np.float64)
        )
        sigma_diag = model._func.g(float(time_value), state_t)[:, :x_dim]
        sigma_norm = torch.norm(sigma_diag, dim=1).detach().cpu().numpy()
        action_density = drift_norm**2 + alpha_g * (growth_t**2) + alpha_sigma * (sigma_norm**2)
        action_curve[time_idx] = float(np.sum(weights * action_density))

        if time_idx < len(dense_times) - 1:
            next_t = x_t.new_full((x_t.shape[0], 1), float(dense_times[time_idx + 1]))
            xt_next = torch.cat([x_t, next_t], dim=1)
            potential_next = model._func.net(xt_next).squeeze(-1).detach().cpu().numpy()
            potential_now = potential_t.detach().cpu().numpy()
            dt = max(float(dense_times[time_idx + 1] - time_value), EPS)
            qphi_curve[time_idx] = float(np.sum(weights * np.abs(potential_next - potential_now) / dt))
        elif time_idx > 0:
            qphi_curve[time_idx] = qphi_curve[time_idx - 1]

        if state_predictor is not None:
            state_nn, state_classes, state_label_indices = state_predictor
            _, predicted_state, state_conf = _weighted_knn_probabilities(
                state_nn,
                state_label_indices,
                state_classes,
                x_np,
            )
        else:
            predicted_state = np.full(x_t.shape[0], "", dtype=object)
            state_conf = np.zeros(x_t.shape[0], dtype=np.float64)

        time_frame = pd.DataFrame(
            {
                "time": float(time_value),
                "sim_index": np.arange(x_t.shape[0], dtype=np.int64),
                "seed_cell_index": seed_cell_index,
                "repeat_index": repeat_index,
                "obs_index": obs_index,
                "mass_norm": weights,
                "state_pred": predicted_state,
                "state_confidence": state_conf,
                "eventual_fate": terminal_fate_labels,
                "drift_norm": drift_norm,
                "growth": growth_t,
                "diffusion_norm": sigma_norm,
                "action_density": action_density,
                "proj_1": proj_np[:, 0],
                "proj_2": proj_np[:, 1],
            }
        )
        for dim_idx in range(x_dim):
            time_frame[f"z{dim_idx + 1}"] = x_np[:, dim_idx]
        per_time_frames.append(time_frame)

        fate_mass = {"time": float(time_value), "Action": float(action_curve[time_idx]), "Q_phi": float(qphi_curve[time_idx])}
        total_weight = max(float(np.sum(weights)), EPS)
        prob_vector = []
        for fate_name in terminal_fate_classes:
            mask = terminal_fate_labels == fate_name
            mass_value = float(np.sum(weights[mask]) / total_weight)
            fate_mass[f"mass_{fate_name}"] = mass_value
            prob_vector.append(mass_value)
        entropy = _entropy_from_probs(np.asarray(prob_vector, dtype=np.float64))
        max_entropy = math.log(max(len(prob_vector), 1)) if prob_vector else 0.0
        commitment = 0.0 if max_entropy <= 0 else 1.0 - entropy / max_entropy
        fate_mass["future_fate_entropy"] = float(entropy)
        fate_mass["future_fate_commitment"] = float(commitment)
        fate_mass_rows.append(fate_mass)

    normalize_anchor_idx = _resolve_anchor_index(dense_times, args.normalize_start_time)
    normalize_anchor_time = float(dense_times[normalize_anchor_idx])
    action_norm = _safe_normalize(action_curve, normalize_anchor_idx)
    qphi_norm = _safe_normalize(qphi_curve, normalize_anchor_idx)
    product_curve = action_norm * qphi_norm
    per_time_df = pd.DataFrame(fate_mass_rows)
    per_time_df["Action_norm"] = action_norm
    per_time_df["Q_phi_norm"] = qphi_norm
    per_time_df["Product"] = product_curve

    window_mask = (per_time_df["time"].to_numpy(dtype=np.float64) >= float(critical_window_start) - 1e-8) & (
        per_time_df["time"].to_numpy(dtype=np.float64) <= float(critical_window_end) + 1e-8
    )
    indicator_column = {
        "product": "Product",
        "qphi": "Q_phi_norm",
        "action": "Action_norm",
    }[str(args.critical_indicator)]
    unrestricted_idx = int(np.nanargmax(per_time_df[indicator_column].to_numpy(dtype=np.float64)))
    if np.any(window_mask):
        window_candidates = np.flatnonzero(window_mask)
        local_idx = int(np.nanargmax(per_time_df.loc[window_mask, indicator_column].to_numpy(dtype=np.float64)))
        critical_idx = int(window_candidates[local_idx])
    else:
        critical_idx = unrestricted_idx
    critical_time = float(per_time_df.iloc[critical_idx]["time"])
    per_time_df["is_critical_time"] = False
    per_time_df.loc[critical_idx, "is_critical_time"] = True

    trajectory_long = pd.concat(per_time_frames, axis=0, ignore_index=True)
    critical_slice = trajectory_long[np.isclose(trajectory_long["time"].to_numpy(dtype=np.float64), critical_time)]
    critical_fate_mass = critical_slice.groupby("eventual_fate", as_index=False)["mass_norm"].sum().sort_values("mass_norm", ascending=False)

    overall_equal = terminal_df["terminal_fate"].value_counts(normalize=True).sort_values(ascending=False)
    overall_mass = terminal_df.groupby("terminal_fate", as_index=True)["terminal_mass_norm"].sum().sort_values(ascending=False)
    overall_mass = overall_mass / max(float(overall_mass.sum()), EPS)

    stem = args.label.lower().replace(" ", "_") if args.label else args.run_name
    summary_path = output_dir / f"{stem}_physics_fate_summary.json"
    per_time_path = output_dir / f"{stem}_physics_fate_per_time.csv"
    terminal_path = output_dir / f"{stem}_physics_fate_terminal.csv"
    seed_path = output_dir / f"{stem}_physics_fate_per_seed.csv"
    critical_slice_path = output_dir / f"{stem}_physics_fate_critical_slice.csv"
    long_path = output_dir / f"{stem}_physics_fate_trajectories.csv"
    figure_path = output_dir / f"{stem}_physics_fate_panel.png"

    per_time_df.to_csv(per_time_path, index=False)
    terminal_df.to_csv(terminal_path, index=False)
    per_seed_df.to_csv(seed_path, index=False)
    critical_fate_mass.to_csv(critical_slice_path, index=False)
    if args.trajectory_csv:
        trajectory_long.to_csv(long_path, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), dpi=220)

    axes[0].plot(
        per_time_df["time"],
        per_time_df["Action_norm"],
        color="black",
        lw=2.0,
        label=f"Action / A({normalize_anchor_time:g})",
    )
    axes[0].plot(
        per_time_df["time"],
        per_time_df["Q_phi_norm"],
        color="#1f77b4",
        lw=2.0,
        label=f"Q_phi / Q_phi({normalize_anchor_time:g})",
    )
    axes[0].plot(per_time_df["time"], per_time_df["Product"], color="#c0392b", lw=2.2, label="Product")
    for observed_t in observed_times:
        axes[0].axvline(float(observed_t), color="0.86", lw=0.8, ls="--", zorder=0)
    axes[0].axvspan(float(critical_window_start), float(critical_window_end), color="#f5f0d7", alpha=0.35, zorder=0)
    axes[0].axvline(critical_time, color="#8e44ad", lw=1.8, ls=":")
    axes[0].set_title(f"Criticality | t*={critical_time:g}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("normalized value")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(observed_proj[:, 0], observed_proj[:, 1], s=10, c="0.87", alpha=0.6, ec="none")
    color_cycle = plt.get_cmap("tab10")
    fate_to_color = {
        label: color_cycle(idx % 10) for idx, label in enumerate(terminal_fate_classes.tolist())
    }
    for fate_name in terminal_fate_classes:
        mask = terminal_fate_labels == fate_name
        if not np.any(mask):
            continue
        subset = trajectory_long.loc[trajectory_long["eventual_fate"] == fate_name]
        sample_subset = subset.iloc[:: max(1, int(len(subset) / 6000))]
        axes[1].scatter(
            sample_subset["proj_1"],
            sample_subset["proj_2"],
            s=8,
            alpha=0.32,
            c=[fate_to_color[fate_name]],
            ec="none",
            label=str(fate_name),
        )
    axes[1].scatter(
        critical_slice["proj_1"],
        critical_slice["proj_2"],
        s=18,
        c=[fate_to_color.get(name, (0, 0, 0, 1)) for name in critical_slice["eventual_fate"]],
        ec="black",
        lw=0.2,
        alpha=0.8,
    )
    axes[1].set_title("Dense manifold trajectories")
    axes[1].set_xlabel("projection 1")
    axes[1].set_ylabel("projection 2")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    equal_labels = overall_equal.index.to_list()
    equal_vals = overall_equal.to_numpy(dtype=np.float64)
    mass_vals = np.asarray([overall_mass.get(label, 0.0) for label in equal_labels], dtype=np.float64)
    x_positions = np.arange(len(equal_labels))
    width = 0.38
    axes[2].bar(x_positions - width / 2, equal_vals, width=width, color="#7f8c8d", alpha=0.9, label="trajectory fraction")
    axes[2].bar(x_positions + width / 2, mass_vals, width=width, color="#27ae60", alpha=0.9, label="mass-weighted")
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(equal_labels, rotation=20, ha="right")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("probability")
    axes[2].set_title("Terminal fate inference")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle(args.label or args.run_name)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "run_name": str(args.run_name),
        "run_dir": str(run_dir),
        "checkpoint_epoch_tag": str(epoch_tag),
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(config["data_path"]),
        "embedding_key": str(embedding_key),
        "time_key": str(time_key),
        "state_key": None if state_key is None else str(state_key),
        "fate_key": None if fate_key is None else str(fate_key),
        "fate_mode": str(fate_mode),
        "start_time": float(start_time),
        "n_start_cells": int(n_start),
        "n_repeats": int(n_repeats),
        "n_simulated_trajectories": int(n_start * n_repeats),
        "observed_times": [float(value) for value in observed_times.tolist()],
        "critical_indicator": str(args.critical_indicator),
        "critical_window": [float(critical_window_start), float(critical_window_end)],
        "normalization_anchor_time": float(normalize_anchor_time),
        "critical_time_unrestricted": float(per_time_df.iloc[unrestricted_idx]["time"]),
        "critical_time": float(critical_time),
        "critical_action_norm": float(per_time_df.iloc[critical_idx]["Action_norm"]),
        "critical_q_phi_norm": float(per_time_df.iloc[critical_idx]["Q_phi_norm"]),
        "critical_product": float(per_time_df.iloc[critical_idx]["Product"]),
        "anchor_min_time": float(anchor_min_time),
        "terminal_fate_classes": [str(value) for value in terminal_fate_classes.tolist()],
        "overall_terminal_fate_fraction": {str(key): float(value) for key, value in overall_equal.items()},
        "overall_terminal_fate_mass_weighted": {str(key): float(value) for key, value in overall_mass.items()},
        "critical_slice_fate_mass": {
            str(row["eventual_fate"]): float(row["mass_norm"]) for _, row in critical_fate_mass.iterrows()
        },
        "prior": prior,
        "outputs": {
            "per_time_csv": str(per_time_path),
            "terminal_csv": str(terminal_path),
            "per_seed_csv": str(seed_path),
            "critical_slice_csv": str(critical_slice_path),
            "trajectory_csv": str(long_path) if args.trajectory_csv else None,
            "figure_png": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote downstream summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
