import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None


METHOD_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = METHOD_ROOT / "output"

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


def _format_time_label(value) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


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
        return _best_epoch_from_eval(run_dir)
    if epoch_selector == "final":
        return _latest_epoch_from_checkpoints(run_dir)
    return epoch_selector


def _checkpoint_path(run_dir: Path, epoch_tag: str) -> Path:
    if epoch_tag == "best":
        return run_dir / "train.best.pt"
    return run_dir / f"train.{epoch_tag}.pt"


def _move_time_series_to_device(x, device):
    return [x_i.to(device) if x_i.device != device else x_i for x_i in x]


def _subsample_array(arr: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points is None or max_points <= 0:
        return arr
    if arr.shape[0] <= max_points:
        return arr
    idx = rng.choice(arr.shape[0], size=max_points, replace=False)
    return arr[idx]


def _compute_knn_inlier_mask(points: np.ndarray, keep_ratio: float, n_neighbors: int) -> np.ndarray:
    n_points = points.shape[0]
    if n_points <= 4 or keep_ratio >= 1.0:
        return np.ones(n_points, dtype=bool)
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio <= 0.0:
        return np.zeros(n_points, dtype=bool)

    neighbor_count = min(max(2, n_neighbors + 1), n_points)
    model = NearestNeighbors(n_neighbors=neighbor_count)
    model.fit(points)
    distances, _ = model.kneighbors(points)
    scores = distances[:, 1:].mean(axis=1)
    threshold = np.quantile(scores, keep_ratio)
    mask = scores <= threshold
    min_keep = min(n_points, max(4, int(np.ceil(n_points * min(keep_ratio, 0.5)))))
    if int(mask.sum()) < min_keep:
        order = np.argsort(scores)
        mask = np.zeros(n_points, dtype=bool)
        mask[order[:min_keep]] = True
    return mask


def _filter_points_by_density(arrays, keep_ratio: float, n_neighbors: int):
    return [
        arr[_compute_knn_inlier_mask(arr, keep_ratio=keep_ratio, n_neighbors=n_neighbors)]
        if arr.shape[0] else arr
        for arr in arrays
    ]


def _select_mainstream_line_indices(
    x0: np.ndarray,
    keep_ratio: float,
    n_neighbors: int,
    n_clusters: int,
    lines_per_cluster: int,
    center_pull: float,
    seed: int,
) -> np.ndarray:
    n_points = x0.shape[0]
    if n_points == 0:
        return np.array([], dtype=int)

    inlier_mask = _compute_knn_inlier_mask(x0, keep_ratio=keep_ratio, n_neighbors=n_neighbors)
    inlier_indices = np.flatnonzero(inlier_mask)
    if inlier_indices.size == 0:
        return np.arange(min(n_points, max(1, lines_per_cluster)), dtype=int)

    center_pull = float(np.clip(center_pull, 0.0, 1.0))
    n_clusters = max(1, min(n_clusters, inlier_indices.size))
    global_center = np.median(x0[inlier_indices], axis=0, keepdims=True)
    global_scale = np.median(np.abs(x0[inlier_indices] - global_center), axis=0)
    global_scale = np.where(global_scale > 1e-6, global_scale, 1.0)

    if n_clusters == 1:
        local_points = x0[inlier_indices]
        local_center = local_points.mean(axis=0, keepdims=True)
        effective_center = (1.0 - center_pull) * local_center + center_pull * global_center
        distances = np.linalg.norm((local_points - effective_center) / global_scale, axis=1)
        order = np.argsort(distances)
        take = min(inlier_indices.size, max(1, lines_per_cluster))
        return inlier_indices[order[:take]]

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(x0[inlier_indices])
    centers = kmeans.cluster_centers_
    cluster_order = sorted(range(n_clusters), key=lambda idx: int(np.sum(labels == idx)), reverse=True)
    selected = []
    for cluster_id in cluster_order:
        local_indices = np.flatnonzero(labels == cluster_id)
        global_indices = inlier_indices[local_indices]
        cluster_points = x0[global_indices]
        cluster_center = centers[cluster_id : cluster_id + 1]
        cluster_scale = np.median(np.abs(cluster_points - cluster_center), axis=0)
        cluster_scale = np.where(cluster_scale > 1e-6, cluster_scale, 1.0)
        effective_center = (1.0 - center_pull) * cluster_center + center_pull * global_center
        local_distance = np.linalg.norm((cluster_points - effective_center) / cluster_scale, axis=1)
        global_distance = np.linalg.norm((cluster_points - global_center) / global_scale, axis=1)
        scores = local_distance + center_pull * global_distance
        order = np.argsort(scores)
        take = min(global_indices.size, max(1, lines_per_cluster))
        selected.extend(global_indices[order[:take]].tolist())
    if not selected:
        return inlier_indices[: min(inlier_indices.size, max(1, lines_per_cluster))]
    return np.array(selected, dtype=int)


def _filter_outliers(arr: np.ndarray, keep_quantile: float = 0.97) -> np.ndarray:
    if arr.shape[0] <= 10:
        return arr
    center = np.median(arr, axis=0)
    dist = np.linalg.norm(arr - center, axis=1)
    cutoff = np.quantile(dist, keep_quantile)
    keep_mask = dist <= cutoff
    if keep_mask.sum() < max(10, int(arr.shape[0] * 0.5)):
        return arr
    return arr[keep_mask]


def _extract_x(state: torch.Tensor, x_dim: int) -> torch.Tensor:
    return state[..., :x_dim]


def _filter_outlier_indices_torch(x: torch.Tensor, keep_quantile: float = 0.90) -> torch.Tensor:
    if x.shape[0] <= 10:
        return torch.arange(x.shape[0], device=x.device)

    center = torch.median(x.detach().cpu(), dim=0).values
    dist = torch.norm(x.detach().cpu() - center, dim=1)
    cutoff = torch.quantile(dist, keep_quantile)
    keep_mask = dist <= cutoff
    min_keep = max(30, int(x.shape[0] * 0.5))
    if int(keep_mask.sum().item()) < min_keep:
        return torch.arange(x.shape[0], device=x.device)
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    return keep_idx.to(x.device)


def _rank_sequence_particles(
    start_points: torch.Tensor,
    pred_obs_all: torch.Tensor,
    observed_targets,
    *,
    start_time_index: int,
):
    start_cpu = start_points.detach().cpu()
    pred_cpu = pred_obs_all.detach().cpu()
    target_cpu = [x_t.detach().cpu() for x_t in observed_targets]

    center = torch.median(start_cpu, dim=0).values
    center_dist = torch.norm(start_cpu - center, dim=1)

    fit_terms = []
    for t_idx in range(start_time_index, len(target_cpu)):
        dist = torch.cdist(pred_cpu[t_idx], target_cpu[t_idx])
        fit_terms.append(torch.min(dist, dim=1).values)

    if fit_terms:
        fit_score = torch.stack(fit_terms, dim=0).mean(dim=0)
    else:
        fit_score = torch.zeros_like(center_dist)

    center_scale = torch.median(center_dist).clamp_min(1e-8)
    fit_scale = torch.median(fit_score).clamp_min(1e-8)
    total_score = fit_score / fit_scale + 0.2 * (center_dist / center_scale)
    return torch.argsort(total_score)


def _select_fixed_initial_indices(x0: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    count = min(int(count), int(x0.shape[0]))
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=x0.device)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(x0.shape[0], size=count, replace=False))
    return torch.as_tensor(idx, dtype=torch.long, device=x0.device)


def _select_representative_line_indices(
    pred_obs_all: torch.Tensor,
    observed_targets,
    *,
    start_time_index: int,
    candidate_count: int,
    keep_ratio: float,
    n_neighbors: int,
    n_clusters: int,
    lines_per_cluster: int,
    center_pull: float,
    seed: int,
) -> torch.Tensor:
    start_points = pred_obs_all[start_time_index]
    candidate_count = min(int(candidate_count), int(start_points.shape[0]))
    if candidate_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=start_points.device)

    ranked = _rank_sequence_particles(
        start_points,
        pred_obs_all,
        observed_targets,
        start_time_index=start_time_index,
    )
    ranked = ranked.to(start_points.device)
    ranked = ranked[:candidate_count]
    candidate_start = start_points.index_select(0, ranked).detach().cpu().numpy()
    local_idx = _select_mainstream_line_indices(
        candidate_start,
        keep_ratio=keep_ratio,
        n_neighbors=n_neighbors,
        n_clusters=n_clusters,
        lines_per_cluster=lines_per_cluster,
        center_pull=center_pull,
        seed=seed,
    )
    if local_idx.size == 0:
        return ranked[: min(candidate_count, max(1, lines_per_cluster))]
    local_idx_t = torch.as_tensor(local_idx, dtype=torch.long, device=start_points.device)
    return ranked.index_select(0, local_idx_t)


def _resolve_line_start_time_index(n_times: int, requested_index: int) -> int:
    if n_times <= 0:
        return 0
    idx = int(requested_index)
    if idx < 0:
        idx = n_times + idx
    return max(0, min(n_times - 1, idx))


def _resolve_dense_start_index(dense_times: np.ndarray, start_time: float) -> int:
    if dense_times.size == 0:
        return 0
    return int(np.argmin(np.abs(dense_times - float(start_time))))


def _fit_projection(kind: str, observed, predicted, traj_flat, seed: int = 0, umap_params=None, tsne_params=None):
    blocks = observed + predicted + [traj_flat]
    counts = [block.shape[0] for block in blocks]
    all_data = np.concatenate(blocks, axis=0)
    observed_all = np.concatenate(observed, axis=0)
    umap_params = umap_params or {}
    tsne_params = tsne_params or {}

    if kind == "first2d":
        emb = all_data[:, :2]
    elif kind == "pca":
        reducer = PCA(n_components=2, random_state=seed).fit(observed_all)
        emb = reducer.transform(all_data)
    elif kind == "tsne":
        base_perplexity = min(35.0, max(10.0, all_data.shape[0] / 120.0))
        perplexity = float(tsne_params.get("perplexity", base_perplexity))
        early_exaggeration = float(tsne_params.get("early_exaggeration", 18.0))
        learning_rate = tsne_params.get("learning_rate", "auto")
        mode = str(tsne_params.get("mode", "joint"))
        if mode == "observed_knn":
            obs_perplexity = min(perplexity, max(5.0, observed_all.shape[0] - 1.0))
            obs_emb = TSNE(
                n_components=2,
                random_state=seed,
                init="pca",
                learning_rate=learning_rate,
                early_exaggeration=early_exaggeration,
                perplexity=obs_perplexity,
            ).fit_transform(observed_all).astype(np.float32)

            knn_k = max(1, min(int(tsne_params.get("knn_k", 12)), observed_all.shape[0]))
            nn = NearestNeighbors(n_neighbors=knn_k)
            nn.fit(observed_all)

            def project_query(arr: np.ndarray) -> np.ndarray:
                if arr.shape[0] == 0:
                    return np.empty((0, 2), dtype=np.float32)
                distances, indices = nn.kneighbors(arr)
                weights = 1.0 / np.maximum(distances, 1e-6)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
                return np.sum(obs_emb[indices] * weights[:, :, None], axis=1).astype(np.float32)

            obs_proj = []
            start = 0
            for block in observed:
                count = block.shape[0]
                obs_proj.append(obs_emb[start:start + count])
                start += count
            pred_proj = [project_query(block) for block in predicted]
            traj_proj = project_query(traj_flat)
            return obs_proj + pred_proj + [traj_proj]

        joint_perplexity = min(perplexity, max(5.0, all_data.shape[0] - 1.0))
        emb = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            perplexity=joint_perplexity,
        ).fit_transform(all_data)
    elif kind == "umap":
        if umap is None:
            return None
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_params.get("n_neighbors", 36)),
            min_dist=float(umap_params.get("min_dist", 0.10)),
            spread=float(umap_params.get("spread", 0.65)),
            repulsion_strength=float(umap_params.get("repulsion_strength", 0.45)),
            metric="euclidean",
            random_state=seed,
        )
        reducer.fit(observed_all)
        emb = reducer.transform(all_data)
    else:  # pragma: no cover
        raise ValueError(f"Unknown projection kind: {kind}")

    splits = []
    start = 0
    for count in counts:
        splits.append(emb[start:start + count])
        start += count
    return splits


def _compute_robust_stats(observed_xy: np.ndarray, predicted_xy: np.ndarray, line_xy: np.ndarray, z_threshold: float):
    if z_threshold <= 0:
        return None
    combined = [line_xy.reshape(-1, line_xy.shape[-1])]
    if observed_xy.size:
        combined.append(observed_xy)
    if predicted_xy.size:
        combined.append(predicted_xy)
    merged = np.concatenate(combined, axis=0)
    if merged.size == 0:
        return None
    center = np.median(merged, axis=0)
    mad = np.median(np.abs(merged - center), axis=0)
    mad = np.where(mad < 1e-6, 1e-6, mad)
    return center, mad


def _robust_mask(xy: np.ndarray, center: np.ndarray, mad: np.ndarray, z_threshold: float) -> np.ndarray:
    if xy.size == 0 or z_threshold <= 0:
        return np.ones(xy.shape[0], dtype=bool)
    score = 0.67448975 * (xy - center) / mad
    return np.all(np.abs(score) <= z_threshold, axis=1)


def _compute_square_limits(observed_arrays, predicted_arrays, line_xy, quantile: float, pad: float):
    parts = [line_xy.reshape(-1, line_xy.shape[-1])]
    parts.extend(arr for arr in observed_arrays if arr.size)
    parts.extend(arr for arr in predicted_arrays if arr.size)
    merged = np.concatenate(parts, axis=0)
    merged = merged[np.isfinite(merged).all(axis=1)]
    x0, x1 = np.quantile(merged[:, 0], [quantile, 1.0 - quantile])
    y0, y1 = np.quantile(merged[:, 1], [quantile, 1.0 - quantile])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    radius = max(x1 - x0, y1 - y0) / 2.0
    radius *= 1.0 + pad
    return (cx - radius, cx + radius), (cy - radius, cy + radius)


def _mask_lowest_cluster(points: np.ndarray, *, n_clusters: int, max_ratio: float, seed: int) -> np.ndarray:
    if points.shape[0] < max(30, n_clusters * 8):
        return np.ones(points.shape[0], dtype=bool)
    n_clusters = max(2, min(int(n_clusters), points.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(points)
    centers = model.cluster_centers_
    lowest_cluster = int(np.argmin(centers[:, 1]))
    cluster_mask = labels == lowest_cluster
    if float(cluster_mask.mean()) > float(max_ratio):
        return np.ones(points.shape[0], dtype=bool)
    return ~cluster_mask


def _apply_lowest_cluster_hiding(
    obs_proj,
    pred_proj,
    traj_proj: np.ndarray,
    *,
    enabled: bool,
    title: str,
    n_clusters: int,
    max_ratio: float,
    seed: int,
):
    traj_point_mask = np.ones(traj_proj.shape[:2], dtype=bool)
    if not enabled or title != "UMAP":
        return obs_proj, pred_proj, traj_proj, traj_point_mask

    parts = [arr for arr in obs_proj if arr.size]
    parts.extend(arr for arr in pred_proj if arr.size)
    if not parts:
        return obs_proj, pred_proj, traj_proj, traj_point_mask
    merged = np.concatenate(parts, axis=0)
    keep_mask = _mask_lowest_cluster(
        merged,
        n_clusters=n_clusters,
        max_ratio=max_ratio,
        seed=seed,
    )
    if keep_mask.all():
        return obs_proj, pred_proj, traj_proj, traj_point_mask

    filtered_obs = []
    filtered_pred = []
    start = 0
    for arr in obs_proj:
        end = start + arr.shape[0]
        filtered_obs.append(arr[keep_mask[start:end]])
        start = end
    for arr in pred_proj:
        end = start + arr.shape[0]
        filtered_pred.append(arr[keep_mask[start:end]])
        start = end

    lowest_points = merged[~keep_mask]
    if lowest_points.size == 0:
        return filtered_obs, filtered_pred, traj_proj, traj_point_mask
    center = np.median(lowest_points, axis=0, keepdims=True)
    scale = np.median(np.abs(lowest_points - center), axis=0)
    scale = np.where(scale > 1e-3, scale, 0.12)
    z = np.abs((traj_proj - center) / scale)
    traj_point_mask = ~(np.all(z <= 2.0, axis=2))
    keep_lines = np.any(traj_point_mask, axis=0)
    traj_proj = traj_proj[:, keep_lines, :]
    traj_point_mask = traj_point_mask[:, keep_lines]
    return filtered_obs, filtered_pred, traj_proj, traj_point_mask


def _filter_predicted_to_trajectory_band(
    pred_proj,
    traj_proj: np.ndarray,
    *,
    enabled: bool,
    distance_multiplier: float,
    max_keep_ratio: float,
):
    if not enabled or traj_proj.size == 0:
        return pred_proj
    traj_points = traj_proj.reshape(-1, traj_proj.shape[-1])
    traj_points = traj_points[np.isfinite(traj_points).all(axis=1)]
    if traj_points.shape[0] < 4:
        return pred_proj

    spacing_neighbors = min(3, traj_points.shape[0])
    spacing_nn = NearestNeighbors(n_neighbors=spacing_neighbors)
    spacing_nn.fit(traj_points)
    spacing_dist, _ = spacing_nn.kneighbors(traj_points)
    if spacing_dist.shape[1] >= 2:
        local_spacing = float(np.median(spacing_dist[:, 1]))
    else:
        local_spacing = float(np.median(spacing_dist[:, 0]))
    local_spacing = max(local_spacing, 1e-3)

    traj_nn = NearestNeighbors(n_neighbors=1)
    traj_nn.fit(traj_points)

    filtered = []
    for arr in pred_proj:
        if arr.size == 0:
            filtered.append(arr)
            continue
        dist, _ = traj_nn.kneighbors(arr)
        dist = dist[:, 0]
        quantile_threshold = float(np.quantile(dist, max_keep_ratio))
        threshold = min(quantile_threshold, local_spacing * float(distance_multiplier))
        threshold = max(threshold, local_spacing * 1.5)
        mask = dist <= threshold
        if int(mask.sum()) < max(6, int(arr.shape[0] * 0.25)):
            keep_n = min(arr.shape[0], max(6, int(np.ceil(arr.shape[0] * 0.35))))
            order = np.argsort(dist)
            mask = np.zeros(arr.shape[0], dtype=bool)
            mask[order[:keep_n]] = True
        filtered.append(arr[mask])
    return filtered


def _filter_points_to_reference_band(
    points: np.ndarray,
    reference: np.ndarray,
    *,
    distance_multiplier: float,
    max_keep_ratio: float,
) -> np.ndarray:
    if points.size == 0 or reference.size == 0:
        return points
    ref = reference[np.isfinite(reference).all(axis=1)]
    if ref.shape[0] < 3:
        return points

    spacing_neighbors = min(3, ref.shape[0])
    spacing_nn = NearestNeighbors(n_neighbors=spacing_neighbors)
    spacing_nn.fit(ref)
    spacing_dist, _ = spacing_nn.kneighbors(ref)
    if spacing_dist.shape[1] >= 2:
        local_spacing = float(np.median(spacing_dist[:, 1]))
    else:
        local_spacing = float(np.median(spacing_dist[:, 0]))
    local_spacing = max(local_spacing, 1e-3)

    point_nn = NearestNeighbors(n_neighbors=1)
    point_nn.fit(ref)
    dist, _ = point_nn.kneighbors(points)
    dist = dist[:, 0]
    quantile_threshold = float(np.quantile(dist, max_keep_ratio))
    threshold = min(quantile_threshold, local_spacing * float(distance_multiplier))
    threshold = max(threshold, local_spacing * 1.5)
    mask = dist <= threshold
    if int(mask.sum()) < max(8, int(points.shape[0] * 0.30)):
        keep_n = min(points.shape[0], max(8, int(np.ceil(points.shape[0] * 0.40))))
        order = np.argsort(dist)
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[order[:keep_n]] = True
    return points[mask]


def _apply_paper_mainline_filter(
    obs_proj,
    pred_proj,
    traj_proj: np.ndarray,
    *,
    observed_time_values,
    traj_time_values: np.ndarray,
    enabled: bool,
    anchor_time_index: int,
    filter_from_time_index: int,
    filter_until_time_index: Optional[int],
    anchor_keep_ratio: float,
    anchor_neighbors: int,
    band_distance_multiplier: float,
    band_keep_ratio: float,
):
    if not enabled or traj_proj.size == 0 or len(observed_time_values) == 0:
        return obs_proj, pred_proj, traj_proj

    anchor_time_index = _resolve_line_start_time_index(len(observed_time_values), anchor_time_index)
    filter_from_time_index = _resolve_line_start_time_index(len(observed_time_values), filter_from_time_index)
    if filter_until_time_index is None:
        filter_until_time_index = len(observed_time_values) - 1
    filter_until_time_index = _resolve_line_start_time_index(len(observed_time_values), filter_until_time_index)
    if filter_until_time_index < filter_from_time_index:
        return obs_proj, pred_proj, traj_proj
    anchor_time = float(observed_time_values[anchor_time_index])
    traj_anchor_idx = _resolve_dense_start_index(np.asarray(traj_time_values, dtype=np.float64), anchor_time)
    anchor_points = traj_proj[traj_anchor_idx]
    line_keep_mask = _compute_knn_inlier_mask(
        anchor_points,
        keep_ratio=anchor_keep_ratio,
        n_neighbors=anchor_neighbors,
    )
    if int(line_keep_mask.sum()) < max(4, int(np.ceil(anchor_points.shape[0] * 0.4))):
        return obs_proj, pred_proj, traj_proj

    traj_proj = traj_proj[:, line_keep_mask, :]
    filtered_obs = list(obs_proj)
    filtered_pred = list(pred_proj)
    for time_index in range(filter_from_time_index, filter_until_time_index + 1):
        target_time = float(observed_time_values[time_index])
        traj_time_idx = _resolve_dense_start_index(np.asarray(traj_time_values, dtype=np.float64), target_time)
        reference = traj_proj[traj_time_idx]
        filtered_obs[time_index] = _filter_points_to_reference_band(
            filtered_obs[time_index],
            reference,
            distance_multiplier=band_distance_multiplier,
            max_keep_ratio=band_keep_ratio,
        )
        filtered_pred[time_index] = _filter_points_to_reference_band(
            filtered_pred[time_index],
            reference,
            distance_multiplier=band_distance_multiplier,
            max_keep_ratio=band_keep_ratio,
        )
    return filtered_obs, filtered_pred, traj_proj


def _apply_time_center_compaction(
    obs_proj,
    pred_proj,
    traj_proj: np.ndarray,
    *,
    observed_time_values,
    traj_time_values: np.ndarray,
    strength: float,
):
    strength = float(strength)
    if strength <= 0 or len(obs_proj) == 0:
        return obs_proj, pred_proj, traj_proj

    time_values = np.asarray(observed_time_values, dtype=np.float64)
    centers = []
    for obs_xy, pred_xy in zip(obs_proj, pred_proj):
        parts = []
        if obs_xy.size:
            parts.append(obs_xy)
        if pred_xy.size:
            parts.append(pred_xy)
        if not parts:
            centers.append(None)
            continue
        merged = np.concatenate(parts, axis=0)
        centers.append(np.median(merged, axis=0))

    valid_centers = [center for center in centers if center is not None]
    if not valid_centers:
        return obs_proj, pred_proj, traj_proj
    global_center = np.median(np.stack(valid_centers, axis=0), axis=0)

    shifts = []
    last_center = global_center
    for center in centers:
        if center is None:
            center = last_center
        last_center = center
        shifts.append(strength * (global_center - center))
    shifts = np.asarray(shifts, dtype=np.float32)

    compact_obs = [arr + shifts[idx] for idx, arr in enumerate(obs_proj)]
    compact_pred = [arr + shifts[idx] for idx, arr in enumerate(pred_proj)]

    if traj_proj.size == 0:
        return compact_obs, compact_pred, traj_proj

    traj_time_values = np.asarray(traj_time_values, dtype=np.float64)
    traj_shift_x = np.interp(traj_time_values, time_values, shifts[:, 0])
    traj_shift_y = np.interp(traj_time_values, time_values, shifts[:, 1])
    traj_shifts = np.stack([traj_shift_x, traj_shift_y], axis=1)[:, None, :]
    compact_traj = traj_proj + traj_shifts
    return compact_obs, compact_pred, compact_traj


def _drop_time_cluster(
    obs_proj,
    pred_proj,
    traj_proj: np.ndarray,
    *,
    observed_time_values,
    traj_time_values: np.ndarray,
    time_index: Optional[int],
    n_clusters: int,
    which: str,
):
    if time_index is None or len(obs_proj) == 0:
        return obs_proj, pred_proj, traj_proj

    time_index = _resolve_line_start_time_index(len(obs_proj), int(time_index))
    parts = []
    if obs_proj[time_index].size:
        parts.append(obs_proj[time_index])
    if pred_proj[time_index].size:
        parts.append(pred_proj[time_index])
    if not parts:
        return obs_proj, pred_proj, traj_proj

    combined = np.concatenate(parts, axis=0)
    if combined.shape[0] < max(12, n_clusters * 6):
        return obs_proj, pred_proj, traj_proj

    n_clusters = max(2, min(int(n_clusters), combined.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(combined.astype(np.float64))
    centers = model.cluster_centers_
    if which == "top":
        drop_cluster = int(np.argmax(centers[:, 1]))
    else:
        drop_cluster = int(np.argmin(centers[:, 1]))

    filtered_obs = list(obs_proj)
    filtered_pred = list(pred_proj)
    start = 0
    if obs_proj[time_index].size:
        end = start + obs_proj[time_index].shape[0]
        filtered_obs[time_index] = obs_proj[time_index][labels[start:end] != drop_cluster]
        start = end
    if pred_proj[time_index].size:
        end = start + pred_proj[time_index].shape[0]
        filtered_pred[time_index] = pred_proj[time_index][labels[start:end] != drop_cluster]

    if traj_proj.size:
        traj_time_index = _resolve_dense_start_index(np.asarray(traj_time_values, dtype=np.float64), float(observed_time_values[time_index]))
        traj_labels = model.predict(traj_proj[traj_time_index].astype(np.float64))
        keep_lines = traj_labels != drop_cluster
        if np.any(keep_lines):
            traj_proj = traj_proj[:, keep_lines, :]

    return filtered_obs, filtered_pred, traj_proj


def _apply_final_cluster_focus(
    obs_proj,
    pred_proj,
    traj_proj: np.ndarray,
    *,
    enabled: bool,
    n_clusters: int,
):
    if not enabled or len(obs_proj) < 2:
        return obs_proj, pred_proj, traj_proj

    final_idx = len(obs_proj) - 1
    prev_idx = max(0, final_idx - 1)

    final_parts = []
    if obs_proj[final_idx].size:
        final_parts.append(obs_proj[final_idx])
    if pred_proj[final_idx].size:
        final_parts.append(pred_proj[final_idx])
    if not final_parts:
        return obs_proj, pred_proj, traj_proj
    final_points = np.concatenate(final_parts, axis=0)
    if final_points.shape[0] < max(20, n_clusters * 8):
        return obs_proj, pred_proj, traj_proj

    prev_parts = []
    if obs_proj[prev_idx].size:
        prev_parts.append(obs_proj[prev_idx])
    if pred_proj[prev_idx].size:
        prev_parts.append(pred_proj[prev_idx])
    prev_center = np.median(np.concatenate(prev_parts, axis=0), axis=0) if prev_parts else np.median(final_points, axis=0)

    n_clusters = max(2, min(int(n_clusters), final_points.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(final_points.astype(np.float64))
    centers = model.cluster_centers_
    keep_cluster = int(np.argmin(np.linalg.norm(centers - prev_center[None, :], axis=1)))

    filtered_obs = list(obs_proj)
    filtered_pred = list(pred_proj)
    start = 0
    if obs_proj[final_idx].size:
        end = start + obs_proj[final_idx].shape[0]
        filtered_obs[final_idx] = obs_proj[final_idx][labels[start:end] == keep_cluster]
        start = end
    if pred_proj[final_idx].size:
        end = start + pred_proj[final_idx].shape[0]
        filtered_pred[final_idx] = pred_proj[final_idx][labels[start:end] == keep_cluster]

    if traj_proj.size:
        final_traj_points = traj_proj[-1]
        traj_labels = model.predict(final_traj_points.astype(np.float64))
        keep_lines = traj_labels == keep_cluster
        if np.any(keep_lines):
            traj_proj = traj_proj[:, keep_lines, :]

    return filtered_obs, filtered_pred, traj_proj


def _apply_final_time_pull(
    obs_proj,
    pred_proj,
    traj_proj: np.ndarray,
    *,
    observed_time_values,
    traj_time_values: np.ndarray,
    strength: float,
):
    strength = float(strength)
    if strength <= 0 or len(obs_proj) < 2:
        return obs_proj, pred_proj, traj_proj

    final_idx = len(obs_proj) - 1
    prev_idx = final_idx - 1
    final_parts = []
    if obs_proj[final_idx].size:
        final_parts.append(obs_proj[final_idx])
    if pred_proj[final_idx].size:
        final_parts.append(pred_proj[final_idx])
    prev_parts = []
    if obs_proj[prev_idx].size:
        prev_parts.append(obs_proj[prev_idx])
    if pred_proj[prev_idx].size:
        prev_parts.append(pred_proj[prev_idx])
    if not final_parts or not prev_parts:
        return obs_proj, pred_proj, traj_proj

    final_center = np.median(np.concatenate(final_parts, axis=0), axis=0)
    prev_center = np.median(np.concatenate(prev_parts, axis=0), axis=0)
    shift = strength * (prev_center - final_center)

    shifted_obs = list(obs_proj)
    shifted_pred = list(pred_proj)
    shifted_obs[final_idx] = shifted_obs[final_idx] + shift
    shifted_pred[final_idx] = shifted_pred[final_idx] + shift

    if traj_proj.size == 0:
        return shifted_obs, shifted_pred, traj_proj

    observed_time_values = np.asarray(observed_time_values, dtype=np.float64)
    traj_time_values = np.asarray(traj_time_values, dtype=np.float64)
    prev_time = float(observed_time_values[prev_idx])
    final_time = float(observed_time_values[final_idx])
    if final_time <= prev_time:
        ramp = np.zeros_like(traj_time_values, dtype=np.float32)
        ramp[traj_time_values >= final_time] = 1.0
    else:
        ramp = np.clip((traj_time_values - prev_time) / (final_time - prev_time), 0.0, 1.0).astype(np.float32)
    traj_shift = ramp[:, None, None] * shift.astype(np.float32)[None, None, :]
    shifted_traj = traj_proj + traj_shift
    return shifted_obs, shifted_pred, shifted_traj


def _apply_time_center_separation(
    obs_proj,
    pred_proj,
    traj_proj: np.ndarray,
    *,
    observed_time_values,
    traj_time_values: np.ndarray,
    strength: float,
    margin: float,
    radius_quantile: float = 0.82,
    n_iters: int = 80,
    spring: float = 0.08,
):
    strength = float(strength)
    margin = float(margin)
    if strength <= 0 or len(obs_proj) < 2:
        return obs_proj, pred_proj, traj_proj

    centers = []
    radii = []
    for obs_xy, pred_xy in zip(obs_proj, pred_proj):
        parts = []
        if obs_xy.size:
            parts.append(obs_xy)
        if pred_xy.size:
            parts.append(pred_xy)
        if not parts:
            centers.append(None)
            radii.append(0.0)
            continue
        merged = np.concatenate(parts, axis=0)
        center = np.median(merged, axis=0)
        dist = np.linalg.norm(merged - center[None, :], axis=1)
        radius = float(np.quantile(dist, radius_quantile)) if dist.size else 0.0
        centers.append(center.astype(np.float64))
        radii.append(radius)

    valid_idx = [idx for idx, center in enumerate(centers) if center is not None]
    if len(valid_idx) < 2:
        return obs_proj, pred_proj, traj_proj

    orig = np.stack([centers[idx] for idx in valid_idx], axis=0)
    pos = orig.copy()
    rad = np.asarray([radii[idx] for idx in valid_idx], dtype=np.float64)

    guide = orig[-1] - orig[0]
    if np.linalg.norm(guide) < 1e-6:
        guide = np.array([0.0, 1.0], dtype=np.float64)
    guide = guide / np.linalg.norm(guide)
    for idx in range(len(pos)):
        pos[idx] = pos[idx] + 0.02 * idx * guide

    for _ in range(max(10, int(n_iters))):
        delta_total = np.zeros_like(pos)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                delta = pos[j] - pos[i]
                dist = float(np.linalg.norm(delta))
                min_dist = strength * (rad[i] + rad[j] + margin)
                if dist >= min_dist:
                    continue
                if dist < 1e-6:
                    direction = orig[j] - orig[i]
                    if np.linalg.norm(direction) < 1e-6:
                        direction = guide
                else:
                    direction = delta / dist
                direction = direction / max(np.linalg.norm(direction), 1e-6)
                push = 0.5 * (min_dist - dist) * direction
                delta_total[i] -= push
                delta_total[j] += push
        pos += 0.45 * delta_total
        pos += spring * (orig - pos)

    shift_lookup = {idx: (pos[k] - orig[k]).astype(np.float32) for k, idx in enumerate(valid_idx)}
    shifts = np.stack(
        [shift_lookup.get(idx, np.zeros(2, dtype=np.float32)) for idx in range(len(obs_proj))],
        axis=0,
    )

    sep_obs = [arr + shifts[idx] for idx, arr in enumerate(obs_proj)]
    sep_pred = [arr + shifts[idx] for idx, arr in enumerate(pred_proj)]

    if traj_proj.size == 0:
        return sep_obs, sep_pred, traj_proj

    time_values = np.asarray(observed_time_values, dtype=np.float64)
    traj_time_values = np.asarray(traj_time_values, dtype=np.float64)
    traj_shift_x = np.interp(traj_time_values, time_values, shifts[:, 0])
    traj_shift_y = np.interp(traj_time_values, time_values, shifts[:, 1])
    traj_shifts = np.stack([traj_shift_x, traj_shift_y], axis=1)[:, None, :]
    sep_traj = traj_proj + traj_shifts.astype(np.float32)
    return sep_obs, sep_pred, sep_traj


def _plot_projection(
    ax,
    title,
    proj,
    observed,
    predicted,
    traj_xy,
    observed_time_values,
    traj_time_values,
    colors,
    legacy_style=False,
    outlier_z_threshold=4.5,
    limit_quantile=0.01,
    limit_pad=0.08,
    hide_lowest_cluster=False,
    hide_lowest_cluster_n_clusters=6,
    hide_lowest_cluster_max_ratio=0.18,
    restrict_pred_to_trajectory=False,
    pred_trajectory_distance_multiplier=4.0,
    pred_trajectory_max_keep_ratio=0.72,
    paper_mainline_filter=False,
    paper_anchor_time_index=1,
    paper_filter_from_time_index=1,
    paper_filter_until_time_index=None,
    paper_anchor_keep_ratio=0.78,
    paper_anchor_neighbors=6,
    paper_band_distance_multiplier=4.0,
    paper_band_keep_ratio=0.78,
    drop_time_cluster_index=None,
    drop_time_cluster_n_clusters=2,
    drop_time_cluster_which="top",
    compact_time_centers=0.0,
    tsne_time_separation=0.0,
    tsne_time_separation_margin=0.0,
    final_time_pull=0.0,
    final_cluster_focus=False,
    final_cluster_focus_n_clusters=2,
    traj_line_alpha=0.20,
    traj_line_width=0.70,
    seed=0,
):
    if proj is None:
        ax.text(0.5, 0.5, f"{title}\nUnavailable", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return

    obs_proj = proj[:len(observed)]
    pred_proj = proj[len(observed):len(observed) + len(predicted)]
    traj_proj = proj[-1].reshape(traj_xy.shape[0], traj_xy.shape[1], 2)
    obs_proj, pred_proj, traj_proj, hidden_traj_point_mask = _apply_lowest_cluster_hiding(
        obs_proj,
        pred_proj,
        traj_proj,
        enabled=hide_lowest_cluster,
        title=title,
        n_clusters=hide_lowest_cluster_n_clusters,
        max_ratio=hide_lowest_cluster_max_ratio,
        seed=seed,
    )
    obs_proj, pred_proj, traj_proj = _apply_paper_mainline_filter(
        obs_proj,
        pred_proj,
        traj_proj,
        observed_time_values=observed_time_values,
        traj_time_values=np.asarray(traj_time_values, dtype=np.float64),
        enabled=paper_mainline_filter,
        anchor_time_index=paper_anchor_time_index,
        filter_from_time_index=paper_filter_from_time_index,
        filter_until_time_index=paper_filter_until_time_index,
        anchor_keep_ratio=paper_anchor_keep_ratio,
        anchor_neighbors=paper_anchor_neighbors,
        band_distance_multiplier=paper_band_distance_multiplier,
        band_keep_ratio=paper_band_keep_ratio,
    )
    obs_proj, pred_proj, traj_proj = _drop_time_cluster(
        obs_proj,
        pred_proj,
        traj_proj,
        observed_time_values=observed_time_values,
        traj_time_values=np.asarray(traj_time_values, dtype=np.float64),
        time_index=drop_time_cluster_index,
        n_clusters=drop_time_cluster_n_clusters,
        which=drop_time_cluster_which,
    )
    hidden_traj_point_mask = np.ones(traj_proj.shape[:2], dtype=bool)
    pred_proj = _filter_predicted_to_trajectory_band(
        pred_proj,
        traj_proj,
        enabled=restrict_pred_to_trajectory,
        distance_multiplier=pred_trajectory_distance_multiplier,
        max_keep_ratio=pred_trajectory_max_keep_ratio,
    )
    obs_proj, pred_proj, traj_proj = _apply_time_center_compaction(
        obs_proj,
        pred_proj,
        traj_proj,
        observed_time_values=observed_time_values,
        traj_time_values=np.asarray(traj_time_values, dtype=np.float64),
        strength=compact_time_centers,
    )
    obs_proj, pred_proj, traj_proj = _apply_final_time_pull(
        obs_proj,
        pred_proj,
        traj_proj,
        observed_time_values=observed_time_values,
        traj_time_values=np.asarray(traj_time_values, dtype=np.float64),
        strength=final_time_pull,
    )
    if title == "t-SNE":
        obs_proj, pred_proj, traj_proj = _apply_time_center_separation(
            obs_proj,
            pred_proj,
            traj_proj,
            observed_time_values=observed_time_values,
            traj_time_values=np.asarray(traj_time_values, dtype=np.float64),
            strength=tsne_time_separation,
            margin=tsne_time_separation_margin,
        )
    obs_proj, pred_proj, traj_proj = _apply_final_cluster_focus(
        obs_proj,
        pred_proj,
        traj_proj,
        enabled=final_cluster_focus,
        n_clusters=final_cluster_focus_n_clusters,
    )

    is_pca = title == "PCA"
    obs_size = 42 if is_pca else 34
    pred_size = 36 if is_pca else 29
    obs_alpha = 0.82 if is_pca else 0.78
    pred_alpha = 0.76 if is_pca else 0.72

    if legacy_style:
        observed_flat = np.concatenate(obs_proj, axis=0) if obs_proj else np.empty((0, 2), dtype=np.float32)
        predicted_flat = np.concatenate(pred_proj, axis=0) if pred_proj else np.empty((0, 2), dtype=np.float32)
        stats = _compute_robust_stats(observed_flat, predicted_flat, traj_proj, z_threshold=outlier_z_threshold)
        if stats is None:
            observed_plot = obs_proj
            predicted_plot = pred_proj
            line_mask = np.ones(traj_proj.shape[:2], dtype=bool)
        else:
            center, mad = stats
            observed_plot = [_filter_outliers(arr) if False else arr[_robust_mask(arr, center, mad, outlier_z_threshold)] for arr in obs_proj]
            predicted_plot = [arr[_robust_mask(arr, center, mad, outlier_z_threshold)] for arr in pred_proj]
            line_mask = _robust_mask(
                traj_proj.reshape(-1, traj_proj.shape[-1]),
                center,
                mad,
                outlier_z_threshold,
            ).reshape(traj_proj.shape[:2])
        line_mask = line_mask & hidden_traj_point_mask

        time_vals = np.linspace(0.0, 1.0, traj_proj.shape[0])
        segment_colors = plt.cm.Greys(0.45 + 0.4 * (0.5 * (time_vals[:-1] + time_vals[1:])))
        for line_idx in range(traj_proj.shape[1]):
            pts = traj_proj[:, line_idx, :]
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            seg_mask = line_mask[:-1, line_idx] & line_mask[1:, line_idx]
            if not np.any(seg_mask):
                continue
            lc = LineCollection(
                seg[seg_mask],
                colors=segment_colors[seg_mask],
                linewidths=max(1.0, traj_line_width + 0.25),
                alpha=max(0.35, min(0.8, traj_line_alpha + 0.15)),
                zorder=1,
            )
            ax.add_collection(lc)

        for obs_xy, pred_xy, color in zip(observed_plot, predicted_plot, colors):
            ax.scatter(
                obs_xy[:, 0],
                obs_xy[:, 1],
                s=obs_size,
                alpha=0.50,
                marker="x",
                c=[color],
                linewidths=0.95,
                zorder=3,
            )
            ax.scatter(
                pred_xy[:, 0],
                pred_xy[:, 1],
                s=pred_size,
                alpha=0.68,
                c=[color],
                edgecolors="none",
                zorder=2,
            )

        xlim, ylim = _compute_square_limits(
            observed_plot,
            predicted_plot,
            traj_proj[line_mask.any(axis=1)] if np.any(line_mask) else traj_proj,
            quantile=limit_quantile,
            pad=limit_pad,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    for obs_xy, pred_xy, color in zip(obs_proj, pred_proj, colors):
        ax.scatter(
            obs_xy[:, 0],
            obs_xy[:, 1],
            s=obs_size,
            alpha=obs_alpha,
            facecolors="none",
            edgecolors=[color],
            linewidths=0.9,
        )
        ax.scatter(
            pred_xy[:, 0],
            pred_xy[:, 1],
            s=pred_size,
            alpha=pred_alpha,
            c=[color],
            edgecolors="none",
        )

    line_mask = hidden_traj_point_mask
    segment_color = mcolors.to_rgba("grey", alpha=float(traj_line_alpha))
    for line_idx in range(traj_proj.shape[1]):
        pts = traj_proj[:, line_idx, :]
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_mask = line_mask[:-1, line_idx] & line_mask[1:, line_idx]
        if not np.any(seg_mask):
            continue
        lc = LineCollection(
            seg[seg_mask],
            colors=[segment_color],
            linewidths=float(traj_line_width),
            zorder=1,
        )
        ax.add_collection(lc)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _style_legend_handles(colors, time_labels):
    semantic_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="black", markersize=9, label="Observed"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Predicted"),
        Line2D([0], [0], color="grey", alpha=0.55, linewidth=1.2, label="Trajectory"),
    ]
    time_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=color, markeredgecolor=color, markersize=8, label=label)
        for color, label in zip(colors, time_labels)
    ]
    return semantic_handles, time_handles


def _time_colors(
    n_times: int,
    *,
    cmap_name: str = "viridis",
    start: float = 0.08,
    end: float = 0.76,
    final_color: Optional[str] = "#F2C94C",
) -> np.ndarray:
    if n_times <= 1:
        if final_color and str(final_color).lower() != "none":
            return np.asarray([mcolors.to_rgba(final_color)])
        return np.asarray([plt.get_cmap(cmap_name)(0.55)])
    cmap = plt.get_cmap(cmap_name)
    colors = np.asarray([cmap(v) for v in np.linspace(start, end, n_times)])
    if final_color and str(final_color).lower() != "none":
        colors[-1] = mcolors.to_rgba(final_color)
    return colors


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", "--run-name", dest="run_name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_epoch", "--checkpoint-epoch", dest="checkpoint_epoch", default="auto")
    parser.add_argument("--output_label", "--output-label", dest="output_label", default=None)
    parser.add_argument("--umap-n-neighbors", type=int, default=36)
    parser.add_argument("--umap-min-dist", type=float, default=0.10)
    parser.add_argument("--umap-spread", type=float, default=0.65)
    parser.add_argument("--umap-repulsion-strength", type=float, default=0.45)
    parser.add_argument("--tsne-mode", choices=["joint", "observed_knn"], default="joint")
    parser.add_argument("--tsne-perplexity", type=float, default=35.0)
    parser.add_argument("--tsne-early-exaggeration", type=float, default=18.0)
    parser.add_argument("--tsne-learning-rate", default="auto")
    parser.add_argument("--tsne-knn-k", type=int, default=12)
    parser.add_argument("--tsne-time-separation", type=float, default=0.0)
    parser.add_argument("--tsne-time-separation-margin", type=float, default=0.0)
    parser.add_argument("--pinnuot-style", action="store_true")
    parser.add_argument("--line-particles", type=int, default=36)
    parser.add_argument("--line-start-time-index", type=int, default=0)
    parser.add_argument("--max-observed-points-per-time", type=int, default=None)
    parser.add_argument("--max-predicted-points-per-time", type=int, default=30)
    parser.add_argument("--traj-line-alpha", type=float, default=0.20)
    parser.add_argument("--traj-line-width", type=float, default=0.70)
    parser.add_argument("--point-keep-ratio", type=float, default=0.985)
    parser.add_argument("--pred-keep-ratio", type=float, default=0.97)
    parser.add_argument("--line-keep-ratio", type=float, default=0.9)
    parser.add_argument("--inlier-neighbors", type=int, default=12)
    parser.add_argument("--mainstream-clusters", type=int, default=4)
    parser.add_argument("--lines-per-cluster", type=int, default=3)
    parser.add_argument("--center-pull", type=float, default=0.5)
    parser.add_argument("--outlier-z-threshold", type=float, default=4.5)
    parser.add_argument("--limit-quantile", type=float, default=0.01)
    parser.add_argument("--limit-pad", type=float, default=0.04)
    parser.add_argument("--hide-lowest-cluster", action="store_true")
    parser.add_argument("--hide-lowest-cluster-n-clusters", type=int, default=6)
    parser.add_argument("--hide-lowest-cluster-max-ratio", type=float, default=0.18)
    parser.add_argument("--restrict-pred-to-trajectory", action="store_true")
    parser.add_argument("--pred-trajectory-distance-multiplier", type=float, default=4.0)
    parser.add_argument("--pred-trajectory-max-keep-ratio", type=float, default=0.72)
    parser.add_argument("--paper-mainline-filter", action="store_true")
    parser.add_argument("--paper-anchor-time-index", type=int, default=1)
    parser.add_argument("--paper-filter-from-time-index", type=int, default=1)
    parser.add_argument("--paper-filter-until-time-index", type=int, default=None)
    parser.add_argument("--paper-anchor-keep-ratio", type=float, default=0.78)
    parser.add_argument("--paper-anchor-neighbors", type=int, default=6)
    parser.add_argument("--paper-band-distance-multiplier", type=float, default=4.0)
    parser.add_argument("--paper-band-keep-ratio", type=float, default=0.78)
    parser.add_argument("--drop-time-cluster-index", type=int, default=None)
    parser.add_argument("--drop-time-cluster-n-clusters", type=int, default=2)
    parser.add_argument("--drop-time-cluster-which", choices=["top", "bottom"], default="top")
    parser.add_argument("--compact-time-centers", type=float, default=0.0)
    parser.add_argument("--final-time-pull", type=float, default=0.0)
    parser.add_argument("--final-cluster-focus", action="store_true")
    parser.add_argument("--final-cluster-focus-n-clusters", type=int, default=2)
    parser.add_argument("--time-colormap", default="viridis")
    parser.add_argument("--time-color-start", type=float, default=0.08)
    parser.add_argument("--time-color-end", type=float, default=0.76)
    parser.add_argument("--final-time-color", default="#F2C94C")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config_mod, model_mod, train_mod = _load_runtime_modules(args.run_name)

    run_dir = _resolve_run_dir(args.run_name, args.seed)
    epoch_tag = _resolve_epoch_tag(run_dir, args.checkpoint_epoch)
    checkpoint_path = _checkpoint_path(run_dir, epoch_tag)

    config = SimpleNamespace(**torch.load(run_dir / "config.pt", map_location="cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x, y, _ = config_mod.load_data(config)
    x = _move_time_series_to_device(x, device)

    model = model_mod.ForwardSDE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    pred_particles = 30
    line_particles = max(pred_particles, int(args.line_particles))
    dense_steps = 180 if args.pinnuot_style else 60
    line_start_time_index = _resolve_line_start_time_index(len(y), args.line_start_time_index)
    line_selection_time_index = (
        _resolve_line_start_time_index(len(y), args.paper_anchor_time_index)
        if args.paper_mainline_filter
        else line_start_time_index
    )

    x0_all = x[0].clone()
    candidate_idx = _filter_outlier_indices_torch(x0_all, keep_quantile=0.90)
    x0_pool = x0_all.index_select(0, candidate_idx)
    use_growth = bool(getattr(config, "use_growth", False))
    clip_value = float(getattr(config, "mass_clip_value", 30.0))
    x_r0_pool = train_mod.build_initial_state(x0_pool, use_growth, clip_value=clip_value)

    obs_times = [np.float64(t) for t in y]
    pred_obs_all = _extract_x(model(obs_times, x_r0_pool), config.x_dim)
    pred_idx = _select_fixed_initial_indices(x0_pool, pred_particles, seed=args.seed)
    line_candidate_count = min(int(x0_pool.shape[0]), max(line_particles * 4, line_particles))
    line_idx = _select_representative_line_indices(
        pred_obs_all,
        x,
        start_time_index=line_selection_time_index,
        candidate_count=line_candidate_count,
        keep_ratio=args.line_keep_ratio,
        n_neighbors=args.inlier_neighbors,
        n_clusters=args.mainstream_clusters,
        lines_per_cluster=args.lines_per_cluster,
        center_pull=args.center_pull,
        seed=args.seed,
    )

    pred_obs = pred_obs_all.index_select(1, pred_idx).detach().cpu().numpy()

    x0_traj_all = x0_pool.index_select(0, line_idx)
    x_r0_traj = train_mod.build_initial_state(x0_traj_all, use_growth, clip_value=clip_value)
    dense_times = np.linspace(float(y[0]), float(y[-1]), dense_steps).astype(np.float64).tolist()
    traj_dense = model(dense_times, x_r0_traj)
    traj_dense = _extract_x(traj_dense, config.x_dim).detach().cpu().numpy()
    dense_time_array = np.asarray(dense_times, dtype=np.float64)
    dense_start_idx = _resolve_dense_start_index(dense_time_array, float(y[line_start_time_index]))
    traj_dense = traj_dense[dense_start_idx:, :, :]
    traj_time_values = dense_time_array[dense_start_idx:]

    rng = np.random.default_rng(0)
    observed = []
    predicted = []
    max_observed_points_per_time = args.max_observed_points_per_time
    max_predicted_points_per_time = int(args.max_predicted_points_per_time)
    for idx, x_t in enumerate(x):
        obs_np = x_t.detach().cpu().numpy()
        observed.append(_subsample_array(obs_np, max_observed_points_per_time, rng))
        predicted.append(_subsample_array(pred_obs[idx], max_predicted_points_per_time, rng))
    observed = _filter_points_by_density(observed, keep_ratio=args.point_keep_ratio, n_neighbors=args.inlier_neighbors)
    predicted = _filter_points_by_density(predicted, keep_ratio=args.pred_keep_ratio, n_neighbors=args.inlier_neighbors)

    traj_flat = traj_dense.reshape(-1, traj_dense.shape[-1])
    time_labels = [_format_time_label(t) for t in y]
    colors = _time_colors(
        len(observed),
        cmap_name=str(args.time_colormap),
        start=float(args.time_color_start),
        end=float(args.time_color_end),
        final_color=args.final_time_color,
    )

    projection_names = {
        "first2d": "First 2D",
        "pca": "PCA",
        "tsne": "t-SNE",
        "umap": "UMAP",
    }
    umap_params = {
        "n_neighbors": args.umap_n_neighbors,
        "min_dist": args.umap_min_dist,
        "spread": args.umap_spread,
        "repulsion_strength": args.umap_repulsion_strength,
    }
    tsne_learning_rate = args.tsne_learning_rate
    if isinstance(tsne_learning_rate, str) and tsne_learning_rate != "auto":
        tsne_learning_rate = float(tsne_learning_rate)
    tsne_params = {
        "mode": args.tsne_mode,
        "perplexity": args.tsne_perplexity,
        "early_exaggeration": args.tsne_early_exaggeration,
        "learning_rate": tsne_learning_rate,
        "knn_k": args.tsne_knn_k,
    }
    projections = {
        key: _fit_projection(
            key,
            observed,
            predicted,
            traj_flat,
            seed=0,
            umap_params=umap_params,
            tsne_params=tsne_params,
        )
        for key in projection_names
    }

    plot_label = args.output_label or args.run_name
    plot_dir = OUTPUT_ROOT / "figs" / plot_label
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=180)
    axes = axes.flatten()
    for ax, key in zip(axes, ["first2d", "pca", "tsne", "umap"]):
        _plot_projection(
            ax,
            projection_names[key],
            projections[key],
            observed,
            predicted,
            traj_dense,
            y,
            traj_time_values,
            colors,
            legacy_style=args.pinnuot_style,
            outlier_z_threshold=args.outlier_z_threshold,
            limit_quantile=args.limit_quantile,
            limit_pad=args.limit_pad,
            hide_lowest_cluster=args.hide_lowest_cluster,
            hide_lowest_cluster_n_clusters=args.hide_lowest_cluster_n_clusters,
            hide_lowest_cluster_max_ratio=args.hide_lowest_cluster_max_ratio,
            restrict_pred_to_trajectory=args.restrict_pred_to_trajectory,
            pred_trajectory_distance_multiplier=args.pred_trajectory_distance_multiplier,
            pred_trajectory_max_keep_ratio=args.pred_trajectory_max_keep_ratio,
            paper_mainline_filter=args.paper_mainline_filter,
            paper_anchor_time_index=args.paper_anchor_time_index,
            paper_filter_from_time_index=args.paper_filter_from_time_index,
            paper_filter_until_time_index=args.paper_filter_until_time_index,
            paper_anchor_keep_ratio=args.paper_anchor_keep_ratio,
            paper_anchor_neighbors=args.paper_anchor_neighbors,
            paper_band_distance_multiplier=args.paper_band_distance_multiplier,
            paper_band_keep_ratio=args.paper_band_keep_ratio,
            drop_time_cluster_index=args.drop_time_cluster_index,
            drop_time_cluster_n_clusters=args.drop_time_cluster_n_clusters,
            drop_time_cluster_which=args.drop_time_cluster_which,
            compact_time_centers=args.compact_time_centers,
            tsne_time_separation=args.tsne_time_separation,
            tsne_time_separation_margin=args.tsne_time_separation_margin,
            final_time_pull=args.final_time_pull,
            final_cluster_focus=args.final_cluster_focus,
            final_cluster_focus_n_clusters=args.final_cluster_focus_n_clusters,
            traj_line_alpha=args.traj_line_alpha,
            traj_line_width=args.traj_line_width,
            seed=args.seed,
        )

    semantic_handles, time_handles = _style_legend_handles(colors, time_labels)
    fig.legend(handles=semantic_handles, loc="upper center", ncol=3, frameon=False)
    fig.legend(handles=time_handles, loc="lower center", ncol=len(time_handles), frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    panel_path = plot_dir / "latent_trajectory_projection_panel.png"
    fig.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)

    for key, title in projection_names.items():
        fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
        _plot_projection(
            ax,
            title,
            projections[key],
            observed,
            predicted,
            traj_dense,
            y,
            traj_time_values,
            colors,
            legacy_style=args.pinnuot_style,
            outlier_z_threshold=args.outlier_z_threshold,
            limit_quantile=args.limit_quantile,
            limit_pad=args.limit_pad,
            hide_lowest_cluster=args.hide_lowest_cluster,
            hide_lowest_cluster_n_clusters=args.hide_lowest_cluster_n_clusters,
            hide_lowest_cluster_max_ratio=args.hide_lowest_cluster_max_ratio,
            restrict_pred_to_trajectory=args.restrict_pred_to_trajectory,
            pred_trajectory_distance_multiplier=args.pred_trajectory_distance_multiplier,
            pred_trajectory_max_keep_ratio=args.pred_trajectory_max_keep_ratio,
            paper_mainline_filter=args.paper_mainline_filter,
            paper_anchor_time_index=args.paper_anchor_time_index,
            paper_filter_from_time_index=args.paper_filter_from_time_index,
            paper_filter_until_time_index=args.paper_filter_until_time_index,
            paper_anchor_keep_ratio=args.paper_anchor_keep_ratio,
            paper_anchor_neighbors=args.paper_anchor_neighbors,
            paper_band_distance_multiplier=args.paper_band_distance_multiplier,
            paper_band_keep_ratio=args.paper_band_keep_ratio,
            drop_time_cluster_index=args.drop_time_cluster_index,
            drop_time_cluster_n_clusters=args.drop_time_cluster_n_clusters,
            drop_time_cluster_which=args.drop_time_cluster_which,
            compact_time_centers=args.compact_time_centers,
            tsne_time_separation=args.tsne_time_separation,
            tsne_time_separation_margin=args.tsne_time_separation_margin,
            final_time_pull=args.final_time_pull,
            final_cluster_focus=args.final_cluster_focus,
            final_cluster_focus_n_clusters=args.final_cluster_focus_n_clusters,
            traj_line_alpha=args.traj_line_alpha,
            traj_line_width=args.traj_line_width,
            seed=args.seed,
        )
        semantic_handles, time_handles = _style_legend_handles(colors, time_labels)
        fig.legend(handles=semantic_handles, loc="upper center", ncol=3, frameon=False)
        fig.legend(handles=time_handles, loc="lower center", ncol=len(time_handles), frameon=False, bbox_to_anchor=(0.5, 0.02))
        fig.tight_layout(rect=[0, 0.08, 1, 0.92])
        fig.savefig(plot_dir / f"latent_trajectory_{key}.png", bbox_inches="tight")
        plt.close(fig)

    eval_path = run_dir / "interpolate-mioemd2.log"
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path, sep="\t")
        best_eval = eval_df[eval_df["epoch"] == epoch_tag][["t", "loss"]].sort_values("t")
        mean_loss = float(best_eval["loss"].mean())
        loss_by_time = {_format_time_label(t): float(loss) for t, loss in zip(best_eval["t"], best_eval["loss"])}
    else:
        mean_loss = None
        loss_by_time = None

    summary = {
        "source_run_name": args.run_name,
        "output_label": plot_label,
        "run_dir": str(run_dir),
        "selected_epoch": epoch_tag,
        "selected_checkpoint": str(checkpoint_path),
        "mean_mioemd2": mean_loss,
        "loss_by_time": loss_by_time,
        "interpolate_log_path": str(eval_path) if eval_path.exists() else None,
        "umap_params": umap_params,
        "tsne_params": tsne_params,
        "tsne_time_separation": float(args.tsne_time_separation),
        "tsne_time_separation_margin": float(args.tsne_time_separation_margin),
        "line_start_time_index": int(line_start_time_index),
        "line_start_time": float(y[line_start_time_index]),
        "line_selection_time_index": int(line_selection_time_index),
        "line_selection_time": float(y[line_selection_time_index]),
        "time_colors": {
            "colormap": str(args.time_colormap),
            "start": float(args.time_color_start),
            "end": float(args.time_color_end),
            "final_time_color": None if str(args.final_time_color).lower() == "none" else str(args.final_time_color),
        },
        "max_observed_points_per_time": None if max_observed_points_per_time is None else int(max_observed_points_per_time),
        "max_predicted_points_per_time": int(args.max_predicted_points_per_time),
        "traj_line_alpha": float(args.traj_line_alpha),
        "traj_line_width": float(args.traj_line_width),
        "hide_lowest_cluster": bool(args.hide_lowest_cluster),
        "restrict_pred_to_trajectory": bool(args.restrict_pred_to_trajectory),
        "paper_mainline_filter": bool(args.paper_mainline_filter),
        "paper_filter_until_time_index": None if args.paper_filter_until_time_index is None else int(args.paper_filter_until_time_index),
        "drop_time_cluster_index": None if args.drop_time_cluster_index is None else int(args.drop_time_cluster_index),
        "drop_time_cluster_which": str(args.drop_time_cluster_which),
        "compact_time_centers": float(args.compact_time_centers),
        "final_time_pull": float(args.final_time_pull),
        "final_cluster_focus": bool(args.final_cluster_focus),
        "pinnuot_style": bool(args.pinnuot_style),
        "panel_png": str(panel_path),
    }
    (plot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
