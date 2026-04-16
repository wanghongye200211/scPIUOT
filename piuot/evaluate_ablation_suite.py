import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import torch


METHOD_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = METHOD_ROOT / "output"

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


def _resolve_src_package(run_name: str) -> str:
    return "src_mps_druot_ablation_suite"


def _load_runtime_modules(run_name: str):
    src_pkg = importlib.import_module(_resolve_src_package(run_name))
    sys.modules["src"] = src_pkg
    config_mod = importlib.import_module("src.config_model")
    model_mod = importlib.import_module("src.model")
    train_mod = importlib.import_module("src.train")
    mio_losses_mod = importlib.import_module("src.mio_losses")
    return config_mod, model_mod, train_mod, mio_losses_mod


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


def _format_time_label(value) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


def _rollout_to_time(config, model, train_mod, x, y, t_idx: int, n_particles: int):
    torch.manual_seed(0)
    np.random.seed(0)

    x_0, a_0 = train_mod.p_samp(x[0], n_particles)
    x_r_0 = train_mod.build_initial_state(
        x_0,
        config.use_growth,
        clip_value=float(getattr(config, "mass_clip_value", 30.0)),
    )

    states = []
    chunk_size = max(1, int(config.ns))
    n_chunks = max(1, math.ceil(int(n_particles) / chunk_size))
    for i in range(n_chunks):
        x_r_0_chunk = x_r_0[i * chunk_size:(i + 1) * chunk_size]
        if x_r_0_chunk.shape[0] == 0:
            continue
        x_r_s = model([np.float64(y[0]), np.float64(y[t_idx])], x_r_0_chunk)
        states.append(x_r_s[-1].detach().cpu())

    x_r_s = torch.cat(states, dim=0)
    pred_x, _, pred_logw = train_mod.unpack_state(x_r_s, config.x_dim, config.use_growth)
    pred_mass = train_mod.normalized_mass_from_logw(pred_logw) if pred_logw is not None else a_0.cpu()
    return pred_x.cpu(), pred_mass.cpu(), x[t_idx].cpu()


def _weighted_mmd(source: torch.Tensor, target: torch.Tensor, source_mass: torch.Tensor = None, target_mass: torch.Tensor = None):
    source = source.float().cpu()
    target = target.float().cpu()
    n = source.shape[0]
    m = target.shape[0]

    if source_mass is None:
        p = torch.full((n,), 1.0 / max(n, 1), dtype=source.dtype)
    else:
        p = source_mass.float().cpu()
        p = p / p.sum().clamp_min(1e-12)

    if target_mass is None:
        q = torch.full((m,), 1.0 / max(m, 1), dtype=target.dtype)
    else:
        q = target_mass.float().cpu()
        q = q / q.sum().clamp_min(1e-12)

    xx = torch.cdist(source, source).pow(2)
    yy = torch.cdist(target, target).pow(2)
    xy = torch.cdist(source, target).pow(2)

    vals = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
    vals = vals[vals > 0]
    sigma2 = torch.median(vals).item() if vals.numel() > 0 else 1.0
    sigma2 = max(float(sigma2), 1e-8)

    kxx = torch.exp(-xx / (2.0 * sigma2))
    kyy = torch.exp(-yy / (2.0 * sigma2))
    kxy = torch.exp(-xy / (2.0 * sigma2))

    mmd2 = p @ (kxx @ p) + q @ (kyy @ q) - 2.0 * (p @ (kxy @ q))
    return float(mmd2.item())


def _w1_w2(source: torch.Tensor, target: torch.Tensor, source_mass: torch.Tensor = None, target_mass: torch.Tensor = None):
    source = source.float().cpu()
    target = target.float().cpu()
    n = source.shape[0]
    m = target.shape[0]

    if source_mass is None:
        mu = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    else:
        mu = source_mass.float().cpu().numpy().astype(np.float64)
        mu = mu / max(mu.sum(), 1e-12)

    if target_mass is None:
        nu = np.full(m, 1.0 / max(m, 1), dtype=np.float64)
    else:
        nu = target_mass.float().cpu().numpy().astype(np.float64)
        nu = nu / max(nu.sum(), 1e-12)

    x_np = source.numpy().astype(np.float64)
    y_np = target.numpy().astype(np.float64)
    cost_w1 = ot.dist(x_np, y_np, metric="euclidean")
    cost_w2 = ot.dist(x_np, y_np, metric="sqeuclidean")
    w1 = float(ot.emd2(mu, nu, cost_w1))
    w2_sq = float(ot.emd2(mu, nu, cost_w2))
    w2 = float(np.sqrt(max(w2_sq, 0.0)))
    return w1, w2


def _evaluate_run(run_name: str, seed: int, n_particles: int, checkpoint_epoch: str):
    config_mod, model_mod, train_mod, mio_losses_mod = _load_runtime_modules(run_name)
    run_dir = _resolve_run_dir(run_name, seed)
    epoch_tag = _resolve_epoch_tag(run_dir, checkpoint_epoch)
    checkpoint_path = _checkpoint_path(run_dir, epoch_tag)

    cfg_dict = torch.load(run_dir / "config.pt", map_location="cpu")
    config = SimpleNamespace(**cfg_dict)
    x, y, config = config_mod.load_data(config)

    model = model_mod.ForwardSDE(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    for t_idx in config.train_t:
        pred_x, pred_mass, target_x = _rollout_to_time(config, model, train_mod, x, y, t_idx, n_particles)
        target_mass = torch.full((target_x.shape[0],), 1.0 / max(target_x.shape[0], 1), dtype=pred_x.dtype)
        mioemd2 = float(
            mio_losses_mod.mioflow_emd2_loss(
                pred_x,
                target_x,
                source_mass=pred_mass,
                target_mass=target_mass,
                detach_weights=config.detach_ot_weights,
            ).item()
        )
        w1, w2 = _w1_w2(pred_x, target_x, pred_mass, target_mass)
        mmd = _weighted_mmd(pred_x, target_x, pred_mass, target_mass)
        rows.append(
            {
                "run_name": run_name,
                "selected_epoch": epoch_tag,
                "time": float(y[t_idx]),
                "time_label": _format_time_label(y[t_idx]),
                "mioemd2": mioemd2,
                "w1": w1,
                "w2": w2,
                "mmd": mmd,
            }
        )

    df = pd.DataFrame(rows)
    means = {
        "mean_mioemd2": float(df["mioemd2"].mean()),
        "mean_w1": float(df["w1"].mean()),
        "mean_w2": float(df["w2"].mean()),
        "mean_mmd": float(df["mmd"].mean()),
    }
    return run_dir, epoch_tag, df, means


def _write_barplot(summary_df: pd.DataFrame, out_path: Path):
    metrics = ["mean_mioemd2", "mean_w1", "mean_w2", "mean_mmd"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=180)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        sub = summary_df.sort_values(metric)
        ax.barh(sub["run_name"], sub[metric], color="#4C72B0")
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_panel(run_names, out_path: Path):
    fig, axes = plt.subplots(math.ceil(len(run_names) / 2), 2, figsize=(16, 5 * math.ceil(len(run_names) / 2)), dpi=160)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, run_name in zip(axes, run_names):
        panel_path = OUTPUT_ROOT / "figs" / run_name / "latent_trajectory_projection_panel.png"
        if panel_path.exists():
            ax.imshow(plt.imread(panel_path))
            ax.set_title(run_name)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"Missing panel\n{run_name}", ha="center", va="center")
            ax.set_title(run_name)
            ax.axis("off")
    for ax in axes[len(run_names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_names", "--run-names", dest="run_names", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_particles", "--n-particles", dest="n_particles", type=int, default=2000)
    parser.add_argument("--label", default="ablation_suite")
    parser.add_argument("--checkpoint_epoch", "--checkpoint-epoch", dest="checkpoint_epoch", default="auto")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    all_frames = []
    summary_rows = []
    for run_name in args.run_names:
        run_dir, epoch_tag, df, means = _evaluate_run(run_name, args.seed, args.n_particles, args.checkpoint_epoch)
        all_frames.append(df)
        summary_rows.append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "selected_epoch": epoch_tag,
                **means,
            }
        )

    per_time_df = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    figs_root = OUTPUT_ROOT / "figs" / args.label
    figs_root.mkdir(parents=True, exist_ok=True)
    csv_path = figs_root / "comparison_metrics.csv"
    json_path = figs_root / "comparison_metrics.json"
    panel_path = figs_root / "compare_ablation_suite_panel.png"
    barplot_path = figs_root / "comparison_barplot.png"
    per_time_path = figs_root / "comparison_metrics_per_time.csv"

    summary_df.to_csv(csv_path, index=False)
    per_time_df.to_csv(per_time_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_df.to_dict(orient="records"),
                "per_time": per_time_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_barplot(summary_df, barplot_path)
    _write_panel(args.run_names, panel_path)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "panel": str(panel_path), "barplot": str(barplot_path)}, indent=2))


if __name__ == "__main__":
    main()
