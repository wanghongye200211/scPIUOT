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


def _resolve_epoch_tag(run_dir: Path, epoch_selector: Optional[str]) -> str:
    if epoch_selector in (None, "", "auto"):
        return _best_epoch_from_eval(run_dir)
    if epoch_selector == "final":
        return _latest_epoch_from_checkpoints(run_dir)
    return epoch_selector


def _checkpoint_path(run_dir: Path, epoch_tag: str) -> Path:
    if epoch_tag == "best":
        return run_dir / "train.best.pt"
    return run_dir / f"train.{epoch_tag}.pt"


def _rollout_segmented(model, config, train_mod, x, y, n_particles: int):
    torch.manual_seed(0)
    np.random.seed(0)

    x0, _ = train_mod.p_samp(x[config.start_t], n_particles)
    state = train_mod.build_initial_state(
        x0,
        config.use_growth,
        clip_value=float(getattr(config, "mass_clip_value", 30.0)),
    )

    records = []

    def record(time_idx: int, state_t: torch.Tensor):
        _, _, logw = train_mod.unpack_state(state_t, config.x_dim, config.use_growth)
        weights = train_mod.stable_exp_weights(
            logw,
            clip_value=float(getattr(config, "mass_clip_value", 30.0)),
        )
        mass = train_mod.normalized_mass_from_logw(
            logw,
            clip_value=float(getattr(config, "mass_clip_value", 30.0)),
        )
        ess = float(1.0 / torch.sum(mass.pow(2)).item())
        target = float(config.relative_mass_by_time[time_idx])
        pred_total = float(weights.sum().item())
        records.append(
            {
                "time": float(y[time_idx]),
                "target_relative_mass": target,
                "pred_total_mass": pred_total,
                "abs_error": abs(pred_total - target),
                "rel_error": abs(pred_total - target) / max(abs(target), 1e-12),
                "pred_min_weight": float(weights.min().item()),
                "pred_max_weight": float(weights.max().item()),
                "pred_mean_weight": float(weights.mean().item()),
                "pred_cv": float(
                    (weights.std(unbiased=False) / weights.mean().clamp_min(1e-12)).item()
                ),
                "pred_ess": ess,
                "pred_ess_ratio": ess / float(weights.shape[0]),
            }
        )

    record(config.start_t, state)

    current_state = state
    for t_prev, t_cur in train_mod.training_segments(config):
        ts = train_mod.segment_time_grid(
            y[t_prev],
            y[t_cur],
            int(getattr(config, "segment_regularization_points", 5)),
        )
        out = model(ts, current_state)
        current_state = out[-1].detach()
        record(t_cur, current_state)

    return records


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", "--run-name", dest="run_name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch_tag", "--epoch-tag", dest="epoch_tag", default=None)
    parser.add_argument("--n_particles", "--n-particles", dest="n_particles", type=int, default=4000)
    parser.add_argument("--output_label", "--output-label", dest="output_label", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    config_mod, model_mod, train_mod = _load_runtime_modules(args.run_name)
    run_dir = _resolve_run_dir(args.run_name, args.seed)
    epoch_tag = _resolve_epoch_tag(run_dir, args.epoch_tag)
    checkpoint_path = _checkpoint_path(run_dir, epoch_tag)

    cfg_dict = torch.load(run_dir / "config.pt", map_location="cpu")
    config = SimpleNamespace(**cfg_dict)
    x, y, config = config_mod.load_data(config)

    model = model_mod.ForwardSDE(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = _rollout_segmented(model, config, train_mod, x, y, args.n_particles)
    df = pd.DataFrame(records)
    summary = {
        "source_run_name": args.run_name,
        "output_label": args.output_label or args.run_name,
        "run_dir": str(run_dir),
        "selected_epoch": epoch_tag,
        "selected_checkpoint": str(checkpoint_path),
        "n_particles": int(args.n_particles),
        "mean_abs_error": float(df["abs_error"].mean()),
        "mean_rel_error": float(df["rel_error"].mean()),
        "records": records,
    }

    plot_dir = OUTPUT_ROOT / "figs" / (args.output_label or args.run_name)
    plot_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(plot_dir / "mass_diagnostics.csv", index=False)
    (plot_dir / "mass_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.plot(df["time"], df["target_relative_mass"], marker="o", linewidth=2, label="Target Mass")
    ax.plot(df["time"], df["pred_total_mass"], marker="o", linewidth=2, label="Predicted Mass")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Mass")
    ax.set_title(f"Mass Curve: {args.run_name}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "mass_curve.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
