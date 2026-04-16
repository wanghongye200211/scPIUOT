from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DEFAULT_CONFIG_PATH, DOWNSTREAM_OUTPUT_ROOT, PIUOT_FIG_ROOT
from yaml_config import dataset_slug_from_config, load_yaml_config


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _show_image(ax: plt.Axes, path: Path, title: str) -> None:
    if not path.exists():
        ax.text(0.5, 0.5, f"Missing\n{path.name}", ha="center", va="center")
        ax.axis("off")
        return
    ax.imshow(mpimg.imread(path))
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a generic downstream focus bundle.")
    parser.add_argument("--yaml-config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.yaml_config)
    run_name = str(cfg["experiment"].get("run_name", "piuot_run"))
    label = str(cfg["data"].get("label") or cfg["experiment"].get("name", run_name))
    dataset_slug = dataset_slug_from_config(cfg, fallback="dataset")
    top_fates = int(cfg["downstream"].get("top_terminal_fates", 5))

    fig_root = PIUOT_FIG_ROOT / run_name
    traj_root = PIUOT_FIG_ROOT / label
    fate_dir = fig_root / "downstream_fate"
    critical_dir = PIUOT_FIG_ROOT / dataset_slug / "criticality_original"
    potential_dir = PIUOT_FIG_ROOT / dataset_slug / "potential_indicator_compare"
    out_dir = DOWNSTREAM_OUTPUT_ROOT / f"{label}_focus"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = label
    per_time_path = fate_dir / f"{stem}_physics_fate_per_time.csv"
    summary_path = fate_dir / f"{stem}_physics_fate_summary.json"
    if not per_time_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Missing downstream fate files. Expected {per_time_path} and {summary_path}."
        )

    per_time = pd.read_csv(per_time_path)
    summary = _load_json(summary_path)
    mass_cols = [col for col in per_time.columns if col.startswith("mass_")]
    if not mass_cols:
        raise ValueError(f"No mass_* columns found in {per_time_path}.")

    terminal = per_time.iloc[-1][mass_cols].sort_values(ascending=False)
    keep = list(terminal.head(top_fates).index)
    if "mass_unresolved" in mass_cols and "mass_unresolved" not in keep:
        keep.append("mass_unresolved")

    stack_df = per_time[["time", *keep]].copy()
    stack_csv = out_dir / f"{label}_future_fate_stackplot.csv"
    stack_df.to_csv(stack_csv, index=False)
    stack_png = out_dir / f"{label}_future_fate_stackplot.png"

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    labels = [name.replace("mass_", "") for name in keep]
    values = [stack_df[name].to_numpy() for name in keep]
    ax.stackplot(stack_df["time"].to_numpy(), values, labels=labels, alpha=0.92)
    ax.set_xlabel("Model time")
    ax.set_ylabel("Mass fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{label} | future-fate composition")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.savefig(stack_png, dpi=220)
    plt.close(fig)

    critical_png = out_dir / f"{label}_criticality_focus.png"
    critical_csv = out_dir / f"{label}_criticality_focus.csv"
    curve_cols = [col for col in ["time", "Action_norm", "Q_phi_norm", "Product"] if col in per_time.columns]
    per_time[curve_cols].to_csv(critical_csv, index=False)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for name in curve_cols:
        if name == "time":
            continue
        ax.plot(per_time["time"], per_time[name], lw=2.0, label=name)
    ax.axvline(float(summary["critical_time"]), color="crimson", lw=1.6, ls="--", label=f"critical t={summary['critical_time']:.2f}")
    ax.set_xlabel("Model time")
    ax.set_ylabel("Normalized value")
    ax.set_title(f"{label} | criticality focus")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(critical_png, dpi=220)
    plt.close(fig)

    board_png = out_dir / f"{label}_downstream_board.png"
    trajectory_png = traj_root / "latent_trajectory_projection_panel.png"
    fate_panel_png = fate_dir / f"{stem}_physics_fate_panel.png"
    qreshape_png = next(iter(sorted(critical_dir.glob("*_qreshape_mass_original_indicator.png"))), None)
    potential_png = next(iter(sorted(potential_dir.glob("*_potential_indicator_comparison.png"))), None)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    _show_image(axes[0, 0], trajectory_png, "Trajectory projection")
    _show_image(axes[0, 1], stack_png, "Future-fate stackplot")
    _show_image(axes[0, 2], critical_png, "Criticality focus")
    _show_image(axes[1, 0], fate_panel_png, "Physics-fate panel")
    _show_image(axes[1, 1], qreshape_png or Path("missing"), "Q_reshape indicator")
    _show_image(axes[1, 2], potential_png or Path("missing"), "Potential indicator compare")
    fig.suptitle(f"{label} | downstream focus bundle", fontsize=18, y=1.01)
    fig.savefig(board_png, dpi=220)
    plt.close(fig)

    manifest = {
        "run_name": run_name,
        "label": label,
        "critical_time": float(summary["critical_time"]),
        "artifacts": {
            "stackplot_png": str(stack_png),
            "stackplot_csv": str(stack_csv),
            "criticality_png": str(critical_png),
            "criticality_csv": str(critical_csv),
            "board_png": str(board_png),
            "trajectory_panel_png": str(trajectory_png),
            "physics_fate_panel_png": str(fate_panel_png),
            "qreshape_png": str(qreshape_png) if qreshape_png is not None else None,
            "potential_compare_png": str(potential_png) if potential_png is not None else None,
        },
    }
    manifest_path = out_dir / f"{label}_focus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_md = out_dir / f"{label}_focus_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                f"# {label} downstream focus",
                "",
                f"- Run name: `{run_name}`",
                f"- Critical time: `{summary['critical_time']:.4f}`",
                f"- Board: `{board_png}`",
                f"- Stackplot: `{stack_png}`",
                f"- Criticality focus: `{critical_png}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
