from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT, PIUOT_ROOT


MANUAL_RUN_NAME = "piuot_run"
MANUAL_SEED = 0
MANUAL_DATA_PATH = str(PIUOT_ROOT / "input" / "input.h5ad")
MANUAL_EMBEDDING_KEY = "X_gae15"
MANUAL_RAW_TIME_KEY = "t"
MANUAL_CHECKPOINT = "auto"
MANUAL_OUTPUT_LABEL = "dataset"
MANUAL_OUTPUT_SLUG = "dataset"
MANUAL_STATE_KEY = "consensus_cluster"
MANUAL_FATE_KEY = "phenotype_facs"
MANUAL_ANALYSIS_DEVICE = "cpu"
MANUAL_PERTURB_DEVICE = "cpu"
MANUAL_CRITICAL_INDICATOR = "product"
MANUAL_CRITICAL_WINDOW_START = None
MANUAL_CRITICAL_WINDOW_END = None
MANUAL_ANCHOR_MIN_TIME = None
MANUAL_NORMALIZE_START_TIME = None
MANUAL_TOP_TERMINAL_FATES = 5
MANUAL_PERTURB_START_TIME = None
MANUAL_PERTURB_END_TIME = None
MANUAL_PERTURB_TARGET_LABEL = None
MANUAL_PERTURB_N_TIMEPOINTS = 25
MANUAL_PERTURB_N_REPEATS = 4
MANUAL_PERTURB_MAX_START_CELLS = 96
MANUAL_PERTURB_SCALE = 2.0


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run downstream analysis with manual path settings.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-name", default=MANUAL_RUN_NAME)
    parser.add_argument("--seed", type=int, default=MANUAL_SEED)
    parser.add_argument("--data-path", type=Path, default=Path(MANUAL_DATA_PATH))
    parser.add_argument("--embedding-key", default=MANUAL_EMBEDDING_KEY)
    parser.add_argument("--raw-time-key", default=MANUAL_RAW_TIME_KEY)
    parser.add_argument("--checkpoint", default=MANUAL_CHECKPOINT)
    parser.add_argument("--output-label", default=MANUAL_OUTPUT_LABEL)
    parser.add_argument("--output-slug", default=MANUAL_OUTPUT_SLUG)
    parser.add_argument("--state-key", default=MANUAL_STATE_KEY)
    parser.add_argument("--fate-key", default=MANUAL_FATE_KEY)
    parser.add_argument("--analysis-device", default=MANUAL_ANALYSIS_DEVICE)
    parser.add_argument("--perturb-device", default=MANUAL_PERTURB_DEVICE)
    parser.add_argument("--critical-indicator", default=MANUAL_CRITICAL_INDICATOR)
    parser.add_argument("--critical-window-start", type=float, default=MANUAL_CRITICAL_WINDOW_START)
    parser.add_argument("--critical-window-end", type=float, default=MANUAL_CRITICAL_WINDOW_END)
    parser.add_argument("--anchor-min-time", type=float, default=MANUAL_ANCHOR_MIN_TIME)
    parser.add_argument("--normalize-start-time", type=float, default=MANUAL_NORMALIZE_START_TIME)
    parser.add_argument("--top-terminal-fates", type=int, default=MANUAL_TOP_TERMINAL_FATES)
    parser.add_argument("--perturb-start-time", type=float, default=MANUAL_PERTURB_START_TIME)
    parser.add_argument("--perturb-end-time", type=float, default=MANUAL_PERTURB_END_TIME)
    parser.add_argument("--perturb-target-label", default=MANUAL_PERTURB_TARGET_LABEL)
    parser.add_argument("--perturb-n-timepoints", type=int, default=MANUAL_PERTURB_N_TIMEPOINTS)
    parser.add_argument("--perturb-n-repeats", type=int, default=MANUAL_PERTURB_N_REPEATS)
    parser.add_argument("--perturb-max-start-cells", type=int, default=MANUAL_PERTURB_MAX_START_CELLS)
    parser.add_argument("--perturb-scale", type=float, default=MANUAL_PERTURB_SCALE)
    parser.add_argument("--skip-perturbation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    py = str(Path(args.python_bin).resolve())

    run(
        [
            py,
            str(PIUOT_ROOT / "plot.py"),
            "--run_name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint_epoch",
            args.checkpoint,
            "--output_label",
            args.output_label,
        ]
    )

    run(
        [
            py,
            str(PIUOT_ROOT / "diagnose.py"),
            "--run_name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--epoch_tag",
            args.checkpoint,
            "--output_label",
            args.output_label,
        ]
    )

    run(
        [
            py,
            str(PROJECT_ROOT / "criticality" / "compute_original_qreshape_mass_indicator.py"),
            "--run-name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.analysis_device,
            "--output-label",
            args.output_label,
            "--output-slug",
            args.output_slug,
        ]
    )

    run(
        [
            py,
            str(PROJECT_ROOT / "criticality" / "compare_potential_related_indicators.py"),
            "--run-name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.analysis_device,
            "--output-label",
            args.output_label,
            "--output-slug",
            args.output_slug,
        ]
    )

    analysis_cmd = [
        py,
        str(PROJECT_ROOT / "downstream" / "analyze_manifold_physics_fates.py"),
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.analysis_device,
        "--data-path",
        str(Path(args.data_path).expanduser().resolve()),
        "--embedding-key",
        args.embedding_key,
        "--label",
        args.output_label,
        "--critical-indicator",
        args.critical_indicator,
    ]
    if str(args.state_key).strip():
        analysis_cmd += ["--state-key", str(args.state_key).strip()]
    if str(args.fate_key).strip():
        analysis_cmd += ["--fate-key", str(args.fate_key).strip()]
    if args.critical_window_start is not None:
        analysis_cmd += ["--critical-window-start", str(args.critical_window_start)]
    if args.critical_window_end is not None:
        analysis_cmd += ["--critical-window-end", str(args.critical_window_end)]
    if args.anchor_min_time is not None:
        analysis_cmd += ["--anchor-min-time", str(args.anchor_min_time)]
    if args.normalize_start_time is not None:
        analysis_cmd += ["--normalize-start-time", str(args.normalize_start_time)]
    run(analysis_cmd)

    run(
        [
            py,
            str(PROJECT_ROOT / "downstream" / "build_focus_bundle.py"),
            "--run-name",
            args.run_name,
            "--output-label",
            args.output_label,
            "--output-slug",
            args.output_slug,
            "--top-terminal-fates",
            str(args.top_terminal_fates),
        ]
    )

    if not args.skip_perturbation:
        perturb_cmd = [
            py,
            str(PROJECT_ROOT / "downstream" / "build_perturbation_dynamic_fraction.py"),
            "--run-name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.perturb_device,
            "--data-path",
            str(Path(args.data_path).expanduser().resolve()),
            "--embedding-key",
            args.embedding_key,
            "--raw-time-key",
            args.raw_time_key,
            "--fate-key",
            args.fate_key,
            "--output-label",
            args.output_label,
            "--n-timepoints",
            str(args.perturb_n_timepoints),
            "--n-repeats",
            str(args.perturb_n_repeats),
            "--max-start-cells",
            str(args.perturb_max_start_cells),
            "--scale",
            str(args.perturb_scale),
            "--top-terminal-fates",
            str(args.top_terminal_fates),
        ]
        if args.perturb_start_time is not None:
            perturb_cmd += ["--start-time", str(args.perturb_start_time)]
        if args.perturb_end_time is not None:
            perturb_cmd += ["--end-time", str(args.perturb_end_time)]
        if args.perturb_target_label:
            perturb_cmd += ["--target-label", str(args.perturb_target_label)]
        run(perturb_cmd)

        run(
            [
                py,
                str(PROJECT_ROOT / "downstream" / "build_perturbation_manifest.py"),
                "--output-label",
                args.output_label,
            ]
        )

    manifest = {
        "run_name": args.run_name,
        "output_label": args.output_label,
        "output_slug": args.output_slug,
        "embedding_key": args.embedding_key,
        "trajectory_dir": str(PIUOT_ROOT / "output" / "figs" / args.output_label),
        "focus_dir": str(DOWNSTREAM_OUTPUT_ROOT / f"{args.output_label}_focus"),
        "perturbation_dir": str(DOWNSTREAM_OUTPUT_ROOT / f"{args.output_label}_perturbation_dynamic_fraction"),
    }
    DOWNSTREAM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (DOWNSTREAM_OUTPUT_ROOT / f"{args.output_label}_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
