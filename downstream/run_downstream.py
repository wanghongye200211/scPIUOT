from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DEFAULT_CONFIG_PATH, DOWNSTREAM_OUTPUT_ROOT, PIUOT_ROOT
from yaml_config import checkpoint_epoch_from_config, device_from_config, embedding_key_from_config, load_yaml_config


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the generic downstream plotting bundle.")
    parser.add_argument("--yaml-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-perturbation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.yaml_config)
    py = str(Path(args.python_bin).resolve())

    run_name = str(cfg["experiment"].get("run_name", "piuot_run"))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    data_path = str(cfg["data"]["path"])
    embedding_key = embedding_key_from_config(cfg)
    checkpoint_epoch = checkpoint_epoch_from_config(cfg, fallback="auto")
    output_label = str(cfg["data"].get("label") or cfg["experiment"].get("name", run_name))

    run(
        [
            py,
            str(PIUOT_ROOT / "plot_trajectory.py"),
            "--run_name",
            run_name,
            "--seed",
            str(seed),
            "--checkpoint_epoch",
            checkpoint_epoch,
            "--output_label",
            output_label,
        ]
    )

    run(
        [
            py,
            str(PIUOT_ROOT / "diagnose_mass.py"),
            "--run_name",
            run_name,
            "--seed",
            str(seed),
            "--epoch_tag",
            checkpoint_epoch,
            "--output_label",
            output_label,
        ]
    )

    run(
        [
            py,
            str(PROJECT_ROOT / "criticality" / "compute_original_qreshape_mass_indicator.py"),
            "--yaml-config",
            str(args.yaml_config),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--checkpoint",
            checkpoint_epoch,
        ]
    )

    run(
        [
            py,
            str(PROJECT_ROOT / "criticality" / "compare_potential_related_indicators.py"),
            "--yaml-config",
            str(args.yaml_config),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--checkpoint",
            checkpoint_epoch,
        ]
    )

    downstream_cfg = cfg["downstream"]
    analysis_cmd = [
        py,
        str(PROJECT_ROOT / "downstream" / "analyze_manifold_physics_fates.py"),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--checkpoint",
        checkpoint_epoch,
        "--device",
        device_from_config(cfg, "analysis", "cpu"),
        "--data-path",
        data_path,
        "--label",
        output_label,
        "--critical-indicator",
        str(downstream_cfg.get("critical_indicator", "product")),
    ]
    state_key = str(cfg["data"].get("state_key") or downstream_cfg.get("state_key") or "").strip()
    fate_key = str(cfg["data"].get("fate_key") or downstream_cfg.get("fate_key") or "").strip()
    if state_key:
        analysis_cmd += ["--state-key", state_key]
    if fate_key:
        analysis_cmd += ["--fate-key", fate_key]
    if downstream_cfg.get("critical_window_start") is not None:
        analysis_cmd += ["--critical-window-start", str(downstream_cfg["critical_window_start"])]
    if downstream_cfg.get("critical_window_end") is not None:
        analysis_cmd += ["--critical-window-end", str(downstream_cfg["critical_window_end"])]
    if downstream_cfg.get("anchor_min_time") is not None:
        analysis_cmd += ["--anchor-min-time", str(downstream_cfg["anchor_min_time"])]
    if downstream_cfg.get("normalize_start_time") is not None:
        analysis_cmd += ["--normalize-start-time", str(downstream_cfg["normalize_start_time"])]
    run(analysis_cmd)

    run([py, str(PROJECT_ROOT / "downstream" / "build_focus_bundle.py"), "--yaml-config", str(args.yaml_config)])

    if not args.skip_perturbation:
        run(
            [
                py,
                str(PROJECT_ROOT / "downstream" / "build_perturbation_dynamic_fraction.py"),
                "--yaml-config",
                str(args.yaml_config),
            ]
        )
        run(
            [
                py,
                str(PROJECT_ROOT / "downstream" / "build_perturbation_manifest.py"),
                "--yaml-config",
                str(args.yaml_config),
            ]
        )

    manifest = {
        "run_name": run_name,
        "output_label": output_label,
        "embedding_key": embedding_key,
        "trajectory_dir": str(PIUOT_ROOT / "output" / "figs" / output_label),
        "focus_dir": str(DOWNSTREAM_OUTPUT_ROOT / f"{output_label}_focus"),
        "perturbation_dir": str(DOWNSTREAM_OUTPUT_ROOT / f"{output_label}_perturbation_dynamic_fraction"),
    }
    DOWNSTREAM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (DOWNSTREAM_OUTPUT_ROOT / f"{output_label}_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
