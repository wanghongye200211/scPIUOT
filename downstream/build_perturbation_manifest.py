from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DEFAULT_CONFIG_PATH, DOWNSTREAM_OUTPUT_ROOT
from yaml_config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a generic perturbation manifest from summary JSON.")
    parser.add_argument("--yaml-config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.yaml_config)
    label = str(cfg["data"].get("label") or cfg["experiment"].get("name", "dataset"))
    pert_root = DOWNSTREAM_OUTPUT_ROOT / f"{label}_perturbation_dynamic_fraction"
    summary_json = pert_root / "perturbation_dynamic_fraction_summary.json"
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    manifest = {
        "label": label,
        "target_label": summary["target_label"],
        "selected_conditions": summary["selected_conditions"],
        "best_positive_shift": summary["best_positive_shift"],
        "best_negative_shift": summary["best_negative_shift"],
        "artifacts": summary["artifacts"],
    }
    (pert_root / "perturbation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (pert_root / "perturbation_summary.md").write_text(
        "\n".join(
            [
                f"# {label} perturbation dynamic fraction",
                "",
                f"- Target label: `{summary['target_label']}`",
                f"- Positive shift condition: `{summary['best_positive_shift']['condition']}`",
                f"- Negative shift condition: `{summary['best_negative_shift']['condition']}`",
                f"- Stackplot: `{summary['artifacts']['dynamic_fraction_stackplot_png']}`",
                f"- Target trajectory: `{summary['artifacts']['target_group_dynamic_fraction_png']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
