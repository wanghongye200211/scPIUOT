from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PIUOT_ROOT = PROJECT_ROOT / "piuot"
PIUOT_CONFIG_ROOT = PIUOT_ROOT / "configs"
PIUOT_INPUT_ROOT = PIUOT_ROOT / "input"
CRITICALITY_ROOT = PROJECT_ROOT / "criticality"
DOWNSTREAM_ROOT = PROJECT_ROOT / "downstream"

PIUOT_OUTPUT_ROOT = PIUOT_ROOT / "output"
PIUOT_FIG_ROOT = PIUOT_OUTPUT_ROOT / "figs"
DOWNSTREAM_OUTPUT_ROOT = DOWNSTREAM_ROOT / "output"

DEFAULT_CONFIG_PATH = PIUOT_CONFIG_ROOT / "default.yaml"
