from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from project_paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT


VALID_DEVICES = {"cpu", "cuda", "mps"}


def _resolve_path(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    data.setdefault("experiment", {})
    data.setdefault("device", {})
    data.setdefault("data", {})
    data.setdefault("reduction", {})
    data.setdefault("training", {})
    data.setdefault("selection", {})
    data.setdefault("criticality", {})
    data.setdefault("downstream", {})
    data.setdefault("perturbation", {})

    data["data"]["path"] = _resolve_path(data["data"].get("path"))
    return data


def device_from_config(config: dict[str, Any], key: str, fallback: str) -> str:
    value = str(config.get("device", {}).get(key, fallback))
    if value not in VALID_DEVICES:
        raise ValueError(f"Unsupported device '{value}'. Choose from {sorted(VALID_DEVICES)}.")
    return value


def checkpoint_epoch_from_config(config: dict[str, Any], fallback: str = "auto") -> str:
    value = config.get("selection", {}).get("checkpoint_epoch", fallback)
    return str(value)


def reduction_method_from_config(config: dict[str, Any], fallback: str = "gae") -> str:
    value = str(config.get("reduction", {}).get("method", fallback)).lower()
    if value not in {"gae", "gaga"}:
        raise ValueError("reduction.method must be one of: gae, gaga")
    return value


def reduction_epoch_from_config(config: dict[str, Any], fallback: int = 15) -> int:
    return int(config.get("reduction", {}).get("epoch", fallback))


def dataset_label_from_config(config: dict[str, Any], fallback: str = "dataset") -> str:
    value = str(
        config.get("data", {}).get("label")
        or config.get("experiment", {}).get("name")
        or fallback
    ).strip()
    return value or fallback


def dataset_slug_from_config(config: dict[str, Any], fallback: str = "dataset") -> str:
    label = dataset_label_from_config(config, fallback=fallback)
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", label).strip("._")
    return slug or fallback


def embedding_key_from_config(config: dict[str, Any]) -> str:
    explicit = str(config.get("data", {}).get("embedding_key", "") or "").strip()
    if explicit:
        return explicit
    method = reduction_method_from_config(config)
    epoch = reduction_epoch_from_config(config)
    prefix = "X_gae" if method == "gae" else "X_gaga"
    return f"{prefix}{epoch}"
