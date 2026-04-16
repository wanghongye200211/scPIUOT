from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = METHOD_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from project_paths import DEFAULT_CONFIG_PATH
from yaml_config import checkpoint_epoch_from_config, device_from_config, embedding_key_from_config, load_yaml_config


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the generic PIUOT trajectory model with YAML-driven dataset selection.",
    )
    parser.add_argument("--config", "--yaml-config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--embedding-key", default=None)
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--raw-time-key", default=None)
    parser.add_argument("--device-type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    args, extra = build_parser().parse_known_args()
    yaml_cfg = load_yaml_config(args.config_path)

    run_name = args.run_name or str(yaml_cfg["experiment"].get("run_name", "piuot_run"))
    data_path = Path(args.data_path or yaml_cfg["data"]["path"])
    embedding_key = args.embedding_key or embedding_key_from_config(yaml_cfg)
    time_key = args.time_key or str(yaml_cfg["data"].get("time_key", "time_bin"))
    raw_time_key = args.raw_time_key or str(yaml_cfg["data"].get("raw_time_key", "t"))
    device_type = args.device_type or device_from_config(yaml_cfg, "cpu")
    seed = int(args.seed if args.seed is not None else yaml_cfg.get("seed", 0))

    src_pkg = importlib.import_module("core")
    sys.modules["src"] = src_pkg

    from src.config_model import config, init_config
    import src.train as train
    from src.evaluation import evaluate_fit

    training = yaml_cfg["training"]
    cli_args = [
        "--run_name",
        run_name,
        "--data_path",
        str(data_path.expanduser().resolve()),
        "--embedding_key",
        embedding_key,
        "--time_key",
        time_key,
        "--raw_time_key",
        raw_time_key,
        "--device_type",
        device_type,
        "--out_dir",
        str((METHOD_ROOT / "output").resolve()),
        "--seed",
        str(seed),
        "--train_epochs",
        str(training.get("train_epochs", 2000)),
        "--train_lr",
        str(training.get("train_lr", 0.005)),
        "--lambda_ot",
        str(training.get("lambda_ot", 1.0)),
        "--lambda_hjb",
        str(training.get("lambda_hjb", 0.05)),
        "--lambda_density",
        str(training.get("lambda_density", 0.05)),
        "--lambda_action",
        str(training.get("lambda_action", 0.01)),
        "--lambda_mass",
        str(training.get("lambda_mass", 0.1)),
        "--growth_mode",
        str(training.get("growth_mode", "bounded")),
        "--growth_scale",
        str(training.get("growth_scale", 0.05)),
        "--hjb_growth_coeff",
        str(training.get("hjb_growth_coeff", 2.0)),
        "--solver_dt",
        str(training.get("solver_dt", 0.1)),
        "--stage_transition_epoch",
        str(training.get("stage_transition_epoch", 200)),
        "--stage2_lr",
        str(training.get("stage2_lr", 0.0005)),
        "--constraint_start_epoch",
        str(training.get("constraint_start_epoch", 230)),
        "--constraint_ramp_epochs",
        str(training.get("constraint_ramp_epochs", 30)),
        "--global_mass_start_epoch",
        str(training.get("global_mass_start_epoch", 200)),
        "--global_mass_ramp_epochs",
        str(training.get("global_mass_ramp_epochs", 0)),
        "--local_mass_loss_mode",
        str(training.get("local_mass_loss_mode", "absolute_l2")),
        "--evaluate_n",
        str(training.get("evaluate_n", 4000)),
        "--ns",
        str(training.get("ns", 2000)),
    ]
    if bool(training.get("detach_ot_weights", True)):
        cli_args.append("--detach_ot_weights")
    cli_args += extra

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + cli_args
        cfg = config()
        if not args.skip_train:
            cfg = train.run(cfg, init_config)
        else:
            cfg = init_config(cfg)

        if not args.skip_eval:
            evaluate_fit(cfg, init_config, use_loss="mioemd2")
    finally:
        sys.argv = old_argv

    if not args.skip_plots:
        python_bin = sys.executable
        checkpoint_epoch = checkpoint_epoch_from_config(yaml_cfg, fallback="auto")
        output_label = str(yaml_cfg["data"].get("label") or yaml_cfg["experiment"].get("name", run_name))
        _run(
            [
                python_bin,
                str(METHOD_ROOT / "plot.py"),
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
        _run(
            [
                python_bin,
                str(METHOD_ROOT / "diagnose.py"),
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


if __name__ == "__main__":
    main()
