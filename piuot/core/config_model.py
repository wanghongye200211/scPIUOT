import argparse
import os
from pathlib import Path
from typing import List, Tuple

import anndata as ad
import numpy as np
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]


def _sync_hjb_aliases(args):
    lambda_hjb = float(getattr(args, "lambda_hjb", getattr(args, "train_lambda", 0.0)))
    args.lambda_hjb = lambda_hjb
    args.train_lambda = lambda_hjb
    return args


def _constraint_schedule_tag(args):
    for loss_name in ("density", "action", "hjb"):
        start_epoch = getattr(args, f"{loss_name}_start_epoch", None)
        ramp_epochs = getattr(args, f"{loss_name}_ramp_epochs", None)
        if start_epoch is not None or ramp_epochs is not None:
            return "-late"
    return ""


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument("--device_type", default="auto")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--out_dir", default=str(METHOD_ROOT / "output"))
    parser.add_argument("--run_name", default="piuot_run")

    # data options
    parser.add_argument(
        "--data_path",
        default=str(METHOD_ROOT / "data" / "input" / "input.h5ad"),
    )
    parser.add_argument("--embedding_key", default="X_latent")
    parser.add_argument("--time_key", default="time_bin")
    parser.add_argument("--raw_time_key", default="t")

    # model options
    parser.add_argument("--k_dims", nargs="+", default=[400, 400], type=int)
    parser.add_argument("--activation", default="softplus")
    parser.add_argument("--sigma_type", default="const")
    parser.add_argument("--sigma_const", default=0.1, type=float)
    parser.add_argument("--solver_dt", default=0.1, type=float)
    parser.add_argument("--use_growth", action="store_true", default=True)
    parser.add_argument("--growth_mode", default="bounded", choices=["free", "bounded"])
    parser.add_argument("--growth_scale", default=0.05, type=float)
    parser.add_argument("--hjb_growth_coeff", default=2.0, type=float)

    # train options
    parser.add_argument("--train_epochs", default=500, type=int)
    parser.add_argument("--train_lr", default=0.005, type=float)
    parser.add_argument("--lambda_ot", default=1.0, type=float)
    parser.add_argument("--lambda_hjb", "--train_lambda", dest="lambda_hjb", default=0.05, type=float)
    parser.add_argument("--train_batch", default=1.0, type=float)
    parser.add_argument("--train_clip", default=0.1, type=float)
    parser.add_argument("--save", default=50, type=int)
    parser.add_argument("--use_deepruot_curriculum", action="store_true", default=False)
    parser.add_argument("--pretrain_epochs", default=0, type=int)
    parser.add_argument("--pretrain_refine_epochs", default=0, type=int)
    parser.add_argument("--pretrain_lr", default=1e-4, type=float)
    parser.add_argument("--pretrain_refine_lr", default=1e-5, type=float)
    parser.add_argument("--pretrain_lambda_ot", default=1.0, type=float)
    parser.add_argument("--pretrain_lambda_mass", default=0.01, type=float)
    parser.add_argument("--pretrain_refine_lambda_ot", default=1.0, type=float)
    parser.add_argument("--pretrain_refine_lambda_mass", default=0.01, type=float)
    parser.add_argument(
        "--pretrain_use_density",
        dest="pretrain_use_density",
        action="store_true",
    )
    parser.add_argument(
        "--no_pretrain_use_density",
        dest="pretrain_use_density",
        action="store_false",
    )
    parser.add_argument("--pretrain_use_hjb", action="store_true", default=False)
    parser.add_argument("--pretrain_use_action", action="store_true", default=False)
    parser.add_argument(
        "--pretrain_refine_use_density",
        dest="pretrain_refine_use_density",
        action="store_true",
    )
    parser.add_argument(
        "--no_pretrain_refine_use_density",
        dest="pretrain_refine_use_density",
        action="store_false",
    )
    parser.add_argument("--pretrain_refine_use_hjb", action="store_true", default=False)
    parser.add_argument("--pretrain_refine_use_action", action="store_true", default=False)
    parser.add_argument("--curriculum_detach_ot_weights", action="store_true", default=False)
    parser.add_argument("--init_model_checkpoint", default=None)
    parser.add_argument("--lambda_density", default=0.05, type=float)
    parser.add_argument("--lambda_action", default=0.01, type=float)
    parser.add_argument("--action_alpha_g", default=1.0, type=float)
    parser.add_argument("--action_alpha_sigma", default=1e-4, type=float)
    parser.add_argument("--density_top_k", default=5, type=int)
    parser.add_argument("--density_hinge_value", default=0.01, type=float)
    parser.add_argument("--detach_ot_weights", action="store_true", default=True)
    parser.add_argument("--lambda_mass", default=0.1, type=float)
    parser.add_argument("--lambda_global_mass", default=1.0, type=float)
    parser.add_argument("--lambda_local_mass", default=1.0, type=float)
    parser.add_argument(
        "--local_mass_loss_mode",
        default="absolute_l2",
        choices=["absolute_l2", "distribution_l1", "distribution_kl"],
    )
    parser.add_argument("--local_mass_smoothing", default=1e-6, type=float)
    parser.add_argument("--mass_start_epoch", default=0, type=int)
    parser.add_argument("--mass_ramp_epochs", default=0, type=int)
    parser.add_argument("--global_mass_start_epoch", default=200, type=int)
    parser.add_argument("--global_mass_ramp_epochs", default=0, type=int)
    parser.add_argument("--mass_clip_value", default=30.0, type=float)
    parser.add_argument("--lambda_g_reg", default=0.0, type=float)
    parser.add_argument("--use_segment_training", action="store_true", default=True)
    parser.add_argument("--segment_regularization_points", default=5, type=int)
    parser.add_argument("--stage_transition_epoch", default=200, type=int)
    parser.add_argument("--stage2_lr", default=0.0005, type=float)
    parser.add_argument("--constraint_start_epoch", default=230, type=int)
    parser.add_argument("--constraint_ramp_epochs", default=30, type=int)
    parser.add_argument("--density_start_epoch", default=None, type=int)
    parser.add_argument("--density_ramp_epochs", default=None, type=int)
    parser.add_argument("--action_start_epoch", default=None, type=int)
    parser.add_argument("--action_ramp_epochs", default=None, type=int)
    parser.add_argument("--hjb_start_epoch", default=None, type=int)
    parser.add_argument("--hjb_ramp_epochs", default=None, type=int)
    parser.add_argument(
        "--reload_best_on_stage_transition",
        dest="reload_best_on_stage_transition",
        action="store_true",
    )
    parser.add_argument(
        "--no_reload_best_on_stage_transition",
        dest="reload_best_on_stage_transition",
        action="store_false",
    )
    parser.add_argument(
        "--reset_optimizer_on_stage_transition",
        dest="reset_optimizer_on_stage_transition",
        action="store_true",
    )
    parser.add_argument(
        "--no_reset_optimizer_on_stage_transition",
        dest="reset_optimizer_on_stage_transition",
        action="store_false",
    )
    parser.add_argument(
        "--reset_best_on_stage_transition",
        dest="reset_best_on_stage_transition",
        action="store_true",
    )
    parser.add_argument(
        "--no_reset_best_on_stage_transition",
        dest="reset_best_on_stage_transition",
        action="store_false",
    )

    # test options
    parser.add_argument("--evaluate_n", default=4000, type=int)
    parser.add_argument("--evaluate_data")
    parser.add_argument("--evaluate-baseline", action="store_true")

    # run options
    parser.add_argument("--task", default="fate")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate")
    parser.add_argument("--config")

    # loss / solver options
    parser.add_argument("--sinkhorn_scaling", default=0.7, type=float)
    parser.add_argument("--sinkhorn_blur", default=0.1, type=float)
    parser.add_argument("--ns", default=2000, type=float)

    parser.add_argument("--start_t", default=0, type=int)
    parser.add_argument("--train_t", nargs="*", type=int, default=None)

    parser.set_defaults(
        pretrain_use_density=True,
        pretrain_refine_use_density=True,
        reload_best_on_stage_transition=True,
        reset_optimizer_on_stage_transition=True,
        reset_best_on_stage_transition=False,
    )

    args = parser.parse_known_args()[0]
    args = _sync_hjb_aliases(args)
    args.layers = len(args.k_dims)
    return args


def init_config(args):
    args = _sync_hjb_aliases(args)
    args.layers = len(args.k_dims)
    args.kDims = "_".join(map(str, args.k_dims))
    args.constraint_schedule_tag = _constraint_schedule_tag(args)

    name = (
        "{activation}-{kDims}-"
        "hjb{lambda_hjb}-{sigma_type}-{sigma_const}-"
        "dt{solver_dt}-"
        "{train_clip}-{train_lr}-"
        "ot{lambda_ot}-"
        "curr{use_deepruot_curriculum}-"
        "pte{pretrain_epochs}-"
        "prte{pretrain_refine_epochs}-"
        "density{lambda_density}-"
        "action{lambda_action}-"
        "growth{use_growth}-"
        "gmode{growth_mode}-"
        "gscale{growth_scale}-"
        "hgb{hjb_growth_coeff}-"
        "seg{use_segment_training}-"
        "dotw{detach_ot_weights}-"
        "lmloss{local_mass_loss_mode}-"
        "greg{lambda_g_reg}-"
        "stage{stage_transition_epoch}-"
        "s2lr{stage2_lr}-"
        "cstart{constraint_start_epoch}-"
        "cramp{constraint_ramp_epochs}"
        "{constraint_schedule_tag}-"
        "gms{global_mass_start_epoch}-"
        "gmr{global_mass_ramp_epochs}-"
        "mass{lambda_mass}"
    ).format(**args.__dict__)

    args.out_dir = os.path.join(args.out_dir, args.run_name, name, "seed_{}".format(args.seed))
    if args.task == "leaveout":
        args.out_dir = os.path.join(args.out_dir, args.leaveout_t)
    else:
        args.out_dir = os.path.join(args.out_dir, "alltime")

    if not os.path.exists(args.out_dir):
        print("Making directory at {}".format(args.out_dir))
        os.makedirs(args.out_dir)
    else:
        print("Directory exists at {}".format(args.out_dir))

    args.train_pt = os.path.join(args.out_dir, "train.{}.pt")
    args.done_log = os.path.join(args.out_dir, "done.log")
    args.config_pt = os.path.join(args.out_dir, "config.pt")
    args.train_log = os.path.join(args.out_dir, "train.log")
    return args


def _to_dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _load_h5ad_latent(args) -> Tuple[List[torch.Tensor], List[np.float64], argparse.Namespace]:
    adata = ad.read_h5ad(args.data_path)

    if args.embedding_key == "X":
        embedding = _to_dense_float32(adata.X)
    elif args.embedding_key in adata.obsm:
        embedding = _to_dense_float32(adata.obsm[args.embedding_key])
    else:
        raise KeyError(f"Embedding key '{args.embedding_key}' not found in h5ad.")

    if args.time_key not in adata.obs.columns:
        raise KeyError(f"time key '{args.time_key}' not found in h5ad.")

    time_index_values = np.asarray(adata.obs[args.time_key], dtype=np.float64)
    raw_time_values = (
        np.asarray(adata.obs[args.raw_time_key], dtype=np.float64)
        if args.raw_time_key in adata.obs.columns
        else time_index_values.copy()
    )

    unique_times = np.sort(np.unique(time_index_values))
    x = []
    y = []
    for t_value in unique_times:
        mask = np.isclose(time_index_values, t_value)
        x.append(torch.tensor(embedding[mask], dtype=torch.float32))
        y.append(np.float64(np.median(raw_time_values[mask])))

    args.time_index_values = [float(v) for v in unique_times.tolist()]
    args.raw_time_values = [float(v) for v in y]
    args.x_dim = x[0].shape[-1]
    args.group_sizes = [int(x_i.shape[0]) for x_i in x]
    args.relative_mass_by_time = [float(size / max(args.group_sizes[0], 1)) for size in args.group_sizes]

    if args.train_t is None or len(args.train_t) == 0:
        args.train_t = list(range(1, len(x)))

    return x, y, args


def load_data(args, base_dir="."):
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)
    args.data_path = data_path

    if data_path.endswith(".pt"):
        data_pt = torch.load(data_path)
        x = data_pt["xp"]
        y = data_pt["y"]
        args.x_dim = x[0].shape[-1]
        if args.train_t is None or len(args.train_t) == 0:
            args.train_t = list(range(1, len(x)))
        return x, y, args

    if data_path.endswith(".h5ad"):
        return _load_h5ad_latent(args)

    raise ValueError("Unsupported data format. Expected .pt or .h5ad.")
