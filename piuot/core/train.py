import torch
from torch import optim
import numpy as np
import tqdm
import shutil
import torch.nn.functional as F
from pathlib import Path
from src.model import ForwardSDE
import os
from src.config_model import load_data
from src.mio_losses import mioflow_density_loss, mioflow_emd2_loss


def sync_hjb_aliases(config):
    lambda_hjb = float(getattr(config, "lambda_hjb", getattr(config, "train_lambda", 0.0)))
    setattr(config, "lambda_hjb", lambda_hjb)
    setattr(config, "train_lambda", lambda_hjb)
    return lambda_hjb


def p_samp(p, num_samp, w=None):
    num_samp = int(num_samp)
    repflag = p.shape[0] < num_samp
    if repflag:
        p_sub = torch.randint(p.shape[0], (num_samp,), device=p.device)
    else:
        p_sub = torch.randperm(p.shape[0], device=p.device)[:num_samp]

    if w is None:
        w_ = torch.full((num_samp,), 1.0 / max(num_samp, 1), dtype=p.dtype, device=p.device)
    else:
        w_ = w.index_select(0, p_sub).clone()
        w_ = w_ / w_.sum().clamp_min(1e-12)

    return p.index_select(0, p_sub).clone(), w_


def _move_time_series_to_device(x, device):
    return [x_i.to(device) if x_i.device != device else x_i for x_i in x]


def clip_logw(logw: torch.Tensor, clip_value: float = 30.0) -> torch.Tensor:
    logw = torch.nan_to_num(
        logw,
        nan=0.0,
        posinf=float(clip_value),
        neginf=-float(clip_value),
    )
    return torch.clamp(logw, min=-float(clip_value), max=float(clip_value))


def build_initial_state(x_i, use_growth: bool, logw_i: torch.Tensor = None, clip_value: float = 30.0):
    r_i = x_i.new_zeros((x_i.shape[0], 1))
    if use_growth:
        if logw_i is None:
            init_mass = 1.0 / max(x_i.shape[0], 1)
            logw_i = x_i.new_full((x_i.shape[0], 1), float(np.log(init_mass)))
        else:
            logw_i = clip_logw(logw_i, clip_value=clip_value)
        return torch.cat([x_i, r_i, logw_i], dim=1)
    return torch.cat([x_i, r_i], dim=1)


def unpack_state(state, x_dim: int, use_growth: bool):
    x = state[:, :x_dim]
    r = state[:, x_dim:x_dim + 1]
    logw = state[:, x_dim + 1:x_dim + 2] if use_growth else None
    return x, r, logw


def normalized_mass_from_logw(logw: torch.Tensor, clip_value: float = 30.0) -> torch.Tensor:
    logw = clip_logw(logw, clip_value=clip_value)
    return torch.softmax(logw.squeeze(-1), dim=0)


def stable_exp_weights(logw: torch.Tensor, clip_value: float = 30.0) -> torch.Tensor:
    logw = clip_logw(logw, clip_value=clip_value)
    weights = torch.exp(logw).squeeze(-1)
    return torch.nan_to_num(weights, nan=0.0, posinf=1e12, neginf=0.0)


def compute_mass_losses(
    target: torch.Tensor,
    pred_x: torch.Tensor,
    pred_logw: torch.Tensor,
    relative_mass_now: float,
    clip_value: float,
    local_mass_loss_mode: str = "absolute_l2",
    local_mass_smoothing: float = 1e-6,
):
    weights = stable_exp_weights(pred_logw, clip_value=clip_value)
    relative_mass_now = pred_x.new_tensor(float(relative_mass_now)).clamp_min(1e-12)
    pred_total_mass = weights.sum().clamp_min(1e-12)
    global_mass_loss = (pred_total_mass - relative_mass_now).pow(2)

    distances = torch.cdist(target, pred_x)
    indices = torch.argmin(distances, dim=1)
    count = torch.zeros_like(weights)
    count.scatter_add_(0, indices, torch.ones_like(indices, dtype=weights.dtype))
    relative_count = count / max(target.shape[0], 1)
    if local_mass_loss_mode == "distribution_l1":
        pred_distribution = weights / pred_total_mass
        local_mass_loss = 0.5 * torch.sum(torch.abs(pred_distribution - relative_count))
    elif local_mass_loss_mode == "distribution_kl":
        pred_distribution = weights / pred_total_mass
        eps = float(max(local_mass_smoothing, 1e-12))
        pred_distribution = pred_distribution.clamp_min(eps)
        pred_distribution = pred_distribution / pred_distribution.sum().clamp_min(1e-12)
        target_distribution = relative_count.clamp_min(eps)
        target_distribution = target_distribution / target_distribution.sum().clamp_min(1e-12)
        local_mass_loss = F.kl_div(
            torch.log(pred_distribution),
            target_distribution,
            reduction="batchmean",
        )
    else:
        local_mass_loss = torch.norm(weights - relative_mass_now * relative_count, p=2).pow(2)
    return global_mass_loss, local_mass_loss


def training_segments(config):
    anchors = [int(config.start_t)] + [int(t) for t in config.train_t]
    return list(zip(anchors[:-1], anchors[1:]))


def segment_time_grid(t_start: float, t_end: float, n_points: int):
    n_points = max(2, int(n_points))
    return [np.float64(t) for t in np.linspace(float(t_start), float(t_end), n_points)]


def mass_scale_for_epoch(config, epoch: int) -> float:
    base = float(getattr(config, "lambda_mass", 0.0))
    if base <= 0:
        return 0.0

    start_epoch = int(getattr(config, "mass_start_epoch", 0))
    ramp_epochs = int(getattr(config, "mass_ramp_epochs", 0))
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return base

    progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
    return base * progress


def ot_scale(config) -> float:
    return float(getattr(config, "lambda_ot", 1.0))


def global_mass_weight_for_epoch(config, epoch: int) -> float:
    base = float(getattr(config, "lambda_global_mass", 0.0))
    if base <= 0:
        return 0.0

    start_epoch = int(getattr(config, "global_mass_start_epoch", getattr(config, "mass_start_epoch", 0)))
    ramp_epochs = int(getattr(config, "global_mass_ramp_epochs", 0))
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return base

    progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
    return base * progress


def constraint_scale_for_epoch(config, epoch: int) -> float:
    start_epoch = int(getattr(config, "constraint_start_epoch", 0))
    ramp_epochs = int(getattr(config, "constraint_ramp_epochs", 0))
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
    return progress


def _constraint_schedule_for_attr(config, attr_name: str):
    loss_prefix = {
        "lambda_density": "density",
        "lambda_action": "action",
        "lambda_hjb": "hjb",
        "train_lambda": "hjb",
    }.get(attr_name)
    if loss_prefix is None:
        return (
            int(getattr(config, "constraint_start_epoch", 0)),
            int(getattr(config, "constraint_ramp_epochs", 0)),
        )

    start_epoch = getattr(config, f"{loss_prefix}_start_epoch", None)
    ramp_epochs = getattr(config, f"{loss_prefix}_ramp_epochs", None)
    if start_epoch is None:
        start_epoch = getattr(config, "constraint_start_epoch", 0)
    if ramp_epochs is None:
        ramp_epochs = getattr(config, "constraint_ramp_epochs", 0)
    return int(start_epoch), int(ramp_epochs)


def constraint_scale_for_attr(config, epoch: int, attr_name: str) -> float:
    start_epoch, ramp_epochs = _constraint_schedule_for_attr(config, attr_name)
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
    return progress


def weighted_constraint_value(config, epoch: int, attr_name: str) -> float:
    if attr_name == "lambda_hjb":
        sync_hjb_aliases(config)
    base = float(getattr(config, attr_name, 0.0))
    if base <= 0:
        return 0.0
    return base * constraint_scale_for_attr(config, epoch, attr_name)


def growth_regularization(model, trajectory, ts, config):
    if (
        not getattr(config, "use_growth", False)
        or float(getattr(config, "lambda_g_reg", 0.0)) <= 0
    ):
        return trajectory[0].new_tensor(0.0)

    values = []
    for idx, t_value in enumerate(ts):
        state_t = trajectory[idx]
        x_t, _, _ = unpack_state(state_t, config.x_dim, config.use_growth)
        t_batch = x_t.new_full((x_t.shape[0], 1), float(t_value))
        xt = torch.cat([x_t, t_batch], dim=1)
        g_t = model._func._growth(xt).squeeze(-1)
        values.append(g_t.pow(2).mean())

    interval = max(float(ts[-1] - ts[0]), 1e-6)
    return torch.stack(values).mean() * interval


def action_regularization(model, trajectory, ts, config):
    if float(getattr(config, "lambda_action", 0.0)) <= 0:
        return trajectory[0].new_tensor(0.0)

    alpha_g = float(getattr(config, "action_alpha_g", 1.0))
    alpha_sigma = float(getattr(config, "action_alpha_sigma", 1e-4))
    clip_value = float(getattr(config, "mass_clip_value", 30.0))
    terms = []
    for idx in range(len(ts) - 1):
        state_t = trajectory[idx]
        z_t, _, logw_t = unpack_state(state_t, config.x_dim, config.use_growth)
        t_batch = z_t.new_full((z_t.shape[0], 1), float(ts[idx]))
        xt = torch.cat([z_t, t_batch], dim=1)
        drift_t = model._func._drift(xt)
        growth_t = (
            model._func._growth(xt).squeeze(-1)
            if getattr(config, "use_growth", False)
            else z_t.new_zeros((z_t.shape[0],))
        )
        sigma_diag = model._func.g(float(ts[idx]), state_t)[:, :config.x_dim]

        if logw_t is not None:
            w = stable_exp_weights(logw_t, clip_value=clip_value)
            w_bar = w / w.sum().clamp_min(1e-12)
        else:
            w_bar = z_t.new_full((z_t.shape[0],), 1.0 / max(z_t.shape[0], 1))

        dt = float(ts[idx + 1] - ts[idx])
        term = (
            torch.sum(drift_t.pow(2), dim=-1)
            + alpha_g * growth_t.pow(2)
            + alpha_sigma * torch.sum(sigma_diag.pow(2), dim=-1)
        )
        terms.append(dt * torch.sum(w_bar * term))

    return torch.stack(terms).sum() if terms else trajectory[0].new_tensor(0.0)


def combined_mass_loss(config, target, pred_x, pred_logw, relative_mass_now):
    if (
        (not getattr(config, "use_growth", False))
        or pred_logw is None
    ):
        zero = pred_x.new_tensor(0.0)
        return zero, zero, zero, zero

    current_lambda_mass = mass_scale_for_epoch(config, int(getattr(config, "train_epoch", 0)))
    if current_lambda_mass <= 0:
        zero = pred_x.new_tensor(0.0)
        return zero, zero, zero, zero
    current_global_mass_weight = global_mass_weight_for_epoch(
        config,
        int(getattr(config, "train_epoch", 0)),
    )

    global_mass_loss, local_mass_loss = compute_mass_losses(
        target=target,
        pred_x=pred_x,
        pred_logw=pred_logw,
        relative_mass_now=relative_mass_now,
        clip_value=float(getattr(config, "mass_clip_value", 30.0)),
        local_mass_loss_mode=str(getattr(config, "local_mass_loss_mode", "absolute_l2")),
        local_mass_smoothing=float(getattr(config, "local_mass_smoothing", 1e-6)),
    )
    weighted_mass_loss = current_lambda_mass * (
        current_global_mass_weight * global_mass_loss
        + config.lambda_local_mass * local_mass_loss
    )
    return (
        weighted_mass_loss,
        global_mass_loss,
        local_mass_loss,
        pred_x.new_tensor(current_lambda_mass),
    )



def init_device(args):
    device_type = str(getattr(args, "device_type", "auto")).lower()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if device_type == "cuda":
        args.cuda = args.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    elif device_type == "mps":
        args.cuda = False
        device = torch.device("mps" if has_mps else "cpu")
    elif device_type == "cpu":
        args.cuda = False
        device = torch.device("cpu")
    else:
        args.cuda = args.use_cuda and torch.cuda.is_available()
        if args.cuda:
            device = torch.device("cuda:{}".format(args.device))
        elif has_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return device





class ObservationLoss:

    def __init__(self, config):
        self.config = config

    def __call__(self, source_mass, source, target_mass, target):
        ot_loss = mioflow_emd2_loss(
            source,
            target,
            source_mass=source_mass,
            target_mass=target_mass,
            detach_weights=self.config.detach_ot_weights,
        )

        density_loss = torch.tensor(0.0, device=source.device)
        if self.config.lambda_density > 0:
            density_loss = mioflow_density_loss(
                source,
                target,
                top_k=self.config.density_top_k,
                hinge_value=self.config.density_hinge_value,
            )

        return ot_loss, density_loss


def _zero_metrics():
    return {
        "loss_xy": 0.0,
        "loss_density": 0.0,
        "loss_mass": 0.0,
        "loss_global_mass": 0.0,
        "loss_local_mass": 0.0,
        "mass_scale": 0.0,
        "loss_g_reg": 0.0,
        "loss_action": 0.0,
        "loss_r": 0.0,
        "train_objective": 0.0,
    }


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _checkpoint_payload(model, epoch: int):
    return {
        'model_state_dict': model.state_dict(),
        'epoch': int(epoch),
    }


def _extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _load_model_checkpoint(model, checkpoint_path):
    if checkpoint_path in (None, ""):
        return None
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(_extract_model_state_dict(checkpoint), strict=True)
    return path


def _config_snapshot(config, names):
    return {name: getattr(config, name) for name in names}


def _apply_config_overrides(config, overrides):
    for name, value in overrides.items():
        setattr(config, name, value)


def _phase_overrides(config, *, use_density, use_hjb, use_action, use_global_mass):
    constraint_on = bool(use_density or use_hjb or use_action)
    original_global_mass = float(getattr(config, "lambda_global_mass", 0.0))
    original_lambda_hjb = sync_hjb_aliases(config)
    return {
        "detach_ot_weights": bool(
            getattr(config, "curriculum_detach_ot_weights", getattr(config, "detach_ot_weights", False))
        ),
        "lambda_density": float(getattr(config, "lambda_density", 0.0)) if use_density else 0.0,
        "lambda_hjb": original_lambda_hjb if use_hjb else 0.0,
        "train_lambda": original_lambda_hjb if use_hjb else 0.0,
        "lambda_action": float(getattr(config, "lambda_action", 0.0)) if use_action else 0.0,
        "lambda_global_mass": original_global_mass if use_global_mass else 0.0,
        "global_mass_start_epoch": 0 if use_global_mass else 999999,
        "global_mass_ramp_epochs": 0 if use_global_mass else 0,
        "constraint_start_epoch": 0 if constraint_on else 999999,
        "constraint_ramp_epochs": 0 if constraint_on else 0,
        "density_start_epoch": 0 if use_density else 999999,
        "density_ramp_epochs": 0 if use_density else 0,
        "action_start_epoch": 0 if use_action else 999999,
        "action_ramp_epochs": 0 if use_action else 0,
        "hjb_start_epoch": 0 if use_hjb else 999999,
        "hjb_ramp_epochs": 0 if use_hjb else 0,
    }


def _train_epoch_full(model, loss_fn, config, x, y, optimizer):
    clip_value = float(getattr(config, "mass_clip_value", 30.0))
    train_t = list(config.train_t)
    y_ts = [np.float64(y[config.start_t])] + [np.float64(y[ts_i]) for ts_i in train_t]

    losses_xy = []
    losses_density = []
    losses_mass = []
    losses_global_mass = []
    losses_local_mass = []
    losses_r = []
    losses_g_reg = []
    losses_action = []
    mass_scales = []
    current_epoch = int(getattr(config, "train_epoch", 0))
    ot_weight = ot_scale(config)
    density_weight = weighted_constraint_value(config, current_epoch, "lambda_density")
    hj_weight = weighted_constraint_value(config, current_epoch, "lambda_hjb")
    action_weight = weighted_constraint_value(config, current_epoch, "lambda_action")

    dat_prev = x[config.start_t]
    x_i, a_i = p_samp(dat_prev, int(dat_prev.shape[0] * config.train_batch))
    x_r_i = build_initial_state(x_i, config.use_growth, clip_value=clip_value)
    x_r_s = model(y_ts, x_r_i)

    total_loss = None
    for position, t_cur in enumerate(train_t, start=1):
        dat_cur = x[t_cur]
        y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * config.train_batch))
        pred_x, pred_r, pred_logw = unpack_state(x_r_s[position], config.x_dim, config.use_growth)
        pred_mass = (
            normalized_mass_from_logw(pred_logw, clip_value=clip_value)
            if pred_logw is not None else a_i
        )
        loss_xy, loss_density = loss_fn(pred_mass, pred_x, b_j, y_j)
        loss_mass, global_mass_loss, local_mass_loss, current_mass_scale = combined_mass_loss(
            config=config,
            target=y_j,
            pred_x=pred_x,
            pred_logw=pred_logw,
            relative_mass_now=config.relative_mass_by_time[t_cur],
        )

        losses_xy.append(loss_xy.item())
        losses_density.append(loss_density.item())
        losses_mass.append(loss_mass.item())
        losses_global_mass.append(global_mass_loss.item())
        losses_local_mass.append(local_mass_loss.item())
        mass_scales.append(current_mass_scale.item())

        if (hj_weight > 0) and (t_cur == train_t[-1]):
            _, final_r, _ = unpack_state(x_r_s[-1], config.x_dim, config.use_growth)
            loss_r = torch.mean(final_r * hj_weight)
            losses_r.append(loss_r.item())
            loss_term = ot_weight * loss_xy + density_weight * loss_density + loss_mass + loss_r
        else:
            loss_term = ot_weight * loss_xy + density_weight * loss_density + loss_mass

        total_loss = loss_term if total_loss is None else total_loss + loss_term

    growth_penalty_raw = growth_regularization(model, x_r_s, y_ts, config)
    growth_penalty = float(getattr(config, "lambda_g_reg", 0.0)) * growth_penalty_raw
    losses_g_reg.append(growth_penalty.item())
    action_penalty = action_weight * action_regularization(model, x_r_s, y_ts, config)
    losses_action.append(action_penalty.item())
    total_loss = total_loss + growth_penalty + action_penalty

    optimizer.zero_grad()
    total_loss.backward()
    if config.train_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
    optimizer.step()
    model.zero_grad()

    metrics = _zero_metrics()
    metrics["loss_xy"] = _mean_or_zero(losses_xy)
    metrics["loss_density"] = _mean_or_zero(losses_density)
    metrics["loss_mass"] = _mean_or_zero(losses_mass)
    metrics["loss_global_mass"] = _mean_or_zero(losses_global_mass)
    metrics["loss_local_mass"] = _mean_or_zero(losses_local_mass)
    metrics["mass_scale"] = _mean_or_zero(mass_scales)
    metrics["loss_g_reg"] = _mean_or_zero(losses_g_reg)
    metrics["loss_action"] = _mean_or_zero(losses_action)
    metrics["loss_r"] = _mean_or_zero(losses_r)
    metrics["train_objective"] = (
        ot_weight * metrics["loss_xy"]
        + density_weight * metrics["loss_density"]
        + metrics["loss_mass"]
        + metrics["loss_g_reg"]
        + metrics["loss_action"]
        + metrics["loss_r"]
    )
    return metrics


def _train_epoch_segmented(model, loss_fn, config, x, y, optimizer):
    clip_value = float(getattr(config, "mass_clip_value", 30.0))
    segment_points = int(getattr(config, "segment_regularization_points", 5))

    losses_xy = []
    losses_density = []
    losses_mass = []
    losses_global_mass = []
    losses_local_mass = []
    losses_r = []
    losses_g_reg = []
    losses_action = []
    mass_scales = []
    current_epoch = int(getattr(config, "train_epoch", 0))
    ot_weight = ot_scale(config)
    density_weight = weighted_constraint_value(config, current_epoch, "lambda_density")
    hj_weight = weighted_constraint_value(config, current_epoch, "lambda_hjb")
    action_weight = weighted_constraint_value(config, current_epoch, "lambda_action")

    dat_prev = x[config.start_t]
    current_x, current_mass = p_samp(dat_prev, int(dat_prev.shape[0] * config.train_batch))
    current_logw = None

    for t_prev, t_cur in training_segments(config):
        ts_segment = segment_time_grid(y[t_prev], y[t_cur], segment_points)
        x_r_i = build_initial_state(
            current_x,
            config.use_growth,
            logw_i=current_logw,
            clip_value=clip_value,
        )
        x_r_s = model(ts_segment, x_r_i)

        target = x[t_cur]
        y_j, b_j = p_samp(target, int(target.shape[0] * config.train_batch))
        pred_x, pred_r, pred_logw = unpack_state(x_r_s[-1], config.x_dim, config.use_growth)
        pred_mass = (
            normalized_mass_from_logw(pred_logw, clip_value=clip_value)
            if pred_logw is not None else current_mass
        )

        loss_xy, loss_density = loss_fn(pred_mass, pred_x, b_j, y_j)
        loss_mass, global_mass_loss, local_mass_loss, current_mass_scale = combined_mass_loss(
            config=config,
            target=y_j,
            pred_x=pred_x,
            pred_logw=pred_logw,
            relative_mass_now=config.relative_mass_by_time[t_cur],
        )
        growth_penalty_raw = growth_regularization(model, x_r_s, ts_segment, config)
        growth_penalty = float(getattr(config, "lambda_g_reg", 0.0)) * growth_penalty_raw
        loss_r = pred_x.new_tensor(0.0)
        if hj_weight > 0:
            loss_r = torch.mean(pred_r * hj_weight)

        action_penalty = action_weight * action_regularization(model, x_r_s, ts_segment, config)
        loss_term = (
            ot_weight * loss_xy
            + density_weight * loss_density
            + loss_mass
            + growth_penalty
            + action_penalty
            + loss_r
        )

        optimizer.zero_grad()
        loss_term.backward()
        if config.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
        optimizer.step()
        model.zero_grad()

        losses_xy.append(loss_xy.item())
        losses_density.append(loss_density.item())
        losses_mass.append(loss_mass.item())
        losses_global_mass.append(global_mass_loss.item())
        losses_local_mass.append(local_mass_loss.item())
        mass_scales.append(current_mass_scale.item())
        losses_g_reg.append(growth_penalty.item())
        losses_action.append(action_penalty.item())
        losses_r.append(loss_r.item())

        current_x = pred_x.detach()
        current_logw = clip_logw(pred_logw.detach(), clip_value=clip_value) if pred_logw is not None else None
        current_mass = pred_mass.detach()

    metrics = _zero_metrics()
    metrics["loss_xy"] = _mean_or_zero(losses_xy)
    metrics["loss_density"] = _mean_or_zero(losses_density)
    metrics["loss_mass"] = _mean_or_zero(losses_mass)
    metrics["loss_global_mass"] = _mean_or_zero(losses_global_mass)
    metrics["loss_local_mass"] = _mean_or_zero(losses_local_mass)
    metrics["mass_scale"] = _mean_or_zero(mass_scales)
    metrics["loss_g_reg"] = _mean_or_zero(losses_g_reg)
    metrics["loss_action"] = _mean_or_zero(losses_action)
    metrics["loss_r"] = _mean_or_zero(losses_r)
    metrics["train_objective"] = (
        ot_weight * metrics["loss_xy"]
        + density_weight * metrics["loss_density"]
        + metrics["loss_mass"]
        + metrics["loss_g_reg"]
        + metrics["loss_action"]
        + metrics["loss_r"]
    )
    return metrics


def _format_train_desc(epoch, metrics, best_train_objective, phase_label="train"):
    desc = "[{}] {}".format(phase_label, epoch + 1)
    desc += " {:.6f}".format(metrics["loss_xy"])
    desc += " {:.6f}".format(metrics["loss_density"])
    desc += " {:.6f}".format(metrics["loss_mass"])
    desc += " {:.6f}".format(metrics["loss_global_mass"])
    desc += " {:.6f}".format(metrics["loss_local_mass"])
    desc += " {:.6f}".format(metrics["mass_scale"])
    desc += " {:.6f}".format(metrics["loss_g_reg"])
    desc += " {:.6f}".format(metrics["loss_action"])
    desc += " {:.6f}".format(metrics["loss_r"])
    desc += " {:.6f}".format(best_train_objective)
    return desc


def _make_optimizer_and_scheduler(config, model, lr=None):
    if lr is None:
        lr = float(config.train_lr)
    optimizer = optim.Adam(list(model.parameters()), lr=float(lr))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    return optimizer, scheduler


def _train_phase(
    *,
    model,
    loss_fn,
    config,
    x,
    y,
    optimizer,
    scheduler,
    n_epochs,
    phase_label,
    log_handle,
    best_checkpoint_path,
    save_epoch_checkpoints=False,
    epoch_offset=0,
):
    best_phase_objective = np.inf
    pbar = tqdm.tqdm(range(int(n_epochs)))
    for local_epoch in pbar:
        config.train_epoch = local_epoch
        if getattr(config, "use_segment_training", False):
            metrics = _train_epoch_segmented(model, loss_fn, config, x, y, optimizer)
        else:
            metrics = _train_epoch_full(model, loss_fn, config, x, y, optimizer)
        scheduler.step()

        desc = _format_train_desc(local_epoch, metrics, best_phase_objective, phase_label=phase_label)
        pbar.set_description(desc)
        log_handle.write(desc + '\n')
        log_handle.flush()

        if metrics["train_objective"] < best_phase_objective:
            best_phase_objective = metrics["train_objective"]
            torch.save(
                _checkpoint_payload(model, epoch_offset + local_epoch + 1),
                best_checkpoint_path,
            )

        if save_epoch_checkpoints and (
            (local_epoch + 1) % config.save == 0 or (local_epoch + 1) == int(n_epochs)
        ):
            epoch_num = epoch_offset + local_epoch + 1
            epoch_tag = f"epoch_{str(epoch_num).rjust(6, '0')}"
            torch.save(
                _checkpoint_payload(model, epoch=epoch_num),
                config.train_pt.format(epoch_tag),
            )

    return best_phase_objective


def _maybe_apply_stage_transition(config, epoch, model, optimizer, scheduler):
    transition_epoch = int(
        getattr(
            config,
            "stage_transition_epoch",
            getattr(config, "global_mass_start_epoch", -1),
        )
    )
    if transition_epoch < 0 or epoch != transition_epoch:
        return optimizer, scheduler, False

    best_path = config.train_pt.format("best")
    stage1_best_path = os.path.join(config.out_dir, "train.stage1_best.pt")
    stage1_transition_path = os.path.join(config.out_dir, "train.stage1_transition.pt")

    if os.path.exists(best_path):
        shutil.copyfile(best_path, stage1_best_path)
        if getattr(config, "reload_best_on_stage_transition", True):
            checkpoint = torch.load(best_path, map_location=next(model.parameters()).device)
            model.load_state_dict(checkpoint["model_state_dict"])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
        },
        stage1_transition_path,
    )

    if getattr(config, "reset_optimizer_on_stage_transition", True):
        optimizer, scheduler = _make_optimizer_and_scheduler(
            config,
            model,
            lr=float(getattr(config, "stage2_lr", config.train_lr)),
        )

    return optimizer, scheduler, True


_CURRICULUM_FIELDS = (
    "lambda_ot",
    "lambda_mass",
    "detach_ot_weights",
    "lambda_density",
    "lambda_hjb",
    "train_lambda",
    "lambda_action",
    "lambda_global_mass",
    "global_mass_start_epoch",
    "global_mass_ramp_epochs",
    "constraint_start_epoch",
    "constraint_ramp_epochs",
    "density_start_epoch",
    "density_ramp_epochs",
    "action_start_epoch",
    "action_ramp_epochs",
    "hjb_start_epoch",
    "hjb_ramp_epochs",
)


def _train_model_curriculum(config, x, y, device, print_model=False):
    final_epoch_tag = "epoch_{}".format(str(config.train_epochs).rjust(6, "0"))
    final_epoch_path = os.path.join(config.out_dir, "train.{}.pt".format(final_epoch_tag))
    if os.path.exists(final_epoch_path):
        print(final_epoch_path, " exists. Skipping.")
        return config

    model = ForwardSDE(config)
    if print_model:
        print(model)
    model.zero_grad()
    model.to(device)
    sync_hjb_aliases(config)
    init_path = _load_model_checkpoint(model, getattr(config, "init_model_checkpoint", None))
    if init_path is not None:
        print(f"Loaded init checkpoint from {init_path}")

    loss_fn = ObservationLoss(config)
    sync_hjb_aliases(config)
    torch.save(config.__dict__, config.config_pt)
    base_snapshot = _config_snapshot(config, _CURRICULUM_FIELDS)
    pretrain_best_path = os.path.join(config.out_dir, "train.pretrain_best.pt")
    refine_best_path = os.path.join(config.out_dir, "train.pretrain_refine_best.pt")

    with open(config.train_log, 'w') as log_handle:
        if init_path is not None:
            log_handle.write(f"# init_checkpoint\t{init_path}\n")

        pretrain_epochs = int(getattr(config, "pretrain_epochs", 0))
        if pretrain_epochs > 0:
            log_handle.write(f"# phase\tpretrain\tepochs={pretrain_epochs}\n")
            _apply_config_overrides(
                config,
                {
                    **_phase_overrides(
                        config,
                        use_density=bool(getattr(config, "pretrain_use_density", True)),
                        use_hjb=bool(getattr(config, "pretrain_use_hjb", False)),
                        use_action=bool(getattr(config, "pretrain_use_action", False)),
                        use_global_mass=False,
                    ),
                    "lambda_ot": float(getattr(config, "pretrain_lambda_ot", getattr(config, "lambda_ot", 1.0))),
                    "lambda_mass": float(getattr(config, "pretrain_lambda_mass", getattr(config, "lambda_mass", 0.0))),
                },
            )
            optimizer, scheduler = _make_optimizer_and_scheduler(
                config,
                model,
                lr=float(getattr(config, "pretrain_lr", config.train_lr)),
            )
            _train_phase(
                model=model,
                loss_fn=loss_fn,
                config=config,
                x=x,
                y=y,
                optimizer=optimizer,
                scheduler=scheduler,
                n_epochs=pretrain_epochs,
                phase_label="pretrain",
                log_handle=log_handle,
                best_checkpoint_path=pretrain_best_path,
            )
            _load_model_checkpoint(model, pretrain_best_path)
            log_handle.write(f"# reload_best\t{pretrain_best_path}\n")

        refine_epochs = int(getattr(config, "pretrain_refine_epochs", 0))
        if refine_epochs > 0:
            log_handle.write(f"# phase\trefine\tepochs={refine_epochs}\n")
            _apply_config_overrides(
                config,
                {
                    **_phase_overrides(
                        config,
                        use_density=bool(getattr(config, "pretrain_refine_use_density", True)),
                        use_hjb=bool(getattr(config, "pretrain_refine_use_hjb", False)),
                        use_action=bool(getattr(config, "pretrain_refine_use_action", False)),
                        use_global_mass=True,
                    ),
                    "lambda_ot": float(getattr(config, "pretrain_refine_lambda_ot", getattr(config, "lambda_ot", 1.0))),
                    "lambda_mass": float(getattr(config, "pretrain_refine_lambda_mass", getattr(config, "lambda_mass", 0.0))),
                },
            )
            optimizer, scheduler = _make_optimizer_and_scheduler(
                config,
                model,
                lr=float(getattr(config, "pretrain_refine_lr", config.train_lr)),
            )
            _train_phase(
                model=model,
                loss_fn=loss_fn,
                config=config,
                x=x,
                y=y,
                optimizer=optimizer,
                scheduler=scheduler,
                n_epochs=refine_epochs,
                phase_label="refine",
                log_handle=log_handle,
                best_checkpoint_path=refine_best_path,
            )
            _load_model_checkpoint(model, refine_best_path)
            log_handle.write(f"# reload_best\t{refine_best_path}\n")

        _apply_config_overrides(config, base_snapshot)
        sync_hjb_aliases(config)
        config.detach_ot_weights = bool(
            getattr(config, "curriculum_detach_ot_weights", getattr(config, "detach_ot_weights", False))
        )
        log_handle.write(f"# phase\ttrain\tepochs={int(config.train_epochs)}\n")
        optimizer, scheduler = _make_optimizer_and_scheduler(config, model)
        _train_phase(
            model=model,
            loss_fn=loss_fn,
            config=config,
            x=x,
            y=y,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=int(config.train_epochs),
            phase_label="train",
            log_handle=log_handle,
            best_checkpoint_path=config.train_pt.format('best'),
            save_epoch_checkpoints=True,
            epoch_offset=0,
        )

    return config


def _train_model(config, x, y, device, print_model=False):
    final_epoch_tag = "epoch_{}".format(str(config.train_epochs).rjust(6, "0"))
    final_epoch_path = os.path.join(config.out_dir, "train.{}.pt".format(final_epoch_tag))
    if os.path.exists(final_epoch_path):
        print(final_epoch_path, " exists. Skipping.")
        return config

    model = ForwardSDE(config)
    if print_model:
        print(model)
    model.zero_grad()
    model.to(device)
    sync_hjb_aliases(config)
    init_path = _load_model_checkpoint(model, getattr(config, "init_model_checkpoint", None))
    if init_path is not None:
        print(f"Loaded init checkpoint from {init_path}")

    loss_fn = ObservationLoss(config)
    sync_hjb_aliases(config)
    torch.save(config.__dict__, config.config_pt)

    optimizer, scheduler = _make_optimizer_and_scheduler(config, model)
    pbar = tqdm.tqdm(range(config.train_epochs))
    best_train_objective = np.inf
    stage2_applied = False

    with open(config.train_log, 'w') as log_handle:
        if init_path is not None:
            log_handle.write(f"# init_checkpoint\t{init_path}\n")
        for epoch in pbar:
            config.train_epoch = epoch
            if not stage2_applied:
                optimizer, scheduler, transition_applied = _maybe_apply_stage_transition(
                    config,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                )
                if transition_applied and getattr(config, "reset_best_on_stage_transition", False):
                    best_train_objective = np.inf
                stage2_applied = stage2_applied or transition_applied
            if getattr(config, "use_segment_training", False):
                metrics = _train_epoch_segmented(model, loss_fn, config, x, y, optimizer)
            else:
                metrics = _train_epoch_full(model, loss_fn, config, x, y, optimizer)
            scheduler.step()

            desc = _format_train_desc(epoch, metrics, best_train_objective)
            pbar.set_description(desc)
            log_handle.write(desc + '\n')
            log_handle.flush()

            if metrics["train_objective"] < best_train_objective:
                best_train_objective = metrics["train_objective"]
                torch.save(_checkpoint_payload(model, epoch=config.train_epoch + 1), config.train_pt.format('best'))

            if (config.train_epoch + 1) % config.save == 0 or (config.train_epoch + 1) == config.train_epochs:
                epoch_ = str(config.train_epoch + 1).rjust(6, '0')
                torch.save(
                    _checkpoint_payload(model, epoch=config.train_epoch + 1),
                    config.train_pt.format('epoch_{}'.format(epoch_)),
                )

    return config


def run(args,initial_config):
    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = initial_config(args)
    x, y, config = load_data(config)
    x = _move_time_series_to_device(x, device)
    if bool(getattr(config, "use_deepruot_curriculum", False)):
        return _train_model_curriculum(config, x, y, device, print_model=True)
    return _train_model(config, x, y, device, print_model=True)


def run_leaveout(args,initial_config,leaveouts=None):
    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.task = 'leaveout'

    Train_ts = args.train_t

    args.leaveout_t = 'leaveout' + '&'.join(map(str, leaveouts))
    args.train_t = list(sorted(set(Train_ts)-set(leaveouts)))
    args.test_t = leaveouts
    print('--------------------------------------------')
    print('----------leaveout_t=',leaveouts,'---------')
    print('----------train_t=', args.train_t)
    print('--------------------------------------------')

    config = initial_config(args)
    x, y, config = load_data(config)
    x = _move_time_series_to_device(x, device)
    if bool(getattr(config, "use_deepruot_curriculum", False)):
        return _train_model_curriculum(config, x, y, device, print_model=False)
    return _train_model(config, x, y, device, print_model=False)
