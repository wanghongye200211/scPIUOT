import torch
import ot


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.sum().clamp_min(1e-12)


def mioflow_emd2_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    source_mass: torch.Tensor = None,
    target_mass: torch.Tensor = None,
    detach_weights: bool = True,
) -> torch.Tensor:
    if source_mass is None:
        mu = torch.full(
            (source.shape[0],),
            1.0 / max(source.shape[0], 1),
            dtype=source.dtype,
            device=source.device,
        )
    else:
        mu = _normalize_weights(source_mass.to(source.device))

    if target_mass is None:
        nu = torch.full(
            (target.shape[0],),
            1.0 / max(target.shape[0], 1),
            dtype=target.dtype,
            device=target.device,
        )
    else:
        nu = _normalize_weights(target_mass.to(target.device))

    cost = torch.cdist(source, target) ** 2
    mu_plan = (mu.detach() if detach_weights else mu).to("cpu")
    nu_plan = (nu.detach() if detach_weights else nu).to("cpu")
    cost_plan = cost.detach().to("cpu")
    plan = ot.emd(mu_plan, nu_plan, cost_plan)
    if not isinstance(plan, torch.Tensor):
        plan = torch.as_tensor(plan, dtype=cost.dtype)
    plan = plan.to(device=cost.device, dtype=cost.dtype)
    if detach_weights:
        plan = plan.detach()
    return torch.sum(plan * cost)


def mioflow_density_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    source_mass: torch.Tensor = None,
    top_k: int = 5,
    hinge_value: float = 0.01,
) -> torch.Tensor:
    k = max(1, min(int(top_k), target.shape[0]))
    c_dist = torch.cdist(source, target)
    values, _ = torch.topk(c_dist, k, dim=1, largest=False, sorted=False)
    values = torch.clamp(values - hinge_value, min=0.0)
    return torch.mean(values)
