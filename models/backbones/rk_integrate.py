"""Shared Runge–Kutta residual integration: x <- RK(f, x, h) with f(y)=block(y)-y."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.func import functional_call


def _rk(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, h: float, method: str | None):
    if method in (None, "RK1", "EULER"):
        return x + h * f(x)
    if method == "RK2":
        k1 = f(x)
        k2 = f(x + h * k1)
        return x + h / 2 * (k1 + k2)
    if method == "RK3":
        k1 = f(x)
        k2 = f(x + h / 2 * k1)
        k3 = f(x - h * k1 + 2 * h * k2)
        return x + h / 6 * (k1 + 4 * k2 + k3)
    if method == "RK4":
        k1 = f(x)
        k2 = f(x + h / 2 * k1)
        k3 = f(x + h / 2 * k2)
        k4 = f(x + h * k3)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"Unknown integration method: {method!r}")


def residual_rk_step(
    block: nn.Module,
    x: torch.Tensor,
    h: float,
    method: str | None = None,
) -> torch.Tensor:
    def f(y: torch.Tensor) -> torch.Tensor:
        return block(y) - y

    return _rk(f, x, h, method)


def _lerp_params(
    block_a: nn.Module,
    block_b: nn.Module,
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Linearly interpolate parameters (and float buffers) of two identical blocks.

    Non-floating buffers (e.g. Swin ``relative_position_index``) are taken from
    ``block_a`` unchanged — they are discrete lookup tables, not lerpable.
    """
    a = (1.0 - alpha)
    params = {
        name: a * pa + alpha * pb
        for (name, pa), (_, pb) in zip(
            block_a.named_parameters(),
            block_b.named_parameters(),
        )
    }
    buffers: dict[str, torch.Tensor] = {}
    for (name, ba), (_, bb) in zip(block_a.named_buffers(), block_b.named_buffers()):
        if ba.is_floating_point() and bb.is_floating_point():
            buffers[name] = a * ba + alpha * bb
        else:
            buffers[name] = ba
    return params, buffers


def call_lerped(block_a: nn.Module, block_b: nn.Module, alpha: float, y: torch.Tensor) -> torch.Tensor:
    """Forward ``y`` through weights θ = (1-α)θ_a + α θ_b (structure of ``block_a``)."""
    if alpha <= 0.0:
        return block_a(y)
    if alpha >= 1.0:
        return block_b(y)
    params, buffers = _lerp_params(block_a, block_b, alpha)
    return functional_call(block_a, (params, buffers), (y,))


def residual_rk_step_lerped(
    block_a: nn.Module,
    block_b: nn.Module,
    alpha: float,
    x: torch.Tensor,
    h: float,
    method: str | None = None,
) -> torch.Tensor:
    """RK step with bilinear weight interpolation θ=(1-α)θ_a + α θ_b."""

    def f(y: torch.Tensor) -> torch.Tensor:
        return call_lerped(block_a, block_b, alpha, y) - y

    return _rk(f, x, h, method)


def bilinear_time_steps(
    n_blocks: int,
    n_steps: int,
    euler_step: float,
) -> list[tuple[int, float]]:
    """Map a uniform Euler grid on [0, n_blocks) to (k, α) for θ=(1-α)θ_k + α θ_{k+1}.

    Last unit interval [n_blocks-1, n_blocks) clamps to θ_{n_blocks-1} (α=0).
    """
    last = n_blocks - 1
    out: list[tuple[int, float]] = []
    for step in range(n_steps):
        t = step * euler_step
        k = int(t)
        if k >= last:
            out.append((last, 0.0))
        else:
            out.append((k, t - k))
    return out
