"""Shared Runge–Kutta residual integration: x <- RK(f, x, h) with f(y)=block(y)-y."""

from __future__ import annotations

import torch
from torch import nn


def residual_rk_step(
    block: nn.Module,
    x: torch.Tensor,
    h: float,
    method: str | None = None,
) -> torch.Tensor:
    def f(y: torch.Tensor) -> torch.Tensor:
        return block(y) - y

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
