import math

import torch


class CosineWithWarmup:
    """Linear warmup followed by cosine decay, advanced once per optimizer step.

    Stepping per iteration instead of per epoch matters for the ConvNeXt recipe: warmup
    covers a handful of epochs out of hundreds, so an epoch-granular schedule would run
    the whole warmup at only a few distinct learning rates.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self.set_step(0)

    def lr_at(self, step: int, base_lr: float) -> float:
        if step < self.warmup_steps:
            progress = step / max(self.warmup_steps, 1)
            return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / decay_steps, 1.0)
        return self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def set_step(self, step: int):
        """Jump the schedule to an absolute step, used when resuming a run."""
        self.step_count = step
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.lr_at(step, base_lr)

    def step(self):
        self.set_step(self.step_count + 1)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]
