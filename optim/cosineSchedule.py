import torch
from timm.scheduler import CosineLRScheduler


class CosineWithWarmup:
    """Linear warmup followed by cosine decay, advanced once per optimizer step.

    Stepping per iteration instead of per epoch matters for the ConvNeXt recipe: warmup
    covers a handful of epochs out of hundreds, so an epoch-granular schedule would run
    the whole warmup at only a few distinct learning rates.

    Backed by timm's CosineLRScheduler, wrapped because timm drives its schedulers with
    step_update(absolute_step) while the Trainer expects the torch-style step()/set_step().
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
        self.step_count = 0
        self.scheduler = CosineLRScheduler(
            optimizer,
            # warmup_prefix keeps the cosine starting at the peak LR once warmup ends,
            # so t_initial counts the decay only. Without it timm folds warmup into the
            # cosine and the peak is never actually reached.
            t_initial=max(total_steps - warmup_steps, 1),
            lr_min=min_lr,
            warmup_t=warmup_steps,
            warmup_lr_init=warmup_start_lr,
            warmup_prefix=True,
            t_in_epochs=False,
            cycle_limit=1,
        )
        self.set_step(0)

    def set_step(self, step: int):
        """Jump the schedule to an absolute step, used when resuming a run."""
        self.step_count = step
        self.scheduler.step_update(step)

    def step(self):
        self.set_step(self.step_count + 1)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]
