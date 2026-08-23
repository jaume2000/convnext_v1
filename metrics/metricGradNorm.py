import torch
from metrics.metric import Metric


class MetricGradNorm(Metric):
    def __init__(self):
        super().__init__()
        self.total_grad_norm = 0.0
        self.last_grad_norm = None

    def accumulate(self, grad_norm: torch.Tensor | float | None = None, **kwargs):
        # Handed over by the trainer, which already computes the global norm every step
        # to catch non-finite gradients. Rebuilding it here from the model would mean
        # concatenating every gradient into one tensor just to reduce it to a scalar,
        # i.e. copying the whole gradient buffer once per step.
        if grad_norm is None:
            return
        self.last_grad_norm = float(grad_norm)
        self.total_grad_norm += self.last_grad_norm
        self.total_samples += 1

    def compute_metric(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_grad_norm / self.total_samples

    def pbar_value(self) -> str | None:
        if self.last_grad_norm is None:
            return None
        return f"{self.last_grad_norm:.3f}"
