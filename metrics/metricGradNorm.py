import torch
from metrics.metric import Metric


class MetricGradNorm(Metric):
    def __init__(self):
        super().__init__()
        self.minibatch_gradHistory: list[float] = []

    def accumulate(self, model: torch.nn.Module | None = None, **kwargs):
        if model is None:
            return
        grads = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        self.minibatch_gradHistory.append(float(torch.stack(grads).mean()))
        self.total_samples += 1

    def compute_metric(self) -> float:
        if not self.minibatch_gradHistory:
            return 0.0
        return sum(self.minibatch_gradHistory) / len(self.minibatch_gradHistory)
