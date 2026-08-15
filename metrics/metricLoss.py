import torch
from metrics.metric import Metric


class MetricLoss(Metric):
    def __init__(self):
        super().__init__()
        self.total_loss = 0.0

    def accumulate(self, loss: torch.Tensor | None = None, pred: torch.Tensor | None = None, **kwargs):
        if loss is None:
            return
        batch_size = pred.shape[0] if pred is not None else 1
        self.total_samples += batch_size
        value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        self.total_loss += value * batch_size

    def compute_metric(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples
