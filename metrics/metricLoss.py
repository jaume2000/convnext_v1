import torch
from metrics.metric import Metric


class MetricLoss(Metric):
    def __init__(self):
        super().__init__()
        self.total_loss = 0.0
        self.last_loss = None

    def accumulate(self, loss: torch.Tensor | None = None, pred: torch.Tensor | None = None, **kwargs):
        if loss is None:
            return
        batch_size = pred.shape[0] if pred is not None else 1
        self.total_samples += batch_size
        value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        self.total_loss += value * batch_size
        self.last_loss = value

    def compute_metric(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples

    def pbar_value(self) -> str | None:
        # The last batch rather than the epoch mean: the progress bar is there to show
        # what the model is doing now, and print_last() already reports the mean.
        if self.last_loss is None:
            return None
        return f"{self.last_loss:.3f}"
