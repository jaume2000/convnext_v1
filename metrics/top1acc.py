import torch
from metrics.metric import Metric


class Top1AccMetric(Metric):
    def __init__(self):
        super().__init__()
        self.total_correct = 0.0
        self.total_samples = 0.0

    def accumulate(self, pred: torch.Tensor | None = None, target: torch.Tensor | None = None,
                   y_labels: torch.Tensor | None = None, **kwargs):
        if pred is None:
            return
        # pred and target are (B, C); soft Mixup labels still work via argmax
        target = target if target is not None else y_labels
        if target is None:
            return
        batch_size = pred.shape[0]
        self.total_samples += batch_size
        correct = (pred.argmax(1) == target.argmax(1)).sum()
        self.total_correct += correct.item()

    def compute_metric(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples
