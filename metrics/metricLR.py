from metrics.metric import Metric


class MetricLR(Metric):
    """Tracks the learning rate over an epoch (mean of per-step values)."""

    def __init__(self):
        super().__init__()
        self.total_lr = 0.0

    def accumulate(self, lr: float | None = None, **kwargs):
        if lr is None:
            return
        self.total_lr += float(lr)
        self.total_samples += 1

    def compute_metric(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_lr / self.total_samples
