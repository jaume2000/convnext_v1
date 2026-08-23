from metrics.metric import Metric


class MetricLR(Metric):
    """Tracks the learning rate over an epoch (mean of per-step values)."""

    def __init__(self):
        super().__init__()
        self.total_lr = 0.0
        self.last_lr = None

    def accumulate(self, lr: float | None = None, **kwargs):
        if lr is None:
            return
        self.last_lr = float(lr)
        self.total_lr += self.last_lr
        self.total_samples += 1

    def compute_metric(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_lr / self.total_samples

    def pbar_value(self) -> str | None:
        # The epoch mean would lag a warmup or a cosine decay by half an epoch.
        if self.last_lr is None:
            return None
        return f"{self.last_lr:.2e}"
