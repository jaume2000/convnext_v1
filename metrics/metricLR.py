from metrics.metric import Metric


class MetricLR(Metric):
    """Tracks the learning rate over an epoch (mean of per-step values)."""

    def __init__(self):
        super().__init__()
        self.minibatch_lrHistory: list[float] = []

    def accumulate(self, lr: float | None = None, **kwargs):
        if lr is None:
            return
        self.minibatch_lrHistory.append(float(lr))
        self.total_samples += 1

    def compute_metric(self) -> float:
        if not self.minibatch_lrHistory:
            return 0.0
        return sum(self.minibatch_lrHistory) / len(self.minibatch_lrHistory)
