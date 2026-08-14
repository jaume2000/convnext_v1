import torch


class Metric:

    def __init__(self):
        self.total_loss = 0
        self.total_samples = 0

    def accumulate(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pass

    def compute_metric(self) -> torch.Tensor:
        pass

    def compute_loss(self):
        if self.total_samples == 0:
            return 0
        return self.total_loss / self.total_samples
