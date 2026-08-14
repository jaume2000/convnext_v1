from typing_extensions import override
import torch
from metrics.metric import Metric


class Top1AccMetric(Metric):
    def __init__(self):
        super().__init__()
        self.total_correct = 0
        self.total_samples = 0

    def accumulate(self, pred: torch.Tensor, target: torch.Tensor, loss: torch.Tensor):
        # pred and target are (B, C); soft Mixup labels still work via argmax
        batch_size = pred.shape[0]
        self.total_samples += batch_size
        correct = (pred.argmax(1) == target.argmax(1)).sum()
        if pred.is_cuda:
            self.add_device_sum("correct", correct)
            self.add_device_sum("loss", loss.detach() * batch_size)
        else:
            self.total_correct += correct.item()
            self.total_loss += loss.item() * batch_size

    @override
    def apply_device_sums(self, sums: dict[str, float]):
        self.total_correct += sums.get("correct", 0.0)
        self.total_loss += sums.get("loss", 0.0)

    def compute_metric(self):
        self.sync()
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples
