import torch


class Metric:

    def __init__(self):
        self.total_loss = 0
        self.total_samples = 0
        self.lr: float | None = None
        self._device_sums: dict[str, torch.Tensor] = {}

    def accumulate(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pass

    def compute_metric(self) -> torch.Tensor:
        pass

    def add_device_sum(self, name: str, value: torch.Tensor) -> None:
        """Add a scalar tensor to a running sum kept on the tensor's own device.

        Reading a CUDA scalar with .item() blocks until the queue drains, so doing it
        per step stops the host from queueing the next iteration. Sums stay on device
        and are read once per epoch by sync().
        """
        value = value.detach().to(torch.float64)
        if name in self._device_sums:
            self._device_sums[name] += value
        else:
            self._device_sums[name] = value.clone()

    def sync(self) -> None:
        """Move pending device sums into the plain-Python totals."""
        if not self._device_sums:
            return
        sums = {name: value.item() for name, value in self._device_sums.items()}
        self._device_sums = {}
        self.apply_device_sums(sums)

    def apply_device_sums(self, sums: dict[str, float]) -> None:
        pass

    def compute_loss(self):
        self.sync()
        if self.total_samples == 0:
            return 0
        return self.total_loss / self.total_samples

    def __getstate__(self):
        # Never pickle CUDA tensors into the history files.
        self.sync()
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Points pickled before the LR was tracked simply do not carry it.
        self.lr = state.get("lr")
        self._device_sums = {}
