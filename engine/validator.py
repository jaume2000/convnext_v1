import torch
import torch.nn as nn
from tqdm import tqdm
from metrics.metricHistory import DictHistoryMetrics

# torch.amp.autocast exists on newer PyTorch; cineca-ai 4.1 only has torch.cuda.amp.
try:
    from torch.amp import autocast
except ImportError:
    import torch.cuda.amp as _cuda_amp

    def autocast(device_type="cuda", enabled=True, **kwargs):
        return _cuda_amp.autocast(enabled=enabled, **kwargs)



class Validator():

    def __init__(self,
    model: nn.Module,
    criterion: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    batch_transforms,
    device: torch.device,
    num_classes: int = 1000,
    amp: bool = True,
    channels_last: bool = True,
    log_every: int = 20):
        self.model = model
        self.criterion = criterion          # Loss function
        self.val_loader = val_loader
        self.device = device
        self.batch_transforms = batch_transforms
        self.num_classes = num_classes
        self.amp = amp and device.type == "cuda"
        # Depthwise 7x7 convs and channel-wise LayerNorm both prefer NHWC on tensor cores.
        self.channels_last = channels_last and device.type == "cuda"
        self.log_every = log_every
        if device.type == "cuda":
            # Shapes are fixed (drop_last=True), so autotuning conv algos pays off.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._prepare_model()

    def _prepare_model(self):
        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)

    def _to_device(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        if self.channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        return batch

    def _postfix(self, loss: torch.Tensor) -> dict:
        postfix = {
            "loss": f"{loss.item():.3f}",
        }
        if self.device.type == "cuda":
            # YOLO-style: reserved VRAM per visible GPU (GiB).
            postfix["vram"] = " ".join(
                f"{torch.cuda.memory_reserved(i) / (1024 ** 3):.1f}G"
                for i in range(torch.cuda.device_count())
            )
        return postfix

    # Validate for classification
    def validate(self, val_histoy_metrics: DictHistoryMetrics, epoch: int|None=None, epochs: int|None=None):
        self.model.eval()
        desc=f"val {epoch + 1}/{epochs}" if epochs != None and epoch != None else None
        pbar = tqdm(self.val_loader, desc, mininterval=1.0)
        val_histoy_metrics.create_point()
        with torch.inference_mode():
            for step, (batch, y_labels) in enumerate(pbar):
                batch = self._to_device(batch)
                y_labels = y_labels.to(self.device, non_blocking=True)
                y_labels = torch.nn.functional.one_hot(y_labels, num_classes=self.num_classes).float()
                with autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
                    pred = self.model(batch)
                    loss = self.criterion(pred, y_labels)
                val_histoy_metrics.accumulate(pred=pred.detach(), y_labels=y_labels, loss=loss.detach())
                if step % self.log_every == 0:
                    pbar.set_postfix(loss=loss.item())
        val_histoy_metrics.print_last()