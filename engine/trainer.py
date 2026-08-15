from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# torch.amp.GradScaler exists on newer PyTorch; cineca-ai 4.1 only has torch.cuda.amp.
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    import torch.cuda.amp as _cuda_amp

    def autocast(device_type="cuda", enabled=True, **kwargs):
        return _cuda_amp.autocast(enabled=enabled, **kwargs)

    def GradScaler(device="cuda", enabled=True, **kwargs):
        return _cuda_amp.GradScaler(enabled=enabled, **kwargs)

from data.testingDataset import TestingDataset
from data.transforms.transforms import build_train_batch_transforms
from metrics.metricHistory import Metrichistory
from metrics.top1acc import Top1AccMetric
from optim.cosineSchedule import CosineWithWarmup


class Trainer():

    def __init__(self,
    experiment_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    batch_transforms,
    device: torch.device,
    num_classes: int = 1000,
    amp: bool = True,
    amp_dtype: str = "auto",
    channels_last: bool = True,
    log_every: int = 20,
    scheduler=None):
        self.experiment_name = experiment_name
        self.model = model
        self.optimizer = optimizer          # SGD, AdamW, ...
        self.scheduler = scheduler          # May be None
        # Plateau schedulers need the validation loss and react to epoch-level noise, so
        # they are stepped once after validate(); everything else steps per iteration.
        self.scheduler_per_epoch = isinstance(scheduler, ReduceLROnPlateau)
        self.criterion = criterion          # Loss function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.batch_transforms = batch_transforms
        self.num_classes = num_classes
        self.amp = amp and device.type == "cuda"
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        # fp16 needs loss scaling; bf16 has the same exponent range as fp32 and does not,
        # which also removes the per-step inf check (a host sync) that GradScaler does.
        self.scaler = GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        # Depthwise 7x7 convs and channel-wise LayerNorm both prefer NHWC on tensor cores.
        self.channels_last = channels_last and device.type == "cuda"
        self.log_every = log_every
        self.experiment_path = Path(f"outputs/{self.experiment_name}")
        self.weights_path = self.experiment_path / "weights"
        if device.type == "cuda":
            # Shapes are fixed (drop_last=True), so autotuning conv algos pays off.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._prepare_model()

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype:
        if amp_dtype == "auto":
            bf16_ok = self.amp and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if bf16_ok else torch.float16
        return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]

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
            "amp": str(self.amp_dtype).removeprefix("torch.") if self.amp else "off",
        }
        if self.scheduler is not None:
            # Read from the optimizer rather than get_last_lr(): ReduceLROnPlateau only
            # grew that method once it became an LRScheduler subclass.
            postfix["lr"] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
        if self.device.type == "cuda":
            # YOLO-style: reserved VRAM per visible GPU (GiB).
            postfix["vram"] = " ".join(
                f"{torch.cuda.memory_reserved(i) / (1024 ** 3):.1f}G"
                for i in range(torch.cuda.device_count())
            )
        return postfix

    def train_epoch(self, train_histoy_metrics: Metrichistory, epoch: int, epochs: int):
        self.model.train()
        train_histoy_metrics.create_point()
        pbar = tqdm(self.train_loader, desc=f"train {epoch + 1}/{epochs}", mininterval=1.0)
        for step, (batch, y_labels) in enumerate(pbar):
            self.optimizer.zero_grad(set_to_none=True)
            batch = self._to_device(batch)
            y_labels = y_labels.to(self.device, non_blocking=True)
            batch, y_labels = self.batch_transforms(batch, y_labels)
            with autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
                pred = self.model(batch)
                loss = self.criterion(pred, y_labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None and not self.scheduler_per_epoch:
                self.scheduler.step()
            train_histoy_metrics.accumulate_last_point(pred.detach(), y_labels, loss.detach())
            if step % self.log_every == 0:
                pbar.set_postfix(self._postfix(loss))
        train_histoy_metrics.print_last()

    # Validate for classification
    def validate(self, val_histoy_metrics: Metrichistory, epoch: int, epochs: int):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"val {epoch + 1}/{epochs}", mininterval=1.0)
        val_histoy_metrics.create_point()
        with torch.inference_mode():
            for step, (batch, y_labels) in enumerate(pbar):
                batch = self._to_device(batch)
                y_labels = y_labels.to(self.device, non_blocking=True)
                y_labels = torch.nn.functional.one_hot(y_labels, num_classes=self.num_classes).float()
                with autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
                    pred = self.model(batch)
                    loss = self.criterion(pred, y_labels)
                val_histoy_metrics.accumulate_last_point(pred.detach(), y_labels, loss.detach())
                if step % self.log_every == 0:
                    pbar.set_postfix(loss=loss.item())
            val_histoy_metrics.print_last()

    def _load_checkpoint(self, name: str = "last"):
        path = self.weights_path / f"{name}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            epoch = ckpt.get("epoch")
        else:
            # Legacy raw state_dict checkpoints.
            state = ckpt
            epoch = None
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        target.load_state_dict(state)
        if isinstance(ckpt, dict):
            # Without the optimizer state, every restart throws away AdamW's moments and
            # the loss visibly jumps; older checkpoints simply do not carry it.
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and self.scaler.is_enabled():
                self.scaler.load_state_dict(ckpt["scaler"])
            # A plateau schedule cannot be rebuilt from the epoch number: its patience and
            # best-so-far counters are the whole state, so they have to be restored.
            if self.scheduler_per_epoch:
                if "scheduler" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                else:
                    print("Warning: checkpoint has no scheduler state, plateau counters restart")
        return epoch

    def _retake(self, train_histoy_metrics: Metrichistory, val_histoy_metrics: Metrichistory) -> int:
        ckpt_name = "last" if (self.weights_path / "last.pth").is_file() else "best"
        ckpt_epoch = self._load_checkpoint(ckpt_name)
        train_loaded = train_histoy_metrics.load()
        val_loaded = val_histoy_metrics.load()
        if not train_loaded or not val_loaded:
            raise FileNotFoundError(
                f"retake=True requires train/val history under {self.experiment_path / 'history'}"
            )

        train_histoy_metrics.drop_empty_trailing()
        val_histoy_metrics.drop_empty_trailing()

        if ckpt_epoch is not None:
            start_epoch = ckpt_epoch + 1
        else:
            start_epoch = min(len(train_histoy_metrics.metricHistory), len(val_histoy_metrics.metricHistory))

        train_histoy_metrics.truncate(start_epoch)
        val_histoy_metrics.truncate(start_epoch)
        val_histoy_metrics.restore_best()
        train_histoy_metrics.save()
        val_histoy_metrics.save()
        if self.scheduler is not None and not self.scheduler_per_epoch:
            # Derived from the epoch rather than stored, so the schedule stays correct
            # even for checkpoints written before it existed.
            self.scheduler.set_step(start_epoch * len(self.train_loader))
        print(f"Retake: loaded {ckpt_name}.pth, resuming at epoch {start_epoch + 1}")
        return start_epoch

    def _step_scheduler_on_epoch(self, val_histoy_metrics: Metrichistory):
        if self.scheduler is None or not self.scheduler_per_epoch:
            return
        lr_before = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(val_histoy_metrics.metricHistory[-1].compute_loss())
        lr_after = self.optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            # Replaces ReduceLROnPlateau's verbose flag, which newer PyTorch removed.
            print(f"LR reduced: {lr_before:.2e} -> {lr_after:.2e}")

    def fit(self, epochs, retake: bool = False):
        train_histoy_metrics = Metrichistory(self.experiment_path, Top1AccMetric, "train")
        val_histoy_metrics = Metrichistory(self.experiment_path, Top1AccMetric, "val")
        start_epoch = self._retake(train_histoy_metrics, val_histoy_metrics) if retake else 0
        for epoch in range(start_epoch, epochs):
            self.train_epoch(train_histoy_metrics, epoch=epoch, epochs=epochs)
            self.validate(val_histoy_metrics, epoch=epoch, epochs=epochs)
            self._step_scheduler_on_epoch(val_histoy_metrics)
            if val_histoy_metrics.last_is_best():
                self.save_model(name="best", epoch=epoch)
            self.save_model(name="last", epoch=epoch)
        train_histoy_metrics.save()
        val_histoy_metrics.save()
        train_histoy_metrics.plot_history()
        val_histoy_metrics.plot_history()
        train_histoy_metrics.plot_history_loss()
        val_histoy_metrics.plot_history_loss()
        return train_histoy_metrics, val_histoy_metrics

    def save_model(self, name: str, epoch=None):
        self.weights_path.mkdir(parents=True, exist_ok=True)
        state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        if epoch is None:
            torch.save(state, self.weights_path / f"{name}.pth")
            return
        payload = {"model": state, "epoch": epoch, "optimizer": self.optimizer.state_dict()}
        if self.scaler.is_enabled():
            payload["scaler"] = self.scaler.state_dict()
        if self.scheduler_per_epoch:
            payload["scheduler"] = self.scheduler.state_dict()
        torch.save(payload, self.weights_path / f"{name}.pth")

def main():
    num_classes = 4
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train_dataset = TestingDataset(input_shape=(3, 8, 8), num_samples=128, num_classes=num_classes)
    val_dataset = TestingDataset(input_shape=(3, 8, 8), num_samples=128, num_classes=num_classes)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    batch_transforms = build_train_batch_transforms(num_classes=num_classes)
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=2 * len(train_loader),
        total_steps=10 * len(train_loader),
    )
    trainer = Trainer(
        experiment_name="test",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_transforms=batch_transforms,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_classes=num_classes,
        amp=True,
    )
    trainer.fit(epochs=10)

if __name__ == "__main__":
    main()
