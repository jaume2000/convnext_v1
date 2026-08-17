import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from metrics.metricGradNorm import MetricGradNorm
from metrics.metricLoss import MetricLoss
from metrics.metricLR import MetricLR

# torch.amp.autocast exists on newer PyTorch; cineca-ai 4.1 only has torch.cuda.amp.
try:
    from torch.amp import autocast
except ImportError:
    import torch.cuda.amp as _cuda_amp

    def autocast(device_type="cuda", enabled=True, **kwargs):
        return _cuda_amp.autocast(enabled=enabled, **kwargs)

from data.testingDataset import TestingDataset
from data.transforms.transforms import build_train_batch_transforms
from metrics.metricHistory import DictHistoryMetrics
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
    channels_last: bool = True,
    log_every: int = 20,
    scheduler=None,
    gradient_clipping: float | None = None,
    collapse_patience: int | None = 3,
    collapse_margin: float = 0.01,
    retake: bool = False):
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
        # Depthwise 7x7 convs and channel-wise LayerNorm both prefer NHWC on tensor cores.
        self.channels_last = channels_last and device.type == "cuda"
        self.log_every = log_every
        self.gradient_clipping = gradient_clipping
        self.collapse_patience = collapse_patience
        self.collapse_margin = collapse_margin
        self._collapsed_epochs = 0
        self._collapse_armed = False
        self.retake = retake
        self.experiment_path = Path(f"outputs/{self.experiment_name}")
        self.weights_path = self.experiment_path / "weights"
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

    def _current_lr(self) -> float:
        # Read from the optimizer rather than get_last_lr(): ReduceLROnPlateau only
        # grew that method once it became an LRScheduler subclass.
        return self.optimizer.param_groups[0]["lr"]

    def _resume_hint(self) -> str:
        return (
            f"{self.weights_path / 'last.pth'} still holds the last finite epoch, "
            "so retake=True resumes from there."
        )

    def _checked_grad_norm(self, epoch: int, step: int) -> torch.Tensor:
        """Total gradient norm, aborting the run if any gradient is inf/NaN.

        A single inf/NaN gradient makes AdamW write NaN into the parameters *and* into
        its moments, and the moments never recover, so every later epoch trains nothing:
        a previous run spent 80 epochs that way. bf16 needs no loss scaling, so this is
        the only check that aborts before the optimizer corrupts its state.

        Ordering matters: this runs before clip_grad_norm_, which would multiply every
        gradient by a NaN total norm and erase the evidence of which tensor went first.
        """
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        named_grads = [(name, p.grad) for name, p in target.named_parameters() if p.grad is not None]
        # Norm of the per-tensor norms is the global norm, and it also says which tensor
        # is the culprit without a second pass over the gradients.
        norms = torch.stack([grad.float().norm() for _, grad in named_grads])
        total_norm = norms.norm()
        if torch.isfinite(total_norm).item():
            return total_norm
        culprits = [
            name for (name, _), finite in zip(named_grads, torch.isfinite(norms).tolist()) if not finite
        ]
        raise RuntimeError(
            f"Non-finite gradient at epoch {epoch + 1}, step {step}: "
            f"{len(culprits)} of {len(named_grads)} tensors, first {', '.join(culprits[:5])}. "
            + self._resume_hint()
        )

    def _postfix(self, loss: torch.Tensor, grad_norm: torch.Tensor) -> dict:
        postfix = {
            "loss": f"{loss.item():.3f}",
            "grad_norm": f"{grad_norm:.3f}"
        }
        if self.scheduler is not None:
            postfix["lr"] = f"{self._current_lr():.2e}"
        if self.device.type == "cuda":
            # YOLO-style: reserved VRAM per visible GPU (GiB).
            postfix["vram"] = " ".join(
                f"{torch.cuda.memory_reserved(i) / (1024 ** 3):.1f}G"
                for i in range(torch.cuda.device_count())
            )
        return postfix

    def train_epoch(self, train_histoy_metrics: DictHistoryMetrics, epoch: int, epochs: int):
        self.model.train()
        train_histoy_metrics.create_point()
        pbar = tqdm(self.train_loader, desc=f"train {epoch + 1}/{epochs}", mininterval=1.0)
        for step, (batch, y_labels) in enumerate(pbar):
            self.optimizer.zero_grad(set_to_none=True)
            batch = self._to_device(batch)
            y_labels = y_labels.to(self.device, non_blocking=True)
            batch, y_labels = self.batch_transforms(batch, y_labels)
            with autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
                pred = self.model(batch)
                loss = self.criterion(pred, y_labels)
            # A non-finite loss means the weights are already poisoned, so there is
            # nothing left to salvage by running the backward.
            if step % self.log_every == 0 and not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite train loss ({loss.item()}) at epoch {epoch + 1}, step {step}. "
                    + self._resume_hint()
                )
            loss.backward()
            grad_norm = self._checked_grad_norm(epoch, step)
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()
            if self.scheduler is not None and not self.scheduler_per_epoch:
                self.scheduler.step()
            train_histoy_metrics.accumulate(
                grad_norm=grad_norm,
                loss=loss.detach(),
                pred=pred.detach(),
                y_labels=y_labels,
                lr=self._current_lr(),
            )
            if step % self.log_every == 0:
                pbar.set_postfix(self._postfix(loss, grad_norm))
        train_histoy_metrics.print_last()

    # Validate for classification
    def validate(self, val_histoy_metrics: DictHistoryMetrics, epoch: int, epochs: int):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"val {epoch + 1}/{epochs}", mininterval=1.0)
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
            # A plateau schedule cannot be rebuilt from the epoch number: its patience and
            # best-so-far counters are the whole state, so they have to be restored.
            if self.scheduler_per_epoch:
                if "scheduler" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                else:
                    print("Warning: checkpoint has no scheduler state, plateau counters restart")
        return epoch

    def _retake(self, train_histoy_metrics: DictHistoryMetrics, val_histoy_metrics: DictHistoryMetrics) -> int:
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
            start_epoch = min(train_histoy_metrics.num_epochs(), val_histoy_metrics.num_epochs())

        train_histoy_metrics.truncate(start_epoch)
        val_histoy_metrics.truncate(start_epoch)
        val_histoy_metrics.restore_best("top1acc")
        train_histoy_metrics.save()
        val_histoy_metrics.save()
        if self.scheduler is not None and not self.scheduler_per_epoch:
            # Derived from the epoch rather than stored, so the schedule stays correct
            # even for checkpoints written before it existed.
            self.scheduler.set_step(start_epoch * len(self.train_loader))
        print(f"Retake: loaded {ckpt_name}.pth, resuming at epoch {start_epoch + 1}")
        return start_epoch

    def _check_collapse(self, train_histoy_metrics: DictHistoryMetrics, epoch: int):
        """Abort once the model stops beating a constant predictor.

        The inf/NaN guards never fire on a dead network: a run can collapse into a
        constant output with a perfectly finite loss of exactly ln(num_classes) and a
        top-1 of 1/num_classes, because the best a class-independent output can do on a
        balanced set is emit the uniform prior. Everything upstream of the head then
        gets ~zero gradient, so it is a fixed point rather than a dip -- one previous
        run sat there for 19 epochs, which is what this check exists to cut short.

        Only the loss is tested. A collapsed grad norm is the more vivid symptom but a
        noisier rule, whereas "does not beat a constant" is on its own enough to call
        the run dead whatever the mechanism.

        The check stays disarmed until some epoch has beaten the prior, so that a merely
        slow start can never abort a job: heavy Mixup plus a warmup LR of 1e-6 can sit
        near ln(num_classes) for the first epochs perfectly legitimately. What it fires
        on is a regression away from a state that was already learning.
        """
        if self.collapse_patience is None:
            return
        uniform_loss = math.log(self.num_classes)
        train_loss = float(train_histoy_metrics.get("loss").compute_last())
        if train_loss < uniform_loss - self.collapse_margin:
            self._collapsed_epochs = 0
            self._collapse_armed = True
            return
        if not self._collapse_armed:
            return
        self._collapsed_epochs += 1
        grad_norm = float(train_histoy_metrics.get("gradnorm").compute_last())
        print(
            f"Warning: epoch {epoch + 1} train loss {train_loss:.4f} does not beat the "
            f"uniform prior ln({self.num_classes})={uniform_loss:.4f} (grad norm "
            f"{grad_norm:.4f}), {self._collapsed_epochs}/{self.collapse_patience}"
        )
        if self._collapsed_epochs < self.collapse_patience:
            return
        raise RuntimeError(
            f"Model collapsed to a constant output: {self._collapsed_epochs} consecutive "
            f"epochs at or above ln({self.num_classes})={uniform_loss:.4f}, last grad norm "
            f"{grad_norm:.4f}. Lower the peak LR, enable gradient_clipping, or keep "
            f"Mixup/CutMix and label smoothing on, then restart. "
            + self._collapse_resume_hint()
        )

    def _collapse_resume_hint(self) -> str:
        return (
            f"Note that {self.weights_path / 'last.pth'} is already collapsed, so retake=True "
            f"would resume the dead model: move it aside first to fall back to "
            f"{self.weights_path / 'best.pth'}."
        )

    def _step_scheduler_on_epoch(self, val_histoy_metrics: DictHistoryMetrics):
        if self.scheduler is None or not self.scheduler_per_epoch:
            return
        lr_before = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(val_histoy_metrics.get("loss").compute_last())
        lr_after = self.optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            # Replaces ReduceLROnPlateau's verbose flag, which newer PyTorch removed.
            print(f"LR reduced: {lr_before:.2e} -> {lr_after:.2e}")

    def fit(self, epochs):
        train_histoy_metrics = DictHistoryMetrics(self.experiment_path, split="train")
        train_histoy_metrics.addHistoryMetric("top1acc", Top1AccMetric)
        train_histoy_metrics.addHistoryMetric("gradnorm", MetricGradNorm)
        train_histoy_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
        train_histoy_metrics.addHistoryMetric("lr", MetricLR)
        val_histoy_metrics = DictHistoryMetrics(self.experiment_path, split="val")
        val_histoy_metrics.addHistoryMetric("top1acc", Top1AccMetric)
        val_histoy_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
        start_epoch = self._retake(train_histoy_metrics, val_histoy_metrics) if self.retake else 0
        for epoch in range(start_epoch, epochs):
            self.train_epoch(train_histoy_metrics, epoch=epoch, epochs=epochs)
            self.validate(val_histoy_metrics, epoch=epoch, epochs=epochs)
            self._step_scheduler_on_epoch(val_histoy_metrics)
            if val_histoy_metrics.last_is_best("top1acc"):
                self.save_model(name="best", epoch=epoch)
            self.save_model(name="last", epoch=epoch)
            train_histoy_metrics.save()
            val_histoy_metrics.save()
            # After the saves, so a collapsed run still leaves its full history on disk.
            self._check_collapse(train_histoy_metrics, epoch)
        train_histoy_metrics.plot_history()
        val_histoy_metrics.plot_history()
        return train_histoy_metrics, val_histoy_metrics

    def save_model(self, name: str, epoch=None):
        self.weights_path.mkdir(parents=True, exist_ok=True)
        state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        if epoch is None:
            torch.save(state, self.weights_path / f"{name}.pth")
            return
        payload = {"model": state, "epoch": epoch, "optimizer": self.optimizer.state_dict()}
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
