import math
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
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

    def _is_main_process(self) -> bool:
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model

    def _broadcast_from_main(self, flag: bool) -> bool:
        """Give every rank rank 0's answer, so a decision made from the metrics is unanimous."""
        if not (dist.is_available() and dist.is_initialized()):
            return flag
        payload = torch.tensor([1 if flag else 0], device=self.device)
        dist.broadcast(payload, src=0)
        return bool(payload.item())

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
        named_grads = [
            (name, p.grad) for name, p in self._unwrap_model().named_parameters() if p.grad is not None
        ]
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

    def _postfix(self, history_metrics: DictHistoryMetrics) -> dict:
        """Whatever the metrics want shown, plus the one number no metric owns."""
        postfix = history_metrics.pbar()
        if self.device.type == "cuda":
            # YOLO-style: reserved VRAM per visible GPU (GiB).
            postfix["vram"] = " ".join(
                f"{torch.cuda.memory_reserved(i) / (1024 ** 3):.1f}G"
                for i in range(torch.cuda.device_count())
            )
        return postfix

    def train_epoch(self, train_history_metrics: DictHistoryMetrics, epoch: int, epochs: int):
        self.model.train()
        # Without set_epoch, DistributedSampler reuses the same shuffle every epoch.
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        # Metrics that read the weights want the model itself, not the DDP wrapper.
        model = self._unwrap_model()
        # Rank 0 alone keeps the metrics. The weights are identical on every rank after
        # the gradient all-reduce, so the others would recompute the same numbers, and
        # the ones that do differ (loss, top-1) only describe that rank's shard anyway.
        # Nothing here communicates, so skipping it off-rank cannot desync the group.
        main = self._is_main_process()
        if main:
            train_history_metrics.create_point()
        pbar = tqdm(
            self.train_loader,
            desc=f"train {epoch + 1}/{epochs}",
            mininterval=1.0,
            disable=not main,
        )
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
            if not main:
                continue
            train_history_metrics.accumulate(
                model=model,
                grad_norm=grad_norm,
                loss=loss.detach(),
                pred=pred.detach(),
                y_labels=y_labels,
                lr=self._current_lr(),
            )
            if step % self.log_every == 0:
                pbar.set_postfix(self._postfix(train_history_metrics))
        if main:
            train_history_metrics.print_last()

    # Validate for classification
    def validate(self, val_history_metrics: DictHistoryMetrics, epoch: int, epochs: int):
        self.model.eval()
        model = self._unwrap_model()
        pbar = tqdm(
            self.val_loader,
            desc=f"val {epoch + 1}/{epochs}",
            mininterval=1.0,
            disable=not self._is_main_process(),
        )
        val_history_metrics.create_point()
        with torch.inference_mode():
            for step, (batch, y_labels) in enumerate(pbar):
                batch = self._to_device(batch)
                y_labels = y_labels.to(self.device, non_blocking=True)
                y_labels = torch.nn.functional.one_hot(y_labels, num_classes=self.num_classes).float()
                with autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
                    pred = self.model(batch)
                    loss = self.criterion(pred, y_labels)
                val_history_metrics.accumulate(
                    model=model, pred=pred.detach(), y_labels=y_labels, loss=loss.detach()
                )
                if step % self.log_every == 0:
                    pbar.set_postfix(self._postfix(val_history_metrics))
            #if printValidationStats in self.model
            if hasattr(self.model, "printValidationStats"):
                self.model.printValidationStats()
        val_history_metrics.print_last()

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
        self._unwrap_model().load_state_dict(state)
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

    def _retake(self, train_history_metrics: DictHistoryMetrics, val_history_metrics: DictHistoryMetrics) -> int:
        ckpt_name = "last" if (self.weights_path / "last.pth").is_file() else "best"
        ckpt_epoch = self._load_checkpoint(ckpt_name)
        train_loaded = train_history_metrics.load()
        val_loaded = val_history_metrics.load()
        if not train_loaded or not val_loaded:
            raise FileNotFoundError(
                f"retake=True requires train/val history under {self.experiment_path / 'history'}"
            )

        train_history_metrics.drop_empty_trailing()
        val_history_metrics.drop_empty_trailing()

        if ckpt_epoch is not None:
            start_epoch = ckpt_epoch + 1
        else:
            start_epoch = min(train_history_metrics.num_epochs(), val_history_metrics.num_epochs())

        train_history_metrics.truncate(start_epoch)
        val_history_metrics.truncate(start_epoch)
        # Metrics added mid-run start short: pad them so epoch N stays at index N.
        train_history_metrics.pad_to(start_epoch)
        val_history_metrics.pad_to(start_epoch)
        val_history_metrics.restore_best("top1acc")
        if self._is_main_process():
            train_history_metrics.save()
            val_history_metrics.save()
        if self.scheduler is not None and not self.scheduler_per_epoch:
            # Derived from the epoch rather than stored, so the schedule stays correct
            # even for checkpoints written before it existed.
            self.scheduler.set_step(start_epoch * len(self.train_loader))
        if self._is_main_process():
            print(f"Retake: loaded {ckpt_name}.pth, resuming at epoch {start_epoch + 1}")
        return start_epoch

    def _check_collapse(self, train_history_metrics: DictHistoryMetrics, epoch: int):
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

        Only rank 0 holds the metrics, so only rank 0 has an opinion; it broadcasts the
        verdict because a rank that aborted alone would leave the rest hanging on the
        next collective. Every rank must therefore reach the broadcast.
        """
        if self.collapse_patience is None:
            return
        collapsed = self._is_main_process() and self._collapsed_this_epoch(train_history_metrics, epoch)
        if not self._broadcast_from_main(collapsed):
            return
        raise RuntimeError(
            f"Model collapsed to a constant output: {self.collapse_patience} consecutive "
            f"epochs at or above ln({self.num_classes})={math.log(self.num_classes):.4f}. "
            f"Lower the peak LR, enable gradient_clipping, or keep Mixup/CutMix and label "
            f"smoothing on, then restart. "
            + self._collapse_resume_hint()
        )

    def _collapsed_this_epoch(self, train_history_metrics: DictHistoryMetrics, epoch: int) -> bool:
        """Advance the collapse counter on this epoch's metrics, rank 0 only."""
        uniform_loss = math.log(self.num_classes)
        train_loss = float(train_history_metrics.get("loss").compute_last())
        if train_loss < uniform_loss - self.collapse_margin:
            self._collapsed_epochs = 0
            self._collapse_armed = True
            return False
        if not self._collapse_armed:
            return False
        self._collapsed_epochs += 1
        grad_norm = float(train_history_metrics.get("gradnorm").compute_last())
        print(
            f"Warning: epoch {epoch + 1} train loss {train_loss:.4f} does not beat the "
            f"uniform prior ln({self.num_classes})={uniform_loss:.4f} (grad norm "
            f"{grad_norm:.4f}), {self._collapsed_epochs}/{self.collapse_patience}"
        )
        return self._collapsed_epochs >= self.collapse_patience

    def _collapse_resume_hint(self) -> str:
        return (
            f"Note that {self.weights_path / 'last.pth'} is already collapsed, so retake=True "
            f"would resume the dead model: move it aside first to fall back to "
            f"{self.weights_path / 'best.pth'}."
        )

    def _step_scheduler_on_epoch(self, val_history_metrics: DictHistoryMetrics):
        if self.scheduler is None or not self.scheduler_per_epoch:
            return
        lr_before = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(val_history_metrics.get("loss").compute_last())
        lr_after = self.optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            # Replaces ReduceLROnPlateau's verbose flag, which newer PyTorch removed.
            print(f"LR reduced: {lr_before:.2e} -> {lr_after:.2e}")

    def fit(
        self,
        epochs: int,
        train_history_metrics: DictHistoryMetrics,
        val_history_metrics: DictHistoryMetrics,
    ):
        """Train for `epochs`, accumulating into the histories the caller built.

        Which metrics a run tracks is the experiment's choice, not the engine's, so the
        histories come from the script: anything model-specific (delta ratios, say)
        reads what it needs off the `model` handed to accumulate(). Three names are
        still expected: train "loss" and "gradnorm" for the collapse check, and val
        "top1acc" to pick the best checkpoint.
        """
        start_epoch = self._retake(train_history_metrics, val_history_metrics) if self.retake else 0
        for epoch in range(start_epoch, epochs):
            self.train_epoch(train_history_metrics, epoch=epoch, epochs=epochs)
            # Rank 0 alone validates the full set; others wait so the next DDP train step stays synced.
            if self._is_main_process():
                self.validate(val_history_metrics, epoch=epoch, epochs=epochs)
                self._step_scheduler_on_epoch(val_history_metrics)
                if val_history_metrics.last_is_best("top1acc"):
                    self.save_model(name="best", epoch=epoch)
                self.save_model(name="last", epoch=epoch)
                train_history_metrics.save()
                val_history_metrics.save()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            # After the saves, so a collapsed run still leaves its full history on disk.
            # Every rank calls in: rank 0 decides and the others wait for its verdict.
            self._check_collapse(train_history_metrics, epoch)
        if self._is_main_process():
            train_history_metrics.plot_history()
            val_history_metrics.plot_history()
        return train_history_metrics, val_history_metrics

    def save_model(self, name: str, epoch=None):
        if not self._is_main_process():
            return
        self.weights_path.mkdir(parents=True, exist_ok=True)
        state = self._unwrap_model().state_dict()
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
    experiment_path = Path("outputs/test")
    train_history_metrics = DictHistoryMetrics(experiment_path, split="train")
    train_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
    train_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
    train_history_metrics.addHistoryMetric("gradnorm", MetricGradNorm)
    train_history_metrics.addHistoryMetric("lr", MetricLR)
    val_history_metrics = DictHistoryMetrics(experiment_path, split="val")
    val_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
    val_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
    trainer.fit(epochs=10, train_history_metrics=train_history_metrics, val_history_metrics=val_history_metrics)

if __name__ == "__main__":
    main()
