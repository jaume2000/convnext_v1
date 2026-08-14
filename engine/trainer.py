from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
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
    amp: bool = True):
        self.experiment_name = experiment_name
        self.model = model
        self.optimizer = optimizer          # SGD, AdamW, ...
        self.criterion = criterion          # Loss function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.batch_transforms = batch_transforms
        self.num_classes = num_classes
        self.amp = amp and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.amp)
        self.experiment_path = Path(f"outputs/{self.experiment_name}")
        self.weights_path = self.experiment_path / "weights"

    def train_epoch(self, train_histoy_metrics: Metrichistory, epoch: int, epochs: int):
        self.model.train()
        self.model.to(self.device)
        train_histoy_metrics.create_point()
        pbar = tqdm(self.train_loader, desc=f"train {epoch + 1}/{epochs}")
        for batch, y_labels in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            batch = batch.to(self.device, non_blocking=True)
            y_labels = y_labels.to(self.device, non_blocking=True)
            batch, y_labels = self.batch_transforms(batch, y_labels)
            with autocast("cuda", enabled=self.amp):
                pred = self.model(batch)
                loss = self.criterion(pred, y_labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            train_histoy_metrics.accumulate_last_point(pred.float(), y_labels, loss.detach().float())
            postfix = {"loss": f"{loss.item():.3f}", "amp": int(self.amp)}
            if self.device.type == "cuda":
                # YOLO-style: reserved VRAM per visible GPU (GiB).
                mem = " ".join(
                    f"{torch.cuda.memory_reserved(i) / (1024 ** 3):.1f}G"
                    for i in range(torch.cuda.device_count())
                )
                postfix["vram"] = mem
            pbar.set_postfix(postfix)
        train_histoy_metrics.print_last()

    # Validate for classification
    def validate(self, val_histoy_metrics: Metrichistory, epoch: int, epochs: int):
        self.model.eval()
        self.model.to(self.device)
        pbar = tqdm(self.val_loader, desc=f"val {epoch + 1}/{epochs}")
        val_histoy_metrics.create_point()
        with torch.inference_mode():
            for batch, y_labels in pbar:
                batch = batch.to(self.device, non_blocking=True)
                y_labels = torch.nn.functional.one_hot(y_labels, num_classes=self.num_classes).float()
                y_labels = y_labels.to(self.device, non_blocking=True)
                with autocast("cuda", enabled=self.amp):
                    pred = self.model(batch)
                    loss = self.criterion(pred, y_labels)
                val_histoy_metrics.accumulate_last_point(pred.float(), y_labels, loss.detach().float())
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
        print(f"Retake: loaded {ckpt_name}.pth, resuming at epoch {start_epoch + 1}")
        return start_epoch

    def fit(self, epochs, retake: bool = False):
        train_histoy_metrics = Metrichistory(self.experiment_path, Top1AccMetric, "train")
        val_histoy_metrics = Metrichistory(self.experiment_path, Top1AccMetric, "val")
        start_epoch = self._retake(train_histoy_metrics, val_histoy_metrics) if retake else 0
        for epoch in range(start_epoch, epochs):
            self.train_epoch(train_histoy_metrics, epoch=epoch, epochs=epochs)
            self.validate(val_histoy_metrics, epoch=epoch, epochs=epochs)
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
        payload = {"model": state, "epoch": epoch} if epoch is not None else state
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
    trainer = Trainer(
        experiment_name="test",
        model=model,
        optimizer=optimizer,
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
