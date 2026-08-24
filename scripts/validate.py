import os
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_train_batch_transforms, build_val_transforms
from engine.validator import Validator
from metrics.metricHistory import DictHistoryMetrics
from metrics.metricLoss import MetricLoss
from metrics.top1acc import Top1AccMetric
from models.backbones.delta_convnext import DeltaConvNext
from timm.loss import SoftTargetCrossEntropy
import torch

BATCH_SIZE = 128
CHECKPOINT = Path("outputs/outputs/delta_convnextv1_imagenet/weights/last.pth")
OUTPUT_PATH = Path("outputs/delta_convnext_validation")


def available_cpus() -> int:
    """Cores this process may use (SLURM cgroup aware)."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


def load_delta_convnext(checkpoint: Path | str, useDeltas: bool = False) -> DeltaConvNext:
    """DeltaConvNext with the weights of `checkpoint`, left on CPU."""
    model = DeltaConvNext(useDeltas=useDeltas)
    # The checkpoint was saved after rewiring, so its keys are sharedBlock.* / deltifiedStage3.*;
    # those modules only exist once stage3 has been replaced.
    model.rewire()
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict, ckpt_epoch = ckpt["model"], ckpt.get("epoch")
    else:
        state_dict, ckpt_epoch = ckpt, None
    model.load_state_dict(state_dict)
    print(f"Loaded {checkpoint}" + (f" (epoch {ckpt_epoch})" if ckpt_epoch is not None else ""))
    return model


def validate(
    model: nn.Module,
    batch_size: int = BATCH_SIZE,
    output_path: Path | str = OUTPUT_PATH,
    device: torch.device | None = None,
    num_workers: int | None = None,
    num_classes: int = 1000,
    amp: bool = False,
) -> DictHistoryMetrics:
    """Run `model` over the ImageNet validation split and return its metric history."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No DataParallel here: the delta blocks keep the shared block as an unregistered attribute,
    # so the replicas on the other GPUs would still read tensors that live on cuda:0.
    print(f"Using {device}")
    model = model.to(device)

    val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())

    if num_workers is None:
        num_workers = int(available_cpus() // 2)
    print(f"Dataloader workers: {num_workers}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    validator = Validator(
        model=model,
        criterion=SoftTargetCrossEntropy(),
        device=device,
        val_loader=val_loader,
        batch_transforms=build_train_batch_transforms(),
        num_classes=num_classes,
        amp=amp,
    )
    val_history_metrics = DictHistoryMetrics(output_path, split="val")
    val_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
    val_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
    validator.validate(val_history_metrics)
    return val_history_metrics


if __name__ == "__main__":
    validate(
        model=load_delta_convnext(CHECKPOINT),
        batch_size=BATCH_SIZE,
        output_path=OUTPUT_PATH,
    )
