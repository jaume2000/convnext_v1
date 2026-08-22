import os
from pathlib import Path
from torch.utils.data import DataLoader
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_train_batch_transforms, build_val_transforms
from engine.validator import Validator
from metrics.metricHistory import DictHistoryMetrics
from metrics.metricLoss import MetricLoss
from metrics.top1acc import Top1AccMetric
from models.backbones.convnext import ConvNextV1
from timm.loss import SoftTargetCrossEntropy
import torch
import torch.nn as nn

BATCH_SIZE = 128

def available_cpus() -> int:
    """Cores this process may use (SLURM cgroup aware)."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNextV1()
stateDict = torch.load("./outputs/convnextv1_imagenet/weights/last.pth")["model"]
model.load_state_dict(stateDict)
if device.type == "cuda" and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"DataParallel on GPUs: {model.device_ids}")
elif device.type == "cuda":
    print("Using single GPU: cuda:0")
else:
    print("Using CPU")

# train loader and val loader of imagenet with huggingface datasets

val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())

num_workers = available_cpus() // 2
print(f"Dataloader workers: {num_workers}")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

validator = Validator(
    model=model,
    criterion=SoftTargetCrossEntropy(),
    device=device,
    val_loader=val_loader,
    batch_transforms=build_train_batch_transforms(),
    num_classes=1000,
    amp=False,
)
val_histoy_metrics = DictHistoryMetrics(Path(f"outputs/convNext_validation"), split="val")
val_histoy_metrics.addHistoryMetric("top1acc", Top1AccMetric)
val_histoy_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
validator.validate(val_histoy_metrics)
