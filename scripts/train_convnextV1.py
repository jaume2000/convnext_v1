import os
from pathlib import Path

from torch.utils.data import DataLoader
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_identity_batch_transforms, build_train_batch_transforms, build_train_transforms, build_val_transforms
from metrics.metricGradNorm import MetricGradNorm
from metrics.metricHistory import DictHistoryMetrics
from metrics.metricLoss import MetricLoss
from metrics.metricLR import MetricLR
from metrics.top1acc import Top1AccMetric
from models.backbones.convnext import ConvNextV1
from engine.trainer import Trainer
from torch.optim import AdamW
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups
from timm.loss import SoftTargetCrossEntropy
from utils.env import experiment_name, load_dotenv
import torch
import torch.nn as nn

load_dotenv()
EXPERIMENT_NAME = experiment_name("convnextv1_imagenet")
EXPERIMENT_PATH = Path("outputs") / EXPERIMENT_NAME
EPOCHS = 300
WARMUP_EPOCHS = 20
BATCH_SIZE = 1024
# ConvNeXt uses 4e-3 at batch 4096; linear scaling gives the equivalent for our batch.
LR = 5e-4
MIN_LR = 1e-6
# Applied to conv/linear weights only, see build_param_groups.
WEIGHT_DECAY = 0.05
# Chained 12h SLURM jobs set RETAKE=1 in .env to resume without editing this file.
RETAKE = os.environ.get("RETAKE", "0") == "1"


def available_cpus() -> int:
    """Cores this process may use (SLURM cgroup aware)."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNextV1()
if device.type == "cuda" and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"DataParallel on GPUs: {model.device_ids}")
elif device.type == "cuda":
    print("Using single GPU: cuda:0")
else:
    print("Using CPU")

# train loader and val loader of imagenet with huggingface datasets

train_dataset = ImageNetDataset(split="train", transforms=build_train_transforms())
val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())

num_workers = available_cpus()
print(f"Dataloader workers: {num_workers}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

param_groups = build_param_groups(model, weight_decay=WEIGHT_DECAY)
print(
    f"Weight decay {WEIGHT_DECAY} on {sum(p.numel() for p in param_groups[0]['params'])} params, "
    f"none on {sum(p.numel() for p in param_groups[1]['params'])} (norms, biases, layer scale)"
)
optimizer = AdamW(param_groups, lr=LR, betas=(0.9, 0.999))
steps_per_epoch = len(train_loader)
scheduler = CosineWithWarmup(
    optimizer,
    warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
    total_steps=EPOCHS * steps_per_epoch,
    min_lr=MIN_LR,
)
print(f"Peak LR: {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, cosine over {EPOCHS} epochs")

# Registration order is the order of the progress bar and of the end-of-epoch print.
train_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="train")
train_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
train_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
train_history_metrics.addHistoryMetric("gradnorm", MetricGradNorm)
train_history_metrics.addHistoryMetric("lr", MetricLR)
val_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="val")
val_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
val_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)

trainer = Trainer(
    experiment_name=EXPERIMENT_NAME,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=SoftTargetCrossEntropy(),
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    batch_transforms=build_train_batch_transforms(),
    num_classes=1000,
    amp=False,
    gradient_clipping=None,
    retake=True,
)
trainer.fit(EPOCHS, train_history_metrics, val_history_metrics)
