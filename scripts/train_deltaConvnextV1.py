import os
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_train_batch_transforms, build_train_transforms, build_val_transforms
from engine.trainer import Trainer
from metrics.metricDeltaRatio import MetricDeltaRatio
from metrics.metricGradNorm import MetricGradNorm
from metrics.metricHistory import DictHistoryMetrics
from metrics.metricLoss import MetricLoss
from metrics.metricLR import MetricLR
from metrics.top1acc import Top1AccMetric
from torch.optim import AdamW
from models.backbones.delta_convnext import DeltaConvNext
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups_with_delta_weight_decay
from timm.loss import SoftTargetCrossEntropy
import torch


EXPERIMENT_NAME = "delta_convnextv1_imagenet"
EXPERIMENT_PATH = Path("outputs") / EXPERIMENT_NAME
USE_DDP = True
EPOCHS = 300
WARMUP_EPOCHS = 10
BATCH_SIZE = 256
# ConvNeXt uses 4e-3 at batch 4096; linear scaling gives the equivalent for our batch.
LR = 1e-3
MIN_LR = 1e-7
# Applied to conv/linear weights only, see build_param_groups.
WEIGHT_DECAY = 0.05
DELTA_WEIGHT_DECAY = 0.05
# Chained 12h SLURM jobs set RETAKE=1 in .env to resume without editing this file.
RETAKE = os.environ.get("RETAKE", "0") == "1"


def available_cpus() -> int:
    """Cores this process may use (SLURM cgroup aware)."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


def setup_ddp():
    """torchrun sets LOCAL_RANK / RANK / WORLD_SIZE; init NCCL and bind this process to its GPU."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


local_rank, rank, world_size = setup_ddp() if USE_DDP else (0, 0, 1)
use_ddp = USE_DDP and world_size > 1 and torch.cuda.is_available()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")
    use_ddp = False

model = DeltaConvNext()
model.rewire()
model = model.to(device)
if use_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print(f"DDP on {world_size} GPUs (local_rank={local_rank})")
elif device.type == "cuda":
    print("Using single GPU: cuda:0")
else:
    print("Using CPU")

train_dataset = ImageNetDataset(split="train", transforms=build_train_transforms())
val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())

# One process per GPU; split the node CPUs so workers do not oversubscribe.
num_workers = max(1, available_cpus() // world_size) if use_ddp else available_cpus()
if rank == 0:
    print(f"Dataloader workers per process: {num_workers}")

train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=num_workers > 0,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=num_workers > 0,
)

param_groups = build_param_groups_with_delta_weight_decay(model, weight_decay=WEIGHT_DECAY, delta_weight_decay=DELTA_WEIGHT_DECAY)
if rank == 0:
    print(
        f"Weight decay {WEIGHT_DECAY} on {sum(p.numel() for p in param_groups[0]['params'])} params, "
        f"delta weight decay {DELTA_WEIGHT_DECAY} on {sum(p.numel() for p in param_groups[1]['params'])} params, "
        f"none on {sum(p.numel() for p in param_groups[2]['params'])} (norms, biases, layer scale)"
    )
optimizer = AdamW(param_groups, lr=LR, betas=(0.9, 0.999))
steps_per_epoch = len(train_loader)
scheduler = CosineWithWarmup(
    optimizer,
    warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
    total_steps=EPOCHS * steps_per_epoch,
    min_lr=MIN_LR,
)
if rank == 0:
    print(f"Peak LR: {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, cosine over {EPOCHS} epochs")

# Registration order is the order of the progress bar and of the end-of-epoch print.
train_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="train")
train_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
train_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
train_history_metrics.addHistoryMetric("gradnorm", MetricGradNorm)
train_history_metrics.addHistoryMetric("lr", MetricLR)
# Optional: a run resumed from before this metric existed has no history file for it.
train_history_metrics.addHistoryMetric("deltaratio", MetricDeltaRatio, optional=True, sample_every=20)
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
    retake=RETAKE,
)
trainer.fit(EPOCHS, train_history_metrics, val_history_metrics)

if use_ddp:
    dist.destroy_process_group()
