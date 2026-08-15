import os

from torch.utils.data import DataLoader
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_train_batch_transforms, build_train_transforms, build_val_transforms
from models.backbones.convnext import ConvNextV1
from engine.trainer import Trainer
from torch.optim import AdamW
from optim.cosineSchedule import CosineWithWarmup
from timm.loss import SoftTargetCrossEntropy
import torch
import torch.nn as nn

EPOCHS = 600
WARMUP_EPOCHS = 20
BATCH_SIZE = 1024
# ConvNeXt uses 4e-3 at batch 4096; linear scaling gives the equivalent for our batch.
LR = 4e-3 * BATCH_SIZE / 4096
MIN_LR = 1e-6


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

# JPEG decode + RandAugment is the pipeline's bottleneck, so use every allocated core.
# With automatic batching each worker builds a whole batch, so throughput is
# num_workers / (batch_size * seconds_per_image): too few workers starves the GPUs.
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

optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.05)
steps_per_epoch = len(train_loader)
scheduler = CosineWithWarmup(
    optimizer,
    warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
    total_steps=EPOCHS * steps_per_epoch,
    min_lr=MIN_LR,
)
print(f"Peak LR: {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, cosine over {EPOCHS} epochs")

trainer = Trainer(
    experiment_name="convnextv1_imagenet",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=SoftTargetCrossEntropy(),
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    batch_transforms=build_train_batch_transforms(),
    num_classes=1000,
    amp=True,
)

# The 12h wall clock needs several chained jobs, so this is an env var: RETAKE=1 resumes
# from outputs/<experiment>/weights/last.pth instead of having to edit this file.
trainer.fit(EPOCHS, retake=os.environ.get("RETAKE", "0") == "1")
