import os

from torch.utils.data import DataLoader
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_identity_batch_transforms, build_train_batch_transforms, build_train_transforms, build_val_transforms
from engine.trainer import Trainer
from torch.optim import AdamW
from models.backbones.delta_convnext import DeltaConvNext
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups, build_param_groups_with_delta_weight_decay
from timm.loss import SoftTargetCrossEntropy
import torch
import torch.nn as nn

EPOCHS = 100
WARMUP_EPOCHS = 7
BATCH_SIZE = 1024
# ConvNeXt uses 4e-3 at batch 4096; linear scaling gives the equivalent for our batch.
LR = 1e-4
MIN_LR = 1e-7
# Applied to conv/linear weights only, see build_param_groups.
WEIGHT_DECAY = 0.05
DELTA_WEIGHT_DECAY = 1.0
# Chained 12h SLURM jobs set RETAKE=1 in .env to resume without editing this file.
RETAKE = os.environ.get("RETAKE", "0") == "1"
BASELINE_WEIGHTS = os.environ.get(
    "BASELINE_WEIGHTS",
    "outputs/convnextv1_imagenet/weights/last.pth",
)


def available_cpus() -> int:
    """Cores this process may use (SLURM cgroup aware)."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeltaConvNext()
if not RETAKE:
    state_dict = torch.load(BASELINE_WEIGHTS, map_location=device)["model"]
    model.load_state_dict(state_dict, strict=False)
else:
    print("RETAKE=1: skipping baseline ConvNeXt load; Trainer restores delta checkpoint")
# Index 5 = stage3 block at position 5 (0-based), used as the shared backbone.
model.rewire(sharedBlock=5)
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

param_groups = build_param_groups_with_delta_weight_decay(model, weight_decay=WEIGHT_DECAY, delta_weight_decay=DELTA_WEIGHT_DECAY)
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
print(f"Peak LR: {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, cosine over {EPOCHS} epochs")

trainer = Trainer(
    experiment_name="delta_convnextv1_imagenet",
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
trainer.fit(EPOCHS)
