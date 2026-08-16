import os

from torch.utils.data import DataLoader
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_train_batch_transforms, build_train_transforms, build_val_transforms
from models.backbones.convnext import ConvNextV1
from engine.repeatOneBatchTrainer import Trainer
from torch.optim import AdamW
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups
from timm.loss import SoftTargetCrossEntropy
import torch
import torch.nn as nn

EPOCHS = 300
WARMUP_EPOCHS = 10
BATCH_SIZE = 1024
# ConvNeXt uses 4e-3 at batch 4096; linear scaling gives the equivalent for our batch.
LR = 1e-3
MIN_LR = 1e-6
# Applied to conv/linear weights only, see build_param_groups.
WEIGHT_DECAY = 0.05


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

train_dataset = ImageNetDataset(split="train", transforms=None)
val_dataset = ImageNetDataset(split="validation", transforms=None)

# JPEG decode + RandAugment is the pipeline's bottleneck, so use every allocated core.
# With automatic batching each worker builds a whole batch, so throughput is
# num_workers / (batch_size * seconds_per_image): too few workers starves the GPUs.
num_workers = available_cpus() * 2
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

trainer = Trainer(
    experiment_name="convnextv1_imagenet_repeatOneBatch",
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
    gradient_clipping=5.0, # norm clipping
)

# The 12h wall clock needs several chained jobs, so this is an env var: RETAKE=1 resumes
# from outputs/<experiment>/weights/last.pth instead of having to edit this file.
trainer.fit(EPOCHS, retake=os.environ.get("RETAKE", "0") == "1")
