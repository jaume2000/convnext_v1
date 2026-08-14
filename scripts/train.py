from torch.utils.data import DataLoader
from data.imagenet import ImageNetDataset
from data.transforms.transforms import build_train_batch_transforms, build_train_transforms, build_val_transforms
from models.backbones.convnext import ConvNextV1
from engine.trainer import Trainer
from torch.optim import AdamW
from timm.loss import SoftTargetCrossEntropy
import torch

model = ConvNextV1()

# train loader and val loader of imagenet with huggingface datasets

train_dataset = ImageNetDataset(split="train", transforms=build_train_transforms())
val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())

train_loader = DataLoader(train_dataset, batch_size=224, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=224, shuffle=False, num_workers=4)

trainer = Trainer(
    experiment_name="convnextv1_imagenet",
    model=model,
    optimizer=AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.999), weight_decay=0.05),
    criterion=SoftTargetCrossEntropy(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    train_loader=train_loader,
    val_loader=val_loader,
    batch_transforms=build_train_batch_transforms(),
    num_classes=1000
)

trainer.fit(600)