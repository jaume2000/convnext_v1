# Per image transforms

from torchvision.transforms import v2
from timm.data import create_transform, Mixup

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_train_transforms(img_size=224):
    return create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment="rand-m9-mstd0.5",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        color_jitter=0.4,
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )

def build_val_transforms(img_size=224):
    return create_transform(
        input_size=img_size,
        is_training=False,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def build_train_batch_transforms(num_classes: int = 1000):
    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,          # apply mixup/cutmix every batch
        switch_prob=0.5,   # P(CutMix | applying)
        mode="batch",      # mix whole batch pairwise
        label_smoothing=0.1,
        num_classes=num_classes,
    )