# Per image transforms

import torch
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

def build_identity_batch_transforms(num_classes: int = 1000, label_smoothing: float = 0.1):
    """Mixup/CutMix disabled, but still smoothed soft targets.

    Smoothing is not cosmetic here: with hard one-hot targets nothing ever stops
    rewarding larger logits, so logit magnitudes and gradient norms grow without bound,
    and the ConvNeXt peak LR (1e-3 at batch 1024) was tuned with Mixup and smoothing
    supplying that damping. Dropping both at once is what blew up a previous run.
    """
    # timm's Mixup spreads the smoothing mass the same way, so switching between this
    # and build_train_batch_transforms does not change the loss scale.
    off_value = label_smoothing / num_classes
    on_value = 1.0 - label_smoothing + off_value

    def identity_batch_transforms(batch, y_labels):
        if y_labels.dim() == 1:
            y_labels = torch.nn.functional.one_hot(y_labels, num_classes=num_classes).float()
            y_labels = y_labels * (on_value - off_value) + off_value
        return batch, y_labels
    return identity_batch_transforms