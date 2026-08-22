import torch.nn as nn

def build_param_groups_with_delta_weight_decay(model: nn.Module, weight_decay: float, delta_weight_decay: float) -> list[dict]:
    """Split parameters into base decay, delta decay and non-decayed groups."""
    base_decayed, delta_decayed, not_decayed = [], [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(param, "isDelta", False):
            delta_decayed.append(param)
        else:
            if param.ndim >= 2:
                base_decayed.append(param)
            else:
                not_decayed.append(param)
    return [
        {"params": base_decayed, "weight_decay": weight_decay},
        {"params": delta_decayed, "weight_decay": delta_weight_decay},
        {"params": not_decayed, "weight_decay": 0.0},
    ]


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into a decayed and a non-decayed group.

    Weight decay on a 1-D parameter (LayerNorm gain and bias, conv/linear bias,
    LayerScale gamma) does not regularise anything: those parameters set a scale, so
    decay just walks them towards zero at a rate of exp(-sum(lr) * wd) whenever the
    gradient is small. For a LayerNorm gain that is not cosmetic: once the gain
    collapses, the layer's backward multiplies gradients by 1/sqrt(var + eps), up to
    1000x per layer, which is how a perfectly finite forward pass can still produce a
    non-finite gradient. Excluding them is what the ConvNeXt/DeiT recipes do.

    Only weights of rank >= 2 (conv kernels, linear matrices) keep the decay.
    """
    decayed, not_decayed = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decayed if param.ndim >= 2 else not_decayed).append(param)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": not_decayed, "weight_decay": 0.0},
    ]
