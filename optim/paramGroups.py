import torch.nn as nn


def build_param_groups_with_delta_weight_decay(
    model: nn.Module,
    weight_decay: float,
    delta_weight_decay: float,
    delta_norm_weight_decay: float = 0.0,
) -> list[dict]:
    NORM_DELTAS = ("lnWDelta", "lnBDelta", "lsDelta")
    base, delta, delta_norm, none = [], [], [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("Delta"):
            (delta_norm if name.split(".")[-1] in NORM_DELTAS else delta).append(param)
        else:
            (base if param.ndim >= 2 else none).append(param)

    if delta:
        n_blocks = 9
        # 6 por bloque: dw, pw1 y pw2, weight y bias. Los deltas de ln y ls van a delta_norm.
        per_block = 6
        assert len(delta) == per_block * n_blocks, f"Esperaba {per_block*n_blocks} deltas de conv, hay {len(delta)}"

    groups = [{"params": base, "weight_decay": weight_decay}]
    if delta:
        groups.append({"params": delta, "weight_decay": delta_weight_decay})
    if delta_norm:
        groups.append({"params": delta_norm, "weight_decay": delta_norm_weight_decay})
    groups.append({"params": none, "weight_decay": 0.0})
    return groups

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
