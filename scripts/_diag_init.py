"""Throwaway diagnostic: compare this ConvNeXt against timm's at init."""
import torch
import torch.nn as nn
import timm

from models.backbones.convnext import ConvNextV1

torch.manual_seed(0)
dev = "cuda"
B, C, N = 32, 1000, 1000

def report(model, name):
    model = model.to(dev).train()
    x = torch.randn(B, 3, 224, 224, device=dev)
    y = torch.randint(0, N, (B,), device=dev)
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    total = torch.tensor(list(norms.values())).norm().item()
    top = sorted(norms.items(), key=lambda kv: -kv[1])[:8]
    print(f"\n=== {name} ===")
    print(f"  logits std {out.std().item():.4f}  mean|logit| {out.abs().mean().item():.4f}  loss {loss.item():.4f}")
    print(f"  total grad norm {total:.4f}")
    for n, v in top:
        print(f"    {v:9.4f}  {n}")
    return total

report(ConvNextV1(), "ConvNextV1 (repo)")
report(timm.create_model("convnext_tiny", pretrained=False, drop_path_rate=0.1), "timm convnext_tiny")


# --- activation scale through the repo model's residual stream ---
m = ConvNextV1().to(dev).eval()
x = torch.randn(B, 3, 224, 224, device=dev)
with torch.no_grad():
    h = m.stem(x); print(f"\nstem      out std {h.std():.4f} absmax {h.abs().max():.3f}")
    for i, st in enumerate([m.stage1, m.stage2, m.stage3, m.stage4], 1):
        h = st(h); print(f"stage{i}    out std {h.std():.4f} absmax {h.abs().max():.3f}")
    h = m.globalPool(h); print(f"pool+ln   out std {h.std():.4f}")
    h = m.fc(h); print(f"fc        out std {h.std():.4f}")
