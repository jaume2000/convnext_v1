"""Report non-finite tensors and per-tensor magnitude for a checkpoint."""
import sys

import torch


def summarize(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    print(f"=== {path}")
    if isinstance(ckpt, dict):
        print("keys:", list(ckpt.keys()), "epoch:", ckpt.get("epoch"))

    bad, clean = [], []
    for name, tensor in state.items():
        t = tensor.float()
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        if n_nan or n_inf:
            bad.append((name, n_nan, n_inf, t.numel()))
        else:
            clean.append((name, float(t.abs().mean()), float(t.abs().max())))
    print(f"tensors: {len(state)}  with non-finite: {len(bad)}")
    for name, n_nan, n_inf, numel in bad[:12]:
        print(f"  NaN/Inf {name}: nan={n_nan} inf={n_inf} / {numel}")
    if len(bad) > 12:
        print(f"  ... and {len(bad) - 12} more")
    for name, mean_abs, max_abs in clean:
        if "gamma" in name or name.endswith(("stem.1.weight", "fc.weight", "fc.bias")) or "globalPool.1.weight" in name:
            print(f"  {name}: mean|w|={mean_abs:.3e} max|w|={max_abs:.3e}")

    opt = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
    if opt is not None:
        n_nan_m = n_nan_v = 0
        max_v = 0.0
        for st in opt["state"].values():
            if "exp_avg" in st:
                n_nan_m += int(torch.isnan(st["exp_avg"].float()).sum())
            if "exp_avg_sq" in st:
                sq = st["exp_avg_sq"].float()
                n_nan_v += int(torch.isnan(sq).sum())
                finite = sq[torch.isfinite(sq)]
                if finite.numel():
                    max_v = max(max_v, float(finite.max()))
        print(f"optimizer: nan in exp_avg={n_nan_m}, nan in exp_avg_sq={n_nan_v}, max finite exp_avg_sq={max_v:.3e}")
        print("param_group lr/wd:", [(g["lr"], g["weight_decay"]) for g in opt["param_groups"]])
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
