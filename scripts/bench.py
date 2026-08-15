"""Locate the training bottleneck: filesystem, JPEG decode, augmentation or GPU compute.

Each stage is measured in isolation and reported as images/second, so they can be
compared directly against the observed training throughput
(global_batch_size / seconds_per_iteration).

    TRAIN_SCRIPT=scripts/bench.py sbatch --time=00:30:00 \
        --account="$SLURM_ACCOUNT" jobs/train_convnext.sh
"""

import argparse
import io
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from timm.loss import SoftTargetCrossEntropy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from data.transforms.transforms import build_train_batch_transforms, build_train_transforms
from models.backbones.convnext import ConvNextV1

TRAIN_SIZE = 1_281_167


def available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm_cpus) if slurm_cpus else len(os.sched_getaffinity(0))


def undecoded(hf_ds):
    """Same rows, but with the raw JPEG bytes instead of a decoded PIL image."""
    from datasets import Image as HFImage

    return hf_ds.cast_column("image", HFImage(decode=False))


class RawRows(Dataset):
    """Row fetch without decoding: isolates arrow/GPFS read cost."""

    def __init__(self, hf_ds):
        self.ds = undecoded(hf_ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return len(self.ds[index]["image"]["bytes"])


class DecodeOnly(Dataset):
    """Row fetch + full JPEG decode, no augmentation."""

    def __init__(self, hf_ds):
        self.ds = undecoded(hf_ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img = Image.open(io.BytesIO(self.ds[index]["image"]["bytes"])).convert("RGB")
        return img.size[0]


class InMemoryJpegs(Dataset):
    """Full train pipeline over JPEGs already in RAM: the CPU ceiling with no I/O.

    Workers inherit the blobs through fork, so nothing touches the filesystem and the
    number measures what the cores could deliver if the dataset were staged in memory.
    """

    def __init__(self, blobs: list[bytes], transforms, length: int):
        self.blobs = blobs
        self.transforms = transforms
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(io.BytesIO(self.blobs[index % len(self.blobs)])).convert("RGB")
        return self.transforms(img), 0


def load_blobs(hf_ds, count: int, short_side: int | None = None) -> list[bytes]:
    """Sample JPEGs spread across the split, optionally re-encoded to a shorter side."""
    ds = undecoded(hf_ds)
    step = max(len(ds) // count, 1)
    blobs = []
    for index in range(0, count * step, step):
        payload = ds[index]["image"]["bytes"]
        if short_side:
            img = Image.open(io.BytesIO(payload)).convert("RGB")
            width, height = img.size
            scale = short_side / min(width, height)
            if scale < 1:
                img = img.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            payload = buffer.getvalue()
        blobs.append(payload)
    return blobs


def bench_loader(name: str, dataset: Dataset, batch_size: int, workers: int, results: dict):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        drop_last=True,
    )
    # Warm up for one full round of workers, since each worker assembles a whole batch
    # and the pipeline is only saturated once every worker has delivered.
    warmup, measured = workers + 1, 2 * workers
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    start = time.perf_counter()
    for _ in range(measured):
        next(it)
    elapsed = time.perf_counter() - start
    del it, loader
    results[name] = measured * batch_size / elapsed
    print(f"{name:<34} {results[name]:>9.0f} img/s")


def bench_model(
    name: str,
    per_gpu_batch: int,
    steps: int,
    channels_last: bool,
    dtype: torch.dtype,
    gpus: int,
    compile_model: bool,
    results: dict,
):
    model = ConvNextV1().cuda()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if compile_model:
        model = torch.compile(model)
    if gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(gpus)))
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = SoftTargetCrossEntropy()
    mixup = build_train_batch_transforms()
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)

    batch_size = per_gpu_batch * gpus
    batch = torch.randint(0, 255, (batch_size, 3, 224, 224), device="cuda", dtype=torch.uint8).float()
    labels = torch.randint(0, 1000, (batch_size,), device="cuda")
    if channels_last:
        batch = batch.contiguous(memory_format=torch.channels_last)
    batch, targets = mixup(batch, labels)

    # torch.compile pays for graph capture on the first steps, so warm up longer.
    warmup = 25 if compile_model else 8
    for step in range(warmup + steps):
        if step == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            loss = criterion(model(batch), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    results[name] = steps * batch_size / elapsed
    print(f"{name:<34} {results[name]:>9.0f} img/s")
    del model, optimizer, batch, targets
    torch.cuda.empty_cache()


def report(results: dict, global_batch: int):
    print("\n" + "=" * 62)
    print(f"{'stage':<34} {'img/s':>9}  {'s/it @' + str(global_batch):>10}  epoch")
    print("-" * 62)
    for name, rate in results.items():
        epoch_min = TRAIN_SIZE / rate / 60
        print(f"{name:<34} {rate:>9.0f}  {global_batch / rate:>10.2f}  {epoch_min:>5.1f} min")
    print("=" * 62)
    print("The slowest of the data stages and the GPU stage is what caps training.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--data-batch", type=int, default=256,
                        help="Small batches keep the data stages' memory bounded; throughput is batch-size independent.")
    parser.add_argument("--gpu-batch", type=int, default=256, help="Per-GPU batch for the compute stages.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--blobs", type=int, default=512, help="Distinct images held in RAM for the CPU stages.")
    parser.add_argument("--short-side", type=int, default=256, help="Target short side for the re-encoded stage.")
    parser.add_argument("--compile", action="store_true", help="Also measure torch.compile (slow to warm up).")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-io", action="store_true", help="Keep the RAM-only stages, drop the ones hitting GPFS.")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    gpus = torch.cuda.device_count()
    if gpus:
        # Mirror the trainer's backend setup or the conv algos measured here differ.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"GPUs: {gpus}  workers: {args.workers}  torch: {torch.__version__}\n")
    results: dict[str, float] = {}

    if not args.skip_data:
        from data.imagenet import ImageNetDataset

        base = ImageNetDataset(split="train")
        if not args.skip_io:
            bench_loader("data: read rows only", RawRows(base.ds), args.data_batch, args.workers, results)
            bench_loader("data: read + jpeg decode", DecodeOnly(base.ds), args.data_batch, args.workers, results)
            full = ImageNetDataset(split="train", transforms=build_train_transforms())
            bench_loader("data: full train pipeline", full, args.data_batch, args.workers, results)

        transforms = build_train_transforms()
        cached = args.data_batch * (3 * args.workers + 2)
        originals = load_blobs(base.ds, args.blobs)
        print(f"sampled {len(originals)} jpegs, mean {sum(map(len, originals)) / len(originals) / 1024:.0f} KiB")
        bench_loader("cpu: pipeline, originals in RAM",
                     InMemoryJpegs(originals, transforms, cached), args.data_batch, args.workers, results)
        resized = load_blobs(base.ds, args.blobs, short_side=args.short_side)
        print(f"re-encoded to {args.short_side}px short side, mean "
              f"{sum(map(len, resized)) / len(resized) / 1024:.0f} KiB")
        bench_loader(f"cpu: pipeline, {args.short_side}px in RAM",
                     InMemoryJpegs(resized, transforms, cached), args.data_batch, args.workers, results)

    if not args.skip_gpu and gpus:
        bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
        bench_model("gpu: 1x NCHW fp16", args.gpu_batch, args.steps, False, torch.float16, 1, False, results)
        bench_model("gpu: 1x channels_last fp16", args.gpu_batch, args.steps, True, torch.float16, 1, False, results)
        if bf16:
            bench_model("gpu: 1x channels_last bf16", args.gpu_batch, args.steps, True, torch.bfloat16, 1, False, results)
        if args.compile:
            bench_model("gpu: 1x channels_last+compile", args.gpu_batch, args.steps, True, amp_dtype, 1, True, results)
        if gpus > 1:
            bench_model(f"gpu: {gpus}x DataParallel channels_last",
                        args.gpu_batch, args.steps, True, amp_dtype, gpus, False, results)

    report(results, args.gpu_batch * max(gpus, 1))


if __name__ == "__main__":
    main()
