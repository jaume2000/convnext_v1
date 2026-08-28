"""Shared ConvNeXt custom_forward configuration sweep on ImageNet validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from models.backbones.delta_convnext import CustomForwardConfig, DeltaConvNext
from utils.env import load_dotenv

# --- Experiment config (edit here) ---
EXPERIMENT_NAME = "sharedConvnextAblation"
CHECKPOINT = Path("outputs/shared_convnextv1_imagenet/weights/last.pth")
OUTPUT_DIR = Path("outputs") / EXPERIMENT_NAME

BATCH_SIZE = 1024
NUM_WORKERS = 16
MAX_BATCHES: int | None = None  # set e.g. 2 for a quick smoke test
AMP = False
RESUME = False
NUM_CLASSES = 1000


def available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


def load_shared_convnext(checkpoint: Path) -> DeltaConvNext:
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def cfg(
    tail: list[int],
    block_indices: list[int],
    euler_step: float = 1.0,
    method: str | None = None,
) -> CustomForwardConfig:
    out: CustomForwardConfig = {
        "block_indices": list(block_indices) + tail,
        "euler_step": euler_step,
    }
    if method is not None:
        out["method"] = method
    return out


def shared_block_count(block_indices: list[int]) -> int:
    return len(block_indices) - 2


def build_configurations(
    n_blocks: int,
    tail: list[int],
) -> list[tuple[str, CustomForwardConfig | None]]:
    return [
        ("-----------------", None),
        ("baseline (forward)", cfg(tail, list(range(n_blocks)))),
        ("D=1, ES=1", cfg(tail, [0] * 1)),
        ("D=2, ES=1", cfg(tail, [0] * 2)),
        ("D=3, ES=1", cfg(tail, [0] * 3)),
        ("D=4, ES=1", cfg(tail, [0] * 4)),
        ("D=5, ES=1", cfg(tail, [0] * 5)),
        ("D=6, ES=1", cfg(tail, [0] * 6)),
        ("D=7, ES=1", cfg(tail, [0] * 7)),
        ("D=8, ES=1", cfg(tail, [0] * 8)),
        ("D=9, ES=1", cfg(tail, [0] * 9)),
        ("D=10, ES=1", cfg(tail, [0] * 10)),
        ("D=11, ES=1", cfg(tail, [0] * 11)),
        ("D=12, ES=1", cfg(tail, [0] * 12)),
        ("D=18, ES=1", cfg(tail, [0] * 18)),
        ("D=24, ES=1", cfg(tail, [0] * 24)),
        ("D=32, ES=1", cfg(tail, [0] * 32)),
        ("D=48, ES=1", cfg(tail, [0] * 48)),
        ("D=64, ES=1", cfg(tail, [0] * 64)),
        ("D=96, ES=1", cfg(tail, [0] * 96)),
        ("D=128, ES=1", cfg(tail, [0] * 128)),
        ("D=256, ES=1", cfg(tail, [0] * 256)),
        ("D=512, ES=1", cfg(tail, [0] * 512)),
        ("D=1024, ES=1", cfg(tail, [0] * 1024)),
        ("D=10000, ES=1", cfg(tail, [0] * 10000)),
        ("-----------------", None),
        ("D=1, ES=9/1", cfg(tail, [0] * 1, euler_step=9 / 1)),
        ("D=2, ES=9/2", cfg(tail, [0] * 2, euler_step=9 / 2)),
        ("D=3, ES=9/3", cfg(tail, [0] * 3, euler_step=9 / 3)),
        ("D=4, ES=9/4", cfg(tail, [0] * 4, euler_step=9 / 4)),
        ("D=5, ES=9/5", cfg(tail, [0] * 5, euler_step=9 / 5)),
        ("D=6, ES=9/6", cfg(tail, [0] * 6, euler_step=9 / 6)),
        ("D=7, ES=9/7", cfg(tail, [0] * 7, euler_step=9 / 7)),
        ("D=8, ES=9/8", cfg(tail, [0] * 8, euler_step=9 / 8)),
        ("D=9, ES=9/9", cfg(tail, [0] * 9, euler_step=9 / 9)),
        ("D=10, ES=9/10", cfg(tail, [0] * 10, euler_step=9 / 10)),
        ("D=11, ES=9/11", cfg(tail, [0] * 11, euler_step=9 / 11)),
        ("D=12, ES=9/12", cfg(tail, [0] * 12, euler_step=9 / 12)),
        ("D=18, ES=9/18", cfg(tail, [0] * 18, euler_step=9 / 18)),
        ("D=24, ES=9/24", cfg(tail, [0] * 24, euler_step=9 / 24)),
        ("D=32, ES=9/32", cfg(tail, [0] * 32, euler_step=9 / 32)),
        ("D=48, ES=9/48", cfg(tail, [0] * 48, euler_step=9 / 48)),
        ("D=64, ES=9/64", cfg(tail, [0] * 64, euler_step=9 / 64)),
        ("D=96, ES=9/96", cfg(tail, [0] * 96, euler_step=9 / 96)),
        ("D=128, ES=9/128", cfg(tail, [0] * 128, euler_step=9 / 128)),
        ("D=256, ES=9/256", cfg(tail, [0] * 256, euler_step=9 / 256)),
        ("D=512, ES=9/512", cfg(tail, [0] * 512, euler_step=9 / 512)),
        ("D=1024, ES=9/1024", cfg(tail, [0] * 1024, euler_step=9 / 1024)),
        ("D=10000, ES=9/10000", cfg(tail, [0] * 10000, euler_step=9 / 10000)),
        ("-----------------", None),
        ("D=1, ES=9/1 RK2", cfg(tail, [0] * 1, euler_step=9 / 1, method="RK2")),
        ("D=2, ES=9/2 RK2", cfg(tail, [0] * 2, euler_step=9 / 2, method="RK2")),
        ("D=3, ES=9/3 RK2", cfg(tail, [0] * 3, euler_step=9 / 3, method="RK2")),
        ("D=4, ES=9/4 RK2", cfg(tail, [0] * 4, euler_step=9 / 4, method="RK2")),
        ("D=5, ES=9/5 RK2", cfg(tail, [0] * 5, euler_step=9 / 5, method="RK2")),
        ("D=6, ES=9/6 RK2", cfg(tail, [0] * 6, euler_step=9 / 6, method="RK2")),
        ("D=7, ES=9/7 RK2", cfg(tail, [0] * 7, euler_step=9 / 7, method="RK2")),
        ("D=8, ES=9/8 RK2", cfg(tail, [0] * 8, euler_step=9 / 8, method="RK2")),
        ("D=9, ES=9/9 RK2", cfg(tail, [0] * 9, euler_step=9 / 9, method="RK2")),
        ("D=10, ES=9/10 RK2", cfg(tail, [0] * 10, euler_step=9 / 10, method="RK2")),
        ("D=11, ES=9/11 RK2", cfg(tail, [0] * 11, euler_step=9 / 11, method="RK2")),
        ("D=12, ES=9/12 RK2", cfg(tail, [0] * 12, euler_step=9 / 12, method="RK2")),
        ("D=18, ES=9/18 RK2", cfg(tail, [0] * 18, euler_step=9 / 18, method="RK2")),
        ("D=24, ES=9/24 RK2", cfg(tail, [0] * 24, euler_step=9 / 24, method="RK2")),
        ("D=32, ES=9/32 RK2", cfg(tail, [0] * 32, euler_step=9 / 32, method="RK2")),
        ("D=48, ES=9/48 RK2", cfg(tail, [0] * 48, euler_step=9 / 48, method="RK2")),
        ("D=64, ES=9/64 RK2", cfg(tail, [0] * 64, euler_step=9 / 64, method="RK2")),
        ("D=96, ES=9/96 RK2", cfg(tail, [0] * 96, euler_step=9 / 96, method="RK2")),
        ("D=128, ES=9/128 RK2", cfg(tail, [0] * 128, euler_step=9 / 128, method="RK2")),
        ("D=256, ES=9/256 RK2", cfg(tail, [0] * 256, euler_step=9 / 256, method="RK2")),
        ("D=512, ES=9/512 RK2", cfg(tail, [0] * 512, euler_step=9 / 512, method="RK2")),
        ("D=1024, ES=9/1024 RK2", cfg(tail, [0] * 1024, euler_step=9 / 1024, method="RK2")),

        ("-----------------", None),
        ("D=1, ES=9/1 RK4", cfg(tail, [0] * 1, euler_step=9 / 1, method="RK4")),
        ("D=2, ES=9/2 RK4", cfg(tail, [0] * 2, euler_step=9 / 2, method="RK4")),
        ("D=3, ES=9/3 RK4", cfg(tail, [0] * 3, euler_step=9 / 3, method="RK4")),
        ("D=4, ES=9/4 RK4", cfg(tail, [0] * 4, euler_step=9 / 4, method="RK4")),
        ("D=5, ES=9/5 RK4", cfg(tail, [0] * 5, euler_step=9 / 5, method="RK4")),
        ("D=6, ES=9/6 RK4", cfg(tail, [0] * 6, euler_step=9 / 6, method="RK4")),
        ("D=7, ES=9/7 RK4", cfg(tail, [0] * 7, euler_step=9 / 7, method="RK4")),
        ("D=8, ES=9/8 RK4", cfg(tail, [0] * 8, euler_step=9 / 8, method="RK4")),
        ("D=9, ES=9/9 RK4", cfg(tail, [0] * 9, euler_step=9 / 9, method="RK4")),
        ("D=10, ES=9/10 RK4", cfg(tail, [0] * 10, euler_step=9 / 10, method="RK4")),
        ("D=11, ES=9/11 RK4", cfg(tail, [0] * 11, euler_step=9 / 11, method="RK4")),
        ("D=12, ES=9/12 RK4", cfg(tail, [0] * 12, euler_step=9 / 12, method="RK4")),
        ("D=18, ES=9/18 RK4", cfg(tail, [0] * 18, euler_step=9 / 18, method="RK4")),
        ("D=24, ES=9/24 RK4", cfg(tail, [0] * 24, euler_step=9 / 24, method="RK4")),
        ("D=32, ES=9/32 RK4", cfg(tail, [0] * 32, euler_step=9 / 32, method="RK4")),
        ("D=48, ES=9/48 RK4", cfg(tail, [0] * 48, euler_step=9 / 48, method="RK4")),
        ("D=64, ES=9/64 RK4", cfg(tail, [0] * 64, euler_step=9 / 64, method="RK4")),
        ("D=96, ES=9/96 RK4", cfg(tail, [0] * 96, euler_step=9 / 96, method="RK4")),
        ("D=128, ES=9/128 RK4", cfg(tail, [0] * 128, euler_step=9 / 128, method="RK4")),
        ("D=256, ES=9/256 RK4", cfg(tail, [0] * 256, euler_step=9 / 256, method="RK4")),
        ("D=512, ES=9/512 RK4", cfg(tail, [0] * 512, euler_step=9 / 512, method="RK4")),
        ("D=1024, ES=9/1024 RK4", cfg(tail, [0] * 1024, euler_step=9 / 1024, method="RK4")),
        ("-----------------", None),
        ("D=9, ES=0.01", cfg(tail, list(range(n_blocks)), euler_step=0.01)),
        ("D=9, ES=0.125", cfg(tail, list(range(n_blocks)), euler_step=0.125)),
        ("D=9, ES=0.25", cfg(tail, list(range(n_blocks)), euler_step=0.25)),
        ("D=9, ES=0.5", cfg(tail, list(range(n_blocks)), euler_step=0.5)),
        ("D=9, ES=2.0", cfg(tail, list(range(n_blocks)), euler_step=2.0)),
        ("D=9, ES=4.0", cfg(tail, list(range(n_blocks)), euler_step=4.0)),
        ("D=9, ES=8.0", cfg(tail, list(range(n_blocks)), euler_step=8.0)),
        ("D=9, ES=16.0", cfg(tail, list(range(n_blocks)), euler_step=16.0)),
        ("-----------------", None),

        ("D=45, ES=9/2.5", cfg(tail, list(range(n_blocks)), euler_step=9 / 2.5)),
        ("D=45, ES=9/5", cfg(tail, list(range(n_blocks)), euler_step=9 / 5)),
        ("D=45, ES=9/11", cfg(tail, list(range(n_blocks)), euler_step=9 / 11)),
        ("D=45, ES=9/22.5", cfg(tail, list(range(n_blocks)), euler_step=9 / 22.5)),
        ("D=45, ES=9/45", cfg(tail, list(range(n_blocks)), euler_step=9 / 45)),
        ("D=45, ES=9/90", cfg(tail, list(range(n_blocks)), euler_step=9 / 90)),
        ("D=45, ES=9/180", cfg(tail, list(range(n_blocks)), euler_step=9 / 180)),
        ("D=45, ES=9/360", cfg(tail, list(range(n_blocks)), euler_step=9 / 360)),
        ("D=45, ES=9/720", cfg(tail, list(range(n_blocks)), euler_step=9 / 720)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print configurations and exit (no checkpoint or dataset required)",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=9,
        help="Stage-3 block count for --list-only",
    )
    return parser.parse_args()


class AblationRunner:
    def __init__(
        self,
        model: DeltaConvNext,
        *,
        device: torch.device,
        batch_size: int,
        max_batches: int | None,
        amp: bool,
        n_blocks: int,
        num_workers: int,
    ) -> None:
        from data.imagenet import ImageNetDataset
        from data.transforms.transforms import build_val_transforms
        from timm.loss import SoftTargetCrossEntropy
        from torch.utils.data import DataLoader

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.amp = amp
        self.n_blocks = n_blocks
        self.criterion = SoftTargetCrossEntropy()
        self.num_workers = max(0, num_workers)
        self.val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )
        print(f"Dataloader workers: {self.num_workers}")

    @staticmethod
    def _to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch = batch.to(device, non_blocking=True)
        if device.type == "cuda":
            batch = batch.contiguous(memory_format=torch.channels_last)
        return batch

    @torch.inference_mode()
    def evaluate_configuration(self, configuration: CustomForwardConfig) -> dict:
        block_indices = configuration["block_indices"]
        euler_step = configuration.get("euler_step", 1.0)
        method = configuration.get("method")

        total_correct = 0.0
        total_samples = 0.0
        total_loss = 0.0
        n_steps = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        desc = (
            f"blocks={shared_block_count(block_indices)} "
            f"h={euler_step} m={method or 'RK1'}"
        )
        pbar = tqdm(self.val_loader, desc=desc, leave=False)
        for step, (batch, y_labels) in enumerate(pbar):
            if self.max_batches is not None and step >= self.max_batches:
                break
            batch = self._to_device(batch, self.device)
            y_labels = y_labels.to(self.device, non_blocking=True)
            soft_labels = torch.nn.functional.one_hot(
                y_labels, num_classes=NUM_CLASSES
            ).float()

            with torch.autocast(
                "cuda",
                enabled=self.amp and self.device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                pred = self.model.custom_forward(batch, configuration)
                loss = self.criterion(pred, soft_labels)

            bs = pred.shape[0]
            total_samples += bs
            total_correct += (pred.argmax(1) == y_labels).sum().item()
            total_loss += loss.item() * bs
            n_steps += 1
            pbar.set_postfix(
                acc=f"{total_correct / total_samples:.3f}",
                loss=f"{total_loss / total_samples:.3f}",
            )
        pbar.close()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        it_s = n_steps / elapsed if elapsed > 0 else 0.0

        top1acc = total_correct / max(total_samples, 1)
        loss_mean = total_loss / max(total_samples, 1)
        print(
            f"  time={elapsed:.2f}s  {it_s:.2f} it/s  "
            f"top1={top1acc:.4f}  loss={loss_mean:.4f}  n={int(total_samples)}"
        )

        return {
            "configuration": configuration,
            "depth": len([i for i in block_indices if i < self.n_blocks]),
            "euler_step": euler_step,
            "method": method,
            "top1acc": top1acc,
            "loss": loss_mean,
            "n_samples": int(total_samples),
            "time_s": elapsed,
            "it_s": it_s,
        }


RESULT_COLUMNS = [
    "name",
    "depth",
    "euler_step",
    "method",
    "top1acc",
    "loss",
    "time_s",
    "it_s",
    "n_samples",
    "configuration",
]


def row_to_record(name: str, row: dict) -> dict:
    return {
        "name": name,
        "depth": row["depth"],
        "euler_step": row["euler_step"],
        "method": row.get("method") or "RK1",
        "top1acc": row["top1acc"],
        "loss": row["loss"],
        "time_s": row["time_s"],
        "it_s": row["it_s"],
        "n_samples": row["n_samples"],
        "configuration": json.dumps(row["configuration"]),
    }


def load_completed_names(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    df = pd.read_csv(csv_path, usecols=["name"])
    return set(df["name"].astype(str))


def append_result(csv_path: Path, record: dict) -> None:
    pd.DataFrame([record])[RESULT_COLUMNS].to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists() or csv_path.stat().st_size == 0,
        index=False,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    if args.list_only:
        n_blocks = args.n_blocks
        tail = [n_blocks, n_blocks + 1]
        configurations = build_configurations(n_blocks, tail)
        for name, configuration in configurations:
            if configuration is None:
                print(name)
                continue
            method = configuration.get("method") or "RK1"
            depth = len([i for i in configuration["block_indices"] if i < n_blocks])
            print(
                f"{name:22s}  depth={depth:2d}  h={configuration.get('euler_step', 1.0)}  "
                f"m={method}  blocks={shared_block_count(configuration['block_indices'])}"
            )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"experiment={EXPERIMENT_NAME}")
    print(f"output_dir={output_dir.resolve()}")
    print(f"checkpoint={CHECKPOINT.resolve()}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"num_workers={NUM_WORKERS}")
    print(f"max_batches={MAX_BATCHES}")
    print(f"amp={AMP}")
    print(f"resume={RESUME}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = load_shared_convnext(CHECKPOINT)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    model.eval()

    n_blocks = model.stage3_length
    tail = [n_blocks, n_blocks + 1]
    print(f"stage3_length={n_blocks}, tail indices={tail}")

    configurations = build_configurations(n_blocks, tail)

    completed = load_completed_names(csv_path) if RESUME else set()
    if completed:
        print(f"Resuming: {len(completed)} configurations already in {csv_path}")

    runner = AblationRunner(
        model,
        device=device,
        batch_size=BATCH_SIZE,
        max_batches=MAX_BATCHES,
        amp=AMP,
        n_blocks=n_blocks,
        num_workers=NUM_WORKERS,
    )

    for name, configuration in configurations:
        if configuration is None:
            print(name)
            continue
        if name in completed:
            print(f"{name}  (skipped, already in {csv_path.name})")
            continue
        print(name)
        row = runner.evaluate_configuration(configuration)
        append_result(csv_path, row_to_record(name, row))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        print(f"\nSaved {len(df)} rows to {csv_path}")
    else:
        print("No results written.")


if __name__ == "__main__":
    main()
