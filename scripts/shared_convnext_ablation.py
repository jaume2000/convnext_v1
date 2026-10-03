"""Shared ConvNeXt custom_forward configuration sweep on ImageNet validation.

Rows are named ``D{D}_ES{ES}_T{T}_{method}`` (see ``build_configurations`` for the grid).
Existing rows of ``results.csv`` are skipped (RESUME); older free-form labels are migrated.

Leonardo:
  source .env && sbatch --account="$SLURM_ACCOUNT" jobs/shared_convnext_ablation.sh

Local:
  python scripts/shared_convnext_ablation.py --list-only
  python scripts/shared_convnext_ablation.py --batch-size 128 --num-workers 8 --only-section D
"""

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
# Same checkpoint as scripts/feature_map_explorer.py SHARED_CHECKPOINT.
CHECKPOINT = _REPO_ROOT / "outputs" / "shared_convnextv1_imagenet" / "weights" / "last.pth"
OUTPUT_DIR = _REPO_ROOT / "outputs" / EXPERIMENT_NAME

BATCH_SIZE = 1024
NUM_WORKERS = 16
MAX_BATCHES: int | None = None  # set e.g. 2 for a quick smoke test
AMP = False
RESUME = True
NUM_CLASSES = 1000
TRAIN_HORIZON = 9.0  # shared was trained at D=9, ES=1


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
    # ``tail.*`` aliases the LN+downsample already stored under ``deltifiedStage3.*``.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    stale = [k for k in missing if not k.startswith("tail.")]
    if stale or unexpected:
        raise RuntimeError(f"missing={stale} unexpected={unexpected}")
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


def config_name(depth: int, euler_step: float, method: str | None) -> str:
    """Canonical row name: D = shared applications, T = D·ES integration horizon."""
    return f"D{depth}_ES{euler_step:.4g}_T{depth * euler_step:.4g}_{(method or 'RK1').upper()}"


def build_configurations(
    n_blocks: int,
    tail: list[int],
) -> list[tuple[str, CustomForwardConfig | None]]:
    """Shared stage-3 grid (one block applied D times with step ES; horizon T = D·ES).

    Shared was trained at D=9, ES=1 ⇒ T=9. Sections:
      A. refine at the train horizon: ES = 9/D, RK1/2/4
      B. fixed ES=1 (horizon grows with depth)
      C. native depth D=9, varying ES (coarse horizon sweep, step = ES)
      D. fine horizon sweep: T varies at a small fixed step (Euler ES=0.1, RK4 ES=0.25),
         separating the effect of T from discretization error
    """
    configs: list[tuple[str, CustomForwardConfig | None]] = []
    seen: set[str] = set()

    def add(depth: int, es: float, method: str = "RK1") -> None:
        name = config_name(depth, es, method)
        if name not in seen:
            seen.add(name)
            configs.append((name, cfg(tail, [0] * depth, euler_step=es, method=method)))

    configs.append(("--- A. refine at T=9 (ES=9/D) ---", None))
    depths_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24, 32, 48, 64, 96, 100, 128, 256, 512, 1024]
    for method in ("RK1", "RK2", "RK4"):
        for d in depths_a:
            add(d, TRAIN_HORIZON / d, method)
    add(10000, TRAIN_HORIZON / 10000, "RK1")

    configs.append(("--- B. fixed ES=1 (T=D) ---", None))
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24, 32, 48, 64, 96, 100, 128, 256, 512, 1024, 10000]:
        add(d, 1.0)

    configs.append(("--- C. D=9, varying ES (T=9·ES) ---", None))
    for es in [0.01, 0.0125, 0.025, 0.05, 0.1, 0.125, 0.2, 0.25, 0.4, 0.5, 9 / 11, 2.0, 1.8, 3.6, 4.0, 8.0, 16.0]:
        add(n_blocks, es)

    configs.append(("--- D. fine horizon sweep (T varies, small fixed step) ---", None))
    horizons = [0.5, 1, 2, 3, 4.5, 6, 7, 8, 8.5, 9, 9.5, 10, 11, 12, 13.5, 18, 27, 36]
    for method, es in (("RK1", 0.1), ("RK4", 0.25)):
        for T in horizons:
            add(round(T / es), es, method)
    return configs


def migrate_legacy_names(csv_path: Path) -> None:
    """Rename rows of an older results.csv to ``config_name`` and add the ``T`` column.

    Legacy sweeps used free-form labels (e.g. "D=45, ES=9/11" was really D=9). Names are
    rebuilt from each row's stored configuration; duplicate configs keep the first row.
    The original file is kept as ``results_legacy_labels.csv``.
    """
    if not csv_path.is_file():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "configuration" not in df.columns:
        return
    cfgs = df["configuration"].map(json.loads)
    depth = cfgs.map(lambda c: shared_block_count(c["block_indices"]))
    es = cfgs.map(lambda c: float(c.get("euler_step", 1.0)))
    method = cfgs.map(lambda c: (c.get("method") or "RK1").upper())
    names = [config_name(d, e, m) for d, e, m in zip(depth, es, method)]
    if "T" in df.columns and list(df["name"]) == names:
        return
    backup = csv_path.with_name("results_legacy_labels.csv")
    if not backup.exists():
        df.to_csv(backup, index=False)
    df["name"] = names
    df["depth"] = depth
    df["euler_step"] = es
    df["method"] = method
    df["T"] = depth * es
    before = len(df)
    df = df.drop_duplicates("name", keep="first")
    df[RESULT_COLUMNS].to_csv(csv_path, index=False)
    print(f"Migrated {csv_path.name}: {before} rows -> {len(df)} canonical (backup {backup.name})")


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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=None, help="Default: min(NUM_WORKERS, CPUs)")
    parser.add_argument(
        "--only-section",
        choices=["A", "B", "C", "D"],
        nargs="+",
        default=None,
        help="Run only these grid sections (see build_configurations)",
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

        depth = shared_block_count(block_indices)
        return {
            "configuration": configuration,
            "depth": depth,
            "euler_step": euler_step,
            "T": depth * euler_step,
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
    "T",
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
        "T": row["T"],
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


def filter_sections(
    configurations: list[tuple[str, CustomForwardConfig | None]], sections: list[str] | None
) -> list[tuple[str, CustomForwardConfig | None]]:
    """Keep only rows under the ``--- X. ... ---`` headers whose letter is in ``sections``."""
    if not sections:
        return configurations
    out, current = [], None
    for name, configuration in configurations:
        if configuration is None:
            current = name.strip("- ").split(".", 1)[0]
        if current in sections:
            out.append((name, configuration))
    return out


def main() -> None:
    load_dotenv()
    args = parse_args()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    migrate_legacy_names(csv_path)
    completed = load_completed_names(csv_path) if RESUME else set()

    if args.list_only:
        n_blocks = args.n_blocks
        tail = [n_blocks, n_blocks + 1]
        configurations = filter_sections(build_configurations(n_blocks, tail), args.only_section)
        todo = 0
        for name, configuration in configurations:
            if configuration is None:
                print(name)
                continue
            done = name in completed
            todo += not done
            print(f"  {'done' if done else 'TODO'}  {name}")
        print(f"{todo} configurations to run")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"experiment={EXPERIMENT_NAME}")
    print(f"output_dir={output_dir.resolve()}")
    print(f"checkpoint={CHECKPOINT.resolve()}")
    num_workers = args.num_workers if args.num_workers is not None else min(NUM_WORKERS, available_cpus())
    print(f"batch_size={args.batch_size}")
    print(f"num_workers={num_workers}")
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

    configurations = filter_sections(build_configurations(n_blocks, tail), args.only_section)

    if completed:
        print(f"Resuming: {len(completed)} configurations already in {csv_path}")

    runner = AblationRunner(
        model,
        device=device,
        batch_size=args.batch_size,
        max_batches=MAX_BATCHES,
        amp=AMP,
        n_blocks=n_blocks,
        num_workers=num_workers,
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
