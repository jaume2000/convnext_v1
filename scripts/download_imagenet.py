"""Download ImageNet-1k into the HF datasets cache (run on a login node with internet).

Requires:
  - Accepted terms for https://huggingface.co/datasets/ILSVRC/imagenet-1k
  - HF_TOKEN in the environment / .env
  - HF_DATASETS_CACHE pointing at a large filesystem (e.g. $CINECA_SCRATCH/hf/datasets)
"""

import os

from datasets import load_dataset


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (accept the ImageNet-1k terms on Hugging Face first).")

    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    print(f"Downloading ILSVRC/imagenet-1k into cache_dir={cache_dir!r}")
    for split in ("train", "validation"):
        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            token=token,
            cache_dir=cache_dir,
        )
        print(f"{split}: {len(ds)} samples")
    print("Done.")


if __name__ == "__main__":
    main()
