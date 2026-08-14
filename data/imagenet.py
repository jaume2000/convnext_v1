import io
import os

from torch.utils.data import Dataset
from datasets import DownloadConfig, IterableDataset, load_dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from data.transforms.transforms import IMAGENET_MEAN, IMAGENET_STD, build_train_transforms


class ImageNetDataset(Dataset):
    def __init__(self, split, transforms=None, streaming=False):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # Parent of ILSVRC___imagenet-1k (e.g. $WORK/huggingface on Leonardo).
        cache_dir = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME")
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        self.ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            streaming=streaming,
            token=token,
            cache_dir=cache_dir,
            download_config=DownloadConfig(local_files_only=offline),
        )
        self.transforms = transforms
        self.streaming = streaming
        if streaming == False:
            self._len = self.ds.__len__()

    def __len__(self):
        if not self._len:
            raise TypeError("streaming dataloader cannot have length")
        return self._len

    def __getitem__(self, index):
        row = self.ds[index]
        if isinstance(self.ds, IterableDataset):
            row = next(iter(self.ds))
        img = row["image"]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        label = int(row["label"])  # img label index
        if self.transforms:
            img = self.transforms(img)
        return img, label


if __name__ == "__main__":
    transforms = build_train_transforms(224)
    ds = ImageNetDataset("train", transforms, True)
    sample = img, label = ds.__getitem__(0)
    print(label)
    if isinstance(img, Image.Image):
        img.save("./test.jpeg")
    else:  # float CHW tensor (ImageNet-normalized)
        mean = img.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
        std = img.new_tensor(IMAGENET_STD).view(-1, 1, 1)
        to_pil_image((img * std + mean).clamp(0, 1)).save("./test.jpeg")
