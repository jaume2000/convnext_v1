import torch
from torch.utils.data import Dataset

class TestingDataset(Dataset):

    def __init__(self, input_shape, num_samples, num_classes: int):
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, *input_shape)
        # Mixup / one_hot need class indices in [0, num_classes)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.num_samples
