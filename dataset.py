import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

SPLIT_PATH = Path(__file__).parent / "split.json"


def load_split():
    return json.loads(SPLIT_PATH.read_text())


class PizzaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
