"""Dataset classes and shared config."""

from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "dataset"
TRAIN_DIR   = DATA_DIR / "train"
TEST_DIR    = DATA_DIR / "test"
TEST_CSV    = DATA_DIR / "Testing_set_flower.csv"
SAMPLE_SUB  = DATA_DIR / "sample_submission.csv"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR   = BASE_DIR / "models"

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES      = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGE_SIZE   = 224
BATCH_SIZE   = 64
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FlowerTrainDataset(Dataset):
    """Walks train/<class>/ dirs in sorted CLASSES order."""

    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []  # (filepath, label_int)
        for label_int, cls in enumerate(CLASSES):
            cls_dir = TRAIN_DIR / cls
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label_int))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)


class FlowerTestDataset(Dataset):
    """Reads filenames from Testing_set_flower.csv (CSV row order)."""

    def __init__(self, transform=None):
        self.transform = transform
        df = pd.read_csv(TEST_CSV)
        self.filenames = df["filename"].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(TEST_DIR / fname).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname
