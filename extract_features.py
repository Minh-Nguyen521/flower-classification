"""Extract feature using EfficientNet_B0 and cache to disk"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

from dataset import (
    BATCH_SIZE,
    FEATURES_DIR,
    FlowerTestDataset,
    FlowerTrainDataset,
    get_transform,
)

TRAIN_FEAT  = FEATURES_DIR / "train_features.npy"
TRAIN_LABELS = FEATURES_DIR / "train_labels.npy"
TEST_FEAT   = FEATURES_DIR / "test_features.npy"
TEST_FNAMES = FEATURES_DIR / "test_filenames.npy"


def build_backbone():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()  # exposes 1280-dim avgpool output
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def extract(loader, model, device, desc):
    feats, labels, names = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if len(batch) == 3:
                imgs, lbl, paths = batch
                labels.append(lbl.numpy())
                names.extend(paths)
            else:
                imgs, fnames = batch
                names.extend(fnames)
            feats.append(model(imgs.to(device)).cpu().numpy())
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(labels, axis=0) if labels else None,
        np.array(names),
    )


def main():
    if TRAIN_FEAT.exists():
        print("Cache found — skipping extraction. Delete features/ to re-run.")
        feats = np.load(TRAIN_FEAT)
        lbls  = np.load(TRAIN_LABELS)
        tfeats = np.load(TEST_FEAT)
        fnames = np.load(TEST_FNAMES)
        print(f"  train_features : {feats.shape}")
        print(f"  train_labels   : {lbls.shape}")
        print(f"  test_features  : {tfeats.shape}")
        print(f"  test_filenames : {fnames.shape}")
        return

    FEATURES_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_backbone().to(device)
    print(f"Backbone output shape: 1280-dim (EfficientNet_B0, classifier=Identity)")

    transform = get_transform()
    train_loader = DataLoader(
        FlowerTrainDataset(transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        FlowerTestDataset(transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    train_feats, train_labels, _ = extract(train_loader, model, device, "Train")
    test_feats, _, test_fnames   = extract(test_loader,  model, device, "Test")

    np.save(TRAIN_FEAT,   train_feats)
    np.save(TRAIN_LABELS, train_labels)
    np.save(TEST_FEAT,    test_feats)
    np.save(TEST_FNAMES,  test_fnames)

    print("\nSaved to features/")
    print(f"  train_features : {train_feats.shape}")
    print(f"  train_labels   : {train_labels.shape}")
    print(f"  test_features  : {test_feats.shape}")
    print(f"  test_filenames : {test_fnames.shape}")


if __name__ == "__main__":
    main()
