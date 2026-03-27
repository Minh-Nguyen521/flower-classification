import sys

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from dataset import CLASSES, MODELS_DIR, get_transform

MODEL_PATH = MODELS_DIR / "classifier.pkl"


def build_backbone():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def predict(image_path: str):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run train.py first to generate models/classifier.pkl.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone().to(device)
    scaler, clf = joblib.load(MODEL_PATH)
    transform = get_transform()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone(tensor).cpu().numpy()

    features_scaled = scaler.transform(features)
    pred_idx = clf.predict(features_scaled)[0]
    probs = clf.predict_proba(features_scaled)[0]

    print(f"Image      : {image_path}")
    print(f"Prediction : {CLASSES[pred_idx]}")
    print(f"\nClass probabilities:")
    for cls, prob in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob:.1%}  {bar}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_single.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
