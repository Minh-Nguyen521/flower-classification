"""Train LogisticRegression"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import CLASSES, FEATURES_DIR, MODELS_DIR, RANDOM_SEED, VAL_SPLIT

TRAIN_FEAT = FEATURES_DIR / "train_features.npy"
TRAIN_LABELS = FEATURES_DIR / "train_labels.npy"
MODEL_PATH = MODELS_DIR / "classifier.pkl"
CM_PATH = MODELS_DIR / "confusion_matrix.png"


def train_and_evaluate(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_val_s)
    print("\nValidation results (80/20 split):")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

    return scaler, clf, y_val, y_pred


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (validation set)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def main():
    if not TRAIN_FEAT.exists():
        raise FileNotFoundError(
            "Run extract_features.py first to build the feature cache."
        )

    X = np.load(TRAIN_FEAT)
    y = np.load(TRAIN_LABELS)
    print(f"Loaded features: {X.shape}, labels: {y.shape}")

    scaler_val, clf_val, y_val, y_pred = train_and_evaluate(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(y_val, y_pred, CM_PATH)

    # Re-fit on full dataset for the saved model
    print("\nRe-fitting on full dataset...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    clf_final = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    clf_final.fit(X_scaled, y)

    joblib.dump((scaler_final, clf_final), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
