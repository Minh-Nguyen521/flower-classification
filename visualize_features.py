"""Visualize cached .npy feature embeddings.

Plots:
  1. t-SNE of train embeddings colored by class
  2. PCA of train embeddings colored by class
  3. Per-class feature mean heatmap
  4. Feature norm distribution per class
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from dataset import CLASSES, FEATURES_DIR

TRAIN_FEAT   = FEATURES_DIR / "train_features.npy"
TRAIN_LABELS = FEATURES_DIR / "train_labels.npy"

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


def load():
    if not TRAIN_FEAT.exists():
        raise FileNotFoundError("Run extract_features.py first.")
    X = np.load(TRAIN_FEAT)
    y = np.load(TRAIN_LABELS)
    print(f"Loaded: X={X.shape}, y={y.shape}")
    return X, y


def plot_tsne(X, y, ax):
    print("Computing t-SNE (may take ~30s)...")
    X_scaled = StandardScaler().fit_transform(X)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(X_scaled)
    for i, cls in enumerate(CLASSES):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=COLORS[i], label=cls, alpha=0.6, s=15)
    ax.set_title("t-SNE of train embeddings")
    ax.legend(markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_pca(X, y, ax):
    X_scaled = StandardScaler().fit_transform(X)
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    for i, cls in enumerate(CLASSES):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=COLORS[i], label=cls, alpha=0.6, s=15)
    ax.set_title("PCA of train embeddings (PC1 vs PC2)")
    ax.legend(markerscale=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_class_means(X, y, ax):
    means = np.stack([X[y == i].mean(axis=0) for i in range(len(CLASSES))])
    # Show first 100 dims for readability
    im = ax.imshow(means[:, :100], aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Feature dimension (first 100 of 1280)")
    ax.set_title("Per-class mean embedding")
    plt.colorbar(im, ax=ax)


def plot_norm_distribution(X, y, ax):
    for i, cls in enumerate(CLASSES):
        norms = np.linalg.norm(X[y == i], axis=1)
        ax.hist(norms, bins=40, alpha=0.6, color=COLORS[i], label=cls)
    ax.set_xlabel("L2 norm of embedding")
    ax.set_ylabel("Count")
    ax.set_title("Feature norm distribution per class")
    ax.legend()


def main():
    X, y = load()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("EfficientNet_B0 Feature Embeddings — Flower Dataset", fontsize=14)

    plot_pca(X, y, axes[0, 0])
    plot_tsne(X, y, axes[0, 1])
    plot_class_means(X, y, axes[1, 0])
    plot_norm_distribution(X, y, axes[1, 1])

    plt.tight_layout()
    out = FEATURES_DIR / "feature_visualization.png"
    plt.savefig(out, dpi=130)
    print(f"\nSaved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
