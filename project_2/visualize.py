"""
visualize.py — Plots for the SWAG satellite image classification project.

Generates four figures saved to plots/:
  1. sample_images.png       — example images for each land-use class
  2. class_distribution.png  — training vs validation label distribution
  3. confidence_histogram.png — MAP prediction confidence distribution
  4. reliability_diagram.png  — calibration curve (reliability diagram)

Uses MAP inference only (fast: loads pretrained weights, no SWAG training needed).
SWAG improves calibration beyond what is shown here; the reliability diagram
illustrates the baseline MAP behaviour.
"""

import pathlib
import numpy as np
import torch
import torch.utils.data
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from solution import CNN
from util import calculate_calibration_curve, paint_reliability_diagram

DATA_DIR = pathlib.Path(__file__).parent / "data"
PLOTS_DIR = pathlib.Path(__file__).parent / "plots"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Residential",
]


def load_data():
    train_xs = torch.from_numpy(np.load(DATA_DIR / "train_xs.npz")["train_xs"])
    train_meta = np.load(DATA_DIR / "train_ys.npz")
    train_ys = torch.from_numpy(train_meta["train_ys"])

    val_xs = torch.from_numpy(np.load(DATA_DIR / "val_xs.npz")["val_xs"])
    val_meta = np.load(DATA_DIR / "val_ys.npz")
    val_ys = torch.from_numpy(val_meta["val_ys"])
    val_snow = torch.from_numpy(val_meta["val_is_snow"])
    val_cloud = torch.from_numpy(val_meta["val_is_cloud"])
    return train_xs, train_ys, val_xs, val_ys, val_snow, val_cloud


def map_predict(network, xs):
    """Run MAP forward pass and return softmax probabilities (N x 6)."""
    network.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xs),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(torch.softmax(network(batch), dim=-1))
    return torch.cat(preds)


# ── Figure 1: Sample images per class ──────────────────────────────────────
def plot_sample_images(train_xs, train_ys):
    n_per_class = 4
    fig, axes = plt.subplots(
        6, n_per_class, figsize=(n_per_class * 2.2, 6 * 2.2)
    )
    for cls in range(6):
        idxs = (train_ys == cls).nonzero(as_tuple=True)[0][:n_per_class]
        for col, idx in enumerate(idxs):
            img = train_xs[idx].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[cls, col].imshow(img)
            axes[cls, col].axis("off")
            if col == 0:
                axes[cls, col].set_ylabel(
                    CLASS_NAMES[cls], fontsize=9, rotation=0, labelpad=80, va="center"
                )
    fig.suptitle("Sample images per land-use class (training set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sample_images.png", dpi=120)
    plt.close(fig)
    print("Saved sample_images.png")


# ── Figure 2: Class distribution ───────────────────────────────────────────
def plot_class_distribution(train_ys, val_ys):
    non_ambiguous = val_ys[val_ys >= 0]
    ambiguous_count = (val_ys == -1).sum().item()

    train_counts = [(train_ys == c).sum().item() for c in range(6)]
    val_counts = [(non_ambiguous == c).sum().item() for c in range(6)]

    x = np.arange(6)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, train_counts, width, label="Train", color="steelblue")
    ax.bar(x + width / 2, val_counts, width, label="Val (non-ambiguous)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Number of samples")
    ax.set_title(
        f"Class distribution  (val also has {ambiguous_count} ambiguous samples)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)
    print("Saved class_distribution.png")


# ── Figure 3: Confidence histogram ─────────────────────────────────────────
def plot_confidence_histogram(probs, val_ys):
    max_conf, pred_cls = probs.max(dim=-1)
    correct = pred_cls == val_ys
    non_ambiguous = val_ys >= 0

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(
        max_conf[non_ambiguous & correct].numpy(),
        bins=bins,
        alpha=0.7,
        label="Correct",
        color="steelblue",
    )
    ax.hist(
        max_conf[non_ambiguous & ~correct].numpy(),
        bins=bins,
        alpha=0.7,
        label="Incorrect",
        color="tomato",
    )
    ax.hist(
        max_conf[~non_ambiguous].numpy(),
        bins=bins,
        alpha=0.7,
        label="Ambiguous (true label −1)",
        color="goldenrod",
    )
    ax.axvline(2 / 3, color="black", linestyle="--", label="Default threshold (2/3)")
    ax.set_xlabel("Max predicted probability (confidence)")
    ax.set_ylabel("Sample count")
    ax.set_title("MAP prediction confidence distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confidence_histogram.png", dpi=120)
    plt.close(fig)
    print("Saved confidence_histogram.png")


# ── Figure 4: Reliability diagram ──────────────────────────────────────────
def plot_reliability_diagram(probs, val_ys):
    calib_data = calculate_calibration_curve(
        probs.numpy(), val_ys.numpy(), num_bins=20
    )
    ece = calib_data["ece"]
    fig = paint_reliability_diagram(
        calib_data,
        title=f"Reliability Diagram — MAP inference  (ECE = {ece:.3f})",
    )
    fig.savefig(PLOTS_DIR / "reliability_diagram.png", dpi=120)
    plt.close(fig)
    print(f"Saved reliability_diagram.png  (ECE = {ece:.3f})")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading data …")
    train_xs, train_ys, val_xs, val_ys, val_snow, val_cloud = load_data()

    print("Loading pretrained MAP weights …")
    network = CNN(in_channels=3, out_classes=6)
    network.load_state_dict(torch.load(DATA_DIR / "map_weights.pt", weights_only=True))

    print("Running MAP inference on validation set …")
    probs = map_predict(network, val_xs)

    plot_sample_images(train_xs, train_ys)
    plot_class_distribution(train_ys, val_ys)
    plot_confidence_histogram(probs, val_ys)
    plot_reliability_diagram(probs, val_ys)

    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
