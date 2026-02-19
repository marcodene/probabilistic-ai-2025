"""
visualize.py — Plots for the SWAG satellite image classification project.

Generates five figures saved to plots/:
  1. sample_images.png        — example images for each land-use class
  2. class_distribution.png   — training vs validation label distribution
  3. confidence_histogram.png — MAP vs SWAG prediction confidence distribution
  4. reliability_diagram.png  — MAP vs SWAG calibration curves side by side
  5. most_least_confident.png — validation images SWAG is most/least sure about

SWAG predictions are cached to plots/swag_probs.joblib after the first run
(~5 min on CPU). Subsequent runs load the cache and are fast.
"""

import pathlib
import numpy as np
import torch
import torch.utils.data
import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from solution import CNN, SWAInferenceHandler
from util import calculate_calibration_curve

DATA_DIR = pathlib.Path(__file__).parent / "data"
PLOTS_DIR = pathlib.Path(__file__).parent / "plots"
SWAG_CACHE = PLOTS_DIR / "swag_probs.joblib"

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
    train_snow = torch.from_numpy(train_meta["train_is_snow"])
    train_cloud = torch.from_numpy(train_meta["train_is_cloud"])

    val_xs = torch.from_numpy(np.load(DATA_DIR / "val_xs.npz")["val_xs"])
    val_meta = np.load(DATA_DIR / "val_ys.npz")
    val_ys = torch.from_numpy(val_meta["val_ys"])
    val_snow = torch.from_numpy(val_meta["val_is_snow"])
    val_cloud = torch.from_numpy(val_meta["val_is_cloud"])
    return train_xs, train_ys, train_snow, train_cloud, val_xs, val_ys, val_snow, val_cloud


def map_predict(xs):
    """Forward pass with pretrained MAP weights; returns softmax probs (N x 6)."""
    network = CNN(in_channels=3, out_classes=6)
    network.load_state_dict(torch.load(DATA_DIR / "map_weights.pt", weights_only=True))
    network.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xs), batch_size=32, shuffle=False, num_workers=0
    )
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(torch.softmax(network(batch), dim=-1))
    return torch.cat(preds)


def swag_predict(train_xs, train_ys, train_snow, train_cloud, val_xs):
    """Run full SWAG inference; loads from cache if available."""
    if SWAG_CACHE.exists():
        print("Loading cached SWAG predictions …")
        return torch.from_numpy(joblib.load(SWAG_CACHE))

    print("Running SWAG training + BMA inference (~5 min on CPU) …")
    train_dataset = torch.utils.data.TensorDataset(
        train_xs, train_snow, train_cloud, train_ys
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0
    )
    handler = SWAInferenceHandler(train_xs=train_xs, model_dir=DATA_DIR)
    handler.train_model(train_loader)
    probs = handler.predict_probs(val_xs)

    joblib.dump(probs.numpy(), SWAG_CACHE)
    print("SWAG predictions cached to", SWAG_CACHE)
    return probs


# ── Reliability diagram helper (draws into an existing Axes) ────────────────
def _draw_reliability(ax, calib_data, title):
    bins = calib_data["bins"]
    positions = bins[:-1] + (bins[1] - bins[0]) / 2
    width = bins[1] - bins[0]
    acc = calib_data["calib_accuracy"]
    conf = calib_data["calib_confidence"]

    ax.bar(
        positions,
        np.abs(acc - conf),
        bottom=np.minimum(acc, conf),
        width=width,
        color=(240 / 255, 60 / 255, 60 / 255, 0.3),
        edgecolor=(240 / 255, 60 / 255, 60 / 255, 0.3),
        linewidth=1,
        label="Gap",
    )
    ax.bar(positions, 0, bottom=acc, width=width,
           edgecolor="black", color="black", alpha=1.0, linewidth=3, label="Accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)


# ── Figure 1: Sample images per class ──────────────────────────────────────
def plot_sample_images(train_xs, train_ys):
    n_per_class = 4
    fig, axes = plt.subplots(6, n_per_class, figsize=(n_per_class * 2.2, 6 * 2.2))
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
    ax.set_title(f"Class distribution  (val also has {ambiguous_count} ambiguous samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)
    print("Saved class_distribution.png")


# ── Figure 3: Confidence histogram (MAP vs SWAG) ────────────────────────────
def plot_confidence_histogram(map_probs, swag_probs, val_ys):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(0, 1, 21)
    non_ambiguous = val_ys >= 0

    for ax, probs, title in zip(
        axes,
        [map_probs, swag_probs],
        ["MAP", "Full SWAG"],
    ):
        max_conf, pred_cls = probs.max(dim=-1)
        correct = pred_cls == val_ys
        ax.hist(max_conf[non_ambiguous & correct].numpy(),
                bins=bins, alpha=0.7, label="Correct", color="steelblue")
        ax.hist(max_conf[non_ambiguous & ~correct].numpy(),
                bins=bins, alpha=0.7, label="Incorrect", color="tomato")
        ax.hist(max_conf[~non_ambiguous].numpy(),
                bins=bins, alpha=0.7, label="Ambiguous (true −1)", color="goldenrod")
        ax.axvline(2 / 3, color="black", linestyle="--", label="Threshold (2/3)")
        ax.set_xlabel("Max predicted probability (confidence)")
        ax.set_title(f"{title} confidence distribution")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Sample count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confidence_histogram.png", dpi=120)
    plt.close(fig)
    print("Saved confidence_histogram.png")


# ── Figure 4: Reliability diagram (MAP vs SWAG) ─────────────────────────────
def plot_reliability_diagram(map_probs, swag_probs, val_ys):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, probs, label in zip(
        axes,
        [map_probs, swag_probs],
        ["MAP", "Full SWAG"],
    ):
        calib = calculate_calibration_curve(probs.numpy(), val_ys.numpy(), num_bins=20)
        _draw_reliability(ax, calib, f"{label}  (ECE = {calib['ece']:.3f})")
    fig.suptitle("Reliability Diagrams — MAP vs Full SWAG", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reliability_diagram.png", dpi=120)
    plt.close(fig)
    calib_map = calculate_calibration_curve(map_probs.numpy(), val_ys.numpy(), num_bins=20)
    calib_swag = calculate_calibration_curve(swag_probs.numpy(), val_ys.numpy(), num_bins=20)
    print(f"Saved reliability_diagram.png  (MAP ECE={calib_map['ece']:.3f}, SWAG ECE={calib_swag['ece']:.3f})")


# ── Figure 5: Most / least confident SWAG predictions ──────────────────────
def plot_most_least_confident(swag_probs, val_xs, val_ys):
    handler_dummy = type("H", (), {})()  # just for label_prediction logic
    threshold = 2.0 / 3.0
    max_conf, pred_cls = swag_probs.max(dim=-1)
    predicted_labels = torch.where(
        max_conf >= threshold, pred_cls, torch.full_like(pred_cls, -1)
    )

    sorted_idx = torch.argsort(max_conf)
    least_idx = sorted_idx[:8]
    most_idx = sorted_idx[-8:]

    fig, axes = plt.subplots(4, 8, figsize=(18, 9))
    for col, (group_idxs, group_label) in enumerate(
        [(least_idx, "Least confident"), (most_idx, "Most confident")]
    ):
        col_offset = col * 4  # rows 0-1 = least, rows 2-3 = most
        # rearrange: top 2 rows = one group, bottom 2 = other
        row_offset = col * 2
        for i, sample_idx in enumerate(group_idxs):
            img_ax = axes[row_offset, i]
            bar_ax = axes[row_offset + 1, i]

            img = val_xs[sample_idx].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_ax.imshow(img)
            img_ax.axis("off")
            true_label = val_ys[sample_idx].item()
            pred_label = predicted_labels[sample_idx].item()
            img_ax.set_title(
                f"pred={pred_label}  true={true_label}", fontsize=7
            )

            bar_colors = ["steelblue"] * 6
            if true_label >= 0:
                bar_colors[true_label] = "coral"
            bar_ax.bar(np.arange(6), swag_probs[sample_idx].numpy(),
                       color=bar_colors, tick_label=np.arange(6))
            bar_ax.set_ylim(0, 1)
            bar_ax.tick_params(labelsize=7)

        axes[row_offset, 0].set_ylabel(group_label, fontsize=9)

    fig.suptitle("SWAG: most and least confident validation predictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "most_least_confident.png", dpi=120)
    plt.close(fig)
    print("Saved most_least_confident.png")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading data …")
    train_xs, train_ys, train_snow, train_cloud, val_xs, val_ys, val_snow, val_cloud = load_data()

    print("Running MAP inference …")
    map_probs = map_predict(val_xs)

    swag_probs = swag_predict(train_xs, train_ys, train_snow, train_cloud, val_xs)

    plot_sample_images(train_xs, train_ys)
    plot_class_distribution(train_ys, val_ys)
    plot_confidence_histogram(map_probs, swag_probs, val_ys)
    plot_reliability_diagram(map_probs, swag_probs, val_ys)
    plot_most_least_confident(swag_probs, val_xs, val_ys)

    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
