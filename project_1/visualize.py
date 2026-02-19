"""
Visualization script for Project 1: GP Pollution Prediction.

Generates four plots saved to ./plots/:
  0_overview.png       - 2x2 combined summary figure
  1_training_data.png  - Training samples colored by PM2.5 with residential areas
  2_clusters.png       - K-Means spatial clustering of training data
  3_prediction.png     - GP predicted pollution concentration heatmap
  4_uncertainty.png    - GP posterior standard deviation (uncertainty) map

Run from the project_1/ directory:
    python visualize.py
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from solution import Model, get_city_area_data, identify_city_area_flags

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_POINTS = 200   # resolution of the evaluation grid
OUTPUT_DIR  = 'plots'
MODEL_CACHE = 'trained_model.joblib'

CIRCLES = np.array([
    [0.5488135,  0.71518937, 0.17167342],
    [0.79915856, 0.46147936, 0.1567626 ],
    [0.26455561, 0.77423369, 0.10298338],
    [0.6976312,  0.06022547, 0.04015634],
    [0.31542835, 0.36371077, 0.17985623],
    [0.15896958, 0.11037514, 0.07244247],
    [0.82099323, 0.09710128, 0.08136552],
    [0.41426299, 0.0641475,  0.04442035],
    [0.09394051, 0.5759465,  0.08729856],
    [0.84640867, 0.69947928, 0.04568374],
    [0.23789282, 0.934214,   0.04039037],
    [0.82076712, 0.90884372, 0.07434012],
    [0.09961493, 0.94530153, 0.04755969],
    [0.88172021, 0.2724369,  0.04483477],
    [0.9425836,  0.6339977,  0.04979664],
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_circles(ax, edgecolor='white', facecolor='none', alpha=1.0, lw=1.5):
    """Draw residential area circles onto an axis."""
    for cx, cy, r in CIRCLES:
        ax.add_patch(mpatches.Circle(
            (cx, cy), r, linewidth=lw,
            edgecolor=edgecolor, facecolor=facecolor, alpha=alpha,
        ))


def make_grid():
    """Uniform (GRID_POINTS x GRID_POINTS) evaluation grid over [0,1]^2."""
    g = np.linspace(0, 1, GRID_POINTS)
    lon, lat = np.meshgrid(g, g)
    coords = np.stack([lon.flatten(), lat.flatten()], axis=1)
    flags  = identify_city_area_flags(coords)
    return coords, flags


def load_or_train(train_y, train_coords, train_flags):
    if os.path.exists(MODEL_CACHE):
        print(f'Loading cached model from {MODEL_CACHE} ...')
        return joblib.load(MODEL_CACHE)

    print('Training model (this may take a few minutes) ...')
    model = Model()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit_model_on_training_data(train_y, train_coords, train_flags)

    joblib.dump(model, MODEL_CACHE)
    print(f'Model cached to {MODEL_CACHE}')
    return model


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_training_data(train_coords, train_y, out):
    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(
        train_coords[:, 0], train_coords[:, 1],
        c=train_y, cmap='YlOrRd', s=4, alpha=0.75, vmin=0, vmax=65,
    )
    add_circles(ax, edgecolor='crimson', facecolor='crimson', alpha=0.20, lw=1.5)
    add_circles(ax, edgecolor='crimson', facecolor='none',    alpha=1.0,  lw=1.5)

    fig.colorbar(sc, ax=ax, label='PM2.5 Concentration')
    res_patch = mpatches.Patch(facecolor='crimson', edgecolor='crimson', alpha=0.5,
                               label='Residential area (50× penalty)')
    ax.legend(handles=[res_patch], loc='upper left', fontsize=9, framealpha=0.8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Training Data — Pollution Concentration', fontsize=13)

    plt.tight_layout()
    path = os.path.join(out, '1_training_data.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved {path}')


def plot_clusters(train_coords, model, out):
    labels    = model.kmeans.predict(train_coords)
    centroids = model.kmeans.cluster_centers_

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        train_coords[:, 0], train_coords[:, 1],
        c=labels, cmap='tab20', s=3, alpha=0.65,
    )
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        c='black', marker='x', s=55, linewidths=1.8, zorder=5, label='Cluster centroid',
    )
    add_circles(ax, edgecolor='dimgray', facecolor='none', alpha=0.45, lw=1.2)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'K-Means Spatial Clustering ({model.n_clusters} clusters)', fontsize=13)

    plt.tight_layout()
    path = os.path.join(out, '2_clusters.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved {path}')


def plot_prediction(pred_grid, out):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        pred_grid, origin='lower', extent=[0, 1, 0, 1],
        cmap='YlOrRd', vmin=0, vmax=65, aspect='equal',
    )
    add_circles(ax, edgecolor='white', facecolor='none', alpha=1.0, lw=1.8)

    fig.colorbar(im, ax=ax, label='Predicted PM2.5')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('GP Predicted Pollution Concentration', fontsize=13)

    plt.tight_layout()
    path = os.path.join(out, '3_prediction.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved {path}')


def plot_uncertainty(std_grid, out):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        std_grid, origin='lower', extent=[0, 1, 0, 1],
        cmap='Blues', aspect='equal',
    )
    add_circles(ax, edgecolor='crimson', facecolor='none', alpha=1.0, lw=1.8)

    res_patch = mpatches.Patch(facecolor='none', edgecolor='crimson',
                               label='Residential area')
    ax.legend(handles=[res_patch], loc='upper left', fontsize=9, framealpha=0.8)

    fig.colorbar(im, ax=ax, label='GP Posterior Std. Dev.')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('GP Prediction Uncertainty (Posterior Std. Dev.)', fontsize=13)

    plt.tight_layout()
    path = os.path.join(out, '4_uncertainty.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved {path}')


def plot_overview(train_coords, train_y, model, pred_grid, std_grid, out):
    """2×2 summary figure."""
    labels    = model.kmeans.predict(train_coords)
    centroids = model.kmeans.cluster_centers_

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        'Project 1 — Gaussian Process Pollution Prediction',
        fontsize=15, fontweight='bold',
    )

    # Panel 1 — training data
    ax = axes[0, 0]
    sc = ax.scatter(train_coords[:, 0], train_coords[:, 1],
                    c=train_y, cmap='YlOrRd', s=3, alpha=0.7, vmin=0, vmax=65)
    add_circles(ax, edgecolor='crimson', facecolor='crimson', alpha=0.18, lw=1.2)
    add_circles(ax, edgecolor='crimson', facecolor='none',    alpha=1.0,  lw=1.2)
    fig.colorbar(sc, ax=ax, label='PM2.5')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Training Data', fontsize=11)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    # Panel 2 — clusters
    ax = axes[0, 1]
    ax.scatter(train_coords[:, 0], train_coords[:, 1],
               c=labels, cmap='tab20', s=3, alpha=0.65)
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='x', s=45, linewidths=1.5, zorder=5, label='Centroid')
    add_circles(ax, edgecolor='dimgray', facecolor='none', alpha=0.4, lw=1.0)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'K-Means Clustering ({model.n_clusters} clusters)', fontsize=11)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    # Panel 3 — prediction heatmap
    ax = axes[1, 0]
    im = ax.imshow(pred_grid, origin='lower', extent=[0, 1, 0, 1],
                   cmap='YlOrRd', vmin=0, vmax=65, aspect='equal')
    add_circles(ax, edgecolor='white', facecolor='none', alpha=1.0, lw=1.5)
    fig.colorbar(im, ax=ax, label='Predicted PM2.5')
    ax.set_title('GP Prediction', fontsize=11)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    # Panel 4 — uncertainty
    ax = axes[1, 1]
    im = ax.imshow(std_grid, origin='lower', extent=[0, 1, 0, 1],
                   cmap='Blues', aspect='equal')
    add_circles(ax, edgecolor='crimson', facecolor='none', alpha=1.0, lw=1.5)
    fig.colorbar(im, ax=ax, label='Std. Dev.')
    ax.set_title('GP Uncertainty (Std. Dev.)', fontsize=11)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    plt.tight_layout()
    path = os.path.join(out, '0_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading data ...')
    train_x = np.loadtxt('data/train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('data/train_y.csv', delimiter=',', skiprows=1)
    test_x  = np.loadtxt('data/test_x.csv',  delimiter=',', skiprows=1)

    train_coords, train_flags, test_coords, test_flags = get_city_area_data(train_x, test_x)

    model = load_or_train(train_y, train_coords, train_flags)

    print('Building evaluation grid ...')
    grid_coords, grid_flags = make_grid()

    print('Running GP predictions on grid ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        predictions, _, gp_std = model.predict_pollution_concentration(
            grid_coords, grid_flags,
        )

    pred_grid = predictions.reshape(GRID_POINTS, GRID_POINTS)
    std_grid  = gp_std.reshape(GRID_POINTS, GRID_POINTS)

    print('Generating plots ...')
    plot_training_data(train_coords, train_y, OUTPUT_DIR)
    plot_clusters(train_coords, model, OUTPUT_DIR)
    plot_prediction(pred_grid, OUTPUT_DIR)
    plot_uncertainty(std_grid, OUTPUT_DIR)
    plot_overview(train_coords, train_y, model, pred_grid, std_grid, OUTPUT_DIR)

    print(f'\nDone — all plots saved to ./{OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
