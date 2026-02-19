"""Visualise TD3 training and policy behaviour on the continuous CartPole task."""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch

from utils import get_env, run_episode
from solution import Agent

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
CACHE_PATH = os.path.join(PLOTS_DIR, "td3_run_cache.joblib")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_TRAIN = "#2563EB"   # blue  – training returns
C_TEST  = "#16A34A"   # green – test returns
C_FILL  = "#BFDBFE"   # light blue – std band
C_NEG   = "#DC2626"   # red   – negative reward region
C_ACT   = "#7C3AED"   # purple – action heatmap

N_SEEDS          = 3
WARMUP_EPISODES  = 10
TRAIN_EPISODES   = 50
TEST_EPISODES    = 50   # reduced from 300 for fast visualisation


# ── Training run ──────────────────────────────────────────────────────────────
def run_training(seeds):
    """Train one agent per seed and collect per-episode returns + policy data."""
    histories = {}

    for seed in seeds:
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        agent = Agent(env)

        train_returns = []
        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode="warmup", verbose=False, rec=False)
        for _ in range(TRAIN_EPISODES):
            r = run_episode(env, agent, mode="train", verbose=False, rec=False)
            train_returns.append(r)

        test_returns = []
        for _ in range(TEST_EPISODES):
            with torch.no_grad():
                r = run_episode(env, agent, mode="test", verbose=False, rec=False)
            test_returns.append(r)

        # Policy landscape: action and Q-value as fn of (theta, theta_dot)
        theta_vals     = np.linspace(-0.20, 0.20, 40)
        theta_dot_vals = np.linspace(-0.50, 0.50, 40)
        actions  = np.zeros((len(theta_dot_vals), len(theta_vals)))
        q_values = np.zeros((len(theta_dot_vals), len(theta_vals)))

        agent.pi.eval(); agent.q1.eval()
        with torch.no_grad():
            for i, td in enumerate(theta_dot_vals):
                for j, th in enumerate(theta_vals):
                    obs    = np.array([0.0, 0.0, th, td], dtype=np.float32)
                    action = agent.get_action(obs, train=False)
                    actions[i, j] = action[0]
                    obs_t  = torch.tensor(obs, device=agent.device).unsqueeze(0)
                    act_t  = torch.tensor(action, device=agent.device).unsqueeze(0)
                    q_values[i, j] = agent.q1(obs_t, act_t).item()

        histories[seed] = {
            "train": train_returns,
            "test":  test_returns,
            "theta_vals":     theta_vals,
            "theta_dot_vals": theta_dot_vals,
            "actions":        actions,
            "q_values":       q_values,
        }
        print(f"  seed {seed}: avg test return = {np.mean(test_returns):.1f}")

    return histories


def get_or_run(seeds):
    if os.path.exists(CACHE_PATH):
        return joblib.load(CACHE_PATH)
    print("Running training (cached after this run)...")
    result = run_training(seeds)
    joblib.dump(result, CACHE_PATH)
    return result


# ── Figure 1: Environment overview ───────────────────────────────────────────
def plot_environment(save_path):
    """Draw a schematic of CartPole and annotate the state/action space."""
    fig, (ax_schematic, ax_state) = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Schematic diagram ----
    ax_schematic.set_xlim(-3, 3)
    ax_schematic.set_ylim(-0.5, 2.5)
    ax_schematic.set_aspect("equal")
    ax_schematic.axis("off")
    ax_schematic.set_title("Continuous CartPole schematic", fontsize=11)

    # Track
    ax_schematic.plot([-2.4, 2.4], [0, 0], color="black", lw=3)
    ax_schematic.plot([-2.4, -2.4], [-0.1, 0.1], color="black", lw=2)
    ax_schematic.plot([ 2.4,  2.4], [-0.1, 0.1], color="black", lw=2)
    ax_schematic.text(0, -0.35, "track  (±2.4 m)", ha="center", fontsize=9, color="gray")

    # Cart
    cart = plt.Rectangle((-0.3, 0.05), 0.6, 0.3, color="#2563EB", zorder=3)
    ax_schematic.add_patch(cart)
    ax_schematic.text(0, 0.2, "cart", ha="center", va="center",
                      color="white", fontsize=9, fontweight="bold", zorder=4)

    # Pole (tilted ~15 deg)
    pole_angle = np.radians(15)
    px = 0.0 + 1.0 * np.sin(pole_angle)
    py = 0.35 + 1.0 * np.cos(pole_angle)
    ax_schematic.annotate("", xy=(px, py), xytext=(0, 0.35),
                          arrowprops=dict(arrowstyle="-", lw=4, color="#DC2626"))
    ax_schematic.text(px + 0.1, py + 0.05, "pole", color="#DC2626", fontsize=9)

    # Theta annotation
    theta_arc = np.linspace(np.pi / 2, np.pi / 2 - pole_angle, 30)
    ax_schematic.plot(0.4 * np.cos(theta_arc), 0.35 + 0.4 * np.sin(theta_arc),
                      color="orange", lw=1.5)
    ax_schematic.text(0.45, 0.85, "θ", color="orange", fontsize=11)

    # Force arrow
    ax_schematic.annotate("", xy=(0.5, 0.2), xytext=(1.3, 0.2),
                          arrowprops=dict(arrowstyle="<-", lw=2, color="#7C3AED"))
    ax_schematic.text(1.35, 0.22, "F ∈ [−1,1]", color="#7C3AED", fontsize=9)

    # ---- State-space table ----
    rows = [
        ("x",           "cart position",          "±2.4 m"),
        ("ẋ",           "cart velocity",           "unbounded"),
        ("θ",           "pole angle",             "±0.2 rad"),
        ("θ̇",           "pole angular velocity",  "unbounded"),
    ]
    col_labels = ["Variable", "Meaning", "Range"]

    ax_state.axis("off")
    tbl = ax_state.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    ax_state.set_title(
        "State space (dim 4)  ·  Action: continuous force\n"
        "Reward: 0 if pole upright, −1 each step it falls",
        fontsize=10,
    )

    fig.suptitle("Continuous CartPole environment", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Figure 2: Training curves ─────────────────────────────────────────────────
def plot_training_curves(histories, save_path):
    """Mean ± std of training returns across seeds."""
    all_train = np.array([h["train"] for h in histories.values()])  # (seeds, episodes)
    mean_tr = all_train.mean(axis=0)
    std_tr  = all_train.std(axis=0)
    episodes = np.arange(1, len(mean_tr) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(episodes, mean_tr - std_tr, mean_tr + std_tr,
                    alpha=0.25, color=C_FILL, label="±1 std across seeds")
    ax.plot(episodes, mean_tr, color=C_TRAIN, lw=2, label="mean train return")

    # Rolling average
    window = 10
    rolling = np.convolve(mean_tr, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window, len(mean_tr) + 1), rolling,
            color=C_TRAIN, lw=2, ls="--", label=f"{window}-ep rolling avg")

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.set_title(f"TD3 learning curve ({N_SEEDS} seeds, {TRAIN_EPISODES} train episodes each)")
    ax.legend(fontsize=9)
    ax.set_xlim(1, len(mean_tr))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Figure 3: Test return distribution ───────────────────────────────────────
def plot_test_distribution(histories, save_path):
    """Box-plot of test returns per seed + aggregate."""
    seeds = sorted(histories.keys())
    data  = [histories[s]["test"] for s in seeds]
    all_returns = np.concatenate(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Per-seed box plot
    bp = ax1.boxplot(data, tick_labels=[f"seed {s}" for s in seeds],
                     patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_FILL)
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.set_ylabel("Episode return")
    ax1.set_title("Test return distribution per seed")

    # Aggregate histogram
    ax2.hist(all_returns, bins=20, color=C_TEST, edgecolor="white", alpha=0.85)
    ax2.axvline(all_returns.mean(), color=C_TRAIN, lw=2, ls="--",
                label=f"mean = {all_returns.mean():.1f}")
    ax2.axvline(0, color="gray", lw=0.8, ls=":")
    ax2.set_xlabel("Episode return")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Aggregate test returns ({N_SEEDS} seeds × {TEST_EPISODES} episodes)")
    ax2.legend(fontsize=9)

    fig.suptitle("Policy evaluation on continuous CartPole", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Figure 4: Policy & Q-value landscape ─────────────────────────────────────
def plot_policy_landscape(histories, save_path):
    """
    Show the learned policy action and Q-value as heatmaps over
    (theta, theta_dot) — the two dominant CartPole state dimensions.
    Uses seed 0 results.
    """
    h = histories[min(histories.keys())]
    theta     = h["theta_vals"]
    theta_dot = h["theta_dot_vals"]
    actions   = h["actions"]
    q_values  = h["q_values"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Action heatmap
    im1 = ax1.imshow(
        actions,
        origin="lower", aspect="auto",
        extent=[theta[0], theta[-1], theta_dot[0], theta_dot[-1]],
        cmap="RdBu", vmin=-1, vmax=1,
    )
    fig.colorbar(im1, ax=ax1, label="Action (force)")
    ax1.set_xlabel("Pole angle θ (rad)")
    ax1.set_ylabel("Pole angular velocity θ̇ (rad/s)")
    ax1.set_title("Policy: mean action π(s)")
    ax1.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)

    # Q-value heatmap
    im2 = ax2.imshow(
        q_values,
        origin="lower", aspect="auto",
        extent=[theta[0], theta[-1], theta_dot[0], theta_dot[-1]],
        cmap="viridis",
    )
    fig.colorbar(im2, ax=ax2, label="Q-value")
    ax2.set_xlabel("Pole angle θ (rad)")
    ax2.set_ylabel("Pole angular velocity θ̇ (rad/s)")
    ax2.set_title("Critic: Q₁(s, π(s))")
    ax2.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)

    fig.suptitle("Learned policy and value landscape (cart at rest, x = 0)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seeds = list(range(N_SEEDS))

    print("Step 1: environment overview...")
    plot_environment(os.path.join(PLOTS_DIR, "01_environment.png"))

    print("Step 2: running training (using cache if available)...")
    histories = get_or_run(seeds)

    print("Step 3: generating plots...")
    plot_training_curves(histories,    os.path.join(PLOTS_DIR, "02_training_curves.png"))
    plot_test_distribution(histories,  os.path.join(PLOTS_DIR, "03_test_distribution.png"))
    plot_policy_landscape(histories,   os.path.join(PLOTS_DIR, "04_policy_landscape.png"))

    all_test = np.concatenate([h["test"] for h in histories.values()])
    print(f"\nDone. Mean test return: {all_test.mean():.1f} ± {all_test.std():.1f}")
    print(f"Plots saved to {PLOTS_DIR}/")
