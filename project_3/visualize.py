"""Visualize the constrained Bayesian optimization algorithm on the toy problem."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from solution import (
    BO_algo, DOMAIN, SAFETY_THRESHOLD, SAFETY_MARGIN,
    get_initial_safe_point, f, v,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
CACHE_PATH = os.path.join(PLOTS_DIR, "bo_run_cache.joblib")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_OBJ   = "#2563EB"   # blue  – objective f
C_CON   = "#DC2626"   # red   – constraint v
C_SAFE  = "#16A34A"   # green – safe set / safe evaluation
C_UNSAFE = "#F97316"  # orange – unsafe evaluation
C_REC   = "#7C3AED"   # purple – recommendation / solution
C_FILL  = "#BFDBFE"   # light blue – confidence interval


# ── Run the algorithm and collect traces ─────────────────────────────────────
def run_bo(n_steps: int = 20, seed: int = 0):
    np.random.seed(seed)
    agent = BO_algo()

    x_init = get_initial_safe_point()
    agent.add_data_point(x_init, f(x_init), v(x_init))

    history = {
        "x": [float(x_init)],
        "f": [float(f(x_init))],
        "v": [float(v(x_init))],
        "safe": [float(v(x_init)) < SAFETY_THRESHOLD],
        "snapshots": [],   # (step, mu_f, sigma_f, mu_v, sigma_v, S)
    }

    for step in range(n_steps):
        x_rec = agent.next_recommendation()
        x_val = float(np.atleast_1d(x_rec).ravel()[0])

        f_obs = float(f(x_rec)) + np.random.randn() * 0.15
        v_obs = float(v(x_rec)) + np.random.randn() * 0.0001

        agent.add_data_point(x_rec, f_obs, v_obs)
        history["x"].append(x_val)
        history["f"].append(f_obs)
        history["v"].append(v_obs)
        history["safe"].append(v_obs < SAFETY_THRESHOLD)

        # Take snapshots at steps 3, 10, 20
        if (step + 1) in (3, 10, n_steps):
            grid = agent.grid
            mu_f, sf, mu_v, sv = agent._fast_gp_predict(grid)
            t = len(agent.sampled_points) + 1
            beta_f = 2 * np.log(2 * agent.num_samples * (t**2) * (np.pi**2) / (6 * 0.1))
            beta_v = 2 * np.log(2 * agent.num_samples * (t**2) * (np.pi**2) / (6 * agent.delta))
            history["snapshots"].append(
                (step + 1, mu_f.copy(), np.sqrt(beta_f) * sf.copy(),
                 mu_v.copy(), np.sqrt(beta_v) * sv.copy(), agent.S.copy())
            )

    history["solution"] = agent.get_solution()
    history["grid"] = agent.grid
    return history


def get_or_run(n_steps: int = 20):
    if os.path.exists(CACHE_PATH):
        return joblib.load(CACHE_PATH)
    result = run_bo(n_steps)
    joblib.dump(result, CACHE_PATH)
    return result


# ── Plot helpers ──────────────────────────────────────────────────────────────
def true_curves(grid):
    f_true = np.array([float(f(np.array([[x]]))) for x in grid])
    v_true = np.full_like(grid, 2.0)  # dummy v = 2
    return f_true, v_true


# ── Figure 1: True landscape ──────────────────────────────────────────────────
def plot_landscape(grid, f_true, v_true, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(grid, f_true, color=C_OBJ, lw=2)
    ax1.axvline(grid[np.argmax(f_true)], color=C_OBJ, ls="--", lw=1, alpha=0.6,
                label=f"true max at x={grid[np.argmax(f_true)]:.2f}")
    ax1.set_xlabel("x"); ax1.set_ylabel("f(x) — logP")
    ax1.set_title("Objective function f(x)")
    ax1.legend(fontsize=9)

    ax2.plot(grid, v_true, color=C_CON, lw=2, label="v(x) = SA score")
    ax2.axhline(SAFETY_THRESHOLD, color="black", ls="--", lw=1.2, label=f"κ = {SAFETY_THRESHOLD}")
    ax2.axhspan(SAFETY_THRESHOLD, v_true.max() + 0.5, alpha=0.12, color=C_CON)
    ax2.set_xlabel("x"); ax2.set_ylabel("v(x) — SA score")
    ax2.set_title("Constraint function v(x)")
    ax2.legend(fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xlim(*DOMAIN[0])

    fig.suptitle("True problem landscape", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Figure 2: GP posterior snapshots ─────────────────────────────────────────
def plot_gp_snapshots(history, grid, f_true, v_true, save_path):
    snapshots = history["snapshots"]
    n = len(snapshots)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

    for col, (step, mu_f, unc_f, mu_v, unc_v, S) in enumerate(snapshots):
        obs_x = np.array(history["x"][:step + 1])
        obs_f = np.array(history["f"][:step + 1])
        obs_v = np.array(history["v"][:step + 1])

        # --- f posterior ---
        ax = axes[0, col]
        ax.fill_between(grid, mu_f - unc_f, mu_f + unc_f, alpha=0.25, color=C_FILL)
        ax.plot(grid, mu_f, color=C_OBJ, lw=2, label="GP mean")
        ax.plot(grid, f_true, color="gray", lw=1, ls="--", alpha=0.7, label="true f")
        ax.scatter(obs_x, obs_f, color=C_OBJ, s=30, zorder=5)
        ax.set_title(f"f posterior (t={step})", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("f(x)")
        ax.set_xlim(*DOMAIN[0])
        if col == 0:
            ax.legend(fontsize=8)

        # --- v posterior ---
        ax = axes[1, col]
        ax.fill_between(grid, mu_v - unc_v, mu_v + unc_v, alpha=0.25, color="#FCA5A5")
        ax.plot(grid, mu_v, color=C_CON, lw=2, label="GP mean")
        ax.plot(grid, v_true, color="gray", lw=1, ls="--", alpha=0.7, label="true v")
        ax.axhline(SAFETY_THRESHOLD, color="black", ls="--", lw=1.2, alpha=0.8)
        ax.scatter(obs_x, obs_v, color=C_CON, s=30, zorder=5)
        # shade safe set
        safe_x = grid[S]
        if len(safe_x) > 0:
            ax.fill_between(
                grid,
                ax.get_ylim()[0] if ax.get_ylim()[0] > -10 else -0.5,
                SAFETY_THRESHOLD,
                where=np.isin(np.arange(len(grid)), S),
                alpha=0.15, color=C_SAFE, label="safe set S_t",
            )
        ax.set_title(f"v posterior (t={step})", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("v(x)")
        ax.set_xlim(*DOMAIN[0])
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle("GP posteriors at different time steps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Figure 3: Full evaluation trace ──────────────────────────────────────────
def plot_evaluation_trace(history, grid, f_true, save_path):
    xs     = np.array(history["x"])
    fs     = np.array(history["f"])
    safe   = np.array(history["safe"])
    sol_x  = history["solution"]
    sol_f  = float(f(np.array([[sol_x]])))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # Top: trajectory on objective landscape
    ax1.plot(grid, f_true, color="gray", lw=1.5, alpha=0.6, label="true f(x)")
    ax1.scatter(xs[safe],  fs[safe],  color=C_SAFE,   s=50, zorder=5, label="safe eval")
    ax1.scatter(xs[~safe], fs[~safe], color=C_UNSAFE, s=50, marker="x", zorder=5,
                label="unsafe eval")
    ax1.axvline(sol_x, color=C_REC, lw=1.5, ls="--")
    ax1.scatter([sol_x], [sol_f], color=C_REC, s=120, zorder=6, marker="*",
                label=f"solution x̂={sol_x:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Evaluation history on objective landscape")
    ax1.set_xlim(*DOMAIN[0])
    ax1.legend(fontsize=9)

    # Bottom: regret over time
    best_safe_f = np.maximum.accumulate(
        [f_val if s else -np.inf for f_val, s in zip(fs, safe)]
    )
    # replace -inf with running carry-forward
    carry = -np.inf
    for i in range(len(best_safe_f)):
        if best_safe_f[i] > carry:
            carry = best_safe_f[i]
        else:
            best_safe_f[i] = carry

    true_opt = float(f(np.array([[5.0]])))  # dummy f peaks at midpoint
    regret = true_opt - best_safe_f

    ax2.plot(range(len(regret)), regret, color=C_OBJ, lw=2, marker="o", ms=4)
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cumulative best-safe regret")
    ax2.set_title("Regret over time")
    ax2.set_xlim(0, len(regret) - 1)

    fig.suptitle("Constrained BO — evaluation trace", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Figure 4: Safe set vs unsafe evaluations ──────────────────────────────────
def plot_safe_unsafe_summary(history, save_path):
    xs   = np.array(history["x"])
    safe = np.array(history["safe"])
    n_safe   = safe.sum()
    n_unsafe = (~safe).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    ax1.pie([n_safe, n_unsafe], labels=["Safe", "Unsafe"],
            colors=[C_SAFE, C_UNSAFE],
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 11})
    ax1.set_title(f"Evaluation safety ({len(xs)} total)")

    # Step chart of v observations
    vs = np.array(history["v"])
    ax2.step(range(len(vs)), vs, color=C_CON, lw=1.5, where="mid")
    ax2.axhline(SAFETY_THRESHOLD, color="black", ls="--", lw=1.2,
                label=f"κ = {SAFETY_THRESHOLD}")
    ax2.fill_between(range(len(vs)), SAFETY_THRESHOLD, vs.max() + 0.1,
                     where=(vs >= SAFETY_THRESHOLD), step="mid",
                     color=C_UNSAFE, alpha=0.3, label="unsafe")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("v(x) — SA score")
    ax2.set_title("Constraint observations over time")
    ax2.legend(fontsize=9)

    fig.suptitle("Safety summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running BO algorithm (using cache if available)...")
    history = get_or_run(n_steps=20)

    grid    = history["grid"]
    f_true, v_true = true_curves(grid)

    print("Generating plots...")
    plot_landscape(grid, f_true, v_true,
                   os.path.join(PLOTS_DIR, "01_landscape.png"))
    plot_gp_snapshots(history, grid, f_true, v_true,
                      os.path.join(PLOTS_DIR, "02_gp_snapshots.png"))
    plot_evaluation_trace(history, grid, f_true,
                          os.path.join(PLOTS_DIR, "03_trace.png"))
    plot_safe_unsafe_summary(history,
                             os.path.join(PLOTS_DIR, "04_safety_summary.png"))

    sol_x = history["solution"]
    print(f"\nDone. Solution: x̂ = {sol_x:.4f}")
    print(f"Plots saved to {PLOTS_DIR}/")
