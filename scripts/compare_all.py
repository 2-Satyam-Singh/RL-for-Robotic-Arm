# scripts/compare_all.py
"""
Master cross-morphology comparison: combine every model and every robot from the
test logs into single figures. Complements scripts/compare_models.py (which
overlays one robot at a time).

Reads results/log_test_<robot>_<LABEL>_<reward>_*.csv and writes:
  results/compare_all_success_bars.png   — grouped bars: success rate, x=DoF, hue=model
  results/compare_all_success_heatmap.png — model x DoF success-rate heatmap

Usage:
  python scripts/compare_all.py
  python scripts/compare_all.py --models PPO LR SVM RF XGBoost --reward_type sparse
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Robots in ascending DoF; label shown on the axis.
ROBOTS = [("3dof", "3-DoF"), ("4dof", "4-DoF"), ("5dof", "5-DoF"),
          ("6dof", "6-DoF"), ("panda", "7-DoF\n(Panda)")]
DEFAULT_MODELS = ["PPO", "LR", "SVM", "RF", "XGBoost"]
COLORS = {"PPO": "#000000", "LR": "#1f77b4", "RF": "#2ca02c",
          "SVM": "#ff7f0e", "XGBoost": "#d62728"}
RESULTS_DIR = "results"


def find_log(robot, label, reward):
    matches = []
    for path in glob.glob(os.path.join(RESULTS_DIR, f"log_test_{robot}_*.csv")):
        parts = os.path.basename(path)[:-4].split("_")
        if len(parts) < 5:
            continue
        if parts[3].lower() == label.lower() and parts[4].lower() == reward.lower():
            matches.append(path)
    return max(matches, key=os.path.getmtime) if matches else None


def success_rate(path):
    df = pd.read_csv(path)
    per_ep = df.groupby("episode")["is_success"].max()
    return per_ep.mean() * 100.0


def main():
    ap = argparse.ArgumentParser(description="Master cross-model/robot comparison.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--reward_type", default="sparse", choices=["sparse", "dense"])
    args = ap.parse_args()

    # success-rate matrix: rows=models present, cols=robots
    SR = np.full((len(args.models), len(ROBOTS)), np.nan)
    for i, m in enumerate(args.models):
        for j, (rb, _) in enumerate(ROBOTS):
            p = find_log(rb, m, args.reward_type)
            if p:
                SR[i, j] = success_rate(p)
    present = [i for i in range(len(args.models)) if not np.all(np.isnan(SR[i]))]
    models = [args.models[i] for i in present]
    SR = SR[present]

    if not models:
        print(f"No logs found in {RESULTS_DIR}/. Run the evaluations first.")
        return

    # ---- console table ----
    print(f"\nSUCCESS RATE (%) — reward={args.reward_type}")
    print(f"{'Model':9s}" + "".join(f"{lab.splitlines()[0]:>9s}" for _, lab in ROBOTS))
    for i, m in enumerate(models):
        print(f"{m:9s}" + "".join(
            (f"{SR[i,j]:8.1f} " if not np.isnan(SR[i, j]) else f"{'--':>9s}")
            for j in range(len(ROBOTS))))

    # ---- figure 1: grouped bars ----
    plt.close("all")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ROBOTS))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        off = (i - (len(models) - 1) / 2) * w
        bars = ax.bar(x + off, np.nan_to_num(SR[i]), width=w,
                      color=COLORS.get(m), label=m, edgecolor="white", linewidth=0.5)
        for j, b in enumerate(bars):
            if not np.isnan(SR[i, j]):
                ax.text(b.get_x() + b.get_width() / 2, SR[i, j] + 1,
                        f"{SR[i,j]:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in ROBOTS])
    ax.set_ylabel("Task Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("Distilled Students vs. PPO Teacher — Success Across Morphologies",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Policy", ncol=len(models), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "compare_all_success_bars.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📊 grouped bars → {p1}")

    # ---- figure 2: heatmap ----
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(models) + 2))
    im = ax.imshow(SR, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ROBOTS)))
    ax.set_xticklabels([lab.replace("\n", " ") for _, lab in ROBOTS])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(ROBOTS)):
            if not np.isnan(SR[i, j]):
                ax.text(j, i, f"{SR[i,j]:.0f}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
    ax.set_title("Success Rate (%) — Policy × Morphology", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Success Rate (%)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "compare_all_success_heatmap.png")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 heatmap      → {p2}")


if __name__ == "__main__":
    main()
