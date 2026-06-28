# scripts/compare_models.py
"""
Overlay / compare evaluation results across the distilled models (and the PPO teacher).

It reads the per-step test logs that scripts/test.py already writes
(results/log_test_<robot>_<LABEL>_<reward>_s<seed>_<timestamp>.csv) and produces
paper-ready figures:

  1. results/compare_<robot>_cdf.png        — overlaid "cumulative success vs step"
                                               (one curve per model, % of episodes)
  2. results/compare_<robot>_success.png     — success-rate + inference-latency bars

plus a printed summary table (success rate, mean reward, mean steps, per-step latency).

For each requested model it auto-selects the most recent matching log, so you just:

    # 1) run each model through the sim (any order)
    python main.py --mode test --algorithm ml --ml_model LR  --robot 3dof --model_name 3dof.pkl --reward_type sparse --episodes 100
    python main.py --mode test --algorithm ml --ml_model RF  --robot 3dof --model_name 3dof.pkl --reward_type sparse --episodes 100
    python main.py --mode test --algorithm ml --ml_model SVM --robot 3dof --model_name 3dof.pkl --reward_type sparse --episodes 100
    python main.py --mode test --algorithm ml --ml_model XGBoost --robot 3dof --model_name 3dof.pkl --reward_type sparse --episodes 100
    # (optional) the PPO teacher, for a teacher-vs-students comparison:
    python main.py --mode test --algorithm ppo --robot 3dof --model_name <ppo_file> --reward_type sparse --episodes 100

    # 2) overlay them
    python scripts/compare_models.py --robot 3dof
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Default models to look for, teacher first. Any without a log are silently skipped.
DEFAULT_MODELS = ["PPO", "LR", "RF", "SVM", "XGBoost"]

# Stable colour per label so figures are consistent across robots/papers.
COLORS = {
    "PPO": "#000000", "LR": "#1f77b4", "RF": "#2ca02c",
    "SVM": "#ff7f0e", "XGBoost": "#d62728", "DQN": "#9467bd",
}
RESULTS_DIR = "results"


def find_log(robot, label, reward):
    """Most-recent test log for (robot, label, reward); label matched case-insensitively."""
    matches = []
    for path in glob.glob(os.path.join(RESULTS_DIR, f"log_test_{robot}_*.csv")):
        parts = os.path.basename(path)[:-4].split("_")  # log_test_<robot>_<label>_<reward>_...
        # robot may itself be a single token (3dof/4dof/panda); label is the next token.
        if len(parts) < 5:
            continue
        f_label, f_reward = parts[3], parts[4]
        if f_label.lower() == label.lower() and f_reward.lower() == reward.lower():
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def summarize(path, max_steps):
    """Reduce a per-step log to per-episode metrics + the success-step CDF."""
    df = pd.read_csv(path)
    # one row per episode: success flag is constant within an episode; steps = last step_count
    per_ep = df.groupby("episode").agg(
        is_success=("is_success", "max"),
        steps=("step_count", "max"),
        reward=("reward", "sum"),
    ).reset_index()

    total = len(per_ep)
    succ = per_ep[per_ep["is_success"] == 1]
    n_succ = len(succ)

    # cumulative % of all episodes that have succeeded by step x
    cum = np.zeros(max_steps + 1)
    for s in succ["steps"]:
        if 1 <= s <= max_steps:
            cum[s] += 1
    cdf = np.cumsum(cum[1:]) / total * 100.0      # length max_steps, in %

    step_time = float(df["step_time"].mean()) if "step_time" in df.columns else float("nan")
    return {
        "total": total,
        "n_succ": n_succ,
        "success_rate": n_succ / total * 100.0 if total else 0.0,
        "mean_reward": float(per_ep["reward"].mean()),
        "mean_steps_succ": float(succ["steps"].mean()) if n_succ else float("nan"),
        "step_time_ms": step_time * 1000.0,
        "cdf": cdf,
    }


def main():
    ap = argparse.ArgumentParser(description="Overlay distilled-model evaluation results.")
    ap.add_argument("--robot", required=True, help="3dof / 4dof / 5dof / 6dof / panda")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Labels to compare (default: PPO LR RF SVM XGBoost)")
    ap.add_argument("--reward_type", default="sparse", choices=["sparse", "dense"])
    ap.add_argument("--max_steps", type=int, default=100, help="X-axis cap for the CDF")
    args = ap.parse_args()

    # --- collect whatever logs exist ---
    results = {}
    for label in args.models:
        path = find_log(args.robot, label, args.reward_type)
        if path is None:
            print(f"⚠️  no test log found for {args.robot}/{label} — skipping")
            continue
        results[label] = summarize(path, args.max_steps)
        print(f"   {label:8s} ← {os.path.basename(path)}")

    if not results:
        print(f"\nNo matching logs in {RESULTS_DIR}/. Run the evaluations first.")
        return

    # --- printed summary table ---
    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON: {args.robot.upper()} ({args.reward_type})")
    print("=" * 78)
    print(f"{'Model':8s} {'Episodes':>9s} {'Success':>9s} {'MeanRew':>10s} "
          f"{'Steps(✓)':>10s} {'Latency':>11s}")
    print("-" * 78)
    for label in args.models:
        if label not in results:
            continue
        r = results[label]
        lat = f"{r['step_time_ms']:.2f} ms" if np.isfinite(r["step_time_ms"]) else "   n/a"
        steps = f"{r['mean_steps_succ']:.1f}" if np.isfinite(r["mean_steps_succ"]) else "n/a"
        print(f"{label:8s} {r['total']:9d} {r['success_rate']:8.1f}% "
              f"{r['mean_reward']:10.1f} {steps:>10s} {lat:>11s}")
    print("=" * 78 + "\n")

    labels = [l for l in args.models if l in results]

    # --- figure 1: overlaid CDF ---
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, args.max_steps + 1)
    for label in labels:
        r = results[label]
        ax.step(x, r["cdf"], where="post", linewidth=2,
                color=COLORS.get(label), label=f"{label} ({r['success_rate']:.0f}%)")
    ax.set_title(f"Distilled Policy Comparison: {args.robot.upper()} | Success vs Steps",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Episodes Succeeded (%)", fontsize=12)
    ax.set_xlim(0, args.max_steps + 1)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", title="Model (final success)")
    plt.tight_layout()
    cdf_path = os.path.join(RESULTS_DIR, f"compare_{args.robot}_cdf.png")
    fig.savefig(cdf_path, dpi=300)
    plt.close(fig)
    print(f"📊 Overlay CDF saved      → {cdf_path}")

    # --- figure 2: success-rate + latency bars ---
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [COLORS.get(l) for l in labels]

    ax1.bar(labels, [results[l]["success_rate"] for l in labels], color=colors, alpha=0.85)
    ax1.set_title("Success Rate", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    lat = [results[l]["step_time_ms"] for l in labels]
    if np.all([np.isfinite(v) for v in lat]):
        ax2.bar(labels, lat, color=colors, alpha=0.85)
        ax2.set_title("Inference Latency per Step", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Mean step time (ms)", fontsize=12)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "step_time not logged\n(re-run tests to capture latency)",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")
        ax2.set_axis_off()

    fig.suptitle(f"Model Comparison: {args.robot.upper()} ({args.reward_type})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, f"compare_{args.robot}_success.png")
    fig.savefig(bar_path, dpi=300)
    plt.close(fig)
    print(f"📊 Success/latency saved  → {bar_path}")


if __name__ == "__main__":
    main()
