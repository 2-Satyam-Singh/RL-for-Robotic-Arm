"""
Generate the teacher--student transfer-curve figure for the paper.

Plots RF student success vs. PPO teacher success on the 7-DoF Panda (the
within-morphology transfer curve), anchored at the trivial (0,0) point, against
the student=teacher diagonal. Lower-DoF arms are overlaid as the high-competence
regime, where the student falls below the teacher (saturation / crossover).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Panda within-morphology transfer points (measured) ---
# teacher %, [RF run successes]
panda = [
    (0,  [0]),                 # anchor: no successful rollouts -> student 0% by construction
    (4,  [33, 14, 34]),        # weak teacher (s0, 1k ep)
    (23, [71, 43, 52]),        # stronger teacher (s42, 10k ep)
]
pt_teacher = np.array([p[0] for p in panda], dtype=float)
pt_mean = np.array([np.mean(p[1]) for p in panda])
pt_std = np.array([np.std(p[1]) for p in panda])

# --- Lower-DoF arms (other morphologies): high-teacher regime ---
# from the closed-loop success table: teacher %, RF %
low_dof = {
    "3-DoF": (100, 79),
    "4-DoF": (79, 36),
    "5-DoF": (90, 47),
    "6-DoF": (91, 70),
}

fig, ax = plt.subplots(figsize=(7.0, 5.6))

# student = teacher diagonal
ax.plot([0, 100], [0, 100], ls="--", color="gray", lw=1.4, zorder=1,
        label="student = teacher")

# shade regimes
ax.fill_between([0, 100], [0, 100], 100, color="#2ca02c", alpha=0.05, zorder=0)
ax.fill_between([0, 100], 0, [0, 100], color="#d62728", alpha=0.05, zorder=0)
ax.text(11, 78, "amplification\n(student > teacher)", color="#2ca02c",
        fontsize=10, ha="center", va="center", style="italic")
ax.text(62, 12, "saturation\n(student < teacher)", color="#b22222",
        fontsize=10, ha="center", va="center", style="italic")

# illustrative concave guide through the measured Panda means
xs = np.linspace(0, 23, 200)
guide = 55.3 * (xs / 23.0) ** 0.41
ax.plot(xs, guide, color="#1f77b4", lw=1.6, alpha=0.55, zorder=2,
        label="concave fit (guide)")

# individual Panda RF runs (faint)
for tch, runs in panda:
    for r in runs:
        ax.scatter(tch, r, s=22, color="#1f77b4", alpha=0.30, zorder=3)

# Panda means under SAMPLED teacher eval, with std error bars
ax.errorbar(pt_teacher, pt_mean, yerr=pt_std, fmt="o", color="#1f77b4",
            ms=9, capsize=4, lw=1.6, zorder=4,
            label="Panda RF (mean $\\pm$ std)")

# Same RF results, but teacher re-scored GREEDILY (mean action): the teacher's
# x collapses toward 0 while the RF's y is unchanged -> points slide left, deeper
# into the amplification region. Greedy: weak 4->0%, strong 23->1%.
greedy_teacher = np.array([0.0, 1.0])
greedy_rf = np.array([27.0, 55.3])
sampled_x = np.array([4.0, 23.0])
for sx, gx, y in zip(sampled_x, greedy_teacher, greedy_rf):
    ax.annotate("", xy=(gx, y), xytext=(sx, y),
                arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.3,
                                ls="--"), zorder=3)
ax.scatter(greedy_teacher, greedy_rf, s=90, marker="o", facecolor="white",
           edgecolor="#1f77b4", lw=1.8, zorder=5,
           label="teacher re-scored greedily (deployment)")
ax.annotate("greedy eval collapses the\nteacher's success ($\\to$0–1%);\n"
            "the RF student is unchanged", xy=(2, 41), xytext=(22, 30),
            fontsize=8.5, color="#555555", ha="left",
            arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.0))

# lower-DoF arms
for name, (tch, rf) in low_dof.items():
    ax.scatter(tch, rf, s=80, marker="s", color="#ff7f0e",
               edgecolor="black", lw=0.6, zorder=4)
    ax.annotate(name, xy=(tch, rf), xytext=(tch - 3, rf - 7), fontsize=9)
# single legend handle for the squares
ax.scatter([], [], s=80, marker="s", color="#ff7f0e", edgecolor="black",
           lw=0.6, label="lower-DoF arms (other morphologies)")

ax.set_xlabel("PPO teacher success rate (%)", fontsize=12)
ax.set_ylabel("RF student success rate (%)", fontsize=12)
ax.set_title("Teacher–Student Transfer Curve (7-DoF Panda)",
             fontsize=13, fontweight="bold")
ax.set_xlim(-2, 102)
ax.set_ylim(-2, 102)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, ls="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

plt.tight_layout()
out = "Research-Paper/images/transfer_curve.png"
fig.savefig(out, dpi=300)
print(f"Saved {out}")
