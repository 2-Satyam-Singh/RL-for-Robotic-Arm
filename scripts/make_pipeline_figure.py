# scripts/make_pipeline_figure.py
"""
Render the distillation pipeline diagram used as a figure in the paper.
Output: Research-Paper/images/distillation_pipeline.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "Research-Paper/images/distillation_pipeline.png"

# stage: (title, body, facecolor, edgecolor)
STAGES = [
    ("PPO Teacher", "frozen policy\nper morphology\n(3–7 DoF)", "#e8e8e8", "#000000"),
    ("Successful\nRollouts", "log state &\naction; keep\nsolved episodes", "#d6e4f0", "#1f77b4"),
    ("Feature\nEngineering", "$9n{+}5$ kinematic\nfeatures + scaler\ntarget: $\\Delta a_t$", "#fce3cf", "#ff7f0e"),
    ("Student Models", "LR · SVM\nRF · XGBoost", "#d8efd8", "#2ca02c"),
    ("Closed-loop\nControl", "$a_t{=}\\mathrm{clip}(a_{t-1}{+}\\Delta a_t)$", "#f7d6d6", "#d62728"),
]

fig, ax = plt.subplots(figsize=(12, 3.2))
ax.set_xlim(0, len(STAGES) * 2.5)
ax.set_ylim(0, 3)
ax.axis("off")

bw, bh, y = 2.0, 1.7, 0.8
centers = []
for i, (title, body, fc, ec) in enumerate(STAGES):
    x = i * 2.5 + 0.15
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=fc, edgecolor=ec, linewidth=2))
    cx = x + bw / 2
    centers.append((cx, y))
    ax.text(cx, y + bh - 0.42, title, ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(cx, y + 0.55, body, ha="center", va="center", fontsize=8.7)

# forward arrows between stages
for i in range(len(STAGES) - 1):
    x0 = centers[i][0] + bw / 2
    x1 = centers[i + 1][0] - bw / 2
    ax.add_patch(FancyArrowPatch((x0, y + bh / 2), (x1, y + bh / 2),
                                 arrowstyle="-|>", mutation_scale=20,
                                 linewidth=2, color="#444444"))

# feedback loop: deployment -> Gazebo -> back to control (closed loop)
xd = centers[-1][0]
ax.text(xd, y - 0.45, "Ignition Gazebo (push task)", ha="center", va="center",
        fontsize=8.5, style="italic", color="#555555")
ax.add_patch(FancyArrowPatch((xd + 0.5, y), (xd + 0.5, y - 0.30),
                             arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
                             color="#888888", connectionstyle="arc3,rad=0"))
ax.add_patch(FancyArrowPatch((xd - 0.5, y - 0.30), (xd - 0.5, y),
                             arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
                             color="#888888"))

# title banner
ax.text((len(STAGES) * 2.5) / 2, 2.85, "RL→Classical-ML Policy Distillation",
        ha="center", va="center", fontsize=13, fontweight="bold")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT}")
