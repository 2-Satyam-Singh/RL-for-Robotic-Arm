# Project Context: RL Policy Distillation for Robotic Arms

## What this project is

A research project (IEEE conference paper) studying whether a PPO reinforcement-learning teacher trained for robotic manipulation can be **distilled into classical ML models** (LR, SVM, RF, XGBoost) that are smaller, faster, and deployable without a deep-learning runtime. The framework is morphology-agnostic — it covers custom 3-, 4-, 5-, 6-DoF arms and a 7-DoF Franka Panda, all in Gazebo Ignition simulation.

**Paper title:** "Distilling Reinforcement-Learning Manipulation Policies into Lightweight Supervised Models Across Robot Morphologies"
**Authors:** Satyam Singh, Jitendra Adhikari, Emon Barua (MANIT Bhopal)
**Source:** `Research-Paper/paper.tex` (IEEEtran, compiles with pdflatex)

---

## Core Technical Pipeline

### Teacher
- PPO with Gaussian actor (`algorithms/ppo.py`). Stochastic at test time by default — `act()` calls `dist.rsample()`, std ≈ 1.0 (learned `log_std` across all checkpoints).
- A `--deterministic` flag (added this session) switches to mean-action (greedy) eval: `action = mu` instead of `dist.rsample()`.
- Trained once per morphology, sparse reward, Gazebo sim via gz-transport.
- `main.py --mode test` runs eval; saves a CSV log to `results/log_test_<run>.csv`.

### Distillation
- Only **successful episodes** from the teacher's test log are used as training data.
- Target = **delta-action**: `Δaᵢ = aᵢ − aᵢ₋₁` (the change in normalized joint command), integrated online at inference as `clip(prev_action + Δ, −1, 1)`.
- Feature vector per step: `9×DOF + 5` features — `[ee_error_x/y/z, ee_dist, progress]` + per-joint `[sin, cos, vel, accel, jerk, vel×accel, prev_action, prev_action_2, prev_vel]`.
- `vel/accel/jerk` are time-derivatives using `dt = step_time.diff()` floored at `1e-5`. At inference a fixed `DT_INFERENCE = 1e-5` is used (deterministic, reproducible).
- `scripts/distill.py` trains all 4 students from a log CSV and saves to `models/{LR,RF,SVM,XGBoost}/<dof>.pkl` + `models/SCALER/<dof>.pkl`.
- `algorithms/ml_policy.py` (`MLPolicyAgent`) wraps a distilled model as a drop-in eval agent.

### Inference / Evaluation
```bash
# PPO teacher (stochastic, default)
python main.py --mode test --algorithm ppo --robot panda \
  --model_name panda_ppo_sparse_s42_17-03-2026_23-15 \
  --reward_type sparse --episodes 100 --max_steps 100 --seed 42

# PPO teacher (greedy / deployment mode)
python main.py --mode test --algorithm ppo --robot panda \
  --model_name panda_ppo_sparse_s42_17-03-2026_23-15 \
  --reward_type sparse --episodes 100 --max_steps 100 --seed 42 --deterministic

# Distilled ML student
python main.py --mode test --algorithm ml --ml_model RF --robot panda \
  --model_name 7dof.pkl --reward_type sparse --episodes 100 --max_steps 100 --seed 42
```
DOF↔robot: 3dof→3dof.pkl, 4dof→4dof.pkl, 5dof→5dof.pkl, 6dof→6dof.pkl, panda→7dof.pkl.

### Success criterion (strict)
`is_success = total_reward > 0 AND steps < max_steps`  
Hitting `max_steps` exactly means the agent timed out — treated as failure. Sparse reward = +1 on goal contact.

---

## Key Findings (the paper's contributions)

### 1. Accuracy–efficiency trade-off
RF is the most reliable student across all 5 morphologies (47–79% success) but the heaviest (30–127 MB, 110–251 ms/step). LR is orders of magnitude smaller (1.4–4.7 kB, ~0.2 ms/step) but inconsistent. XGBoost 2.1–3.9 MB, 4.5–8.4 ms. PPO actor ≈ 150 kB, ~0.75 ms.

### 2. R² does NOT predict closed-loop success
LR has the best offline R² on 3–6 DoF (~0.50–0.53) yet 0% closed-loop success on 3-DoF. RF wins control despite middling R². One-step imitation fidelity ≠ episode-level control performance.

### 3. Noise-borrowed teacher success (the headline finding)
The PPO teacher's success on the 7-DoF Panda is largely an **exploration-noise artifact**:
- Stochastic eval (rsample, std≈1.0): weak checkpoint 4%, stronger checkpoint 23%
- Greedy eval (mean action, as at deployment): both collapse to **0–1%**
- The teacher's `log_std` never converges — std stays ≈1.0 across all 6 tested checkpoints. The teacher navigates by noise, not by a competent mean policy.
- The deterministic RF student, distilled from the teacher's few successful rollouts, **retains 27–55%** — it recovers a deployable controller the teacher does not itself possess in greedy form.
- On lower-DoF arms the teacher is genuinely competent (79–100% stochastic) — same std≈1.0 noise but good mean policy — so the one-shot student trails the teacher (saturation regime).

### Teacher–student transfer curve
Three Panda competence points measured:
| Teacher checkpoint | Sampled % | Greedy % | RF mean (3 seeds) |
|---|---|---|---|
| none (anchor) | 0% | 0% | 0% |
| weak (s0, ep1000) | 4% | 0% | 27% (runs: 33, 14, 34) |
| stronger (s42, ep10000) | 23% | 1% | 55% (runs: 71, 43, 52) |

The curve is concave/saturating: RF amplifies steeply in the low-teacher regime (noise-borrowed), then flattens as the teacher becomes genuinely competent (lower-DoF arms, upper-right saturation region).

---

## Repository Structure

```
main.py                   # entry point (--mode train|test, --algorithm ppo|dqn|ml)
algorithms/
  ppo.py                  # PPO agent; act() has deterministic flag
  dqn.py
  ml_policy.py            # MLPolicyAgent — drop-in wrapper for distilled sklearn/XGB models
environment/
  environment.py          # PandaEnv (Gazebo Ignition via gz-transport)
scripts/
  test.py                 # evaluation loop (called by main.py)
  distill.py              # train all 4 distilled students from a log CSV
  compare_models.py       # per-robot CDF + success/latency bars
  compare_all.py          # cross-robot heatmap + grouped bars
  make_pipeline_figure.py # distillation pipeline diagram
  make_transfer_curve.py  # teacher–student transfer curve figure
utils/
  logger.py               # CSV + plot logging
config/                   # per-robot YAML configs
models/                   # distilled model .pkl files (gitignored)
results/                  # test log CSVs (gitignored)
  archive_2026-06-28_weak-teacher-experiment/  # labeled CSVs + README for paper experiments
Research-Paper/
  paper.tex               # IEEE paper source (now versioned)
  images/                 # all figures including transfer_curve.png
  IEEE_Conference_Template/IEEEtran.cls
```

---

## Git Branches

| Branch | Contents |
|---|---|
| `main` | Stable baseline (v0.8, multi-robot framework) |
| `feat/deterministic-eval-distillation` | All session work: `--deterministic` flag, `distill.py`, `ml_policy.py`, new scripts, paper updates |
| `release/paper-final` | Same as feat + `Research-Paper/` versioned (paper.tex + figures) — do NOT merge to main yet |

**`results/`, `models/` are gitignored** — disk-only. `Research-Paper/` is now versioned on `release/paper-final`.

---

## Conda Environment

```bash
conda activate RL-for-Robotic-Arm   # Python 3.12, PyTorch, sklearn, xgboost, joblib, pandas
```
Always set `PYTHONPATH=/home/wizsaty/GitHub/Model-Distillation-RL-for-Robotic-Arm` when running scripts outside the repo root (needed for `from config import ...`).

---

## Paper Status

**Complete, compiles clean.** `pdflatex paper.tex` twice → 8-page PDF, no errors, no undefined references.

Sections updated (June 2026 session):
- Abstract: qualitative, no raw numbers (IEEE convention), mentions greedy-collapse finding
- Introduction: fully rewritten around 3 findings
- Experimental Setup: notes teacher is stochastic by default; greedy eval reported separately for 7-DoF
- Section "The teacher–student transfer curve" (`sec:rf_variance`): transfer curve figure + `tab:rf_variance` (5-col, footnotesize to fit IEEE column)
- Discussion: noise-mechanism paragraph (std≈1.0 at test time, greedy collapse, recovery)
- Limitations: single morphology evidence for recovery effect, evaluation consistency caveat
- Conclusion: headline recovery finding

Known reviewer weak spots: single-seed experiments, the recovery effect is supported by one hard morphology + three competence points, dt=1e-5 feature approximation is a known leakage/approximation.

---

## Gazebo / Simulation Notes

- Simulation: Gazebo Ignition (gz sim), communicate via gz-transport
- Start sim before any `--mode test` run: `gz sim sim/serial/robot_panda.sdf` (or appropriate SDF)
- If Steps=1 with no data in output: Gazebo session is stale — restart it
- Greedy teacher runs and ML student runs both require Gazebo running
