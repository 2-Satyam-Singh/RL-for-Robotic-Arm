# RL for Robotic Arm in Gazebo

A modular Reinforcement Learning framework for training robotic arms using **Gazebo** and **PyTorch**. This project features custom implementations of PPO and DQN, advanced sparse/dense reward shaping, and support for multiple robotic environments (3-DOF, 5-DOF, and Franka Panda).

## 🌟 Key Features
* **Multiple Environments:** Seamlessly switch between 3-DOF, 5-DOF, and Franka Panda robotic arms.
* **Custom RL Algorithms:** Includes optimized, from-scratch PyTorch implementations of PPO (Continuous) and DQN (Discrete) tailored for robotic control.
* **Advanced Logging:** Automated, collision-safe CSV logging and professional matplotlib visualizations (moving averages, shaded standard deviation stability regions, and hyperparameter metadata).
* **Reward Shaping:** Easily toggle between dense (distance-based) and sparse (task-completion) reward functions.

---

## 🛠️ Installation & Setup

We recommend using **Conda** to manage dependencies and ensure compatibility with Gazebo Python bindings.

### 1. Create the Environment
```bash
conda create -n RL-for-Robotic-Arm python=3.12 -c conda-forge -y
conda activate RL-for-Robotic-Arm
```

### 2. Install Dependencies
```bash
conda install -c conda-forge -c pytorch -c nvidia --file requirements.txt -y
```

### 3. Verify Gazebo Bindings
```bash
python -c "import gz.transport as gz; print('✅ Gazebo Python bindings work')"
```

---

## 🚀 Usage

The project is controlled entirely through the `main.py` entry point.

### Step 1: Launch the Simulation
```bash
gz sim sim/serial/robot_3dof.sdf  # or robot_5dof.sdf or panda.sdf
```

### Step 2: Train the Agent
```bash
# Train PPO on the Panda arm with sparse rewards
python main.py --mode train --robot panda --algorithm ppo --reward_type sparse --seed 42

# Train DQN on the 3-DOF arm with dense rewards
python main.py --mode train --robot 3dof --algorithm dqn --reward_type dense --seed 999
```

### Step 3: Test a Saved Model
```bash
# Evaluate a saved PPO checkpoint
python main.py --mode test --robot panda --algorithm ppo --reward_type sparse --model_name panda_ppo_sparse_s42_01-01-2025_12-00_ep5000
```

### Command-Line Arguments
| Argument | Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `train`, `test` | `train` | Run training loop or evaluate a saved model. |
| `--robot` | `panda`, `3dof`, `5dof` | `panda` | The robotic arm environment to load. |
| `--algorithm` | `ppo`, `dqn` | `ppo` | The RL algorithm to use. |
| `--reward_type` | `sparse`, `dense` | `dense` | The reward function logic. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--episodes` | `int` | `10000` | Total number of training/testing episodes. |
| `--max_steps` | `int` | `100` | Maximum steps per episode. |
| `--model_name` | `str` | `""` | Saved model name (required for `test` mode). |

---

## 📊 Logging & Results

All outputs are written to the `results/` directory using unique, timestamped filenames to prevent any overwriting.

### File Naming Convention

Every file follows the pattern: `{prefix}_{robot}_{algo}_{reward}_s{seed}_{timestamp}`

| File | Mode | Description |
| :--- | :--- | :--- |
| `log_train_3dof_ppo_dense_s42_....csv` | Train | Per-step log: joint angles, entity positions, actions, rewards |
| `train_3dof_ppo_dense_s42_....png` | Train | Reward curve with rolling average and stability shading |
| `log_test_3dof_ppo_dense_s42_....csv` | Test | Per-episode log: total reward, steps taken, success flag |
| `test_3dof_ppo_dense_s42_....png` | Test | Cumulative success curve across evaluation episodes |

### Training Plot
Generated every N episodes (configurable via `plot_every`). Shows:
- Rolling average reward (50-episode window)
- Shaded ±1 std-dev band for stability tracking
- Hyperparameter metadata box (LR, Gamma, Entropy, Seed)

### Test Plot
Generated once after evaluation completes. Shows:
- Cumulative successes vs episode number (steps up by 1 on each success)
- Final success rate displayed in the plot title
- Seed and DOF metadata box

---

## ⚠️ Known Issues

* **`3t` (Cartesian gantry robot):** Not working correctly yet. Training/testing on this robot is unreliable — under active debugging.

---

## 📁 Project Structure

```text
RL-for-Robotic-Arm/
├── algorithms/           # Custom RL agent implementations (ppo.py, dqn.py, base.py)
├── environment/          # Gazebo environment wrapper (environment.py)
├── results/              # Auto-generated CSV logs and PNG plots
│   ├── log_train_*.csv   # Per-step training data
│   ├── log_test_*.csv    # Per-episode evaluation data
│   ├── train_*.png       # Training reward plots
│   └── test_*.png        # Evaluation success plots
├── scripts/              # Core execution logic (train.py, test.py)
├── sim/                  # Gazebo SDF model files
│   ├── serial/           # Serial kinematic arms (Panda, 3DOF, 5DOF)
│   ├── branched/         # Branched kinematic robots (coming soon)
│   └── parallel/         # Future
├── utils/                # Helper tools (logger.py, math utilities)
├── config.py             # Robot hardware configurations & limits
└── main.py               # CLI entry point
```

---

## 📜 License

Copyright (C) 2026 Satyam Singh

This project is licensed under the **GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`). See the [LICENSE](LICENSE) file for the full license text.

Under the AGPL's network-use clause (§13), if you run a modified version of this software as a network service, you must make the corresponding source available to its users.