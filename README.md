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
Install Gazebo bindings, PyTorch (with GPU support), and other required packages:
```bash
# Gazebo bindings
conda install -c conda-forge gz-transport-python gz-msgs-python protobuf -y

# PyTorch (Update cudatoolkit version based on your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y

# Additional ML libraries
pip install matplotlib numpy
```

### 3. Verify Gazebo Bindings
Run a quick check to ensure Python can talk to Gazebo:
```bash
python -c "import gz.transport as gz; print('✅ Gazebo Python bindings work')"
```

---

## 🚀 Usage

The project is controlled entirely through the `main.py` entry point. 

### Step 1: Launch the Simulation
Before starting the training script, launch the Gazebo simulation for your specific robot. (The training script will print the exact launch command you need when you run it).
```bash
gz sim robot_3dof.sdf  # or robot_5dof or panda
```

### Step 2: Train the Agent
Run `main.py` in a separate terminal. You can customize the training run using command-line arguments.

**Examples:**
```bash
# Train PPO on the Panda arm with sparse rewards
python main.py --mode train --robot panda --algorithm ppo --reward_type sparse --seed 42

# Train DQN on the 3-DOF arm with dense rewards
python main.py --mode train --robot 3dof --algorithm dqn --reward_type dense --seed 999
```

### Command-Line Arguments
| Argument | Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `train`, `test` | `train` | Run training loop or evaluate a saved model. |
| `--robot` | `panda`, `3dof`, `5dof` | `panda` | The robotic arm environment to load. |
| `--algorithm` | `ppo`, `dqn` | `ppo` | The RL algorithm to use. |
| `--reward_type` | `sparse`, `dense` | `dense` | The reward function logic. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--episodes` | `int` | `10000` | Total number of training episodes. |
| `--max_steps` | `int` | `100` | Maximum steps per episode. |

---

## 📊 Logging & Results

The framework automatically logs your runs into the `results/` directory using unique, timestamped filenames to prevent overwriting data.

* **CSV Logs:** Tracks per-step joint angles, entity positions, actions, and rewards.
* **Performance Plots:** Automatically generates PNG plots every N episodes featuring:
  * Rolling average reward curves.
  * Shaded standard deviation for stability tracking.
  * A "Legend Box" detailing specific hyperparameter values (Learning Rate, Gamma, Entropy, etc.).

---

## 📁 Project Structure

```text
RL-for-Robotic-Arm/
├── algorithms/           # Custom RL agent implementations (ppo.py, dqn.py, base.py)
├── environment/          # Gazebo environment wrapper (environment.py)
├── results/              # Auto-generated CSV logs and PNG plots
├── scripts/              # Core execution logic (train.py, test.py)
├── sim/                  # Gazebo SDF model files
├── utils/                # Helper tools (logger.py, math utilities)
├── config.py             # Robot hardware configurations & limits
└── main.py               # CLI entry point
```