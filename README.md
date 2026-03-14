# RL-for-Robotic-Arm

This project contains code for training **reinforcement learning agents** on a **robotic arm** using **Gazebo** and **PyTorch**.

---

# Environment Setup

We use **Conda** to manage dependencies. GPU support for PyTorch can be added later if required.

## 1. Create the Environment

```bash
# Load conda
source ~/anaconda3/etc/profile.d/conda.sh

# Create environment
conda create -n RL-for-Robotic-Arm python=3.12 -c conda-forge -y

# Activate environment
conda activate RL-for-Robotic-Arm
```

---

## 2. Install Gazebo Python Bindings

```bash
conda install -c conda-forge gz-transport-python gz-msgs-python protobuf -y
```

---

## 3. Test the Installation

```bash
python -c "import gz.transport as gz; from gz.msgs.double_pb2 import Double; print('Gazebo Python bindings work')"
```

If the message **"Gazebo Python bindings work"** prints, the setup is successful.

---

## 4. Optional: Install PyTorch with GPU Support

You can install PyTorch later if you plan to train reinforcement learning agents.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
```

---

## 5. Optional: Install Reinforcement Learning Packages

```bash
pip install stable-baselines3 gymnasium[all] matplotlib
```

---

## Fix VS Code Import Warnings

Sometimes VS Code may show **"Import could not be resolved"** even when the code runs correctly.

To fix this:

1. Press **Ctrl + Shift + P**
2. Select **Python: Select Interpreter**
3. Choose the interpreter from the `RL-for-Robotic-Arm` environment.

---

# Running the Example

## 1. Activate the Environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RL-for-Robotic-Arm
```

---

## 2. Start the Gazebo Simulation

Run the following in a terminal:

```bash
gz sim RL-for-Robotic-Arm/sim/model.sdf
```

---

## 3. Run the Controller Script

Run the example control script from the project root:

```bash
python -m examples.control_random_joint
```