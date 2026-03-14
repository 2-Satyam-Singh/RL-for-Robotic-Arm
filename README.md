# RL-for-Robotic-Arm

This project contains code for training reinforcement learning agents on a robotic arm using Gazebo and PyTorch.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RL-for-Robotic-Arm

## Environment Setup

We use **conda** to manage dependencies. GPU support for PyTorch can be added later.

### Steps to create the environment

```bash
# 1. Create and activate the environment
conda create -n RL-for-Robotic-Arm python=3.12 -c conda-forge -y
conda activate RL-for-Robotic-Arm

# 2. Install Gazebo Python bindings
conda install -c conda-forge gz-transport-python gz-msgs-python protobuf -y

# 3. Test the imports
python -c "import gz.transport as gz; from gz.msgs.double_pb2 import Double; print('Gazebo Python bindings work')"

# 4. Optional: Install PyTorch GPU support later
# conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y

# 5. Optional: Install RL packages
# pip install stable-baselines3 gymnasium[all] matplotlib

# To solve the import could not be resolved squiggly lines, when the code is running properly, do Ctrl+Shift+P and select the correct python interpreter in same environment.