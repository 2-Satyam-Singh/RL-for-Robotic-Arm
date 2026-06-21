# Copyright (C) 2026 Satyam Singh
# SPDX-License-Identifier: AGPL-3.0-or-later

# algorithms/dqn.py
"""
Optimized DQN (PyTorch) — Discrete actions for a continuous environment.
Works smoothly by taking discrete steps in normalized observation space.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base import BaseAgent

# ==========================================
# HYPERPARAMETERS
# ==========================================
DEFAULT_LR = 0.0001
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 64
DEFAULT_BUFFER_SIZE = 100_000
DEFAULT_TARGET_UPDATE = 500
DEFAULT_MIN_BUFFER = 1000
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_END = 0.05
DEFAULT_EPSILON_DECAY = 20_000
DEFAULT_HIDDEN = (128, 128)
STEP_SIZE = 0.5  # How much to move a joint in normalized space (-1 to 1) per step


class MLP(nn.Module):
    """Simple Neural Network for Q-Value approximation."""
    def __init__(self, inp_dim, out_dim, hidden=DEFAULT_HIDDEN):
        super().__init__()
        layers = []
        last = inp_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Clean, array-based replay buffer for fast sampling."""
    def __init__(self, obs_dim, size):
        self.size = int(size)
        self.obs_buf = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(self.size, dtype=np.int64)
        self.rews_buf = np.zeros(self.size, dtype=np.float32)
        self.done_buf = np.zeros(self.size, dtype=np.float32)
        self.ptr, self.len = 0, 0

    def push(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.len, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.acts_buf[idxs],
            self.rews_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs]
        )

    def __len__(self):
        return self.len


class DQNAgent(BaseAgent):
    def __init__(self, env, device=None):
        super().__init__(env, algo_name="dqn")
        self.n_joints = len(env.joints)
        self.entity_names = sorted(list(env.entities)) if env.entities else []
        
        # Single source of truth: obs dim comes from the env's Gym observation_space
        # (this also removes the old, mismatched "+ 7 end-effector" assumption that
        #  crashed store_transition on the first step).
        self.obs_dim = int(np.prod(env.observation_space.shape))
        
        # 3 primitive actions per joint: 0 (Decrease), 1 (Stay), 2 (Increase)
        self.n_primitives = 3
        self.n_actions = self.n_joints * self.n_primitives

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net = MLP(self.obs_dim, self.n_actions, DEFAULT_HIDDEN).to(self.device)
        self.target_net = MLP(self.obs_dim, self.n_actions, DEFAULT_HIDDEN).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.opt = optim.Adam(self.q_net.parameters(), lr=DEFAULT_LR)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(self.obs_dim, DEFAULT_BUFFER_SIZE)

        # Exploration params
        self.epsilon = DEFAULT_EPSILON_START
        self.epsilon_step = (DEFAULT_EPSILON_START - DEFAULT_EPSILON_END) / DEFAULT_EPSILON_DECAY
        
        self.train_steps = 0
        self._last_action_idx = 0 # State tracker to prevent reverse-engineering math

    def select_action(self, obs):
        """Picks a discrete index, but returns a continuous array for the environment."""
        # 1. Epsilon-Greedy choice for DISCRETE action index
        if random.random() < self.epsilon:
            idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                idx = int(torch.argmax(self.q_net(obs_t)).item())

        # Decay epsilon
        if self.epsilon > DEFAULT_EPSILON_END:
            self.epsilon = max(DEFAULT_EPSILON_END, self.epsilon - self.epsilon_step)
            
        # Save the index internally so we don't have to guess it later!
        self._last_action_idx = idx

        # 2. Convert index to a continuous target array
        joint_idx = idx // self.n_primitives
        action_type = idx % self.n_primitives

        # Extract current normalized joints from the observation
        current_joints = obs[:self.n_joints].copy()

        # Apply the step directly in normalized space (-1.0 to 1.0)
        if action_type == 0:
            current_joints[joint_idx] -= STEP_SIZE
        elif action_type == 2:
            current_joints[joint_idx] += STEP_SIZE
            
        # Ensure we don't request a position outside the robot's physical limits
        return np.clip(current_joints, -1.0, 1.0)

    def store_transition(self, obs, action, reward, next_obs, done):
        """Stores the experience using our saved discrete index, ignoring the continuous action array."""
        self.buffer.push(obs, self._last_action_idx, float(reward), next_obs, done)

    def learn(self):
        """Standard Double-DQN learning step."""
        if len(self.buffer) < DEFAULT_MIN_BUFFER:
            return

        s, a, r, s2, d = self.buffer.sample(DEFAULT_BATCH_SIZE)
        
        s_t = torch.tensor(s, device=self.device)
        a_t = torch.tensor(a, device=self.device).unsqueeze(1) # For gather
        r_t = torch.tensor(r, device=self.device)
        s2_t = torch.tensor(s2, device=self.device)
        d_t = torch.tensor(d, device=self.device)

        # Current Q values
        q_taken = self.q_net(s_t).gather(1, a_t).squeeze(1)

        # Double DQN target calculation
        with torch.no_grad():
            next_online_actions = self.q_net(s2_t).argmax(1, keepdim=True)
            next_target_q = self.target_net(s2_t).gather(1, next_online_actions).squeeze(1)
            target = r_t + (1.0 - d_t) * DEFAULT_GAMMA * next_target_q

        loss = self.loss_fn(q_taken, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.train_steps += 1
        
        # Sync target network
        if self.train_steps % DEFAULT_TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, name="dqn"):
        p = os.path.join(self.model_dir, f"{name}.pth")
        torch.save(self.q_net.state_dict(), p)

    def load(self, name="dqn"):
        p = os.path.join(self.model_dir, f"{name}.pth")
        if os.path.exists(p):
            self.q_net.load_state_dict(torch.load(p, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            print(f"Warning: Model file {p} not found.")