# algorithms/dqn_0_5.py
"""
Optimized DQN (PyTorch) — discrete actions for joint adjustments.
Compatible with environment_0_5 (normalized obs arrays, array actions).
Saves to models/dqn_models/<name>.pth via BaseAgent.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from algorithms.base import BaseAgent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden=(128, 128)):
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


class NumpyReplay:
    def __init__(self, obs_dim, act_dim, size=100_000):
        self.size = int(size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer = {
            "s": np.zeros((size, obs_dim), dtype=np.float32),
            "s2": np.zeros((size, obs_dim), dtype=np.float32),
            "a": np.zeros(size, dtype=np.int64),
            "r": np.zeros(size, dtype=np.float32),
            "d": np.zeros(size, dtype=np.float32),
        }
        self.ptr = 0
        self.len = 0

    def push(self, s, a, r, s2, done):
        i = self.ptr % self.size
        self.buffer["s"][i] = s
        self.buffer["s2"][i] = s2
        self.buffer["a"][i] = a
        self.buffer["r"][i] = r
        self.buffer["d"][i] = float(done)
        self.ptr += 1
        self.len = min(self.len + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.len, size=batch)
        return (
            self.buffer["s"][idx],
            self.buffer["a"][idx],
            self.buffer["r"][idx],
            self.buffer["s2"][idx],
            self.buffer["d"][idx],
        )

    def __len__(self):
        return self.len


class DQNAgent(BaseAgent):
    def __init__(
        self,
        env,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=100_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20000,
        target_update_freq=500,
        step_frac=0.08,
        min_buffer=500,
        hidden=(128, 128),
        device=None,
    ):
        super().__init__(env, algo_name="dqn")
        self.joint_names = list(env.joints)
        self.joint_limits = env.limits
        self.entity_names = sorted(list(env.entities)) if env.entities else []
        self.n_joints = len(self.joint_names)
        self.n_primitives = 3
        self.n_actions = self.n_joints * self.n_primitives
        self.obs_dim = self.n_joints + 3 * len(self.entity_names)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_frac = step_frac
        self.min_buffer = min_buffer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute deltas
        self.deltas = np.array([(hi - lo) * self.step_frac for (lo, hi) in self.joint_limits], dtype=np.float32)

        # Networks
        self.q = MLP(self.obs_dim, self.n_actions, hidden).to(self.device)
        self.target = MLP(self.obs_dim, self.n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = NumpyReplay(self.obs_dim, self.n_actions, buffer_size)

        # Epsilon
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(1, epsilon_decay_steps)
        self.epsilon_step = (epsilon_start - epsilon_end) / self.epsilon_decay

        self.train_steps = 0

    def _action_idx_to_array(self, idx, current_joints):
        """Convert action index to array of joint positions."""
        ji = idx // self.n_primitives
        prim = idx % self.n_primitives
        curr = np.array([current_joints.get(jn, 0.0) for jn in self.joint_names], dtype=np.float32)
        delta = self.deltas[ji]
        if prim == 0:
            newv = curr[ji] - delta
        elif prim == 1:
            newv = curr[ji]
        else:
            newv = curr[ji] + delta
        lo, hi = self.joint_limits[ji]
        newv = max(lo, min(hi, newv))
        curr[ji] = newv
        return curr

    def select_action(self, obs):
        """Select action (array of joint positions)."""
        if random.random() < self.epsilon:
            idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                qv = self.q(torch.tensor(obs, device=self.device).unsqueeze(0))
                idx = int(torch.argmax(qv).item())
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_step)
        self._last_action_idx = idx
        # Convert to array using denormalized current joints (from raw_obs if needed, but assume obs is normalized; for action, need raw)
        # Note: To get raw joints, we'd need denormalization, but since train loop has info["raw_obs"], but here we don't.
        # Approximation: Denormalize obs joints part
        joint_vals = obs[:self.n_joints]
        current_joints = {jn: low + (val + 1.0) * (high - low) / 2.0 if high > low else low
                          for jn, val, (low, high) in zip(self.joint_names, joint_vals, self.joint_limits)}
        return self._action_idx_to_array(idx, current_joints)

    def store_transition(self, obs, action, reward, next_obs, done):
        s = obs  # Normalized array
        s2 = next_obs
        # Convert action array back to idx for storage (discrete)
        diffs = np.abs(action - [current_joints.get(jn, 0.0) for jn in self.joint_names])  # Approximate from action
        ji = np.argmax(diffs)
        delta = self.deltas[ji]
        if action[ji] <= action[ji] - 0.5 * delta:  # Use original curr from select, but approx
            prim = 0
        elif action[ji] >= action[ji] + 0.5 * delta:
            prim = 2
        else:
            prim = 1
        idx = ji * self.n_primitives + prim
        self.buffer.push(s, idx, float(reward), s2, float(done))

    def learn(self):
        if len(self.buffer) < max(self.min_buffer, self.batch_size):
            return
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, device=self.device)
        s2 = torch.tensor(s2, device=self.device)
        d = torch.tensor(d, device=self.device)

        q_vals = self.q(s)  # (B, A)
        q_taken = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN
        next_online = self.q(s2).argmax(1)
        next_target = self.target(s2).gather(1, next_online.unsqueeze(1)).squeeze(1)
        target = r + (1.0 - d) * self.gamma * next_target

        loss = self.loss_fn(q_taken, target.detach())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.q.state_dict())

    def save(self, name="dqn"):
        p = os.path.join(self.model_dir, f"{name}.pth")
        torch.save(self.q.state_dict(), p)

    def load(self, name="dqn"):
        p = os.path.join(self.model_dir, f"{name}.pth")
        if os.path.exists(p):
            self.q.load_state_dict(torch.load(p, map_location=self.device))
            self.target.load_state_dict(self.q.state_dict())