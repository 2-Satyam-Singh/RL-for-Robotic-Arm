# ppo.py
"""
PPOAgent (PyTorch) — on-policy, compatible with environment_0_5.py.
Works with normalized observation arrays and outputs action arrays.

API:
    algo = PPOAgent(env)
    action = algo.select_action(obs)         # returns np.array of actions
    algo.store_transition(obs, action, r, next_obs, done)
    algo.learn()                             # updates when buffer is ready
    algo.save("name")
    algo.load("name")
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base import BaseAgent

# =============================================
# HYPERPARAMETERS (moved to top for easy tuning)
# =============================================
# These are the exact defaults you had.
# Keep them exactly as-is — they are already good for your small env
# (obs_dim ≈ 10, act_dim = 7, sparse reward).
#
# Recommended values (do NOT change unless you have a reason):
#   hidden=(128, 128)          ← perfect size for this problem
#   rollout_steps=2048         ← keep (updates every ~10-20 episodes)
#   minibatch=64               ← keep
#   epochs=8                   ← keep
#   lr=3e-4                    ← standard for PPO continuous
#   gamma=0.99, lam=0.95       ← keep
#   clip_eps=0.2               ← keep
#   entropy_coef=0.01          ← keep (sparse reward works well with this)
#   value_loss_coef=0.5        ← keep
#
# If training feels slow later, you can lower rollout_steps to 1024.
# Do NOT change hidden layers or buffer size unless you add more entities.

DEFAULT_LR = 0.0003
DEFAULT_GAMMA = 0.99
DEFAULT_LAM = 0.95
DEFAULT_CLIP_EPS = 0.2
DEFAULT_EPOCHS = 8
DEFAULT_MINIBATCH = 64
DEFAULT_ROLLOUT_STEPS = 2048
DEFAULT_HIDDEN = (128, 128)
DEFAULT_VALUE_LOSS_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01

DECIMAL_PLACES = 1


class ActorCritic(nn.Module):
    """Actor-Critic network with Gaussian policy for continuous actions."""
    def __init__(self, obs_dim, act_dim, hidden=DEFAULT_HIDDEN):
        super().__init__()
        # Shared body for actor
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.actor_mean = nn.Sequential(*layers, nn.Linear(last, act_dim))
        # Learnable log-std (independent of state)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Separate critic head
        v_layers = []
        last = obs_dim
        for h in hidden:
            v_layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.critic = nn.Sequential(*v_layers, nn.Linear(last, 1))

    def act(self, x):
        mu = self.actor_mean(x)
        mu = torch.tanh(mu)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, x, action):
        mu = self.actor_mean(x)
        mu = torch.tanh(mu)                    # ← fixed (must match act())
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return log_prob, entropy, value


class PPOAgent(BaseAgent):
    def __init__(
        self,
        env,
        lr=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        lam=DEFAULT_LAM,
        clip_eps=DEFAULT_CLIP_EPS,
        epochs=DEFAULT_EPOCHS,
        minibatch=DEFAULT_MINIBATCH,
        rollout_steps=DEFAULT_ROLLOUT_STEPS,
        hidden=DEFAULT_HIDDEN,
        value_loss_coef=DEFAULT_VALUE_LOSS_COEF,
        entropy_coef=DEFAULT_ENTROPY_COEF,
        device=None,
    ):
        super().__init__(env, algo_name="ppo")
        self.joint_names = list(env.joints)
        self.joint_limits = env.limits
        self.entity_names = sorted(list(env.entities)) if env.entities else []
        self.workspace_range = getattr(env, "workspace_range", 0.85)
        self.n_joints = len(self.joint_names)

        # Dimensions — single source of truth: the env's Gymnasium spaces
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))

        # Hyperparameters (now using the top defaults)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = int(epochs)
        self.minibatch = int(minibatch)
        self.rollout_steps = int(rollout_steps)
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-Critic network and optimizer
        self.ac = ActorCritic(self.obs_dim, self.act_dim, hidden).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

        # Rollout buffer (NumPy arrays)
        self.buffer = {
            "obs": np.zeros((rollout_steps, self.obs_dim), dtype=np.float32),
            "act": np.zeros((rollout_steps, self.act_dim), dtype=np.float32),
            "logp": np.zeros(rollout_steps, dtype=np.float32),
            "rew": np.zeros(rollout_steps, dtype=np.float32),
            "val": np.zeros(rollout_steps, dtype=np.float32),
            "done": np.zeros(rollout_steps, dtype=np.float32),
        }
        self.buffer_size = 0

        # Track last step for bootstrapping
        self.last_next_obs = None
        self.last_done = False

    @property
    def _rollout_buffer(self):
        """Helper property so train.py can safely check len(algo._rollout_buffer)"""
        return [None] * self.buffer_size

    # ---- API used by Training loop ----
    def select_action(self, obs):
        """Return action array."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, value_t = self.ac.act(obs_t)
        action = action_t.squeeze(0).cpu().numpy()
        # action = np.clip(action, -1.0, 1.0)          # ← This line is causing training to become unstable
        action = np.round(action, decimals=DECIMAL_PLACES)
        log_prob = float(log_prob_t.item())
        value = float(value_t.item())
        self.last = (obs, action, log_prob, value)
        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Append transition to buffer.
        """
        if self.buffer_size >= self.rollout_steps:
            return
        idx = self.buffer_size
        self.buffer["obs"][idx] = obs
        self.buffer["act"][idx] = action
        self.buffer["logp"][idx] = self.last[2]
        self.buffer["rew"][idx] = float(reward)
        self.buffer["val"][idx] = self.last[3]
        self.buffer["done"][idx] = float(done)
        self.buffer_size += 1

        self.last_next_obs = next_obs
        self.last_done = bool(done)

    def learn(self):
        """
        Called every step — updates only when buffer ready or episode done.
        """
        if (self.buffer_size < self.rollout_steps):
            return

        if self.last_next_obs is None:
            last_val = 0.0
        else:
            s_t = torch.tensor(self.last_next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                last_val = float(self.ac.critic(s_t).item())

        adv, ret = self._compute_gae_returns(last_val)

        s_t = torch.tensor(self.buffer["obs"][:self.buffer_size], dtype=torch.float32, device=self.device)
        a_t = torch.tensor(self.buffer["act"][:self.buffer_size], dtype=torch.float32, device=self.device)
        logp_t = torch.tensor(self.buffer["logp"][:self.buffer_size], dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)

        n = self.buffer_size
        inds = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, n, self.minibatch):
                mb = inds[start:start + self.minibatch]
                logp, entropy, v = self.ac.evaluate(s_t[mb], a_t[mb])
                ratio = torch.exp(logp - logp_t[mb])
                obj1 = ratio * adv_t[mb]
                obj2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(obj1, obj2).mean()
                value_loss = ((ret_t[mb] - v) ** 2).mean()
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self._clear_buffer()

    # ---- utilities ----
    def _compute_gae_returns(self, last_val):
        r = self.buffer["rew"][:self.buffer_size]
        v = self.buffer["val"][:self.buffer_size]
        d = self.buffer["done"][:self.buffer_size]
        v_next = np.append(v[1:], last_val)
        T = len(r)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - d[t]
            delta = r[t] + self.gamma * v_next[t] * nonterm - v[t]
            lastgaelam = delta + self.gamma * self.lam * nonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + v
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def _clear_buffer(self):
        self.buffer_size = 0
        self.last = None
        self.last_next_obs = None
        self.last_done = False

    # ---- persistence ----
    def save(self, name="ppo"):
        p = os.path.join(self.model_dir, f"{name}.pth")
        torch.save(self.ac.state_dict(), p)

    def load(self, name="ppo"):
        p = os.path.join(self.model_dir, f"{name}.pth")
        if os.path.exists(p):
            self.ac.load_state_dict(torch.load(p, map_location=self.device))


# alias for compatibility with your training factory
AlgorithmXYZ = PPOAgent