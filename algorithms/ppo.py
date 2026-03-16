# algorithms/ppo_0_5.py
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

# Set random seed for reproducibility
SEED = 99
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ActorCritic(nn.Module):
    """Actor-Critic network with Gaussian policy for continuous actions."""
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        """
        Args:
            obs_dim (int): Observation dimension (joints + 3 * entities).
            act_dim (int): Action dimension (number of joints).
            hidden (tuple): Hidden layer sizes for the network.
        """
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
        """
        Sample action and compute log-probability and value.
        Args:
            x (torch.Tensor): Observation tensor of shape (batch, obs_dim).
        Returns:
            tuple: (action, log_prob, value), where action is a tensor (batch, act_dim).
        """
        mu = self.actor_mean(x)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()  # Reparameterized sample
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, x, action):
        """
        Evaluate log-probability, entropy, and value for given action.
        Args:
            x (torch.Tensor): Observation tensor of shape (batch, obs_dim).
            action (torch.Tensor): Action tensor of shape (batch, act_dim).
        Returns:
            tuple: (log_prob, entropy, value).
        """
        mu = self.actor_mean(x)
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
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=8,
        minibatch=64,
        rollout_steps=2048,
        hidden=(128, 128),
        value_loss_coef=0.5,
        entropy_coef=0.01,
        device=None,
    ):
        super().__init__(env, algo_name="ppo")
        self.joint_names = list(env.joints)
        self.joint_limits = env.limits
        self.entity_names = sorted(list(env.entities)) if env.entities else []
        self.workspace_range = getattr(env, "workspace_range", 0.85)
        self.n_joints = len(self.joint_names)

        # Dimensions
        self.obs_dim = self.n_joints + 3 * len(self.entity_names)
        self.act_dim = self.n_joints

        # Hyperparameters
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

    # ---- API used by Training loop ----
    def select_action(self, obs):
        """Return action array."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, value_t = self.ac.act(obs_t)
        action = action_t.squeeze(0).cpu().numpy()
        log_prob = float(log_prob_t.item())
        value = float(value_t.item())
        self.last = (obs, action, log_prob, value)
        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Append transition to buffer.
        """
        if self.buffer_size >= self.rollout_steps:
            return  # wait for learn()
        idx = self.buffer_size
        self.buffer["obs"][idx] = obs
        self.buffer["act"][idx] = action
        self.buffer["logp"][idx] = self.last[2]
        self.buffer["rew"][idx] = float(reward)
        self.buffer["val"][idx] = self.last[3]
        self.buffer["done"][idx] = float(done)
        self.buffer_size += 1

        # store next state for bootstrap
        self.last_next_obs = next_obs
        self.last_done = bool(done)

    def learn(self):
        """
        Called every step — updates only when buffer ready or episode done.
        """
        if (self.buffer_size < self.rollout_steps) and not self.last_done:
            return

        # bootstrap value
        if self.last_next_obs is None:
            last_val = 0.0
        else:
            s_t = torch.tensor(self.last_next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                last_val = float(self.ac.critic(s_t).item())

        # compute GAE + returns
        adv, ret = self._compute_gae_returns(last_val)

        # convert buffers
        s_t = torch.tensor(self.buffer["obs"][:self.buffer_size], dtype=torch.float32, device=self.device)
        a_t = torch.tensor(self.buffer["act"][:self.buffer_size], dtype=torch.float32, device=self.device)
        logp_t = torch.tensor(self.buffer["logp"][:self.buffer_size], dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)

        # updates
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

        # clear
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