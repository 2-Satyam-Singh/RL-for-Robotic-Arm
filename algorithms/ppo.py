# algorithms/ppo_0_4.py
"""
PPOAgent (PyTorch) — on-policy, but safe to call `learn()` every step.
Works with your training loop without any change.

API:
    algo = PPOAgent(env)
    action = algo.select_action(obs)         # returns joint->pos dict
    algo.store_transition(obs, action, r, next_obs, done)
    algo.learn()                             # safe every step; updates when ready
    algo.save("name")
    algo.load("name")
"""

import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim

seed = 99

random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ----- Actor-Critic network (Gaussian policy) -----
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        # shared body for actor
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.actor_mean = nn.Sequential(*layers, nn.Linear(last, act_dim))
        # learnable log-std (shared independent of state)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # separate critic head
        v_layers = []
        last = obs_dim
        for h in hidden:
            v_layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.critic = nn.Sequential(*v_layers, nn.Linear(last, 1))

    def forward(self, x):
        raise NotImplementedError("Use act() or evaluate()")

    def act(self, x):
        # x: tensor (B, obs_dim)
        mu = self.actor_mean(x)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()   # reparameterized sample
        logp = dist.log_prob(a).sum(-1)
        v = self.critic(x).squeeze(-1)
        return a, logp, v

    def evaluate(self, x, a):
        mu = self.actor_mean(x)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(a).sum(-1)
        entropy = dist.entropy().sum(-1)
        v = self.critic(x).squeeze(-1)
        return logp, entropy, v


# ----- PPO Agent -----
class PPOAgent:
    def __init__(self,
                 env,
                 lr=3e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.2,
                 epochs=8,
                 minibatch=64,
                 rollout_steps=2048,
                 hidden=(128, 128),
                 model_dir="ppo_torch_models",
                 device=None):
        self.env = env
        self.joint_names = list(env.joints)
        self.joint_limits = env.limits
        self.entity_names = sorted(list(env.entities)) if env.entities else []
        self.n_joints = len(self.joint_names)

        # dims
        self.obs_dim = self.n_joints + 3 * len(self.entity_names)
        self.act_dim = self.n_joints  # continuous joint values

        # hyperparams
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = int(epochs)
        self.minibatch = int(minibatch)
        self.rollout_steps = int(rollout_steps)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # networks + optim
        self.ac = ActorCritic(self.obs_dim, self.act_dim, hidden).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

        # rollout buffer (lists)
        self.buf_s = []
        self.buf_a = []
        self.buf_logp = []
        self.buf_r = []
        self.buf_v = []
        self.buf_d = []

        # helpers
        self.steps_collected = 0
        self.last = None          # stores (s_vec, a_vec, logp, v) from last select_action
        self.last_next_s = None   # store next state vector for bootstrapping when updating
        self.last_done = False

        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir

    # ---- observation / action helpers ----
    def _obs_to_vec(self, obs):
        v = np.zeros((self.obs_dim,), dtype=np.float32)
        joints = obs.get("joints", {})
        for i, jn in enumerate(self.joint_names):
            val = joints.get(jn)
            if val is None:
                lo, hi = self.joint_limits[i]; val = 0.5 * (lo + hi)
            v[i] = float(val)
        base = self.n_joints
        ents = obs.get("entities", {})
        for ei, en in enumerate(self.entity_names):
            p = ents.get(en, [0.0, 0.0, 0.0])
            v[base + 3 * ei: base + 3 * ei + 3] = [float(p[0]), float(p[1]), float(p[2])]
        return v

    def _vec_to_action(self, a_vec, current_joints):
        """Clamp continuous action vector to joint limits and return dict."""
        out = {}
        for i, jn in enumerate(self.joint_names):
            lo, hi = self.joint_limits[i]
            val = float(np.clip(a_vec[i], lo, hi))
            out[jn] = val
        return out

    # ---- API used by Training loop ----
    def select_action(self, obs):
        """Return joint->pos dict. Also stash last (s,a,logp,v) for store_transition."""
        s_vec = self._obs_to_vec(obs)
        s_t = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a_t, logp_t, v_t = self.ac.act(s_t)
        a_np = a_t.squeeze(0).cpu().numpy()
        logp = float(logp_t.item())
        v = float(v_t.item())
        self.last = (s_vec, a_np, logp, v)
        return self._vec_to_action(a_np, obs.get("joints", {}))

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Append last collected (s,a,logp,v) and reward/done.
        Keep next_obs vector for bootstrapping the final value when updating.
        """
        if self.last is None:
            # select_action wasn't called (shouldn't happen)
            return
        s_vec, a_vec, logp, v = self.last
        self.buf_s.append(s_vec)
        self.buf_a.append(a_vec)
        self.buf_logp.append(logp)
        self.buf_r.append(float(reward))
        self.buf_v.append(v)
        self.buf_d.append(float(done))
        self.steps_collected += 1

        # store next state vector for bootstrap value at update time
        self.last_next_s = self._obs_to_vec(next_obs)
        self.last_done = bool(done)

    def learn(self):
        """
        Called every step — updates only when:
          - buffer length >= rollout_steps, or
          - episode ended (last_done True)
        After update, buffer is cleared.
        """
        ready = (len(self.buf_s) >= self.rollout_steps) or (self.last_done and len(self.buf_s) > 0)
        if not ready:
            return

        # bootstrap value for last-next-state
        if self.last_next_s is None:
            last_val = 0.0
        else:
            s_t = torch.tensor(self.last_next_s, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                last_val = float(self.ac.critic(s_t).item())

        # compute GAE advantages + returns
        adv, ret = self._compute_gae_returns(last_val)

        # convert buffers to arrays
        s_arr = np.array(self.buf_s, dtype=np.float32)
        a_arr = np.array(self.buf_a, dtype=np.float32)
        logp_arr = np.array(self.buf_logp, dtype=np.float32)
        adv_arr = adv.astype(np.float32)
        ret_arr = ret.astype(np.float32)

        # multiple epochs of minibatch updates
        n = len(s_arr)
        inds = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, n, self.minibatch):
                mb = inds[start:start + self.minibatch]
                mb_s = torch.tensor(s_arr[mb], dtype=torch.float32, device=self.device)
                mb_a = torch.tensor(a_arr[mb], dtype=torch.float32, device=self.device)
                mb_logp_old = torch.tensor(logp_arr[mb], dtype=torch.float32, device=self.device)
                mb_adv = torch.tensor(adv_arr[mb], dtype=torch.float32, device=self.device)
                mb_ret = torch.tensor(ret_arr[mb], dtype=torch.float32, device=self.device)

                logp, entropy, v = self.ac.evaluate(mb_s, mb_a)
                ratio = torch.exp(logp - mb_logp_old)

                # clipped surrogate loss
                obj1 = ratio * mb_adv
                obj2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(obj1, obj2).mean()

                value_loss = ((mb_ret - v) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        # clear buffers after update
        self._clear_buffer()

    # ---- utilities ----
    def _compute_gae_returns(self, last_val):
        """
        Compute GAE advantages and returns.
        last_val: bootstrapped value for final next state (0 if terminal).
        """
        r = np.array(self.buf_r, dtype=np.float32)
        v = np.array(self.buf_v, dtype=np.float32)
        d = np.array(self.buf_d, dtype=np.float32)
        # append last_val for v_{t+1} at final step
        v_next = np.append(v[1:], last_val)
        T = len(r)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - d[t]
            delta = r[t] + self.gamma * (v_next[t]) * nonterm - v[t]
            lastgaelam = delta + self.gamma * self.lam * nonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + v
        # normalize advantages
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def _clear_buffer(self):
        self.buf_s.clear(); self.buf_a.clear(); self.buf_logp.clear()
        self.buf_r.clear(); self.buf_v.clear(); self.buf_d.clear()
        self.steps_collected = 0
        self.last = None
        self.last_next_s = None
        self.last_done = False

    # ---- persistence ----
    def save(self, name="ppo"):
        torch.save(self.ac.state_dict(), os.path.join(self.model_dir, f"{name}.pth"))

    def load(self, name="ppo"):
        p = os.path.join(self.model_dir, f"{name}.pth")
        if os.path.exists(p):
            self.ac.load_state_dict(torch.load(p, map_location=self.device))


# alias for compatibility with your training factory
AlgorithmXYZ = PPOAgent
