# algorithms/dqn_0_4.py
"""
Optimized DQN (PyTorch version).
Features:
- Numpy ring buffer
- Double DQN target
- Compact MLP
- API compatible with TF version
"""

import os, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ---------- fast numpy replay ----------
class NumpyReplay:
    def __init__(self, obs_dim, size=100_000):
        self.size = int(size); self.obs_dim = obs_dim
        self.s  = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.a  = np.zeros((self.size,), dtype=np.int64)
        self.r  = np.zeros((self.size,), dtype=np.float32)
        self.d  = np.zeros((self.size,), dtype=np.float32)
        self.ptr = 0; self.len = 0
    def push(self, s,a,r,s2,done):
        i = self.ptr % self.size
        self.s[i] = s; self.s2[i] = s2; self.a[i] = a; self.r[i] = r; self.d[i] = float(done)
        self.ptr += 1; self.len = min(self.len+1, self.size)
    def sample(self, batch):
        idx = np.random.randint(0, self.len, size=batch)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx])
    def __len__(self): return self.len

# ---------- small MLP ----------
class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden=(128,128)):
        super().__init__()
        layers = []
        last = inp_dim
        for h in hidden:
            layers.append(nn.Linear(last,h)); layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last,out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

# ---------- DQN agent ----------
class DQNAgent:
    def __init__(self, env,
                 lr=1e-3, gamma=0.99, batch_size=64, buffer_size=100_000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=20000,
                 target_update_freq=500, step_frac=0.08, min_buffer=500,
                 hidden=(128,128), model_dir="dqn_torch_models", device=None):
        self.env = env
        self.joint_names = list(env.joints)
        self.joint_limits = env.limits
        self.entity_names = sorted(list(env.entities)) if env.entities else []
        self.n_joints = len(self.joint_names)
        self.n_primitives = 3
        self.n_actions = self.n_joints * self.n_primitives
        self.obs_dim = self.n_joints + 3*len(self.entity_names)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_frac = step_frac
        self.min_buffer = min_buffer

        # device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # precompute deltas
        self.deltas = np.array([(hi - lo) * self.step_frac for (lo, hi) in self.joint_limits], dtype=np.float32)

        # networks
        self.q = MLP(self.obs_dim, self.n_actions, hidden).to(self.device)
        self.target = MLP(self.obs_dim, self.n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = NumpyReplay(self.obs_dim, buffer_size)

        # epsilon
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(1, epsilon_decay_steps)
        self.epsilon_step = (epsilon_start - epsilon_end)/self.epsilon_decay

        self.train_steps = 0
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir

    # ---------- obs/action helpers ----------
    def _obs_to_vec(self, obs):
        vec = np.zeros((self.obs_dim,), dtype=np.float32)
        joints = obs.get("joints", {})
        for i, jn in enumerate(self.joint_names):
            v = joints.get(jn)
            if v is None:
                lo, hi = self.joint_limits[i]; v = 0.5*(lo+hi)
            vec[i] = float(v)
        base = self.n_joints
        ent = obs.get("entities", {})
        for ei, en in enumerate(self.entity_names):
            p = ent.get(en)
            if p and len(p)>=3:
                vec[base+3*ei:base+3*ei+3] = [float(p[0]), float(p[1]), float(p[2])]
        return vec

    def _action_idx_to_dict(self, idx, current_joints):
        ji = idx // self.n_primitives; prim = idx % self.n_primitives
        curr = [None]*self.n_joints
        for i, jn in enumerate(self.joint_names):
            v = current_joints.get(jn)
            if v is None:
                lo, hi = self.joint_limits[i]; v = 0.5*(lo+hi)
            curr[i] = float(v)
        delta = float(self.deltas[ji])
        if prim==0: newv = curr[ji]-delta
        elif prim==1: newv = curr[ji]
        else: newv = curr[ji]+delta
        lo, hi = self.joint_limits[ji]
        newv = max(lo, min(hi, newv))
        act = {jn: curr[i] for i, jn in enumerate(self.joint_names)}
        act[self.joint_names[ji]] = newv
        return act

    # ---------- policy ----------
    def select_action(self, obs):
        s = self._obs_to_vec(obs)
        if random.random() < self.epsilon:
            idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                qv = self.q(torch.tensor(s,device=self.device).unsqueeze(0))
                idx = int(torch.argmax(qv).item())
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_step)
        self._last_action_idx = idx
        return self._action_idx_to_dict(idx, obs.get("joints", {}))

    def store_transition(self, obs, action, reward, next_obs, done):
        s = self._obs_to_vec(obs); s2 = self._obs_to_vec(next_obs)
        if hasattr(self, "_last_action_idx"): idx = int(self._last_action_idx)
        else:
            diffs=[abs((action.get(jn,s[i]))-s[i]) for i,jn in enumerate(self.joint_names)]
            ji=int(np.argmax(diffs)); prim=1
            if action.get(self.joint_names[ji],s[ji])<=s[ji]-0.5*self.deltas[ji]: prim=0
            elif action.get(self.joint_names[ji],s[ji])>=s[ji]+0.5*self.deltas[ji]: prim=2
            idx=ji*self.n_primitives+prim
        self.buffer.push(s, idx, float(reward), s2, float(done))

    # ---------- train ----------
    def learn(self):
        if len(self.buffer) < max(self.min_buffer,self.batch_size): return
        s,a,r,s2,d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s,device=self.device)
        a = torch.tensor(a,device=self.device)
        r = torch.tensor(r,device=self.device)
        s2 = torch.tensor(s2,device=self.device)
        d = torch.tensor(d,device=self.device)

        q_vals = self.q(s)                       # (B,A)
        q_taken = q_vals.gather(1,a.unsqueeze(1)).squeeze(1)

        # Double DQN
        next_online = self.q(s2).argmax(1)
        next_target = self.target(s2).gather(1,next_online.unsqueeze(1)).squeeze(1)
        target = r + (1.0-d)*self.gamma*next_target

        loss = self.loss_fn(q_taken, target.detach())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.q.state_dict())

    # ---------- save/load ----------
    def save(self,name="dqn"):
        torch.save(self.q.state_dict(), os.path.join(self.model_dir,f"{name}.pth"))
    def load(self,name="dqn"):
        p = os.path.join(self.model_dir,f"{name}.pth")
        if os.path.exists(p):
            self.q.load_state_dict(torch.load(p,map_location=self.device))
            self.target.load_state_dict(self.q.state_dict())
