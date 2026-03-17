import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(self, joints, limits, entities, plot_every=50, workspace_range=0.85):

        ts = datetime.now().strftime("%d-%m-%Y_%H-%M")
        os.makedirs("results", exist_ok=True)

        self.csv_file = f"results/log_{ts}.csv"
        self.plot_file = f"results/plot_{ts}.png"
        self.plot_every = plot_every

        self.rewards = []
        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.workspace_range = workspace_range

        joint_cols = [f"joint_{j}" for j in range(len(joints))]
        entity_cols = [f"ent_{i}_{ax}" for i in range(len(entities)) for ax in "xyz"]
        action_cols = [f"action_{j}" for j in range(len(joints))]

        self.header = ["episode", "step", "reward", "done"] + joint_cols + entity_cols + action_cols

        with open(self.csv_file, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

        self.fig, self.ax = plt.subplots()

    def flatten_state_action(self, state, action, joints, entities):
        state = np.asarray(state)

        num_joints = len(joints)
        num_entity_coords = len(entities) * 3

        joint_vals = state[:num_joints]
        entity_vals = state[num_joints:num_joints + num_entity_coords]

        limits = np.asarray(self.limits)
        lows = limits[:, 0]
        highs = limits[:, 1]

        denorm_joint_vals = np.where(
            highs > lows,
            lows + (joint_vals + 1.0) * (highs - lows) / 2.0,
            lows
        )

        denorm_entity_vals = entity_vals * self.workspace_range

        if isinstance(action, dict):
            act_vals = np.array([action.get(j, 0.0) for j in joints])
        else:
            act_vals = np.asarray(action[:num_joints])

        return np.round(np.concatenate([denorm_joint_vals, denorm_entity_vals, act_vals]), decimals=4).tolist()

    def log_episode(self, ep, ep_data, total_reward):

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            for row in ep_data:
                writer.writerow(row)

        self.rewards.append(total_reward)

        if (ep + 1) % self.plot_every == 0:
            self.ax.clear()
            self.ax.plot(self.rewards)
            self.ax.set(xlabel="Episode", ylabel="Reward", title="Rewards vs Episodes")
            self.fig.savefig(self.plot_file)