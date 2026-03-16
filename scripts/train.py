# train_0_5.py
"""
Training loop using PandaEnv.
Updated for environment_0_5 with normalized observations.
"""

import random, csv, os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from environment.environment import PandaEnv
from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent
# later, we can add SAC, A2C, etc.

random.seed(42)

JOINTS = [f"panda_joint{i}" for i in range(1, 8)]   # + ["panda_finger_joint1", "panda_finger_joint2"]
LIMITS = [(-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9), (-3.07, -0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)] #  , (0, 0.04), (0, 0.04)]
ENTITIES = {
    "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure"
}
WORKSPACE_RANGE = 0.85  # Default workspace range in meters

def make_algo(name, env):
    return {"dqn": DQNAgent, "ppo": PPOAgent}[name.lower()](env)


class Logger:
    def __init__(self, joints, limits, entities, plot_every=50, workspace_range=0.85):
        """
        Initialize logger with joint limits and workspace range for denormalization.
        """
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

        # Build CSV header: episode, step, reward, done + flattened state/action
        joint_cols = [f"joint_{j}" for j in range(len(joints))]
        entity_cols = [f"ent_{i}_{ax}" for i in range(len(entities)) for ax in "xyz"]
        action_cols = [f"action_{j}" for j in range(len(joints))]
        self.header = ["episode", "step", "reward", "done"] + joint_cols + entity_cols + action_cols

        # Init CSV
        with open(self.csv_file, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

        plt.ion()
        self.fig, self.ax = plt.subplots()

    def flatten_state_action(self, state, action, joints, entities):
        """
        Convert normalized flat state to denormalized values for logging.
        State is a flat array: [normalized_joints, normalized_entity_positions].
        """
        # Split state into joints and entities
        num_joints = len(joints)
        num_entity_coords = len(entities) * 3
        joint_vals = state[:num_joints]
        entity_vals = state[num_joints:num_joints + num_entity_coords]

        # Denormalize joints: from [-1, 1] to [low, high]
        denorm_joint_vals = []
        for val, (low, high) in zip(joint_vals, self.limits):
            if high > low:
                denorm_val = low + (val + 1.0) * (high - low) / 2.0
            else:
                denorm_val = low
            denorm_joint_vals.append(denorm_val)

        # Denormalize entity positions: multiply by workspace_range
        denorm_entity_vals = [v * self.workspace_range for v in entity_vals]

        # Actions: assume raw joint angles
        act_vals = [action.get(j, 0.0) if isinstance(action, dict) else action[i] for i, j in enumerate(joints)]

        return denorm_joint_vals + denorm_entity_vals + act_vals

    def log_episode(self, ep, ep_data, total_reward):
        # Write episode data to CSV
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            for row in ep_data:
                writer.writerow(row)

        # Update plot
        self.rewards.append(total_reward)
        if (ep + 1) % self.plot_every == 0:
            self.ax.clear()
            self.ax.plot(self.rewards)
            self.ax.set(xlabel="Episode", ylabel="Reward", title="Rewards vs Episodes")
            self.fig.savefig(self.plot_file)  # overwrite old plot


def main():
    env = PandaEnv(JOINTS, LIMITS, ENTITIES, workspace_range=WORKSPACE_RANGE)
    algo = make_algo("ppo", env)  # "dqn" or "ppo"
    log = Logger(JOINTS, LIMITS, ENTITIES, plot_every=1, workspace_range=WORKSPACE_RANGE)

    episodes = 10000
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        ep_data = []

        while not done:
            # Generic training loop for any algo
            action = algo.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            algo.store_transition(obs, action, reward, next_obs, done)
            algo.learn()  # each algo internally decides when to update

            if (ep + 1) % 100 == 0:
                algo.save(f"episode_{ep+1}")

            flat = log.flatten_state_action(obs, action, JOINTS, ENTITIES)
            ep_data.append([ep, env.step_count, reward, int(done)] + flat)

            obs = next_obs

        print(f"[ep {ep}] total_reward={total_reward:.2f}")
        log.log_episode(ep, ep_data, total_reward)


if __name__ == "__main__":
    main()