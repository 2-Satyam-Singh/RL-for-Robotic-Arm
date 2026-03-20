import csv
import os
import time  # ← Added to track wall-clock time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(self, joints, limits, entities, plot_every=50, workspace_range=0.85, algo_name="PPO", env_name="7 DOF arm"):
        
        self.start_time = time.time()  # ← Start the stopwatch!
        self.times = []                # ← List to store elapsed time per episode

        ts = datetime.now().strftime("%d-%m-%Y_%H-%M")
        os.makedirs("results", exist_ok=True)

        self.algo_name = algo_name.upper()
        self.env_name = env_name

        # Updated Nomenclature to differentiate Episode plots and Time plots
        self.csv_file = f"results/log_{algo_name}_{ts}.csv"
        self.plot_ep_file = f"results/reward_vs_ep_{algo_name}_{ts}.png"
        self.plot_time_file = f"results/reward_vs_time_{algo_name}_{ts}.png"
        
        self.plot_every = plot_every
        self.window_size = 100 

        self.rewards = []
        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.workspace_range = workspace_range

        joint_cols = [f"joint_{j}" for j in range(len(joints))]
        entity_cols = [f"ent_{i}_{ax}" for i in range(len(entities)) for ax in "xyz"]
        action_cols = [f"action_{j}" for j in range(len(joints))]

        self.header = ["episode", "reward", "done"] + joint_cols + entity_cols + action_cols

        with open(self.csv_file, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

        # Create two separate graphs so they don't overwrite each other
        self.fig_ep, self.ax_ep = plt.subplots()
        self.fig_time, self.ax_time = plt.subplots()

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
        
        # Calculate elapsed time in hours (e.g., 0.5 hours)
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        self.times.append(elapsed_hours)

        # Plotting logic
        if (ep + 1) % self.plot_every == 0:
            
            # ==========================================
            # 1. PLOT: REWARD VS EPISODE
            # ==========================================
            self.ax_ep.clear()
            self.ax_ep.plot(self.rewards, color='lightblue', alpha=0.4, label='Raw Reward')
            
            if len(self.rewards) >= self.window_size:
                moving_avg = np.convolve(self.rewards, np.ones(self.window_size)/self.window_size, mode='valid')
                x_axis = np.arange(self.window_size - 1, len(self.rewards))
                self.ax_ep.plot(x_axis, moving_avg, color='blue', linewidth=2, label=f'{self.window_size}-Ep Moving Avg')
            else:
                early_avg = [np.mean(self.rewards[:i+1]) for i in range(len(self.rewards))]
                self.ax_ep.plot(early_avg, color='blue', linewidth=2, label='Cumulative Avg')

            self.ax_ep.set(xlabel="Episode", ylabel="Reward", title=f"Rewards vs Episodes for {self.env_name} using {self.algo_name}")
            self.ax_ep.legend(loc="upper left")
            self.fig_ep.savefig(self.plot_ep_file)

            # ==========================================
            # 2. PLOT: REWARD VS TIME (HOURS)
            # ==========================================
            self.ax_time.clear()
            self.ax_time.plot(self.times, self.rewards, color='lightgreen', alpha=0.4, label='Raw Reward')
            
            if len(self.rewards) >= self.window_size:
                moving_avg = np.convolve(self.rewards, np.ones(self.window_size)/self.window_size, mode='valid')
                # Align the time array to match the moving average offset
                time_axis = self.times[self.window_size - 1:]
                self.ax_time.plot(time_axis, moving_avg, color='green', linewidth=2, label=f'{self.window_size}-Ep Moving Avg')
            else:
                early_avg = [np.mean(self.rewards[:i+1]) for i in range(len(self.rewards))]
                self.ax_time.plot(self.times, early_avg, color='green', linewidth=2, label='Cumulative Avg')

            self.ax_time.set(xlabel="Time (Hours)", ylabel="Reward", title=f"Rewards vs Time for {self.env_name} using {self.algo_name}")
            self.ax_time.legend(loc="upper left")
            self.fig_time.savefig(self.plot_time_file)