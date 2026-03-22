# utils/logger.py
import csv
import os
import time  
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, run_name, algo_info, joints, limits, entities, plot_every=50, workspace_range=0.85, algo_name="PPO", env_name="3 DOF"):
        self.start_time = time.time()  
        self.times = []                
        self.rewards = []

        # Store metadata
        self.run_name = run_name
        self.algo_info = algo_info
        self.algo_name = algo_name
        self.env_name = env_name
        self.plot_every = plot_every
        self.window_size = 50 # Rolling window for average and std-dev
        
        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.workspace_range = workspace_range

        # Files nomenclature
        os.makedirs("results", exist_ok=True)
        self.csv_file = f"results/log_{run_name}.csv"
        self.plot_ep_file = f"results/plot_{run_name}.png"
        
        # CSV Setup
        joint_cols = [f"joint_{j}" for j in range(len(joints))]
        entity_cols = [f"ent_{i}_{ax}" for i in range(len(entities)) for ax in "xyz"]
        action_cols = [f"action_{j}" for j in range(len(joints))]
        self.header = ["episode", "step_count", "reward", "done"] + joint_cols + entity_cols + action_cols

        with open(self.csv_file, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

    def flatten_state_action(self, state, action, joints, entities):
        state = np.asarray(state)
        num_joints = len(joints)
        num_entity_coords = len(entities) * 3

        joint_vals = state[:num_joints]
        entity_vals = state[num_joints:num_joints + num_entity_coords]

        limits = np.asarray(self.limits)
        lows, highs = limits[:, 0], limits[:, 1]

        denorm_joints = np.where(highs > lows, lows + (joint_vals + 1.0) * (highs - lows) / 2.0, lows)
        denorm_entities = entity_vals * self.workspace_range
        act_vals = np.asarray(action[:num_joints])

        return np.round(np.concatenate([denorm_joints, denorm_entities, act_vals]), decimals=4).tolist()

    def _draw_info_box(self, ax, elapsed_time):
        """Adds a professional metadata box to the top right of the plot."""
        info_text = (
            f"Robot: {self.env_name} ({self.algo_info['DOF']} DOF)\n"
            f"Algo: {self.algo_name}\n"
            f"Seed: {self.algo_info['Seed']}\n"
            f"LR: {self.algo_info['LR']}\n"
            f"Gamma: {self.algo_info['Gamma']}\n"
            f"Entropy: {self.algo_info['Entropy Coeff']}\n"
            f"Time: {elapsed_time:.2f}h"
        )
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    def log_episode(self, ep, ep_data, total_reward):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(ep_data)

        self.rewards.append(total_reward)
        self.times.append((time.time() - self.start_time) / 3600.0)

        if (ep + 1) % self.plot_every == 0:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            rewards_arr = np.array(self.rewards)
            
            # --- SHADED STABILITY LOGIC ---
            if len(rewards_arr) >= self.window_size:
                # Calculate rolling mean and std
                means = np.array([np.mean(rewards_arr[max(0, i-self.window_size):i+1]) for i in range(len(rewards_arr))])
                stds = np.array([np.std(rewards_arr[max(0, i-self.window_size):i+1]) for i in range(len(rewards_arr))])
                
                x = np.arange(len(rewards_arr))
                ax.plot(x, means, color='#1f77b4', label=f'{self.window_size}-Ep Moving Avg', linewidth=2)
                ax.fill_between(x, means - stds, means + stds, color='#1f77b4', alpha=0.2, label='Stability (±1 std)')
            else:
                ax.plot(rewards_arr, color='#1f77b4', alpha=0.3, label='Raw Reward')
                cum_avg = np.cumsum(rewards_arr) / (np.arange(len(rewards_arr)) + 1)
                ax.plot(cum_avg, color='#1f77b4', linewidth=2, label='Cumulative Avg')

            # --- STYLING ---
            ax.set_title(f"Training Performance: {self.env_name} | {self.algo_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Episode", fontsize=12)
            ax.set_ylabel("Total Reward", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper left')

            # Add Info Box
            self._draw_info_box(ax, self.times[-1])

            plt.tight_layout()
            fig.savefig(self.plot_ep_file)
            plt.close(fig)