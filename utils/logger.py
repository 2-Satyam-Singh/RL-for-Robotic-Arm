# Copyright (C) 2026 Satyam Singh
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    def __init__(self, run_name, algo_info, joints, limits, entities,
                 plot_every=50, workspace_range=0.85,
                 algo_name="PPO", env_name="3 DOF", mode="train"):

        self.start_time = time.time()
        self.times = []
        self.rewards = []

        # Metadata
        self.run_name = run_name
        self.algo_info = algo_info
        self.algo_name = algo_name
        self.env_name = env_name
        self.plot_every = plot_every
        self.window_size = 50  # Rolling window for average and std-dev
        self.mode = mode  # "train" or "test"

        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.workspace_range = workspace_range

        os.makedirs("results", exist_ok=True)

        # --- FILE NAMING ---
        self.csv_file        = f"results/log_{mode}_{run_name}.csv"
        self.plot_file       = f"results/{mode}_{run_name}.png"
        self.plot_hist_file  = f"results/{mode}_{run_name}_histogram.png"

        # --- CSV SETUP (train and test share the core schema — enables future ML use) ---
        joint_cols = [f"joint_{j}" for j in range(len(joints))]
        entity_cols = [f"ent_{i}_{ax}" for i in range(len(entities)) for ax in "xyz"]
        action_cols = [f"action_{j}" for j in range(len(joints))]
        self.header = ["episode", "step_count", "reward", "done"]
        # Test mode also records per-step wall-clock latency and an episode-level success flag
        if mode == "test":
            self.header += ["step_time"]
        self.header += joint_cols + entity_cols + action_cols
        if mode == "test":
            self.header += ["is_success"]

        with open(self.csv_file, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

        # --- TEST-MODE STATE ---
        self._total_episodes = 0        # total episodes run
        self.success_step_counts = []   # step number at which each successful episode ended

    # ------------------------------------------------------------------
    # SHARED HELPERS
    # ------------------------------------------------------------------

    def flatten_state_action(self, state, action, joints, entities):
        state = np.asarray(state)
        num_joints = len(joints)
        num_entity_coords = len(entities) * 3

        joint_vals = state[:num_joints]
        entity_vals = state[num_joints:num_joints + num_entity_coords]

        limits = np.asarray(self.limits)
        lows, highs = limits[:, 0], limits[:, 1]

        denorm_joints = np.where(
            highs > lows,
            lows + (joint_vals + 1.0) * (highs - lows) / 2.0,
            lows
        )
        denorm_entities = entity_vals * self.workspace_range
        act_vals = np.asarray(action[:num_joints])

        return np.round(
            np.concatenate([denorm_joints, denorm_entities, act_vals]), decimals=4
        ).tolist()

    def _draw_info_box(self, ax, elapsed_time):
        """Metadata box shown in the bottom-right corner of every plot."""
        dof_str = f"({self.algo_info['DOF']} DOF)" if "DOF" in self.algo_info else ""
        lines = [
            f"Robot: {self.env_name} {dof_str}",
            f"Algo:  {self.algo_name}",
            f"Seed:  {self.algo_info.get('Seed', 'N/A')}",
        ]
        if self.mode == "train":
            lines += [
                f"LR:    {self.algo_info.get('LR', 'N/A')}",
                f"Gamma: {self.algo_info.get('Gamma', 'N/A')}",
                f"Entropy: {self.algo_info.get('Entropy Coeff', 'N/A')}",
            ]
        lines.append(f"Time:  {elapsed_time:.1f}s")
        info_text = "\n".join(lines)

        ax.text(
            0.98, 0.02, info_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def log_episode(self, ep, ep_data, total_reward):
        """Append per-step rows to the train CSV and redraw the reward plot."""
        with open(self.csv_file, "a", newline="") as f:
            csv.writer(f).writerows(ep_data)

        self.rewards.append(total_reward)
        self.times.append(time.time() - self.start_time)

        if (ep + 1) % self.plot_every == 0:
            self._save_train_plot()

    def _save_train_plot(self):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))
        rewards_arr = np.array(self.rewards)

        if len(rewards_arr) >= self.window_size:
            means = np.array([
                np.mean(rewards_arr[max(0, i - self.window_size):i + 1])
                for i in range(len(rewards_arr))
            ])
            stds = np.array([
                np.std(rewards_arr[max(0, i - self.window_size):i + 1])
                for i in range(len(rewards_arr))
            ])
            x = np.arange(len(rewards_arr))
            ax.plot(x, means, color='#1f77b4',
                    label=f'{self.window_size}-Ep Moving Avg', linewidth=2)
            ax.fill_between(x, means - stds, means + stds,
                            color='#1f77b4', alpha=0.2, label='Stability (±1 std)')
        else:
            ax.plot(rewards_arr, color='#1f77b4', alpha=0.3, label='Raw Reward')
            cum_avg = np.cumsum(rewards_arr) / (np.arange(len(rewards_arr)) + 1)
            ax.plot(cum_avg, color='#1f77b4', linewidth=2, label='Cumulative Avg')

        ax.set_title(f"Training Performance: {self.env_name} | {self.algo_name}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Total Reward", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')
        self._draw_info_box(ax, self.times[-1])

        plt.tight_layout()
        fig.savefig(self.plot_file, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # TESTING
    # ------------------------------------------------------------------

    def log_test_episode(self, ep, ep_data, total_reward, steps, is_success):
        """Append per-step rows to the test CSV and record success step count.

        ep_data rows come from flatten_state_action(), same shape as training rows.
        Each row gets is_success appended so the file is self-contained for future ML use.
        """
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            for row in ep_data:
                writer.writerow(row + [int(is_success)])

        self._total_episodes += 1
        if is_success:
            self.success_step_counts.append(steps)

        self.times.append(time.time() - self.start_time)

    def save_test_plot(self):
        """Save both the CDF curve and the histogram after evaluation is complete."""
        self._save_cdf_plot()
        self._save_histogram_plot()

    def _save_cdf_plot(self):
        """Cumulative successes by step — how many successes achieved by step X."""
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))

        max_steps   = int(self.algo_info.get("Max Steps", 100))
        n_success   = len(self.success_step_counts)
        total_eps   = self._total_episodes
        success_rate = (n_success / total_eps * 100) if total_eps else 0

        if n_success > 0:
            # Count how many episodes succeeded at each exact step number
            counts = np.zeros(max_steps + 1, dtype=int)
            for s in self.success_step_counts:
                if 1 <= s <= max_steps:
                    counts[s] += 1

            x = np.arange(1, max_steps + 1)
            y = np.cumsum(counts[1:])               # cumulative successes up to each step

            ax.step(x, y, color='#2ca02c', linewidth=2, where='post', label='Successes by Step')
            ax.fill_between(x, y, step='post', color='#2ca02c', alpha=0.15)
            ax.set_ylim(0, n_success + 1)
        else:
            ax.text(0.5, 0.5, "No successes recorded", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')

        ax.set_title(
            f"Evaluation Performance: {self.env_name} | {self.algo_name}\n"
            f"Success Rate: {n_success}/{total_eps} ({success_rate:.1f}%)",
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Successes by Step", fontsize=12)
        ax.set_xlim(0, max_steps + 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self._draw_info_box(ax, self.times[-1] if self.times else 0.0)

        plt.tight_layout()
        fig.savefig(self.plot_file, dpi=300)
        plt.close(fig)
        print(f"📊 CDF plot saved       → {self.plot_file}")

    def _save_histogram_plot(self):
        """Histogram — how many episodes succeeded at exactly step X."""
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))

        max_steps    = int(self.algo_info.get("Max Steps", 100))
        n_success    = len(self.success_step_counts)
        total_eps    = self._total_episodes
        success_rate = (n_success / total_eps * 100) if total_eps else 0

        if n_success > 0:
            counts = np.zeros(max_steps + 1, dtype=int)
            for s in self.success_step_counts:
                if 1 <= s < max_steps:          # max_steps already filtered but guard anyway
                    counts[s] += 1

            x = np.arange(1, max_steps)         # 1 .. max_steps-1
            y = counts[1:max_steps]

            ax.bar(x, y, color='#1f77b4', alpha=0.75, width=0.8,
                   label='Episodes Succeeding at Exact Step')
            ax.set_ylim(0, y.max() + 1)
        else:
            ax.text(0.5, 0.5, "No successes recorded", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')

        ax.set_title(
            f"Success Step Distribution: {self.env_name} | {self.algo_name}\n"
            f"Success Rate: {n_success}/{total_eps} ({success_rate:.1f}%)",
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel("Step at Which Success Was Achieved", fontsize=12)
        ax.set_ylabel("Number of Episodes", fontsize=12)
        ax.set_xlim(0, max_steps)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self._draw_info_box(ax, self.times[-1] if self.times else 0.0)

        plt.tight_layout()
        fig.savefig(self.plot_hist_file, dpi=300)
        plt.close(fig)
        print(f"📊 Histogram plot saved → {self.plot_hist_file}")