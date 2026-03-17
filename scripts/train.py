"""
Training loop using PandaEnv.
Logger moved to utils/logger.py for cleaner code and better plotting. Each episode's data is logged to a CSV file and plotted in real-time.
"""

import random, csv, os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from environment.environment import PandaEnv, JOINTS, LIMITS
from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent
from utils.logger import Logger

random.seed(42)

JOINTS = JOINTS[:-2]      # panda_joint1..7 only
LIMITS = LIMITS[:-2]
ENTITIES = {
    "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure"
}
WORKSPACE_RANGE = 0.85  # Default workspace range in meters

def make_algo(name, env):
    return {"dqn": DQNAgent, "ppo": PPOAgent}[name.lower()](env)

def main():
    env = PandaEnv(JOINTS, LIMITS, ENTITIES, workspace_range=WORKSPACE_RANGE)
    algo = make_algo("ppo", env)  # "dqn" or "ppo"
    log = Logger(JOINTS, LIMITS, ENTITIES, plot_every=1, workspace_range=WORKSPACE_RANGE)

    episodes = 100000
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