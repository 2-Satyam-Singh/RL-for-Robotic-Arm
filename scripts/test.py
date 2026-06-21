# scripts/test.py
"""
Evaluation script for a trained RL agent.
Called by main.py. Loads a saved model and runs it for a few episodes without training.
"""

import time
import torch
import random
import numpy as np
from datetime import datetime

from environment.environment import RobotEnv
from algorithms.ppo import PPOAgent
from algorithms.dqn import DQNAgent
from utils.logger import Logger


def set_seed(seed):
    """Sets the seed for perfectly reproducible testing runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_algo(name, env):
    return {"dqn": DQNAgent, "ppo": PPOAgent}[name.lower()](env)


def test_agent(cfg, args):
    """Called by main.py. Runs the evaluation loop."""

    set_seed(args.seed)

    # --- UNIQUE IDENTIFICATION (mirrors train.py convention) ---
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M")
    run_name = f"{args.robot}_{args.algorithm}_{args.reward_type}_s{args.seed}_{ts}"

    print(f"Initializing {args.robot.upper()} environment...")
    env = RobotEnv(
        joints=cfg["joints"],
        limits=cfg["limits"],
        entities=cfg["entities"],
        workspace_range=cfg["workspace_range"],
        model_name=cfg["model_name"],
        world_name=cfg["world_name"],
        ee_link_name=cfg["ee_link_name"],
        reward_type=args.reward_type,
        max_steps=args.max_steps
    )

    clean_model_name = args.model_name.replace(".pth", "")

    print(f"Loading trained {args.algorithm.upper()} Agent '{clean_model_name}'...")
    algo = make_algo(args.algorithm, env)
    algo.load(clean_model_name)

    if hasattr(algo, 'ac'):
        algo.ac.eval()
    elif hasattr(algo, 'q_net'):
        algo.q_net.eval()

    # --- LOGGER SETUP (test mode) ---
    algo_info = {
        "Seed": args.seed,
        "Max Steps": args.max_steps,
        "DOF": len(cfg["joints"]),
        "Episodes": args.episodes,
    }
    log = Logger(
        run_name=run_name,
        algo_info=algo_info,
        joints=cfg["joints"],
        limits=cfg["limits"],
        entities=cfg["entities"],
        workspace_range=cfg["workspace_range"],
        algo_name=args.algorithm.upper(),
        env_name=args.robot.upper(),
        mode="test"                         # <-- produces log_test_... and test_...
    )

    # --- METRIC TRACKERS ---
    successes = 0
    all_rewards = []
    all_steps = []

    print(f"\n🚀 Starting evaluation for {args.episodes} episodes...\n")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        ep_data = []

        while not done:
            step_start = time.time()
            action = algo.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_time = time.time() - step_start
            total_reward += reward
            steps += 1

            flat = log.flatten_state_action(obs, action, cfg["joints"], cfg["entities"])
            ep_data.append([ep, steps, reward, int(done), round(step_time, 4)] + flat)
            obs = next_obs

        all_rewards.append(total_reward)
        all_steps.append(steps)

        # --- SUCCESS CHECK ---
        # Success = the episode terminated on the task goal (object knocked off),
        # as opposed to truncating at max_steps. The env reports this in info.
        is_success = bool(info.get("is_success", terminated))

        if is_success:
            successes += 1

        # --- LOG TO CSV + UPDATE CUMULATIVE TRACKER ---
        log.log_test_episode(ep, ep_data, total_reward, steps, is_success)

        print(f"[Test Ep {ep + 1:03d}] Reward: {total_reward:8.2f} | Steps: {steps:3d} | {'✅' if is_success else '❌'}")

    # --- SAVE CUMULATIVE SUCCESS PLOT ---
    log.save_test_plot()

    # --- TERMINAL REPORT ---
    mean_reward  = np.mean(all_rewards)
    std_reward   = np.std(all_rewards)
    min_reward   = np.min(all_rewards)
    max_reward   = np.max(all_rewards)
    avg_steps    = np.mean(all_steps)
    success_rate = (successes / args.episodes) * 100

    print("\n" + "=" * 50)
    print(f"🏆 EVALUATION REPORT: {args.robot.upper()} | {args.algorithm.upper()}")
    print("=" * 50)
    print(f"Total Episodes Tested : {args.episodes}")
    print(f"Success Rate          : {successes}/{args.episodes} ({success_rate:.1f}%)")
    print("-" * 50)
    print(f"Mean Reward           : {mean_reward:.2f}")
    print(f"Reward Std Deviation  : ±{std_reward:.2f} (Stability)")
    print(f"Min / Max Reward      : {min_reward:.2f} / {max_reward:.2f}")
    print(f"Average Episode Length: {avg_steps:.1f} steps")
    print("=" * 50 + "\n")
    print(f"📄 Test log saved  → results/log_test_{run_name}.csv")