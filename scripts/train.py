# scripts/train.py
import random, csv, os
from datetime import datetime
import numpy as np
import torch

from environment.environment import PandaEnv
from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent
from utils.logger import Logger

def set_seed(seed):
    """Sets the seed for perfectly reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_algo(name, env):
    return {"dqn": DQNAgent, "ppo": PPOAgent}[name.lower()](env)

def train_agent(cfg, args):
    """Called by main.py. Runs the training loop."""

    set_seed(args.seed)

    # --- UNIQUE IDENTIFICATION ---
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M")
    run_name = f"{args.robot}_{args.algorithm}_{args.reward_type}_s{args.seed}_{ts}"

    env = PandaEnv(
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

    algo = make_algo(args.algorithm, env)

    # --- GATHER HYPERPARAMETERS FOR PLOT ---
    algo_info = {
        "LR": algo.opt.param_groups[0]['lr'],
        "Gamma": algo.gamma,
        "Entropy Coeff": getattr(algo, "entropy_coef", "N/A"),
        "Seed": args.seed,
        "Max Steps": args.max_steps,
        "DOF": len(cfg["joints"])
    }

    log = Logger(
        run_name=run_name,
        algo_info=algo_info,
        joints=cfg["joints"],
        limits=cfg["limits"],
        entities=cfg["entities"],
        plot_every=10,
        workspace_range=cfg["workspace_range"],
        algo_name=args.algorithm.upper(),
        env_name=args.robot.upper(),
        mode="train"                        # <-- produces log_train_... and train_...
    )

    # --- TRACKER FOR ROLLING CHECKPOINTS ---
    last_saved_model = None

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        ep_data = []

        while not done:
            action = algo.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            algo.store_transition(obs, action, reward, next_obs, done)

            if args.algorithm == "ppo":
                if algo.buffer_size >= algo.rollout_steps:
                    algo.learn()
            else:
                algo.learn()

            flat = log.flatten_state_action(obs, action, cfg["joints"], cfg["entities"])
            ep_data.append([ep, env.step_count, reward, int(done)] + flat)
            obs = next_obs

        print(f"[{args.algorithm.upper()} | {args.robot.upper()} | ep {ep}] total_reward={total_reward:.2f}")
        log.log_episode(ep, ep_data, total_reward)

        # --- ROLLING CHECKPOINT LOGIC ---
        if (ep + 1) % args.save_interval == 0:
            current_save_name = f"{run_name}_ep{ep+1}"

            # 1. Save new model
            algo.save(current_save_name)

            # 2. Delete old model to save disk space
            if last_saved_model:
                try:
                    model_dir = getattr(algo, "model_dir", "models")
                    old_path = os.path.join(model_dir, f"{last_saved_model}.pth")
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except Exception as e:
                    print(f"Warning: Could not delete old checkpoint: {e}")

            # 3. Update tracker
            last_saved_model = current_save_name

    print("✅ Training finished!")