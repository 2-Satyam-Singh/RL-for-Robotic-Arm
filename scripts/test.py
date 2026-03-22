# scripts/test.py
"""
Evaluation script for a trained RL agent.
Called by main.py. Loads a saved model and runs it for a few episodes without training.
"""

import time
import torch
import random
import numpy as np

from environment.environment import PandaEnv
from algorithms.ppo import PPOAgent
from algorithms.dqn import DQNAgent

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
    
    # 1. Apply the seed
    set_seed(args.seed)

    print(f"Initializing {args.robot.upper()} environment...")
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
    
    # --- SAFETY NET: Clean the string in case the user typed '.pth' ---
    clean_model_name = args.model_name.replace(".pth", "")
    
    print(f"Loading trained {args.algorithm.upper()} Agent '{clean_model_name}'...")
    algo = make_algo(args.algorithm, env)
    
    # BaseAgent automatically looks in 'models/' (or dqn equivalent)
    algo.load(clean_model_name) 
    
    # Put the neural network in evaluation mode
    if hasattr(algo, 'ac'):
        algo.ac.eval()
    elif hasattr(algo, 'q_net'):
        algo.q_net.eval()

    successes = 0

    print("\nStarting evaluation...\n")
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # 1. Ask the trained network for the best action
            action = algo.select_action(obs)
            
            # 2. Step the environment
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Optional: Add a tiny sleep so the simulation doesn't run too fast to watch
            # time.sleep(0.05) 

        print(f"[Test Episode {ep + 1}] Total Reward: {total_reward:.2f}")
        
        # Check success condition (adjust thresholds based on your reward function)
        if args.reward_type == "sparse" and total_reward > 0.0:
            successes += 1
        elif args.reward_type == "dense" and total_reward >= 1000.0:
            successes += 1

    print(f"\n✅ Evaluation Complete!")
    print(f"Success Rate: {successes}/{args.episodes} ({(successes/args.episodes)*100:.1f}%)")