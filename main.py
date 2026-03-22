# main.py
import argparse
from config import ROBOT_CONFIGS
from scripts.train import train_agent
from scripts.test import test_agent

def main():
    parser = argparse.ArgumentParser(description="Main entry point for RL Robotic Arm")
    
    # Mode selection
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Run training or testing")
    
    # Environment & Algo arguments
    parser.add_argument("--robot", choices=["panda", "3dof", "5dof"], default="panda")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--reward_type", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Hyperparameters
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100)
    
    # Testing specific
    parser.add_argument("--model_name", type=str, default="", help="Name of saved model (required for test mode)")

    args = parser.parse_args()

    # Load configuration for the selected robot
    cfg = ROBOT_CONFIGS[args.robot]
    
    print(f"🚀 Running in {args.mode.upper()} mode for {args.robot.upper()}")
    print(f"   Launch simulation with:\n   {cfg['sdf_launch_cmd']}\n")

    # Dispatch to the correct script
    if args.mode == "train":
        train_agent(cfg, args)
    elif args.mode == "test":
        if not args.model_name:
            print("Error: --model_name is required when running in test mode.")
            return
        test_agent(cfg, args)

if __name__ == "__main__":
    main()