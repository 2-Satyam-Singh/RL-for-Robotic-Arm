# test.py
"""
Evaluation script for a trained PPO agent in the PandaEnv.
Loads a saved model and runs it for a few episodes without training.
"""

import time
import torch
from environment.environment import PandaEnv, JOINTS, LIMITS
from algorithms.ppo import PPOAgent

# Match the exact settings you used during training
JOINTS = JOINTS[:-2]      
LIMITS = LIMITS[:-2]
ENTITIES = {
    "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure"
}
WORKSPACE_RANGE = 0.85  
MAX_STEPS = 25

def main():
    print("Initializing environment...")
    env = PandaEnv(JOINTS, LIMITS, ENTITIES, workspace_range=WORKSPACE_RANGE, max_steps=MAX_STEPS)
    
    print("Loading trained PPO Agent...")
    algo = PPOAgent(env)
    
    # BaseAgent automatically looks in 'models/ppo_models/' and adds the '.pth'
    algo.load("episode_10000") 
    
    # Put the neural network in evaluation mode
    algo.ac.eval()

    num_test_episodes = 100
    successes = 0

    print("\nStarting evaluation...\n")
    for ep in range(num_test_episodes):
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
        
        # If your success reward is 1000, we can easily track the win rate
        if total_reward >= 1000.0:
            successes += 1

    print(f"\nEvaluation Complete!")
    print(f"Success Rate: {successes}/{num_test_episodes} ({(successes/num_test_episodes)*100:.1f}%)")

if __name__ == "__main__":
    main()