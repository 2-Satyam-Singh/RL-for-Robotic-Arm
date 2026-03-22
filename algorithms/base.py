# algorithms/base.py
"""
Base class for RL algorithms, providing standard save/load interface.
Saves models to models/<algo_name>_models/<name>.pth.
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class BaseAgent:
    def __init__(self, env, algo_name, model_base_dir=None):
        """
        Args:
            env: The environment instance.
            algo_name (str): Name of the algorithm (e.g., 'ppo', 'dqn').
            model_base_dir (str, optional): Base directory for models (default: 'models/<algo_name>_models/').
        """
        self.env = env
        self.algo_name = algo_name
        self.model_dir = model_base_dir or f"models/{algo_name}_models/"
        os.makedirs(self.model_dir, exist_ok=True)

    def save(self, name):
        """Save model to self.model_dir/<name>.pth. To be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement save()")

    def load(self, name):
        """Load model from self.model_dir/<name>.pth. To be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement load()")