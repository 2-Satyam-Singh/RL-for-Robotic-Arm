# utils/utils.py
import logging
from math import sqrt

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

def validate_obs(obs, joints, entities, joint_limits):
    """
    Validate observation, filling missing/None values.
    Args:
        obs (dict): {"joints": {name: value}, "entities": {name: [x,y,z]}, "ee_pose": ...}
        joints (list): List of joint names
        entities (list): List of entity names
        joint_limits (list): List of (low, high) tuples
    Returns:
        dict: Validated observation with defaults
    """
    obs = obs or {"joints": {}, "entities": {}}
    obs["joints"] = obs.get("joints", {})
    obs["entities"] = obs.get("entities", {})

    for j in joints:
        if j not in obs["joints"] or obs["joints"][j] is None:
            logging.warning(f"Joint {j} missing or None, using 0.0")
            obs["joints"][j] = 0.0

    for e in entities:
        if e not in obs["entities"] or obs["entities"][e] is None:
            logging.warning(f"Entity {e} missing or None, using [0, 0, 0]")
            obs["entities"][e] = [0.0, 0.0, 0.0]

    # Ensure ee_pose exists so downstream tasks don't break
    if "ee_pose" not in obs or obs["ee_pose"] is None:
        obs["ee_pose"] = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}

    return obs

def euclidean_distance(a, b):
    """Compute Euclidean distance between two 3D points."""
    if a is None or b is None:
        return 0.0
    return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def clamp(val, min_val=0.0, max_val=1.0):
    """Clamp value between min_val and max_val."""
    return max(min_val, min(val, max_val))