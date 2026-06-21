# reward.py
"""
Reward shaping for RobotEnv (v0_5) - UPDATED VERSION
Massive reward for success, plus a 100-point reward for significant object movement.
"""

from utils.utils import euclidean_distance, clamp, validate_obs

# Module state
_last_entity_pos = None
_last_joint_pos = None
_success_this_ep = False

# Metrics for terminal printing
_total_reward = 0.0
_step_count = 0

# Tunable parameters
GROUND_Z = 0.10
TABLE_Z = 1.025
TOUCH_DISP_THRESH = 0.08  # ← Increased to 8cm to require significant movement

# === YOUR CONTROL HYPERPARAMETERS ===
TOUCH_REWARD = 100.0    # ← 100 points for moving the object >= 8cm in a single step
LIVING_PENALTY = 0.0    
DROP_REWARD = 0.0       
SUCCESS_REWARD = 1000.0 # ← MASSIVE reward only on ground hit

def reset_reward():
    """Clear history for new episode."""
    global _last_entity_pos, _last_joint_pos, _success_this_ep
    global _total_reward, _step_count

    _last_entity_pos = None
    _last_joint_pos = None
    _success_this_ep = False
    _total_reward = 0.0
    _step_count = 0

def _safe_midpoint(joint_idx, joint_limits):
    lo, hi = joint_limits[joint_idx]
    return 0.5 * (lo + hi)

def compute_reward(obs, entities, joint_limits):
    """
    Calculates the reward based on significant movement (touch) and ground hit (success).
    """
    global _last_entity_pos, _last_joint_pos, _success_this_ep
    global _total_reward, _step_count

    obs = validate_obs(obs, [], entities, joint_limits)
    obs_entities = obs["entities"]

    # Find target entity
    target_name = next((name for name in obs_entities if name in entities), None)
    if not target_name:
        _last_entity_pos = None
        _last_joint_pos = obs["joints"]
        return LIVING_PENALTY

    curr_ent = [float(p) if p is not None else 0.0 for p in obs_entities[target_name][:3]]
    curr_z = curr_ent[2]

    if _last_entity_pos is None or _last_joint_pos is None:
        _last_entity_pos = curr_ent[:]
        _last_joint_pos = dict(obs["joints"])
        return 0.0

    # === TOUCH / MOVEMENT REWARD ===
    # Measure how far the object moved this exact step
    ent_disp = euclidean_distance(curr_ent, _last_entity_pos)
    
    # Give the 100 reward ONLY if it moved a significant distance
    touch_reward = TOUCH_REWARD if ent_disp >= TOUCH_DISP_THRESH else 0.0

    # === SUCCESS REWARD ===
    fall_distance = TABLE_Z - curr_z
    success_bonus = 0.0
    if curr_z <= GROUND_Z and not _success_this_ep:
        # We keep the small clamp calculation here just for the vertical drop bonus multiplier
        success_bonus = SUCCESS_REWARD + clamp(fall_distance / (TABLE_Z - GROUND_Z)) * 2.0
        _success_this_ep = True

    # Total = massive success + touch reward
    total = success_bonus + touch_reward + LIVING_PENALTY
    
    # Fixed line to allow rewards over 1.0!
    total = max(total, 0.0)

    # Update state 
    _last_entity_pos = curr_ent[:]
    _last_joint_pos = dict(obs["joints"])

    # Update metrics for printing
    _total_reward += float(total)
    _step_count += 1
    
    # Prints the average reward per step for the current episode
    # (Commented out by default so it doesn't flood your terminal every single step. 
    # Uncomment if you want to see the rapid output).
    # print(f"Average rewards: {_total_reward / _step_count:.4f}")

    return float(total)