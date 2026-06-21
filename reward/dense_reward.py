# reward/dense_reward.py
"""
Reward shaping for RobotEnv - DENSE VERSION
Uses Potential-Based Reward Shaping to guide the agent to the object,
push it, and drop it, without allowing it to "farm" points by shaking.
"""

from utils.utils import euclidean_distance, clamp, validate_obs

# Module state
_last_ee_dist = None
_last_obj_pos = None
_success_this_ep = False

# Metrics for terminal printing
_total_reward = 0.0
_step_count = 0

# Tunable parameters
GROUND_Z = 0.10
TABLE_Z = 1.025

# === DENSE WEIGHTS (RE-SCALED FOR PPO) ===
W_REACH = 10.0      # Scaled down. Total reaching reward will cap around ~10-15
W_MOVE = 20.0       # Scaled down. Total moving/dropping reward will cap around ~40-50
SUCCESS_REWARD = 100.0  # Scaled down from 1000. Still the ultimate goal!

def reset_reward():
    """Clear history for new episode."""
    global _last_ee_dist, _last_obj_pos, _success_this_ep
    global _total_reward, _step_count

    _last_ee_dist = None
    _last_obj_pos = None
    _success_this_ep = False
    _total_reward = 0.0
    _step_count = 0

def compute_reward(obs, entities, joint_limits):
    """
    Calculates a dense, potential-based reward to guide the agent smoothly.
    """
    global _last_ee_dist, _last_obj_pos, _success_this_ep
    global _total_reward, _step_count

    obs = validate_obs(obs, [], entities, joint_limits)
    obs_entities = obs["entities"]

    # 1. Find Target Object
    target_name = next((name for name in obs_entities if name in entities), None)
    if not target_name:
        return 0.0

    curr_obj_pos = [float(p) if p is not None else 0.0 for p in obs_entities[target_name][:3]]
    curr_z = curr_obj_pos[2]

    # 2. Find End Effector (Gripper) Position
    # FIXED: Extract from the "position" key of the dictionary returned by the controller
    if "ee_pose" in obs and obs["ee_pose"] is not None:
        curr_ee_pos = [float(p) for p in obs["ee_pose"]["position"]]
    else:
        # Fallback just in case it's missing or commented out
        curr_ee_pos = [0.0, 0.0, 0.0] 
        print("This should never happem. This reaults in always end_effector position assumed to be 0, hence no reward ever.")

    # Calculate current distance between gripper and object
    curr_ee_dist = euclidean_distance(curr_ee_pos, curr_obj_pos)

    # Initialize history on the first step of the episode
    if _last_obj_pos is None or _last_ee_dist is None:
        _last_obj_pos = curr_obj_pos[:]
        _last_ee_dist = curr_ee_dist
        return 0.0

    # ==========================================
    # REWARD 1: DENSE REACHING
    # ==========================================
    # Positive if it got closer, negative if it moved away.
    reach_reward = (_last_ee_dist - curr_ee_dist) * W_REACH

    # ==========================================
    # REWARD 2: DENSE PUSHING & DROPPING
    # ==========================================
    # Reward any horizontal XY movement (pushing) and downward Z movement (falling)
    xy_dist_moved = euclidean_distance(curr_obj_pos[:2], _last_obj_pos[:2])
    z_dist_dropped = max(0.0, _last_obj_pos[2] - curr_z) 
    
    # We multiply z_drop by 2 so falling off the table is highly incentivized
    move_reward = (xy_dist_moved + (z_dist_dropped * 2.0)) * W_MOVE

    # ==========================================
    # REWARD 3: SPARSE SUCCESS JACKPOT
    # ==========================================
    success_bonus = 0.0
    if curr_z <= GROUND_Z and not _success_this_ep:
        success_bonus = SUCCESS_REWARD
        _success_this_ep = True

    # Combine all rewards
    total = reach_reward + move_reward + success_bonus

    # Update history for the next step's math
    _last_obj_pos = curr_obj_pos[:]
    _last_ee_dist = curr_ee_dist

    # Update metrics
    _total_reward += float(total)
    _step_count += 1

    return float(total)
