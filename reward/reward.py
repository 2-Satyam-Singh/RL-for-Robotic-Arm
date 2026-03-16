# reward/reward_0_5.py
"""
Reward shaping for PandaEnv (v0_5) - Push entity off table to ground.
Design goals:
- Reward initial contact/displacement (moderate, no early termination).
- High reward for dropping entity off table (proportional to drop distance/speed).
- Bonus for success (entity on ground, z <= GROUND_Z).
- Encourage large actions/efficiency: Scale movement by entity effect, small step penalty.
- Penalize unnecessary actions via living penalty and mobility factor.
- No high reward on touch alone; allow fall time.

Stateful: Tracks last positions for deltas.
Call reset_reward() in env.reset().
"""

from utils.utils import euclidean_distance, clamp, validate_obs

# Module state
_last_entity_pos = None
_last_joint_pos = None

# Tunable parameters
GROUND_Z = 0.10  # z <= this = success (on ground)
TABLE_Z = 1.025  # Approximate table height (from Gazebo poses)
TOUCH_DISP_THRESH = 0.02  # Min displacement for "touch" (meters)
DROP_SCALE = 2.0  # Reward per meter of vertical drop
SPEED_BONUS_THRESH = 0.1  # Min drop speed (m/step) for bonus
MOVEMENT_SCALE = 0.5  # Weight for large joint movements
LIVING_PENALTY = -0.01  # Small per-step penalty for efficiency
TOUCH_REWARD = 0.5  # Moderate for initial contact (avoids early done)
DROP_REWARD = 2.0  # Base for significant drop
SUCCESS_REWARD = 5.0  # High for ground hit

def reset_reward():
    """Clear history for new episode."""
    global _last_entity_pos, _last_joint_pos
    _last_entity_pos = None
    _last_joint_pos = None

def _safe_midpoint(joint_idx, joint_limits):
    """Midpoint if joint value missing."""
    lo, hi = joint_limits[joint_idx]
    return 0.5 * (lo + hi)

def _compute_drop_speed(curr_z, last_z, dt=0.1):
    """Drop speed (m/s), positive downward."""
    if last_z is None:
        return 0.0
    return clamp((last_z - curr_z) / dt, max_val=1.0)

def compute_reward(obs, entities, joint_limits):
    """
    Compute shaped reward for pushing entity off table.
    Args:
        obs: dict {"joints": {name: val}, "entities": {name: [x,y,z]}}
        entities: list of entity names
        joint_limits: list of (low, high) tuples
    Returns:
        float: Reward in [0, 5+], high for ground success.
    """
    global _last_entity_pos, _last_joint_pos
    obs = validate_obs(obs, [], entities, joint_limits)
    obs_entities = obs["entities"]

    # Find target entity (first match)
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
        return 0.0  # No reward on first step

    # Joint movement (encourage large changes, normalized)
    total_norm_move = sum(
        abs(float(obs["joints"].get(jn, _safe_midpoint(idx, joint_limits))) -
            float(_last_joint_pos.get(jn, _safe_midpoint(idx, joint_limits)))) / max(1e-4, hi - lo)
        for idx, (jn, (lo, hi)) in enumerate(zip(list(obs["joints"].keys()), joint_limits))
    )
    movement_norm = clamp(total_norm_move / len(joint_limits), 0.0, 2.0)  # Up to 2x for large moves
    movement_reward = MOVEMENT_SCALE * movement_norm

    # Entity displacement and drop
    ent_disp = euclidean_distance(curr_ent, _last_entity_pos)
    vert_drop = max(0.0, _last_entity_pos[2] - curr_z)
    drop_speed = _compute_drop_speed(curr_z, _last_entity_pos[2])

    # Mobility factor: Scale movement by entity effect (prefer effective large actions)
    mobility_factor = clamp(0.2 + 0.8 * (ent_disp / max(TOUCH_DISP_THRESH, ent_disp)))

    # Touch/displacement reward (moderate)
    touch_reward = TOUCH_REWARD if ent_disp >= TOUCH_DISP_THRESH else 0.0

    # Drop reward: Proportional to drop + speed bonus
    drop_reward = DROP_REWARD * clamp(vert_drop / (TABLE_Z - GROUND_Z)) + drop_speed * 1.0

    # Success: On ground, bonus for distance fallen
    fall_distance = TABLE_Z - curr_z
    success_bonus = SUCCESS_REWARD if curr_z <= GROUND_Z else 0.0
    if success_bonus > 0:
        success_bonus += clamp(fall_distance / (TABLE_Z - GROUND_Z)) * 2.0  # Extra for full fall

    # Total: Movement (scaled), touch, drop, success + living penalty
    total = (
        movement_reward * mobility_factor +
        touch_reward +
        drop_reward +
        success_bonus +
        LIVING_PENALTY
    )
    total = clamp(total, 0.0)  # No negatives

    # Update state
    _last_entity_pos = curr_ent[:]
    _last_joint_pos = dict(obs["joints"])

    return float(total)