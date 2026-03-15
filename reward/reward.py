# reward/reward_0_3.py
"""
Reward shaping for PandaEnv.

Design goals (summary):
- Encourage the robot to move (reduce "hesitation").
- Prefer movement that affects the entity (so robot doesn't just flail to get movement reward).
- Give higher reward for touching/displacing the entity.
- Highest reward when the entity is thrown to the ground (success).
- Keep a small structure so you can tune weights/thresholds easily.

This module is stateful: it keeps last-step joint/entity poses to compute deltas.
Call reset_reward() when the environment is reset (PandaEnv.reset()) so the history clears.
"""

from math import sqrt

# ---- module state (last-step info) ----
_last_entity_pos = None   # e.g. [x,y,z] for the tracked target entity (first matching entity)
_last_joint_pos = None    # dict: {joint_name: value}

# ---- tunable parameters (small, understandable) ----
GROUND_Z = 0.10                 # z <= this => considered "thrown off table / on ground"
TOUCH_DISP_THRESH = 0.02        # entity displacement >= 2 cm -> treat as 'touch/move'
DROP_NORM = 0.20                # normalize significant drop amount (meters)
# weights
W_MOVEMENT = 0.30               # reward weight for joint movement (when it helps the object)
W_DROP     = 1.0                # reward weight for vertical drop of the object
W_TOUCH    = 1.0                # reward for touching / displacing the object
W_THROW    = 5.0                # big reward for throwing object to ground (success)
STEP_PENALTY = 0.0              # optional small step penalty (negative) to discourage pointless moves

# ---- helper functions ----
def _euclid(a, b):
    if a is None or b is None:
        return 0.0
    return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def reset_reward():
    """Clear internal last-step memory. Call this in PandaEnv.reset()."""
    global _last_entity_pos, _last_joint_pos
    _last_entity_pos = None
    _last_joint_pos = None

def _safe_midpoint(joint_name, joint_idx, joint_limits):
    """Return midpoint of joint range if value missing."""
    lo, hi = joint_limits[joint_idx]
    return 0.5 * (lo + hi)

# ---- main API ----
def compute_reward(obs, target_entities, joint_limits=None):
    """
    Compute shaped reward.

    Args:
      obs: dict with keys "joints" (dict joint_name->value) and "entities" (dict name->[x,y,z]).
      target_entities: set/list of entity names to consider (same as before).
      joint_limits: optional list of (low,high) tuples in same order as env.joints.
                    If provided, movement normalization is better. If not provided,
                    movement normalization will be approximate.

    Returns:
      float reward >= 0 (higher is better). Very large reward W_THROW returned
      when entity hits ground (curr_z <= GROUND_Z).
    """

    global _last_entity_pos, _last_joint_pos

    # Quick guards
    if not obs or "entities" not in obs:
        # If no entity data, give zero (or a tiny movement reward if you prefer)
        _last_entity_pos = None
        _last_joint_pos = obs.get("joints", {}) if obs else None
        return 0.0

    # --- pick the target entity (first one found in target_entities) ---
    entities = obs.get("entities", {})
    target_name = None
    for name in entities:
        if name in target_entities:
            target_name = name
            break

    if target_name is None:
        # no tracked entity found
        _last_entity_pos = None
        _last_joint_pos = obs.get("joints", {})
        return 0.0

    curr_ent = entities.get(target_name)
    # If entity pose is missing or malformed, bail
    if not curr_ent or len(curr_ent) < 3:
        _last_entity_pos = None
        _last_joint_pos = obs.get("joints", {})
        return 0.0

    # Convert to (x,y,z) floats
    curr_ent = [float(curr_ent[0]), float(curr_ent[1]), float(curr_ent[2])]
    curr_z = curr_ent[2]

    # If this is the first call (no history) — initialize memory and return 0
    if _last_entity_pos is None or _last_joint_pos is None:
        _last_entity_pos = list(curr_ent)
        _last_joint_pos = dict(obs.get("joints", {}))
        return 0.0

    # --- Joint movement magnitude (normalized) --------------------------------
    # We want "movement reward" to discourage inaction, **but** we only want it to
    # matter if the movement actually affects the object. So we 1) compute
    # normalized joint motion, then 2) scale it by how much the entity moved.
    curr_joints = obs.get("joints", {})
    # compute normalized movement: sum over joints of abs(delta)/range, averaged
    total_norm = 0.0
    n_joints = 0
    for idx, (jn, last_val) in enumerate(_last_joint_pos.items()):
        n_joints += 1
        curr_val = curr_joints.get(jn)
        if curr_val is None:
            # fallback to midpoint if missing and joint_limits provided
            if joint_limits is not None and idx < len(joint_limits):
                curr_val = _safe_midpoint(jn, idx, joint_limits)
            else:
                curr_val = last_val if last_val is not None else 0.0
        lo_hi = (None if joint_limits is None or idx >= len(joint_limits) else joint_limits[idx])
        if lo_hi is not None:
            lo, hi = lo_hi
            rng = max(1e-4, hi - lo)
        else:
            # generic fallback range
            rng = 1.0
        total_norm += abs(curr_val - (last_val if last_val is not None else curr_val)) / rng

    movement_norm = (total_norm / max(1, n_joints))  # average normalized change, ~0..(maybe >1)
    # clamp to 0..1 for safety
    if movement_norm < 0.0: movement_norm = 0.0
    if movement_norm > 1.0: movement_norm = 1.0

    # --- Entity displacement & vertical drop ----------------------------------
    ent_disp = _euclid(curr_ent, _last_entity_pos)   # meters
    vert_drop = max(0.0, _last_entity_pos[2] - curr_z)  # positive if entity moved down

    # --- mobility_factor: scale movement reward by how much the object moved ---
    # If the object didn't move, movement is less valuable. This prevents the agent
    # from just flailing to accumulate movement reward without affecting the object.
    # We keep a small base factor (0.1) so some exploratory motion is still rewarded.
    mobility_factor = 0.1 + 0.9 * min(1.0, ent_disp / max(1e-6, TOUCH_DISP_THRESH))

    # --- reward components ----------------------------------------------------
    movement_reward = W_MOVEMENT * movement_norm * mobility_factor
    # drop_reward: reward proportional to how much the object dropped this step (clamped)
    drop_reward = W_DROP * min(1.0, vert_drop / DROP_NORM)
    # touch_reward: discrete bonus if the object was noticeably displaced (touch/interaction)
    touch_reward = W_TOUCH if ent_disp >= TOUCH_DISP_THRESH else 0.0
    # throw_reward: success condition if object hits floor / ground (big reward)
    if curr_z <= GROUND_Z:
        # If object is on the ground, give the largest reward. We still add other terms
        # to keep numeric continuity, but the throw reward dominates.
        total = W_THROW + movement_reward + drop_reward + touch_reward + STEP_PENALTY
    else:
        total = movement_reward + drop_reward + touch_reward + STEP_PENALTY

    # clamp (no negative rewards unless STEP_PENALTY negative)
    if total < 0.0:
        total = 0.0

    # --- update history for next step and return --------------------------------
    _last_entity_pos = list(curr_ent)
    _last_joint_pos = dict(curr_joints)

    return float(total)
