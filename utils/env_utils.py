# env_utils.py
import numpy as np

def normalize_obs(obs, joints, limits, entities, workspace_range):
    """
    Convert dict observation -> flat normalized vector.
    - Joint angles normalized by joint limits to [-1, 1].
    - Entity positions normalized by workspace_range to [-1, 1].
    """
    # Validate observation
    for j in joints:
        if j not in obs["joints"] or obs["joints"][j] is None:
            print(f"[Warning] Joint {j} missing or None in observation, using 0.0")
            obs["joints"][j] = 0.0
    for e in entities:
        if e not in obs["entities"] or obs["entities"][e] is None:
            print(f"[Warning] Entity {e} missing or None in observation, using [0, 0, 0]")
            obs["entities"][e] = [0.0, 0.0, 0.0]

    # Normalize joints
    joint_vals = []
    for (j, (low, high)) in zip(joints, limits):
        val = obs["joints"].get(j, 0.0)
        if high > low:
            norm_val = 2.0 * (val - low) / (high - low) - 1.0  # scale to [-1, 1]
            norm_val = float(np.clip(norm_val, -1.0, 1.0))     # keep obs truthful to the Box
        else:
            norm_val = 0.0
        joint_vals.append(norm_val)

    # Normalize entities
    entity_vals = []
    for e in entities:
        pos = obs["entities"].get(e, [0.0, 0.0, 0.0])
        entity_vals.extend([max(-1.0, min(1.0, p / workspace_range)) for p in pos])

    flat_obs = np.array(joint_vals + entity_vals, dtype=np.float32)
    return flat_obs


def denormalize_action(action, joints, limits):
    if isinstance(action, dict):
        return action  # Already raw
    denorm_vals = []
    for val, (low, high) in zip(action, limits):
        denorm_vals.append(low + (val + 1.0) * (high - low) / 2.0 if high > low else low)
    return dict(zip(joints, denorm_vals))


def action_to_dict(action, joints):
    """
    Convert action (array or dict) to dict expected by RobotController.
    If array, assume same order as joints.
    """
    if isinstance(action, dict):
        return {j: action.get(j, 0.0) for j in joints}
    elif isinstance(action, (list, np.ndarray)):
        if len(action) != len(joints):
            raise ValueError(f"Action array length {len(action)} does not match joints {len(joints)}")
        return dict(zip(joints, action))
    else:
        raise ValueError(f"Unsupported action type: {type(action)}")