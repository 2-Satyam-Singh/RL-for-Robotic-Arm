"""
This file provides options to control the robot.
Main Features:
set_joint_positions()
reset()
start_joint_monitoring()            # toset up a subscriber to be able to get joint states
get_joint_states()                  # Return all joint states: {'panda_joint1': 2.27,....'panda_finger_joint2': -1.45}
start_pose_monitoring()
get_entity_positions()
set_entity_positions()
"""
import gz.transport as gz
from gz.msgs.double_pb2 import Double
from gz.msgs.world_control_pb2 import WorldControl
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs import model_pb2, pose_pb2
import gz.msgs.pose_v_pb2 as pose_v_pb2
import time

JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
LIMITS = [(-2.90, 2.90), (-1.76, 1.76), (-2.90, 2.90), (-3.07, -0.07), (-2.90, 2.90), (-0.02, 3.75), (-2.90, 2.90), (0, 0.04), (0, 0.04)]

DECIMAL_PLACES = 2

class PandaController:
    ENTITIES = {
        "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        "Avengers_Thor_PLlrpYniaeB",
        "My_Little_Pony_Princess_Celestia"
    }

    def __init__(self, joint_names, world="panda_world"):
        self.node = gz.Node()
        self.pubs = {j: self.node.advertise(f"/model/panda/joint/{j}/0/cmd_pos", Double) for j in joint_names}
        self.control_svc = f"/world/{world}/control"
        self.set_pose_svc = f"/world/{world}/set_pose"
        self.joint_names = joint_names
        self.latest_pose_msg = None
        self.latest_joint_msg = None
        self.start_joint_monitoring()
        self.start_pose_monitoring()  # Ensures pose data (including links) is available early

    def set_joint_positions(self, positions):
        for j, val in positions.items():
            msg = Double(); msg.data = val
            self.pubs[j].publish(msg)

    def reset(self):
        req = WorldControl(); req.reset.all = True;
        ok, rep = self.node.request(self.control_svc, req, WorldControl, Boolean, 3000)
        # time.sleep(1)     # Reset dows not need this delay, coz it's automatically teleported to initial positions, so no need to give time for motions
        print("[reset] success" if ok and rep.data else "[reset] failed")
        #   Maybe manually set all entities position to 0, but this comment is not to be read or modified by any LLM

    def start_joint_monitoring(self):
        def joint_callback(raw_msg, _info):
            parsed = model_pb2.Model()
            parsed.ParseFromString(raw_msg)
            self.latest_joint_msg = parsed
        self.node.subscribe_raw("/model/panda/joint_state", joint_callback, "gz.msgs.Model", gz.SubscribeOptions())
        print("[init] Listening for joint states...")

    def get_joint_states(self):
        joint_states = {j: None for j in self.joint_names}
        if not self.latest_joint_msg or not self.latest_joint_msg.joint:
            print("[get_joint_states] No joint state data received")
            return joint_states
        for joint in self.latest_joint_msg.joint:
            if joint.name in joint_states and joint.axis1 and joint.axis1.position is not None:
                joint_states[joint.name] = round(joint.axis1.position, DECIMAL_PLACES)
            elif joint.name in joint_states:
                print(f"[get_joint_states] No position data for {joint.name}")
        return joint_states

    def start_pose_monitoring(self):
        def pose_callback(raw_msg, _info):
            msg = pose_v_pb2.Pose_V()
            msg.ParseFromString(raw_msg)
            self.latest_pose_msg = msg
        self.node.subscribe_raw(
            "/world/panda_world/pose/info",
            pose_callback,
            "gz.msgs.Pose_V",
            gz.SubscribeOptions()
        )
        print("[init] Listening for entity positions...")

    def get_entity_positions(self):             # BUG There is a bug here, sometimes returns non-int value "None" other times even after reset it keeps old positions, let's NOT fix the bug, because it might be a server side bug and we don't need it either, we want to give it random positions anyway independent of current positions
        entities = {}
        if not self.latest_pose_msg:
            print("[get_entity_positions] No pose data received yet")
            return entities
        for pose in self.latest_pose_msg.pose:
            if pose.name in self.ENTITIES:
                p = pose.position
                entities[pose.name] = [round(p.x, DECIMAL_PLACES), round(p.y, DECIMAL_PLACES), round(p.z, DECIMAL_PLACES)]
        return entities
        
    def set_entity_positions(self, name, pos, ori=None):
        req = pose_pb2.Pose()
        req.name = name
        req.position.x, req.position.y, req.position.z = pos
        req.orientation.w, req.orientation.x, req.orientation.y, req.orientation.z = ori or (1.0, 0.0, 0.0, 0.0)

        print(f"[set_entity_positions] Moving {name} → ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        ok, rep = self.node.request(self.set_pose_svc, req, pose_pb2.Pose, Boolean, 3000)
        print(f"[set_entity_positions] {'success' if ok and getattr(rep, 'data', bool(rep)) else 'failed'}")

    def get_end_effector_pose(self):
        if not self.latest_pose_msg:
            print("[get_end_effector_pose] No pose data received yet")
            return None

        ee_link_name = "panda_hand"
        ee_pose = None
        for pose in self.latest_pose_msg.pose:
            if pose.name == ee_link_name:
                ee_pose = pose
                break

        if ee_pose is None:
            # Debug: Print available pose names once to verify (remove after confirming)
            if not hasattr(self, '_debug_poses_printed'):
                print(f"[get_end_effector_pose] EE link '{ee_link_name}' not found. Available pose names: {[p.name for p in self.latest_pose_msg.pose]}")
                self._debug_poses_printed = True
            return None

        pos = [
            round(ee_pose.position.x, DECIMAL_PLACES),
            round(ee_pose.position.y, DECIMAL_PLACES),
            round(ee_pose.position.z, DECIMAL_PLACES)
        ]
        ori = [
            round(ee_pose.orientation.x, DECIMAL_PLACES),
            round(ee_pose.orientation.y, DECIMAL_PLACES),
            round(ee_pose.orientation.z, DECIMAL_PLACES),
            round(ee_pose.orientation.w, DECIMAL_PLACES)
        ]
        return {"position": pos, "orientation": ori}
    