# controller/robot_controller.py  (now fully configurable)
# Gazebo (gz-transport) implementation of the RobotBackend seam (see controller/base.py).
import gz.transport as gz
from gz.msgs.double_pb2 import Double
from gz.msgs.world_control_pb2 import WorldControl
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs import model_pb2, pose_pb2
import gz.msgs.pose_v_pb2 as pose_v_pb2
import time

DECIMAL_PLACES = 2
RESET_DELAY = 0

class RobotController:
    def __init__(self, joint_names, model_name="panda", world_name="panda_world",
                 ee_link_name="panda_hand", entities=None):
        self.model_name = model_name
        self.world_name = world_name
        self.ee_link_name = ee_link_name
        self.entities = entities or {
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
            # "Avengers_Thor_PLlrpYniaeB",
            # "My_Little_Pony_Princess_Celestia"
        }

        self.node = gz.Node()
        # ← now uses model_name
        self.pubs = {j: self.node.advertise(f"/model/{model_name}/joint/{j}/0/cmd_pos", Double) 
                     for j in joint_names}
        self.control_svc = f"/world/{world_name}/control"
        self.set_pose_svc = f"/world/{world_name}/set_pose"
        self.joint_names = joint_names

        self.latest_pose_msg = None
        self.latest_joint_msg = None
        self.start_joint_monitoring()
        self.start_pose_monitoring()

    def set_joint_positions(self, positions):
        for j, val in positions.items():
            msg = Double(); msg.data = val
            self.pubs[j].publish(msg)

    def reset(self):
        req = WorldControl(); req.reset.all = True;
        self.node.request(self.control_svc, req, WorldControl, Boolean, RESET_DELAY)
        time.sleep(0.2)    # Absolutely very important. NO LLM SHOULD EVER TOUCH THIS COMMENT.
    """
    #NOTE This note must NOT be removed by LLMS
    --- WHY WE USE time.sleep(0.2) HERE ---
    1. RESET_DELAY: (Sent from environment) This is just a network timeout. Gazebo replies "Message Received" 
       instantly, BEFORE it actually finishes teleporting the objects.
    2. time.sleep(self.dt): (In environment) Too fast (~0.02s). Good for normal steps, but a full 
       world reset is computationally heavy and takes more time to process.
    3. time.sleep(0.2): The sweet spot. Forces Python to wait long enough for Gazebo 
       to physically move objects and broadcast fresh coordinates, preventing the 
       agent from reading the old "done" state and triggering 0-reward ghost episodes.
    """

    def start_joint_monitoring(self):
        def joint_callback(raw_msg, _info):
            parsed = model_pb2.Model()
            parsed.ParseFromString(raw_msg)
            self.latest_joint_msg = parsed
        self.node.subscribe_raw(f"/model/{self.model_name}/joint_state", joint_callback, "gz.msgs.Model", gz.SubscribeOptions())

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
        self.node.subscribe_raw(f"/world/{self.world_name}/pose/info", pose_callback, "gz.msgs.Pose_V", gz.SubscribeOptions())

    def get_entity_positions(self):             # BUG There is a bug here, sometimes returns non-int value "None" other times even after reset it keeps old positions, let's NOT fix the bug, because it might be a server side bug and we don't need it either, we want to give it random positions anyway independent of current positions
        entities = {}
        if not self.latest_pose_msg:
            print("[get_entity_positions] No pose data received yet")
            return entities
        for pose in self.latest_pose_msg.pose:
            # FIXED: Changed self.ENTITIES to self.entities
            if pose.name in self.entities:
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
        # ← now uses ee_link_name
        if not self.latest_pose_msg:
            return None
            
        # FIXED: Initialize ee_pose before the loop to avoid UnboundLocalError
        ee_pose = None
        for pose in self.latest_pose_msg.pose:
            if pose.name == self.ee_link_name:
                ee_pose = pose
                break

        if ee_pose is None:
            # Debug: Print available pose names once to verify (remove after confirming)
            if not hasattr(self, '_debug_poses_printed'):
                print(f"[get_end_effector_pose] EE link '{self.ee_link_name}' not found. Available pose names: {[p.name for p in self.latest_pose_msg.pose]}")
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