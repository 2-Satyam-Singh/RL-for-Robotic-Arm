# Copyright (C) 2026 Satyam Singh
# SPDX-License-Identifier: AGPL-3.0-or-later

ROBOT_CONFIGS = {
    "panda": {
        "model_name": "panda",
        "world_name": "panda_world",
        "ee_link_name": "panda_hand",
        "joints": [f"panda_joint{i}" for i in range(1, 8)],
        "limits": [
            (-2.90, 2.90), (-1.76, 1.76), (-2.90, 2.90),
            (-3.07, -0.07), (-2.90, 2.90), (-0.02, 3.75), (-2.90, 2.90)
        ],
        "entities": {
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        },
        "workspace_range": 0.85,
        "sdf_launch_cmd": "gz sim sim/serial/panda.sdf"
    },
    
    "3dof": {
        "model_name": "robot_3dof",             
        "world_name": "world_3dof",             
        "ee_link_name": "link3",         
        "joints": ["joint1", "joint2", "joint3"], 
        "limits": [
            (-3.14, 3.14), (-1.57, 1.57), (-1.57, 1.57)
        ],
        "entities": {  
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        },
        "workspace_range": 0.5,
        "sdf_launch_cmd": "gz sim sim/serial/robot_3dof.sdf"
    },

    "4dof": {
        "model_name": "robot_4dof",             
        "world_name": "world_4dof",             
        "ee_link_name": "link4",         
        "joints": ["joint1", "joint2", "joint3", "joint4"], 
        "limits": [
            (-3.14, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57)
        ],
        "entities": {  
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        },
        "workspace_range": 0.60,
        "sdf_launch_cmd": "gz sim sim/serial/robot_4dof.sdf"
    },

    "5dof": {
        "model_name": "robot_5dof",             
        "world_name": "world_5dof",             
        "ee_link_name": "link5",         
        "joints": ["joint1", "joint2", "joint3", "joint4", "joint5"], 
        "limits": [
            (-3.14, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57)
        ],
        "entities": {  
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        },
        "workspace_range": 0.75,
        "sdf_launch_cmd": "gz sim sim/serial/robot_5dof.sdf"
    },

    "6dof": {
        "model_name": "robot_6dof",             
        "world_name": "world_6dof",             
        "ee_link_name": "link6",         
        "joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], 
        "limits": [
            (-3.14, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57)
        ],
        "entities": {  
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        },
        "workspace_range": 0.90,
        "sdf_launch_cmd": "gz sim sim/serial/robot_6dof.sdf"
    },

   # KNOWN ISSUE (as of v0.9 dev): 3t (Cartesian gantry) is not working correctly yet.
   # Training/testing on this robot is unreliable until this is debugged further.
   "3t": {
        "model_name": "robot_3t",             
        "world_name": "world_3t",             
        "ee_link_name": "link_z",         
        "joints": ["joint_x", "joint_y", "joint_z"], 
        "limits": [
            (0.0, 0.8),    # X slider moves out
            (-0.45, 0.45), # Y slider moves left/right
            (-0.7, 0.0)    # Z yellow gripper plunges DOWN the blue rail
        ],
        "entities": {  
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        },
        "workspace_range": 0.8,
        "sdf_launch_cmd": "gz sim sim/cartesian/robot_3t.sdf"
    }
}
