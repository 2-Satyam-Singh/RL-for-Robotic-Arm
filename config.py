# config.py

ROBOT_CONFIGS = {
    "panda": {
        "model_name": "panda",                  # <model name="panda"> in your SDF
        "world_name": "panda_world",
        "ee_link_name": "panda_hand",
        "joints": [f"panda_joint{i}" for i in range(1, 8)],  # 7 arm joints
        "limits": [
            (-2.90, 2.90), (-1.76, 1.76), (-2.90, 2.90),
            (-3.07, -0.07), (-2.90, 2.90), (-0.02, 3.75), (-2.90, 2.90)
        ],
        "entities": {
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
            # "Avengers_Thor_PLlrpYniaeB",
            # "My_Little_Pony_Princess_Celestia"
        },
        "workspace_range": 0.85,
        "sdf_launch_cmd": "gz sim RL-for-Robotic-Arm/sim/model.sdf"
    },
    
    "3dof": {
        "model_name": "robot_3dof",             
        "world_name": "world_3dof",             
        "ee_link_name": "link_ee_3dof",         
        "joints": ["joint1", "joint2", "joint3"], 
        "limits": [
            (-3.14, 3.14), (-1.57, 1.57), (-3.14, 3.14)  # <-- FIXED: Replaced low1/high1 with actual numbers
        ],
        "entities": {  
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
            # "Avengers_Thor_PLlrpYniaeB",
            # "My_Little_Pony_Princess_Celestia"
        },
        "workspace_range": 0.5,                 
        "sdf_launch_cmd": "gz sim path/to/your_3dof_robot.sdf"
    },

    "5dof": {
        "model_name": "robot_5dof",             
        "world_name": "world_5dof",             
        "ee_link_name": "link_ee_5dof",         
        "joints": ["joint1", "joint2", "joint3", "joint4", "joint5"], 
        "limits": [
            (-3.14, 3.14), (-1.57, 1.57), (-3.14, 3.14), 
            (-1.57, 1.57), (-3.14, 3.14)
        ],
        "entities": {  
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
            # "Avengers_Thor_PLlrpYniaeB",
            # "My_Little_Pony_Princess_Celestia"
        },
        "workspace_range": 0.65,                
        "sdf_launch_cmd": "gz sim path/to/your_5dof_robot.sdf"
    }
}