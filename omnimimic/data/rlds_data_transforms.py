from typing import Any, Dict
from functools import partial
from .data_utils import *
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import rlds

GNM_NORMALIZE_DICT = {
    'recon': 0.25,
    'scand': 0.38,
    'tartan_drive': 0.72,
    'go_stanford': 0.12,
    'cory_hall': 0.06,
    'seattle': 0.35,
    'sacson': 0.255
}


def gnm_dataset_transform(
    trajectory: Dict[str, Any],
    config,
    min_goal_distance=20
) -> Dict[str, Any]:
    #Decode Image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"]["image"])

    #Relabel actions
    traj_len = tf.shape(trajectory["action"])[0]
    len_seq_pred = config['seq_length']

    #Pad trajectory states
    padding = tf.tile(trajectory['observation']['state'][-1:, :], 
            [len_seq_pred, 1])
    trajectory['observation']['state'] = tf.concat(
            (trajectory['observation']['state'], padding),
            axis=0)

    #Get next len_seq_pred indices
    indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(1, len_seq_pred + 1)
    global_waypoints = tf.gather(trajectory['observation']['state'], indices)[:, :, :2]

    #Get current position indices
    curr_pos_indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(0, len_seq_pred)
    if config['gnm_delta_actions']:
        curr_pos = tf.gather(trajectory['observation']['state'], curr_pos_indices)[:, :, :2] #delta waypoints
    else:
        curr_pos = tf.expand_dims(trajectory['observation']['state'][:traj_len, :2], 1) #waypoints
    global_waypoints -= curr_pos
    global_waypoints = tf.expand_dims(global_waypoints, 2)
    trajectory["action_sequence"] = tf.squeeze(tf.linalg.matmul(global_waypoints, tf.expand_dims(
        trajectory['observation']['yaw_rotmat'][:, :2, :2], 1)), 2)
    
    #Transform into manipulation coords
    trajectory["action_sequence"] = tf.concat((
            tf.zeros((traj_len, len_seq_pred, 1), dtype=tf.float64),
            trajectory["action_sequence"][:,:, 1:2],
            -trajectory["action_sequence"][:,:, 0:1],
            tf.zeros((traj_len, len_seq_pred, 4), dtype=tf.float64)
            ), axis=-1)
    trajectory["action"] = trajectory["action_sequence"][:, 0, :]

    #Get normalization factor
    normalization_factor = 1.0
    for dataset_name, value in GNM_NORMALIZE_DICT.items():
        if tf.strings.regex_full_match(trajectory['traj_metadata']['episode_metadata']['file_path'][0],
             f'.*{dataset_name}.*'):
            normalization_factor = value
    normalization_factor = tf.cast(normalization_factor, tf.float64)
    trajectory["action_sequence"] /= normalization_factor
    
    #Relabel random goals
    high, low, goal_idxs = get_goal_idxs(traj_len, min_goal_distance_indices=0, max_goal_distance_indices=20)
    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    trajectory["goal_state"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["state"]
        )[:, :2]
    trajectory["goal_idxs"] = goal_idxs
    trajectory["goal_state"] -= trajectory['observation']['state'][:traj_len, :2]
    trajectory["goal_state"] = tf.expand_dims(trajectory["goal_state"], 1)
    trajectory["goal_state"] = tf.squeeze(tf.linalg.matmul(trajectory["goal_state"],
        trajectory['observation']['yaw_rotmat'][:, :2, :2]), 1)
    trajectory["goal_state"] /= normalization_factor
    trajectory["goal_distance_indices"] = tf.cast(goal_idxs, tf.float32) - tf.cast(tf.range(traj_len), tf.float32) 
   
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 0
    return trajectory


def polybot_dataset_transform(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:

    #Decode Image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"]["image"])

    #Relabel random goals
    traj_len = tf.shape(trajectory["action"])[0]
    high, low, goal_idxs = get_goal_idxs(traj_len)
    '''
    trajectory["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["image"]
    )
    trajectory["goal_state"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        trajectory["observation"]["state"]
    )[:, 1:3]
    '''
    trajectory["goal_idxs"] = goal_idxs
    #trajectory["goal_pose"] = tf.nest.map_structure(
    #    lambda x: tf.gather(x, goal_idxs),
    #    trajectory["observation"]["cartesian_pose"]
    #)
    #trajectory["goal_distance"] = torch_linalg.norm(
    #    trajectory["goal_pose"][:, :3] - trajectory["observation"]["cartesian_pose"][:, :3],
    #    dim=-1
    #)
    trajectory["goal_distance_indices"] = tf.cast(goal_idxs - low, tf.float32) 
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 1
    return trajectory


def bridge_dataset_transform(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:

    #Set image key
    image_keys = []
    for key in trajectory["observation"].keys():
        if "image" in key:
            image_keys.append(key)
    if config['wrist_image_only']:
        image_keys = ['image_3']
    key = np.random.choice(image_keys)

    #Decode Image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"][key])

    #Resize Image
    trajectory["observation"]["image"] = tf.cast(trajectory["observation"]["image"], dtype='float32') / 255.0
    trajectory["observation"]["image"] = tf.image.resize(trajectory["observation"]["image"],
            size=(64, 64), method='bicubic')
    trajectory["observation"]["image"] = tf.cast(trajectory["observation"]["image"] * 255.0, dtype='uint8')
 
    #Mirror Wrist Image
    if key == 'image_3':
        trajectory["observation"]["image"] = trajectory["observation"]['image'][:,::-1,::-1,:]    

    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 2
    return trajectory


def r2d2_dataset_transform(
    trajectory: Dict[str, Any],
    config
) -> Dict[str, Any]:
    
    #Set image key
    image_keys = []
    for key in trajectory["observation"].keys():
        if "image" in key:
            image_keys.append(key)
    if config['wrist_image_only']:
        image_keys = ['wrist_image_left', 'wrist_image_right']
    key = np.random.choice(image_keys)
    
    #Decode Image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"][key])

    #Resize Image
    trajectory["observation"]["image"] = tf.cast(trajectory["observation"]["image"], dtype='float32') / 255.0
    trajectory["observation"]["image"] = tf.image.resize(trajectory["observation"]["image"],
            size=(64, 64), method='bicubic')
    trajectory["observation"]["image"] = tf.cast(trajectory["observation"]["image"] * 255.0, dtype='uint8')

    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 3
    return trajectory


def jaco_play_dataset_transform(trajectory, config) -> Dict[str, Any]:
    #Set image key
    image_keys = ['image', 'image_wrist']
    if config['wrist_image_only']:
        image_keys = ['image_wrist']
    key = np.random.choice(image_keys)
    
    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"][key])
    
    trajectory["observation"]["state_eef"] = trajectory["observation"][
        "end_effector_cartesian_pos"
    ][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"][
        "end_effector_cartesian_pos"
    ][:, -1:]

    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            trajectory["action"]["gripper_closedness_action"],
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 4
    return trajectory


def kuka_dataset_transform(trajectory, config) -> Dict[str, Any]:
    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"]["image"])
    
    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            trajectory["action"]["gripper_closedness_action"],
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 5
    return trajectory


def taco_play_dataset_transform(trajectory, config):
    #Set image key
    image_keys = ['rgb_gripper', 'rgb_static']
    if config['wrist_image_only']:
        image_keys = ['rgb_static']
    key = np.random.choice(image_keys) 

    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"][key])

    #Set action
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 6
    return trajectory


def berkeley_cable_routing_dataset_transform(
    trajectory,
    config
):
    #Set image key
    image_keys = ['image', 'top_image', 'wrist45_image']
    if config['wrist_image_only']:
        image_keys = ['wrist45_image']
    key = np.random.choice(image_keys)

    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"][key])

    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 7
    return trajectory


def roboturk_dataset_transform(trajectory, config):
    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"]["front_rgb"])

    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"],
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 8
    return trajectory


def nyu_door_opening_dataset_transform(trajectory, config):
    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"]["image"])

    #Set action
    trajectory["action"] = tf.concat(
    (    
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"],
        ),
        axis=-1,
    ) 
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 9
    return trajectory


def viola_dataset_transform(trajectory, config):
    #Set image key
    image_keys = ['agentview_rgb', 'eye_in_hand_rgb']
    if config['wrist_image_only']:
        image_keys = ['eye_in_hand_rgb']
    key = np.random.choice(image_keys)

    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"][key])

    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 10
    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory, config):
    #Set image key
    image_keys = ['image', 'hand_image']
    if config['wrist_image_only']:
        image_keys = ['hand_image']
    key = np.random.choice(image_keys)

    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"][key])

    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]    
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 11
    return trajectory

def toto_dataset_transform(trajectory, config):
    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"]["image"])

    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 12
    return trajectory


def rt1_dataset_transform(trajectory, config):
    #Decode image
    trajectory["observation"]["image"] = decode_images(trajectory["observation"]["image"])

    #Set action
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"],
        ),
        axis=-1,
    )
    traj_len = tf.shape(trajectory['action'])[0]
    trajectory['dataset_idx'] = tf.ones([traj_len]) * 13
    return trajectory


RLDS_TRANSFORM_DICT = {
    'gnm_dataset': gnm_dataset_transform,
    'polybot_dataset': polybot_dataset_transform,
    'bridge_dataset': bridge_dataset_transform,
    'r2_d2': r2d2_dataset_transform,
    'kuka': kuka_dataset_transform,
    'fractal20220817_data': rt1_dataset_transform,
    'jaco_play': jaco_play_dataset_transform,
    'taco_play': taco_play_dataset_transform,
    'berkeley_cable_routing': berkeley_cable_routing_dataset_transform,
    'roboturk': roboturk_dataset_transform,
    'nyu_door_opening_surprising_effectiveness': nyu_door_opening_dataset_transform,
    'viola': viola_dataset_transform,
    'berkeley_autolab_ur5': berkeley_autolab_ur5_dataset_transform,
    'toto': toto_dataset_transform 
}

