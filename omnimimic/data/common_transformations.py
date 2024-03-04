import tensorflow as tf
from .data_utils import *


def resize_image(traj, size=(256, 256)):
    image_shape = tf.shape(traj["observation"]["image"])
    if image_shape[1] == size[0] and image_shape[2] == size[1]:
        return traj
    traj["observation"]["image"] = tf.cast(traj["observation"]["image"], dtype='float32') / 255.0
#    traj["observation"]["image"] = tf.image.resize_with_pad(traj["observation"]["image"],
#            size[0], size[1], method='bicubic')
    traj["observation"]["image"] = tf.image.resize(traj["observation"]["image"], size, method='bicubic')
    traj["observation"]["image"] = tf.cast(traj["observation"]["image"] * 255.0, dtype='uint8')

    return traj


def relabel_goal_image(traj, metadata):
    #Relabel random goals
    traj_len = tf.shape(traj["action"])[0]
    high, low, goal_idxs = get_goal_idxs(traj_len)

    if 'goal_idxs' in traj.keys() and 'goal_distance_indices' in traj.keys():
        goal_idxs = traj['goal_idxs']
    else:
        traj['goal_idxs'] = goal_idxs
        traj["goal_distance_indices"] = tf.cast(goal_idxs, tf.float32) - tf.cast(tf.range(traj_len), tf.float32)

    traj["goal_image"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["observation"]["image"]
    )

    #traj["goal_distance_indices"] /= tf.cast(metadata['traj_len'], tf.float32)

    return traj


def normalize_obs_and_actions(traj, action_keys, metadata, eps=1e-9):
    '''
    For now, only normalize appropriate action keys
    '''
    normal_keys = []
    min_max_keys = action_keys
    
    for key in normal_keys:
        map_nested_dict_index(
            traj,
            key,
            lambda x: (x - metadata[key]["mean"]) / metadata[key]["std"]
        )

    for key in min_max_keys:
        map_nested_dict_index(
            traj,
            key,
            lambda x: tf.clip_by_value((2 * (x - metadata[key]["min"]) + eps)
                / (metadata[key]["max"] - metadata[key]["min"] + eps) - 1,
                -1,
                1)
        )
    
    return traj

def discretize_actions(traj, discretize_keys, metadata, eps=1e-9, discretize=False, num_bins=256):
    for key in discretize_keys:
        num_bins = tf.constant(num_bins, dtype=tf.float32)
        def discretize_func(x):
            x = tf.cast(x, dtype=tf.float32)
            print('min:', metadata[key]["min"])
            action_range = metadata[key]["max"] - metadata[key]["min"] 
            x = tf.clip_by_value((x - metadata[key]["min"]) / (action_range + eps), 0, 1)
            bin_id = (x * (num_bins - 2) + 1) * tf.cast(action_range > 0, dtype=tf.float32)
            return tf.cast(bin_id, dtype=tf.int64)

        map_nested_dict_index(
            traj,
            key,
            discretize_func  
        )

    return traj


def random_dataset_sequence_transform_v2(traj, frame_stack, seq_length,
    pad_frame_stack, pad_seq_length):
    '''
    Extract a random subsequence of the data given sequence_length given keys
    '''
    traj_len = tf.shape(traj["action"])[0]

    if 'action_sequence' in traj.keys():
        traj_len = tf.shape(traj["action"])[0]
        traj['action'] = traj['action_sequence']
    else:
        #Pad actions
        padding = tf.tile(tf.zeros_like(traj['action'][-1:, :]),
            [seq_length - 1, 1])
        traj['action'] = tf.concat(
            (traj['action'], padding),
            axis=0)

        #Get sequence length actions
        indices = tf.reshape(tf.range(traj_len), [-1, 1])  \
            + tf.range(0, seq_length)
        traj['action'] = tf.gather(traj['action'], indices)

    #Pad frames
    padding = tf.repeat([traj['observation']['image'][0]], repeats=[frame_stack], axis=0)
    traj['observation']['image'] = tf.concat(
        (padding, traj['observation']['image']),
        axis=0)

    #Get sequence length images
    if frame_stack != 0:
        indices = tf.reshape(tf.range(traj_len), [-1, 1])  \
            + tf.range(0, frame_stack + 1)
        obs = tf.gather(traj['observation']['image'], indices)
        obs = tf.transpose(obs, perm=[0, 2, 3, 1, 4])
        os = tf.shape(obs)
        traj['observation']['image'] = tf.reshape(obs, (os[0], os[1], os[2], 
            os[3] * os[4]))
    return traj


def random_dataset_sequence_transform(traj, frame_stack, seq_length,
    pad_frame_stack, pad_seq_length):
    '''
    Extract a random subsequence of the data given sequence_length given keys
    '''

    if 'action_sequence' in traj.keys():
        traj['action'] = traj['action_sequence']
        traj['observation']['image'] = traj['observation']['image'][index_in_demo][None]
        for k in traj.keys():
            if 'goal' in k:
                traj[k] = traj[k][index_in_demo][None]
        return traj

    traj_len = tf.shape(traj["action"])[0]
    seq_begin_pad, seq_end_pad = 0, 0
    if pad_frame_stack:
        seq_begin_pad = frame_stack - 1
    if pad_seq_length:
        seq_end_pad = seq_length - 1
    index_in_demo = tf.random.uniform(shape=[],
            maxval=traj_len + seq_end_pad - (seq_length - 1),
          dtype=tf.int32)
    pad_mask = tf.concat((tf.repeat(0, repeats=seq_begin_pad),
                        tf.repeat(1, repeats=traj_len),
                        tf.repeat(0, repeats=seq_end_pad)), axis=0)[:, None]
    traj['pad_mask'] = tf.cast(pad_mask, dtype=tf.bool)
    keys = ["observation", "action",  "goal"]

    def random_sequence_func(x):
        begin_padding = tf.repeat([x[0]], repeats=[seq_begin_pad], axis=0)
        end_padding = tf.repeat([x[-1]], repeats=[seq_end_pad], axis=0)
        sequence = tf.concat((begin_padding, x, end_padding), axis=0)
        return sequence[index_in_demo: index_in_demo + seq_length + frame_stack - 1]

    traj = selective_tree_map(
        traj,
        match=lambda x: any([key in str(x) for key in keys]),
        map_fn=random_sequence_func
    )
    return traj


def process_batch_transform_v2(traj, use_goal_state, discrete):
    new_traj = dict()
    print(traj.keys())
    new_traj['traj_idx'] = tf.range(tf.shape(traj['action'])[0])
    new_traj['observation'] = dict()
    new_traj['observation']['image'] = traj['observation']['image']
    new_traj['goal_image'] = traj['goal_image']
    if use_goal_state:
        new_traj['goal_state'] = traj['goal_state']
    new_traj['goal_distance_indices'] = traj['goal_distance_indices']
    new_traj['goal_idxs'] = traj['goal_idxs']
    new_traj['action'] = traj['action']
    if not discrete:
        new_traj['action'] = tf.cast(traj['action'], dtype=tf.float32)
    new_traj['dataset_idx'] = traj['dataset_idx']
    #new_traj['file_path'] = traj['traj_metadata']['episode_metadata']['file_path']
    return new_traj

def process_batch_transform(traj):
    new_traj = dict()
    new_traj['observation'] = dict()
    new_traj['observation']['image'] = traj['observation']['image'][0]
    new_traj['goal_image'] = traj['goal_image'][0]
    new_traj['goal_state'] = traj['goal_state'][0]
    new_traj['goal_distance_indices'] = traj['goal_distance_indices'][0]
    new_traj['action'] = traj['action']
    new_traj['file_path'] = traj['episode_metadata']['file_path']
    '''
    new_traj['observation'] = dict()
    new_traj['observation']['image'] = traj['observation']['image']
    new_traj['goal_image'] = traj['goal_image']
    new_traj['goal_state'] = traj['goal_state']
    new_traj['goal_distance_indices'] = traj['goal_distance_indices']
    new_traj['action'] = traj['action']
    '''
    return new_traj

