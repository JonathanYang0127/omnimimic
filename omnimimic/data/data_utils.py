from typing import Any, Callable, Dict, Sequence, Union
from functools import partial
import tensorflow as tf
import numpy as np

def get_goal_idxs(traj_len, min_goal_distance_prop=0.2, max_goal_distance_prop=0.8,
    min_goal_distance_indices=None, max_goal_distance_indices=40):
    rand = tf.random.uniform(shape=[traj_len])
    if min_goal_distance_indices is None:
        low = tf.cast(tf.range(traj_len), dtype=tf.float32) + tf.cast(traj_len, dtype=tf.float32) * min_goal_distance_prop
    else:
        low = tf.cast(tf.range(traj_len), dtype=tf.float32) + min_goal_distance_indices
    if max_goal_distance_indices is None:
        high = tf.cast(tf.range(traj_len), dtype=tf.float32) + tf.cast(traj_len, dtype=tf.float32) * max_goal_distance_prop
    else:
        high = tf.cast(tf.range(traj_len), dtype=tf.float32) + max_goal_distance_indices
    goal_idxs = tf.cast((high-low) * rand + low, dtype=tf.int32)
    goal_idxs = tf.maximum(goal_idxs, tf.range(traj_len))
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)
    low = tf.cast(low, dtype=tf.int32)
    low = tf.maximum(low, tf.range(traj_len))
    high = tf.cast(high, dtype=tf.int32)
    high = tf.minimum(high, traj_len - 1)

    return high, low, goal_idxs


def unnormalize_action(action, metadata, key='action', eps=1e-9, discrete=False, num_bins=256):
    if not discrete:
        action_range = (metadata[key]["max"] - metadata[key]["min"] + eps) / 2
        action = (action + 1) * action_range + metadata[key]["min"] - eps
    else:
        action_range = (metadata[key]["max"] - metadata[key]["min"])
        pads = (action_range > 0).astype(np.float32)
        action = action.astype(np.float32)
        action = ((action - 1) / (num_bins - 2)) * (action_range + eps) + metadata[key]["min"]
        action = action * pads
    return action   


def frame_map(map_fn, tensor):
    tdataset = tf.data.Dataset.from_tensor_slices(tensor)
    tdataset = tdataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(
                tf.dtypes.int64.max,
                num_parallel_calls=tf.data.AUTOTUNE,
                drop_remainder=False,
            )
    return tdataset.get_single_element()


def decode_images(images):
    if images.dtype == tf.uint8:
        return images
    images = frame_map(
        partial(
            tf.io.decode_image,
            expand_animations=False
        ), images
    )
    return images


def decode_dataset_images(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "image"
) -> Dict[str, Any]:
    if isinstance(match, str):
        match = [match]

    def match_fn(keypath, value):
        image_in_keypath = any([s in keypath for s in match])
        return image_in_keypath

    return selective_tree_map(
        x,
        match=match_fn,
        map_fn=partial(tf.io.decode_image, expand_animations=False),
    )


def selective_tree_map(
    x: Dict[str, Any],
    match: Union[str, Sequence[str], Callable[[str, Any], bool]],
    map_fn: Callable,
    *,
    _keypath: str = "",
) -> Dict[str, Any]:
    """Maps a function over a nested dictionary, only applying it leaves that match a criterion.

    Args:
        x (Dict[str, Any]): The dictionary to map over.
        match (str or Sequence[str] or Callable[[str, Any], bool]): If a string or list of strings, `map_fn` will only
        be applied to leaves whose key path contains one of `match`. If a function, `map_fn` will only be applied to
        leaves for which `match(key_path, value)` returns True.
        map_fn (Callable): The function to apply.
    """
    if not callable(match):
        if isinstance(match, str):
            match = [match]
        match_fn = lambda keypath, value: any([s in keypath for s in match])
    else:
        match_fn = match

    out = {}
    for key in x:
        if isinstance(x[key], dict):
            out[key] = selective_tree_map(
                x[key], match_fn, map_fn, _keypath=_keypath + key + "/"
            )
        elif match_fn(_keypath + key, x[key]):
            out[key] = map_fn(x[key])
        else:
            out[key] = x[key]
    return out

def index_nested_dict(d: Dict[str, Any], index: int):
    """
    Indexes a nested dictionary with backslashes separating hierarchies
    """
    indices = index.split("/")
    for i in indices:
        if i not in d.keys():
            raise ValueError(f"Index {index} not found")
        d = d[i]
    return d

def set_nested_dict_index(d: Dict[str, Any], index: int, value):
    """
    Sets an index in a nested dictionary with a value
    Indexes have backslashes separating hierarchies
    """
    indices = index.split("/")
    for i in indices[:-1]:
        if i not in d.keys():
            raise ValueError(f"Index {index} not found")
        d = d[i]
    d[indices[-1]] = value


def map_nested_dict_index(d: Dict[str, Any], index: int, map_func):
    """
    Maps an index in a nested dictionary with a function
    Indexes have backslashes separating hierarchies
    """
    indices = index.split("/")
    for i in indices[:-1]:
        if i not in d.keys():
            raise ValueError(f"Index {index} not found")
        d = d[i]
    d[indices[-1]] = map_func(d[indices[-1]])


