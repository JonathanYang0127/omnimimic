import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from .rlds_data_transforms import RLDS_TRANSFORM_DICT
from .common_transformations import *
from .data_utils import *
from .data_splits import VERSION_DICT
from tqdm import tqdm
import dlimp as dl
import numpy as np
import pickle
import torch


class RLDSTorchDataset:
    def __init__(self, dataset_iterator, try_to_use_cuda=True, dataset_length=1000,
            convert_action_dim=False, diffusion_action_dim=False):
        self.dataset_iterator = dataset_iterator
        self.device = torch.device('cuda')
        self.dataset_length = dataset_length
        self.convert_action_dim = convert_action_dim
        self.diffusion_action_dim = diffusion_action_dim

    def __iter__(self):
        for batch in self.dataset_iterator:
            image = batch['observation']['image']
            goal_image = batch['goal_image']
            obs_image = torch.from_numpy(image.astype(np.float32)) / 255.0
            goal_image = torch.from_numpy(goal_image.astype(np.float32)) / 255.0
            obs_image = obs_image.to(self.device)
            goal_image = goal_image.to(self.device)
            obs_image = torch.einsum('b h w c -> b c h w', obs_image)
            goal_image = torch.einsum('b h w c -> b c h w', goal_image)

            if self.convert_action_dim:
                batch['action'] = np.concatenate((-batch['action'][:, :, 2:3], 
                    batch['action'][:, :, 1:2]), axis=-1)
            if self.diffusion_action_dim:
                action_concat_shape = list(batch['action'].shape)
                action_concat_shape[-1] = 1
                batch['action'] = np.concatenate((batch['action'],
                    np.zeros(action_concat_shape)), axis=-1)
            action_label = torch.from_numpy(batch['action']).float().to(self.device)
            dist_label = torch.from_numpy(batch['goal_distance_indices']).to(self.device)
            if 'goal_state' in batch.keys():
                goal_pos = torch.from_numpy(batch['goal_state']).to(self.device)
            else:
                goal_pos = torch.zeros((action_label.shape[0], 2)).to(self.device)
            dataset_index = torch.ones(action_label.shape[0]).to(self.device) * 8
            action_mask = torch.ones(action_label.shape[0]).to(self.device)
    
            yield obs_image, goal_image, action_label, dist_label, goal_pos, dataset_index, action_mask 

    def __len__(self):
        return self.dataset_length

def decode_trajectory(builder, episode):
    steps = episode['steps']
    for key in steps:
        steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key])
    if 'file_path' in episode['episode_metadata'].keys():
        steps['file_path'] = builder.info.features[
            'episode_metadata']['file_path'].decode_batch_example(
            episode['episode_metadata']['file_path'])
    return steps

def apply_common_transforms(dataset, config, 
    obs_action_metadata=None):
    #Resize image
    dataset = dataset.map(
        partial(
            resize_image,
            size=config['image_size'],
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    #Relabel goals
    dataset = dataset.map(
        partial(
            relabel_goal_image,
            metadata=obs_action_metadata
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    '''
    #Get random action sequence
    dataset = dataset.map(
        partial(
            random_dataset_sequence_transform_v2,
            frame_stack=config['context_size'],
            seq_length=config['seq_length'],
            pad_frame_stack=True,
            pad_seq_length=True
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    '''
    #Normalize actions
    if obs_action_metadata is not None and not config['no_normalization'] \
            and not config['discrete']:
        dataset = dataset.map(
            partial(
                normalize_obs_and_actions,
                action_keys=['action'],
                metadata=obs_action_metadata,
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    #Get random action sequence
    dataset = dataset.map(
        partial(
            random_dataset_sequence_transform_v2,
            frame_stack=config['context_size'],
            seq_length=config['seq_length'],
            pad_frame_stack=True,
            pad_seq_length=True
        ),
     
        num_parallel_calls=tf.data.AUTOTUNE
    )

    #Discretize actions
    if config['discrete']:
        dataset = dataset.map(
            partial(
                discretize_actions,
                discretize_keys=['action'],
                metadata=obs_action_metadata,
                num_bins=config['num_bins']
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    #Process correct keys
    dataset = dataset.map(
        partial(
            process_batch_transform_v2,
            use_goal_state=config['visualize'],
            discrete=config['discrete']
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


def get_obs_action_metadata(
    name, builder, dataset, keys,
    load_if_exists=True
):
    # get statistics file path --> embed unique hash that catches if dataset info changed
    data_info_hash = name
    path = tf.io.gfile.join(
        builder.info.data_dir, f"obs_action_stats_{data_info_hash}.pkl"
    )

    # check if stats already exist and load, otherwise compute
    all_keys_present = True
    path_exists = tf.io.gfile.exists(path)
    print(path_exists, load_if_exists)
    if path_exists and load_if_exists:
        print(f"Loading existing statistics for normalization from {path}.")
        with tf.io.gfile.GFile(path, "rb") as f:
            metadata = pickle.load(f)
        print(metadata.keys())
        for key in keys:
            all_keys_present = (key in metadata.keys()) and all_keys_present
        all_keys_present = 'traj_len' in metadata.keys() and all_keys_present
    if load_if_exists and path_exists and all_keys_present:
        return metadata
    else:
        print("Computing obs/action statistics for normalization...")
        eps_by_key = {key: [] for key in keys}

        i, n_samples = 0, 1000
        dataset_iter = dataset.repeat().as_numpy_iterator()
        traj_len_metadata = []
        for _ in tqdm(range(n_samples)):
            episode = next(dataset_iter)
            i = i + 1
            for key in keys:
                eps_by_key[key].append(index_nested_dict(episode, key))
            traj_len_metadata.append(episode['action'].shape[0])
        eps_by_key = {key: np.concatenate(values) for key, values in eps_by_key.items()}

        metadata = {}
        for key in keys:
            metadata[key] = {
                "mean": eps_by_key[key].mean(0),
                "std": eps_by_key[key].std(0),
                "max": eps_by_key[key].max(0),
                "min": eps_by_key[key].min(0),
            }
        metadata['traj_len'] = np.mean(traj_len_metadata)
        print(metadata['traj_len'])
        with tf.io.gfile.GFile(path, "wb") as f:
            pickle.dump(metadata, f)
        print("Done!")

    return metadata


def make_dataloader(dataset_name, split, dataloader_config, data_dir=None, validation=False, version=None):
    dataset_name_and_version = dataset_name
    if version is not None:
        dataset_name_and_version = dataset_name + ':' + version
    builder = tfds.builder(dataset_name_and_version, data_dir=data_dir)
    '''
    dataset = builder.as_dataset(
            split=split,
            decoders={"steps": tfds.decode.SkipDecoding()},
            shuffle_files=True,
        )
    '''
    dataloader = dl.DLataset.from_rlds(builder, split=split, shuffle=True,
        num_parallel_reads=8)

    #dataloader = decode_dataloader(dataloader)
    if dataset_name in RLDS_TRANSFORM_DICT.keys():
        dataloader = dataloader.map(partial(
            RLDS_TRANSFORM_DICT[dataset_name],
            config=dataloader_config),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    action_keys = ['action']
    metadata = get_obs_action_metadata(
        dataset_name,
        builder,
        dataloader,
        action_keys,
        load_if_exists=True or validation,
    )
    print(f'{dataset_name}: {metadata}')
    dataloader = apply_common_transforms(dataloader, dataloader_config, 
            obs_action_metadata=metadata)
    #Ensure observations are not 0
    dataloader = dataloader.filter(lambda e:
        tf.math.reduce_mean(e['observation']['image'][0]) > 0)
    dataloader = dataloader.flatten(num_parallel_calls=8)
    dataloader = dataloader.repeat()
    
    return dataloader, metadata

def shuffle_batch_and_prefetch_dataloader(dataloader, batch_size, shuffle_size=None):
    if shuffle_size is not None:
        dataloader = dataloader.shuffle(shuffle_size)
    dataloader = dataloader.batch(batch_size)
    dataloader = dataloader.prefetch(1)
    return dataloader

