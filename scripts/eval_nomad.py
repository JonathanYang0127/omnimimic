from iris_robots.robot_env import RobotEnv
from iris_robots.transformations import add_angles, angle_diff, pose_diff
import matplotlib.pyplot as plt

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import yaml
import pickle as pkl

from omnimimic.policies import *
from omnimimic.data.data_utils import unnormalize_action
import omnimimic.torch.pytorch_util as ptu
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.nn as nn


def process_image(image, downsample=False):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    if downsample:
        image = image[:,::2, ::2]
    return image.flatten()

def process_obs(obs, task, use_robot_state, prev_obs=None, downsample=False, image_idx=0):
    if use_robot_state:
        observation_keys = ['image', 'desired_pose', 'current_pose', 'task_embedding']
    else:
        observation_keys = ['image', 'task_embedding']

    if prev_obs:
        observation_keys = ['previous_image'] + observation_keys

    if task is None:
        observation_keys = observation_keys[:-1]

    obs['image'] = process_image(obs['images'][image_idx]['array'], downsample=downsample)
    if prev_obs is not None:
        obs['previous_image'] = process_image(prev_obs['images'][image_idx]['array'], downsample=downsample)
    obs['task_embedding'] = task
    return ptu.from_numpy(np.concatenate([obs[k] for k in observation_keys]))

def stack_obs(context):
    return np.concatenate(context, axis=-1)

def process_action(action):
    return np.clip(action, -1, 1)


if __name__ == '__main__':
    num_trajs = 100
    full_image = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-v", "--video-save-dir", type=str, default="")
    parser.add_argument("-d", "--data-save-dir", type=str, default=None)
    parser.add_argument("-n", "--num-timesteps", type=int, default=15)
    parser.add_argument("--q-value-eval", default=False, action='store_true')
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--task-embedding", default=False, action="store_true")
    parser.add_argument("--task-encoder", default=None)
    parser.add_argument("--sample-trajectory", type=str, default=None)
    parser.add_argument("--use-checkpoint-encoder", action='store_true', default=False)
    parser.add_argument("--use-robot-state", action='store_true', default=False)
    parser.add_argument("--action-relabelling", type=str, choices=("achieved, cdp"))
    parser.add_argument("--normalize-relabelling", action="store_true", default=False)
    parser.add_argument("--robot-model", type=str, choices=('wx250s', 'franka'), default='wx250s')
    parser.add_argument("--downsample-image", action='store_true', default=False)
    parser.add_argument("--multi-head-idx", type=int, default=-1)
    parser.add_argument("--blocking", action="store_true", default=False)
    parser.add_argument("--image-idx", type=int, default=2)
    parser.add_argument("--policy-class", type=str, required=True)
    parser.add_argument("--policy-config", type=str, required=True)
    parser.add_argument('--goal_image', type=str, default='goal_image.png')
    parser.add_argument('--normalization-path', type=str, 
        default='/iris/u/jyang27/rlds_data/polybot_dataset/1.0.3/obs_action_stats_polybot_dataset.pkl')
    args = parser.parse_args()

    ptu.set_gpu_mode(True)
    torch.cuda.empty_cache()

    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    if args.robot_model == 'wx250s':
    	env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=args.blocking)
    else:
        env = RobotEnv('172.16.0.21', use_robot_cameras=True, reverse_image=True, blocking=args.blocking)
    obs = env.reset()
    #env.step(np.array([0, 0, 0, 0, 0, 0, 1]))

    with open(args.policy_config, 'rb') as f:
        policy_config = yaml.safe_load(f) 
    policy_config['num_bins'] = policy_config.get('num_bins', 0)
    policy_config['discrete'] = policy_config.get('discrete', False)
    # policy = make_policy(args.policy_class, policy_config)

    config = policy_config
    model, noise_scheduler = load_model(args.checkpoint_path, config, 'cuda')
    model = model.cuda()
    policy = model.eval()
    num_diffusion_iters = config["num_diffusion_iters"]
    wrapper = get_numpy_wrapper('nomad')
    policy = wrapper(policy, noise_scheduler, policy_config)
    '''
    print("Loading model")
    with open(args.checkpoint_path, 'rb') as f:
        print("Opened f")
        state_dict = torch.load(f)
        print("Loaded state dict")
        try:
            try:
                state_dict = state_dict['model'].module.state_dict()
                policy.load_state_dict(state_dict, strict=False)
            except:
                state_dict = state_dict['model'].state_dict()
                policy.load_state_dict(state_dict, strict=False)
        except:
            policy.load_state_dict(state_dict)
    policy = policy.to(ptu.device)
    policy = policy.eval()
    wrapper = get_numpy_wrapper(args.policy_class)
    policy = wrapper(policy, policy_config)
    '''

    goal_image = Image.open(args.goal_image)
    goal_image = np.asarray(goal_image, dtype=np.uint8)
    import matplotlib.pyplot as plt
    print("Showing Goal Image")
    fig = plt.figure()
    plt.imshow(goal_image)
    plt.savefig('out.png')

    #Get Action Metadata
    with open(args.normalization_path, 'rb') as f:
        action_metadata = pickle.load(f)

    eval_policy = policy

    distance_preds = []
    for i in range(num_trajs):
        obs = env.reset()
        images = []
        context_size = policy_config['context_size']
        if context_size != 0:
            context = [obs['images'][0]['array'][::2, ::2, :] for _ in range(context_size + 1)]

        if not args.task_embedding:
            if args.num_tasks != 0:
                valid_task_idx = False
                while not valid_task_idx:
                    task_idx = "None"
                    while not task_idx.isnumeric():
                        task_idx = input("Enter task idx to continue...")
                    task_idx = int(task_idx)
                    valid_task_idx = task_idx in list(range(args.num_tasks))
                task = np.array([0] * args.num_tasks)
                task[task_idx] = 1
            else:
                task = None
        else:
            if args.task_encoder or args.use_checkpoint_encoder:
                if args.sample_trajectory is None:
                    env.move_to_state([-0.11982477,  0.2200,  0.07], 0, duration=1)
                    input("Press enter to take image")
                    obs = ptu.from_numpy(env.get_observation()['image'].reshape(1, 1, -1))
                else:
                    with open(args.sample_trajectory, 'rb') as f:
                        path = pickle.load(f)
                    obs = ptu.from_numpy(path['observations'][-1]['image'].reshape(1, 1, -1))
                task = ptu.get_numpy(task_encoder(obs[0]))
                task = task.reshape(-1)
                print("task: ", task)
                input("Press enter to continue")
                obs = env.reset()
            else:
                task = "None"
                while not isinstance(eval(task), list):
                    task = input("Enter task embedding to continue...")
                task = np.array(eval(task))
        print("Eval Traj {}".format(i))


        if args.data_save_dir is not None:
            trajectory = []

        for j in range(args.num_timesteps):
            obs = env.get_observation()
            img = obs['images'][0]['array'][::2, ::2, :]
            if context_size != 0:
                context.pop(-1) 
                context = [img] + context
                print(len(context))
                img = stack_obs(context)
            action = eval_policy.get_action(img[None], goal_image[::2, ::2, :][None], task)[0]
            action = unnormalize_action(action, action_metadata, discrete=policy_config['discrete'],
                num_bins=policy_config['num_bins']) 
            action = process_action(action)[0]
            env.step(action)

            if args.video_save_dir:
                image = obs['images'][0]['array']
                images.append(Image.fromarray(image))

        #Save Trajectory
        if args.data_save_dir is not None:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

            with open(os.path.join(args.data_save_dir, 'traj_{}.pkl'.format(dt_string)), 'wb+') as f:
                pickle.dump(trajectory, f)

        # Save Video
        if args.video_save_dir:
            print("Saving Video")
            images[0].save('{}/eval_{}.MOV'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
