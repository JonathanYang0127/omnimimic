import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from omnimimic.train_utils.visualize_utils import (visualize_train_images,
    plot_trajs_and_points, VISUALIZATION_IMAGE_SIZE)
from omnimimic.torch.pytorch_util import to_numpy, from_numpy
from omnimimic.train_utils.logger import Logger
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# Train utils for NOMAD
def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions, batch_action_label, dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions, start_dim=1),
        torch.flatten(batch_action_label, start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions, batch_action_label, dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions, start_dim=1),
        torch.flatten(batch_action_label, start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results


def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
            ) = data
            if i >= num_batches:
                break        
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)
            device = batch_obs_images.device

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            #goal_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            # Get distance label
            distance = distance.float().to(device)

            
            # Normalization is done on the dataloader side
            # By default, the dataloader outputs deltas
            naction = actions.to(device)

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            loss = alpha * (0.01) * dist_loss + (1-alpha) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})


            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                 visualize_train_images(
                    to_numpy(batch_obs_images),
                    project_folder,
                    epoch,
                    num_images_log=num_images_log
                )               


def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data
            if i >= num_batches:
                break
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)
            device = batch_obs_images.device

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

            distance = distance.to(device)

            naction = actions.to(device)

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            ### RANDOM MASK ERROR ###
            # Predict the noise residual
            rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
            
            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
            
            ### NO MASK ERROR ###
            # Predict the noise residual
            no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
            
            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
            
            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
            
            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
            wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
            wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def get_action(diffusion_output):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 7)
    ndeltas = to_numpy(ndeltas)
    
    return from_numpy(ndeltas).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    uc_actions = get_action(diffusion_output)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }


def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)


