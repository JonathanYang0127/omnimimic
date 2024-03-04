import numpy as np
import torch

import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from torch.distributions import constraints, Distribution, Independent, Normal
import torch
import omnimimic.torch.pytorch_util as ptu


class NomadWrapper:
    def __init__(self, policy, noise_scheduler, config):
        self.policy = policy
        self.noise_scheduler = noise_scheduler
        self.action_dim = 7
        self.pred_horizon = 8

    def forward_np(self, image, goal_image, task_embedding=None):
        '''
        Image is (batch, height, width, channels)
        Data type is uint8 and normalized
        '''
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).to(ptu.device)
        if task_embedding is not None:
            task_embedding = torch.from_numpy(task_embedding)

        goal_image = goal_image.astype(np.float32) / 255.0
        goal_image = torch.from_numpy(goal_image).to(ptu.device)

        obs_images = torch.einsum('b h w c -> b c h w', image)
        goal_image = torch.einsum('b h w c -> b c h w', goal_image)

        mask = torch.zeros((goal_image.shape[0],)).long().to(ptu.device)

        obsgoal_cond = self.policy('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), 
            goal_img=goal_image, input_goal_mask=mask)
        noisy_diffusion_output = torch.randn(
            (len(obsgoal_cond), self.pred_horizon, self.action_dim), device=ptu.device)
        diffusion_output = noisy_diffusion_output

        for k in self.noise_scheduler.timesteps[:]:
            # predict noise
            noise_pred = self.policy(
                "noise_pred_net",
                sample=diffusion_output,
                timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(ptu.device),
                global_cond=obsgoal_cond
            )

            # inverse diffusion step (remove noise)
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output
            ).prev_sample

        return diffusion_output


    def compute_loss_np(self, image, goal_image, actions, goal_distances, task_embedding=None):
        raise NotImplementedError

    def get_action(self, image, goal_image, task_embedding=None):
        action_pred = self.forward_np(image, goal_image, task_embedding)
        return action_pred.detach().cpu().numpy()

    def get_distance_pred(self, image, goal_image, task_embedding=None):
        raise NotImplementedError


