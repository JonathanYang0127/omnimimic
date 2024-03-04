import torch
import torch.nn as nn
from .nomad_wrapper import NomadWrapper
from omnimimic.policies.nomad.nomad import NoMaD
from omnimimic.policies.nomad.nomad import NoMaD, DenseNetwork
from omnimimic.policies.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""

    model = make_policy(config['model_type'], config)
    checkpoint = torch.load(model_path)#, map_location=device)
    if model_type == "nomad":
        state_dict = checkpoint
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            #For models trained with data parallelism
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def make_policy(model_type, config):
    if config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        noise_pred_net = ConditionalUnet1D(
                input_dim=7,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise NotImplementedError

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    return model, noise_scheduler

def get_numpy_wrapper(policy_class):
    if policy_class == 'nomad':
        wrapper = NomadWrapper
    else:
        raise NotImplementedError
    return wrapper

