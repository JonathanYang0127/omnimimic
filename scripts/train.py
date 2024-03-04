import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from omnimimic.train_utils.train_eval_loop import train_eval_loop_nomad
from omnimimic.policies import load_model, make_policy
from omnimimic.data.dataset import *
from omnimimic.data.data_splits import DATASET_SPLITS, VERSION_DICT


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    
    TFDS_DATA_DIR = config['data_dir']
    datasets = config['datasets']
    if 'all' in datasets:
        datasets = list(DATASET_SPLITS.keys())
        print("Using /scr data dir")
    if 'no_gnm' in datasets:
        datasets = list([k for k in DATASET_SPLITS.keys() 
            if k != 'gnm_dataset'])
        print("Using /scr data dir")
    train_dataloaders = []
    train_dataloader_names = []
    wrist_image_only = config.get('wrist_image_only', False)
    dataloader_config = {'wrist_image_only': wrist_image_only,
          'seq_length': config["len_traj_pred"],
          'context_size': config["context_size"],
          'visualize': False,
          'no_normalization': False,
          'image_size': config['image_size'],
          'discrete': False,
          'num_bins': 0,
          'gnm_delta_actions': True}
    with tf.device('/cpu'):
        for dataset in datasets:
            train_split='train'
            for version in VERSION_DICT.get(dataset, [None]):
                try:
                    TFDS_DATA_DIR = '/scr/jonathan/rlds_data'
                    train_dataloader, _ = make_dataloader(dataset, train_split, dataloader_config,
                        data_dir=TFDS_DATA_DIR, version=version)
                except:
                    TFDS_DATA_DIR = '/iris/u/jyang27/rlds_data'
                    train_dataloader, _ = make_dataloader(dataset, train_split, dataloader_config,
                        data_dir=TFDS_DATA_DIR, version=version)
                train_dataloaders.append(train_dataloader)
                train_dataloader_names.append(dataset)

            splits = ['val', 'train[80%:]', 'train']
            val_dataloader_splits = []
            for split in splits:
                try:
                    val_split, val_meta = make_dataloader(dataset, split, dataloader_config,
                        data_dir=TFDS_DATA_DIR, validation=True)
                    val_dataloader_splits.append(val_split)
                    break
                except:
                    print(f"Error: can't use {split} split for validating dataset {dataset}")
            val_dataloader = tf.data.Dataset.sample_from_datasets(val_dataloader_splits)
            val_dataloader = shuffle_batch_and_prefetch_dataloader(val_dataloader,
                 64, shuffle_size=10000)
            test_dataloaders[dataset] = RLDSTorchDataset(val_dataloader.as_numpy_iterator(),
                    dataset_length=50)
            
        sample_weights = [DATASET_SPLITS[d] for d in train_dataloader_names]
        sample_weights /= tf.reduce_sum(sample_weights)
        train_dataloader = tf.data.Dataset.sample_from_datasets(
            train_dataloaders, sample_weights)
        train_dataloader = shuffle_batch_and_prefetch_dataloader(train_dataloader,
            config['batch_size'], shuffle_size=10000)
        train_loader = RLDSTorchDataset(train_dataloader.as_numpy_iterator())    
            

    # Create the model
    model, noise_scheduler = make_policy(config["model_type"], config) 

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        if config['model_type'] == 'nomad':
            state_dict = latest_checkpoint
            try:
                model.load_state_dict(state_dict, strict=True)
            except:
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=True)
        else:
            load_model(model, latest_checkpoint)
        current_epoch = latest_checkpoint.get("epoch", -1) + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        try:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
            if scheduler is not None:
                scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
        except:
            print("Error loading optimizer and scheduler")

    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    parser.add_argument(
        "--use-rlds",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True
    )
    args = parser.parse_args()

    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(file_path, "../config/defaults.yaml"), "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config['use_rlds'] = True
    config['datasets'] = args.datasets
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="jyang27", # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    main(config)
