# Omnimimic
Omnimimic is a codebase for training cross-embodiment generalist robot policies.

 * [Project Page](https://extreme-cross-embodiment.github.io)
 * [Video](https://www.youtube.com/watch?v=ljZ48IoaPTY)


## Setup
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:  \
    ```
    conda create --name omnimimic python=3.10
    ```

First, install the required packages:   \
    ```
    pip install -r requirements.txt
    ```

There are additional dependencies that need to be installed manually. Please follow the instructions in [dlimp](https://github.com/kvablack/dlimp), [diffusion_policy](https://github.com/real-stanford/diffusion_policy), and [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr).


Finally, install the omnimimic module:   \
    ```
    pip install -e .
    ```

## Training
Firstly, modify the appropriate training configuration found in configs. \
To train a small policy on a single dataset, run \
    ```
    python train_scripts/train.py -c config/omnimimic_single_dataset.yaml --datasets polybot_dataset
    ```

## Dataloading
Our codebase uses the [RLDS](https://github.com/google-research/rlds) dataset format. To download the datasets, follow the instructions from the [Open-X Embodiment](https://github.com/google-deepmind/open_x_embodiment) codebase. \
To link the training script to the datasets, add all of the datasets to a single directory, then modify the data\_dir field in the config.


To add a new dataset, follow these steps:
1. Convert your dataset to RLDS format. 
2. Add a dataset-specific transform in [this script](https://github.com/JonathanYang0127/omnimimic/blob/release/omnimimic/data/rlds_data_transforms.py). The transform should return a trajectory with the following keys:
    * trajectory["observation"]["image"]
    * trajectory["observation"]["state"]
    * trajectory["observation"]["action"]
3. Add this dataset specific transform to the [RLDS_TRANSFORM_DICT](https://github.com/JonathanYang0127/omnimimic/blob/3628392798924f5261d38118cc1bc548cfe3315e/omnimimic/data/rlds_data_transforms.py#L408)
4. Add the dataset name to the [data splits](https://github.com/JonathanYang0127/omnimimic/blob/release/omnimimic/data/data_splits.py) file. Modify the frequency of the data in the training batch.



