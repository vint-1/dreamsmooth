# DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing

This code is built on top of the official DreamerV3 implementation (https://github.com/danijar/embodied/tree/unstable).

## Prerequisites

* Ubuntu 22.04
* Python 3.9+


## Installation
* Install [DreamerV3 dependencies](https://github.com/danijar/dreamerv3)
* Install dependencies
    ```
    pip install -r requirements.txt
    ```

## Environments
* Modified robodesk and hand environments can be found in `embodied/envs/robodesk.py` and `embodied/envs/hand.py`


## Important directories and files
* `embodied/core/smoothing.py`: reward smoothing implementation
* `embodied/agents/dreamerv3/configs.yaml`: configs
* `scripts`: scripts for running experiments


## Run experiments
Replace [EXP_NAME] with name of the experiment, [GPU] with the GPU number you wish to use, and [WANDB_ENTITY] and [WANDB_PROJECT] with the W&B entity/project you want to log to. [SMOOTHING_METHOD] should be `gaussian`, `uniform`, `exp`, or `no` (for no smoothing).
* Running experiments on Robodesk
    ```
    source scripts/d3_robodesk_train.sh [EXP_NAME] [GPU] [SEED] [SMOOTHING_METHOD] [SMOOTHING_PARAMETER] [WANDB_ENTITY] [WANDB_PROJECT]
    ```

* Running experiments on Hand
    ```
    source scripts/d3_hand_train.sh [EXP_NAME] [GPU] [SEED] [SMOOTHING_METHOD] [SMOOTHING_PARAMETER] [WANDB_ENTITY] [WANDB_PROJECT]
    ```

* Running experiments on Crafter
    ```
    source scripts/d3_crafter_train.sh [EXP_NAME] [GPU] [SEED] [SMOOTHING_METHOD] [SMOOTHING_PARAMETER] [WANDB_ENTITY] [WANDB_PROJECT]
    ```

* Running experiments on Atari
    ```
    source scripts/d3_atari_train.sh [EXP_NAME] [TASK] [GPU] [SEED] [SMOOTHING_METHOD] [SMOOTHING_PARAMETER] [WANDB_ENTITY] [WANDB_PROJECT]
    ```

* Running experiments on Deepmind Control
    ```
    source scripts/d3_dmc_train.sh [EXP_NAME] [TASK] [GPU] [SEED] [SMOOTHING_METHOD] [SMOOTHING_PARAMETER] [WANDB_ENTITY] [WANDB_PROJECT]
    ```

## Examples
* Gaussian Smoothing with sigma = 3 on Robodesk
    ```
    source scripts/d3_robodesk_train.sh example_01 [GPU] 1 gaussian 3 [WANDB_ENTITY] [WANDB_PROJECT]
    ```

* Uniform Smoothing with delta = 5 on Hand
    ```
    source scripts/d3_hand_train.sh example_03 [GPU] 1 uniform 5 [WANDB_ENTITY] [WANDB_PROJECT]
    ```
