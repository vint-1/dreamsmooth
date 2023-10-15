# DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing
[Project Website](https://vint-1.github.io/dreamsmooth)

## Overview

### Reward Prediction is Important in MBRL

Reward models, which predict the rewards that an agent would have obtained for some imagined trajectory, play a vital role in state-of-the-art MBRL algorithms like DreamerV3 and TD-MPC because the policy learns from predicted rewards.

### Reward Prediction is Challenging

Reward prediction in sparse-reward environments, especially those with **partial observability** or **stochastic rewards**, is surprisingly challenging.

The following plots show predicted and ground truth rewards over a single episode, in several environments (including [Robodesk](https://github.com/google-research/robodesk), [ShadowHand](https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block/), and [Crafter](https://github.com/danijar/crafter)), with mispredicted sparse rewards highlighted in yellow.

![Reward prediction is challenging](/assets/unsmooth_trajectories.png "trajectories for different environments, showing poor reward prediction by dreamerv3")


### Our Solution: Temporally Smoothed Rewards

We propose DreamSmooth, which performs temporal smoothing of the rewards obtained in each rollout before adding them to the replay buffer. Our method makes learning a reward model easier, especially when rewards are ambiguous or sparse.

With our method, the reward models no longer omit sparse rewards from its output, predicting them accurately.

![Dreamsmooth improves reward prediction](/assets/smooth_trajectories.png "trajectories for different environments, showing accurate reward prediction by dreamsmooth")

Moreover, the improved reward predictions of DreamSmooth translates to better performance. We studied several different smoothing techniques (Gaussian, uniform, exponential moving average) on many sparse-reward environments, and find that our method outperforms the base DreamerV3 model.

![Dreamsmooth improves performance](/assets/performance.png "learning curves for different environments, showing dreamsmooth outperforms dreamerv3")

## Quickstart

This code is built on top of the official [DreamerV3 implementation](https://github.com/danijar/embodied/tree/unstable).

### Prerequisites

* Ubuntu 22.04
* Python 3.9+


### Installation
* Install [DreamerV3 dependencies](https://github.com/danijar/dreamerv3)
* Install dependencies
    ```
    pip install -r requirements.txt
    ```

### Environments
* Modified robodesk and hand environments can be found in [`embodied/envs/robodesk.py`](embodied/envs/robodesk.py) and [`embodied/envs/hand.py`](embodied/envs/hand.py)


### Important directories and files
* [`embodied/core/smoothing.py`](embodied/core/smoothing.py): reward smoothing implementation
* [`embodied/agents/dreamerv3/configs.yaml`](embodied/agents/dreamerv3/configs.yaml): configs
* [`scripts`](scripts/): scripts for running experiments


### Run experiments
Replace `[EXP_NAME]` with name of the experiment, `[GPU]` with the GPU number you wish to use, and `[WANDB_ENTITY]` and `[WANDB_PROJECT]` with the W&B entity/project you want to log to. `[SMOOTHING_METHOD]` should be `gaussian`, `uniform`, `exp`, or `no` (for no smoothing).
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

### Examples
* Gaussian Smoothing with sigma = 3 on Robodesk
    ```
    source scripts/d3_robodesk_train.sh example_01 [GPU] 1 gaussian 3 [WANDB_ENTITY] [WANDB_PROJECT]
    ```

* Uniform Smoothing with delta = 5 on Hand
    ```
    source scripts/d3_hand_train.sh example_03 [GPU] 1 uniform 5 [WANDB_ENTITY] [WANDB_PROJECT]
    ```

## Citation

```
@article{lee2023dreamsmooth,
  author    = {Vint Lee and Pieter Abbeel and Youngwoon Lee},
  title     = {DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing},
  journal   = {????},
  year      = {2023},
}
```
