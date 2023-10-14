NAME=$1
TASK=$2
GPU=$3
SEED=$4
SMOOTHING=$5
AMT=$6
WANDB_ENT=$7
WANDB_PROJ=$8

CUDA_VISIBLE_DEVICES=${GPU} \
python embodied/agents/dreamerv3/train.py \
--configs dmc_vision ${SMOOTHING}_smoothing \
--logdir ./log/dreamer3 \
--method ${NAME} \
--task dmc_${TASK} \
--run.wandb True \
--run.wandb_entity ${WANDB_ENT} \
--run.wandb_project ${WANDB_PROJ} \
--run.rew_smoothing_amt ${AMT} \
--seed ${SEED}