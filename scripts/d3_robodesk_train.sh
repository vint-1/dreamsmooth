NAME=$1
GPU=$2
SEED=$3
SMOOTHING=$4
AMT=$5
WANDB_ENT=$6
WANDB_PROJ=$7

CUDA_VISIBLE_DEVICES=${GPU} \
python embodied/agents/dreamerv3/train.py \
--configs robodesk ${SMOOTHING}_smoothing \
--logdir ./log/dreamer3 \
--method ${NAME} \
--run.wandb True \
--run.wandb_entity ${WANDB_ENT} \
--run.wandb_project ${WANDB_PROJ} \
--run.rew_smoothing_amt ${AMT} \
--seed ${SEED}
