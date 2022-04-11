#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=sc_800ep_cls_eval
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --mem=64G

DATASET_PATH="scratch/imagenet/"
EXPERIMENT_PATH="scratch/sc_experiments/sc_100ep_cls_eval"
PRETRAINED_PATH="scratch/sc_experiments/sc_100ep_train/model_100.pth.tar"
mkdir -p $EXPERIMENT_PATH

python -u ./src/cls_eval.py \
-j 2 \
-b 12 \
--print-freq 16 \
--cls-size 1000 2000 4000 8000 \
--num-cls 4 \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--tau 0.1 \
--use-bn \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
${DATASET_PATH}
