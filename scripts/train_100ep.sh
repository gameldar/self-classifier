#!/usr/bin/env bash
#SBATCH --nodes=11
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sc_100ep_train
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr
#SBATCH --mem=64G

DATASET_PATH="scratch/imagenet/"
EXPERIMENT_PATH="scratch/sc_experiments/sc_100ep_train"
mkdir -p $EXPERIMENT_PATH

#srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label
python -u ./src/train.py \
-j 2 \
-b 8 \
--print-freq 16 \
--epochs 100 \
--lr 4.8 \
--start-warmup 0.3 \
--final-lr 0.0048 \
--lars \
--sgd \
--cos \
--wd 1e-6 \
--cls-size 1000 2000 4000 8000 \
--num-cls 4 \
--queue-len 262144 \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--row-tau 0.1 \
--col-tau 0.05 \
--global-crops-scale 0.4 1.0 \
--local-crops-scale 0.05 0.4 \
--local-crops-number 6 \
--use-bn \
--save-path ${EXPERIMENT_PATH} \
${DATASET_PATH}
