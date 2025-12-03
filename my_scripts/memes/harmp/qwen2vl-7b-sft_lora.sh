#!/bin/bash
#SBATCH -J qwen2vl-7b-harmp-sft
#SBATCH -A GVDD-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
model_name="qwen2vl-7B"
dataset_name="harmp"
mode="lora_sft"
additional_args="" # Rember to add _ before the additional args

current_date=$(date +"%Y-%m-%d")
Name="${mode}${additional_args}"
export WANDB_PROJECT="LLAMAFACTORY_hateful"
#export WANDB_NAME=$Name
export RUN_NAME=$Name
export DATE=$current_date
export WANDB_RUN_GROUP="Finetuning_${model_name}_${dataset_name}_${current_date}"

which python

envsubst < my_configs/memes/${dataset_name}/${model_name}_${mode}${additional_args}.yaml > my_configs/memes/${dataset_name}/${model_name}_${mode}${additional_args}_temp.yaml

llamafactory-cli train my_configs/memes/${dataset_name}/${model_name}_${mode}${additional_args}_temp.yaml

rm my_configs/memes/${dataset_name}/${model_name}_${mode}${additional_args}_temp.yaml