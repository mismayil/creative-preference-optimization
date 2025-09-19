#!/bin/bash

config_file=${1:-"configs/dpo/dpo_lora.yaml"}
accelerate=${2:-"False"}
accelerate_config_file=${3:-"configs/dpo/dpo_lora_accelerate_config.yml"}
num_gpus=$(nvidia-smi --list-gpus | wc -l)

printf "Using config file: ${config_file}\n"
printf "Using ${num_gpus} GPUs\n"

if [ ${accelerate} == "True" ]; then
    accelerate launch --num-processes $num_gpus --config_file ${accelerate_config_file} -m crpo.train_dpo --config ${config_file}
else
    python -m crpo.train_dpo --config ${config_file}
fi