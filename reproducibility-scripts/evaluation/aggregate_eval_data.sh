#!/bin/bash

baselines=(
    # "claude-3-7-sonnet-20250219"
    # "gpt-4o"
    # "gemini-2.0-flash"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-Small-24B-Instruct-2501"
    # "mistralai/Mistral-7B-Instruct-v0.3"
)

cpo_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre"
)

extra_cpo_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-fa-msd5-mm10-nov"
)

ms11_cpo_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-cre-qua"
)

tulu_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-tulu3-lora-fa-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-tulu3-lora-fa-msd5-mm10-ms11-nov"
)

new_cpo_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-ms30"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-cre"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-mistral-small-24b-instruct-lora-fa-ms30"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-small-24b-instruct-lora-fa-msd5-mm10-ms11"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-mult-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-mult-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-mult-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre-qua"
)

ms30_cpo_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-div-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-sur-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-qua"
)

ms20_cpo_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua"
)

all_cpo_models=(
    # ${cpo_models[@]}
    # ${extra_cpo_models[@]}
    # ${ms11_cpo_models[@]}
    # ${tulu_models[@]}
    ${new_cpo_models[@]}
)

models=(
    ${baselines[@]}
    # ${cpo_models[@]}
    ${extra_cpo_models[@]}
)

ms20_lambda_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l0.5"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l1.5"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l2.0"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l0.5"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l1.5"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l2.0"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l0.5"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l1.5"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l2.0"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l0.5"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l1.5"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l2.0"
)

ms20_s12_lambda_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l0.5-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l1.5-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l2.0-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l0.5-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l1.5-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l2.0-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l0.5-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l1.5-s12"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l2.0-s12"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-s12"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l0.5-s12"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l1.5-s12"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l2.0-s12"
)

ms20_s92_lambda_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l0.5-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l1.5-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-l2.0-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l0.5-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l1.5-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-l2.0-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l0.5-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l1.5-s92"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-l2.0-s92"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-s92"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l0.5-s92"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l1.5-s92"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua-l2.0-s92"
)

cpo_orpo_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua"
)

mistral_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-mistral-7b-instruct-lora-fa-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-mistral-7b-instruct-lora-fa-msd5-mm10-ms20-qua"
)

eval_parameters=(
    "-t 0.7 -p 0.95"
    "-t 0.9 -p 0.99"
    "-t 0.7 -k 50"
    "-t 0.8 -p 0.97"
    # "-t 0.7 -y 0.95"
    # "-t 0.75 -p 1.0"
    # "-t 0.85 -p 1.0"
    # "-t 0.95 -p 1.0"
)

min_p_parameters=(
    "-t 1.0 -mp 0.1"
    "-t 1.5 -mp 0.1"
    "-t 1.0 -mp 0.05"
    "-t 1.5 -mp 0.05"
)

#################### filtered eval data experiments ######################
# for model in "${baselines[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_scr$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_scr_heldout16/cpo_en_multitask_text_heldout_eval_data_scr16.json \
#             -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#             -np 1 \
#             -fs "heldout_item_lm_data" "heldout_task_lm_data"
# done

# for model in "${baselines[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${min_p_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_scr$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_scr_heldout16_min_p/cpo_en_multitask_text_heldout_eval_data_scr16_min_p.json \
#             -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#             -np 1 \
#             -fs "heldout_item_lm_data" "heldout_task_lm_data"
# done

# for model in "${baselines[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_test_stories/en$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_test_stories/cpo_en_stories_text_test_eval_data_default.json
# done

# for model in "${ms20_s12_lambda_models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_heldout16/cpo_en_multitask_text_heldout_eval_data_default16.json \
#             -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#             -np 1 \
#             -fs "heldout_item_lm_data" "heldout_task_lm_data"
# done

# for model in "${new_cpo_models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_test_stories/en$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_test_stories/cpo_en_stories_text_test_eval_data_default.json
# done

################ sentence completion experiments ######################
# models=(
#     ${baselines[@]}
#     ${final_cpo_models[@]}
# )

# for model in "${ms20_s12_lambda_models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/en$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json
# done

# for model in "${baselines[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${min_p_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/en$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_sent_comp_sr_min_p/cpo_en_sent_comp_text_test_eval_data_sr_min_p.json
# done

################ aut experiments ######################
baselines=(
    "gpt-4o"
)

aut_bs_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua"
)

aut_bs_parameters=(
    "-t 0.7 -p 0.95"
    "-t 0.9 -p 0.99"
    "-t 0.7 -k 50"
    "-t 0.8 -p 0.97"
)

for model in "${baselines[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths=()
    for param in "${aut_bs_parameters[@]}"; do
        suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
        input_paths+=("experiments/lm/results/$model_name/en_aut_bs_scr/en$suffix")
    done
    python aggregate_eval_data.py \
            -i "${input_paths[@]}" \
            -o experiments/lm/results/$model_name/en_aut_bs_scr/cpo_en_aut_text_test_eval_data_bs_scr.json
done

for model in "${aut_bs_models[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths=()
    for param in "${aut_bs_parameters[@]}"; do
        suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
        input_paths+=("experiments/lm/results/$model_name/en_aut/en$suffix")
    done
    python aggregate_eval_data.py \
            -i "${input_paths[@]}" \
            -o experiments/lm/results/$model_name/en_aut/cpo_en_aut_text_test_eval_data_default.json
done

# ##################### jokes experiments ######################
# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${human_eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/jokes/jokes$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/jokes/jokes_eval_default_all.json
# done


##################### writing prompts experiments ######################

# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${human_eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/writing_prompts/wp$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/writing_prompts/writing_prompts_eval_default_all.json
# done

##################### orpo experiments ######################
# cpo_orpo_models=(
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre"
# )

# for model in "${cpo_orpo_models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${human_eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_heldout/cpo_en_multitask_text_heldout_eval_data_default.json \
#             -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
# done

#################### test stories experiments ######################
# models=(
#     ${baselines[@]}
#     ${ms11_cpo_models[@]}
# )

# eval_parameters=(
#     "-t 0.7 -p 0.95"
#     "-t 0.9 -p 0.99"
#     "-t 0.7 -k 50"
#     "-t 0.8 -p 0.97"
#     # "-t 0.7 -y 0.95"
#     # "-t 0.75 -p 1.0"
#     # "-t 0.85 -p 1.0"
#     # "-t 0.95 -p 1.0"
# )

# for model in "${models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths=()
#     for param in "${eval_parameters[@]}"; do
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_test_stories/en$suffix")
#     done
#     python aggregate_eval_data.py \
#             -i "${input_paths[@]}" \
#             -o experiments/lm/results/$model_name/en_test_stories/cpo_en_stories_text_test_eval_data_default.json
# done