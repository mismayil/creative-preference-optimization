#!/bin/bash

################### for final human evaluation ##############

baselines=(
    # "claude-3-7-sonnet-20250219"
    "gpt-4o"
    # "gemini-2.0-flash"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-Small-24B-Instruct-2501"
    # "mistralai/Mistral-7B-Instruct-v0.3"
)

cpo_models=(
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre-qua"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-fa-msd5-mm10-nov"
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
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua"
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
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua"
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

parameters=(
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
# for model in "${baselines[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval_scr/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_scr.json \
#                experiments/lm/data/en/test/eval_scr/cpo_en_multitask_text_raw_heldout_task_lm_data_eval_scr.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_scr${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

# for model in "${baselines[@]}"; do
#     for param in "${min_p_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval_scr/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_scr.json \
#                experiments/lm/data/en/test/eval_scr/cpo_en_multitask_text_raw_heldout_task_lm_data_eval_scr.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_scr${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

# for model in "${baselines[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_stories_text_raw_test_lm_data_eval_default.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_test_stories/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

# for model in "${ms20_s12_lambda_models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default.json \
#                experiments/lm/data/en/test/eval/cpo_en_multitask_text_raw_heldout_task_lm_data_eval_default.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

# for model in "${cpo_orpo_models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default.json \
#                experiments/lm/data/en/test/eval/cpo_en_multitask_text_raw_heldout_task_lm_data_eval_default.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

# for model in "${new_cpo_models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_stories_text_raw_test_lm_data_eval_default.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_test_stories/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

############################## sentence completion experiments ######################
# models=(
#     ${baselines[@]}
#     # ${final_cpo_models[@]}
# )

# for model in "${ms20_s12_lambda_models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_sent_comp_text_raw_test_lm_data_eval_sr.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_sent_comp_sr/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 128 \
#             -nf 4
#     done
# done

# for model in "${baselines[@]}"; do
#     for param in "${min_p_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_sent_comp_text_raw_test_lm_data_eval_sr.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_sent_comp_sr/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 128 \
#             -nf 4
#     done
# done

############################## aut bs experiments ######################
# models=(
#     ${baselines[@]}
#     # ${final_cpo_models[@]}
# )

aut_bs_parameters=(
    "-t 0.7 -p 0.95"
    "-t 0.9 -p 0.99"
    "-t 0.7 -k 50"
    "-t 0.8 -p 0.97"
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

# for model in "${baselines[@]}"; do
#     for param in "${aut_bs_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval_scr/cpo_en_aut_text_raw_lm_data_eval_aut_bs_scr.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_aut_bs_scr/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 512 \
#             -nf 4
#     done
# done

for model in "${aut_bs_models[@]}"; do
    for param in "${aut_bs_parameters[@]}"; do
        model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
        suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
        echo "Running inference for $model with parameters: $param"
        python inference_lm.py \
            -d experiments/lm/data/en/test/eval/cpo_en_aut_text_raw_lm_data_eval_default.json \
            -m "$model" \
            -o "experiments/lm/results/${model_name}/en_aut/en${suffix}" \
            -b 32 \
            $param \
            -g 128 \
            -nf 4
    done
done
######### jokes experiments ######################

##########  to remove duplicates
# python inference_lm.py \
#     -d experiments/lm/data/en/test/eval_scr_man/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_scr_man.json \
#     -m claude-3-7-sonnet-20250219 \
#     -o "experiments/lm/results/claude-3-7-sonnet-20250219/en_scr_t1.0_p1.0" \
#     -b 32 \
#     -t 1.0 \
#     -p 1.0 \
#     -g 1024 \
#     -nf 4

# python inference_lm.py \
#     -d experiments/lm/data/en/test/eval_scr_man/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_scr_man.json \
#     -m "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30" \
#     -o "experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_scr_t1.0_p1.0" \
#     -b 32 \
#     -t 1.0 \
#     -p 1.0 \
#     -g 1024 \
#     -nf 2

# python inference_lm.py \
#     -d experiments/lm/data/en/test/eval_scr_man/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_scr_man.json \
#     -m "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur" \
#     -o "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en_scr_t1.0_p1.0" \
#     -b 32 \
#     -t 1.0 \
#     -p 1.0 \
#     -g 1024 \
#     -nf 2

# python inference_lm.py \
#     -d experiments/lm/data/en/test/eval_scr_man/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_scr_man.json \
#     -m "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre" \
#     -o "experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en_scr_t1.0_p1.0" \
#     -b 32 \
#     -t 1.0 \
#     -p 1.0 \
#     -g 1024 \
#     -nf 2
##################################

# cpo_pa_models=(
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-pa-ms30"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-pa-msd5-mm10"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-pa-msd5-mm10-div"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-pa-msd5-mm10-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-pa-msd5-mm10-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-pa-msd5-mm10-cre"
# )

# pa_parameters=(
#     "-t 0.7 -p 0.95"
# )

# for model in "${cpo_pa_models[@]}"; do
#     for param in "${pa_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done


################ jokes experiments ######################
# baselines=(
#     "claude-3-7-sonnet-20250219"
#     "gpt-4o"
#     "gemini-2.0-flash"
#     "meta-llama/Llama-3.1-8B-Instruct"
# )

# cpo_models=(
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua"
# )

# parameters=(
#     "-t 0.7 -p 0.95"
#     "-t 0.9 -p 0.99"
#     "-t 0.7 -k 50"
#     "-t 0.8 -p 0.97"
#     # "-t 0.7 -y 0.95"
#     # "-t 0.75 -p 1.0"
#     # "-t 0.85 -p 1.0"
#     # "-t 0.95 -p 1.0"
# )

# models=(
#     ${baselines[@]}
#     ${cpo_models[@]}
# )

# for model in "${models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/jokes/eval \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/jokes/jokes${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

###########################################

# ################ writing prompts experiments ######################
# baselines=(
#     "claude-3-7-sonnet-20250219"
#     "gpt-4o"
#     "gemini-2.0-flash"
#     "meta-llama/Llama-3.1-8B-Instruct"
# )

# cpo_models=(
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua"
# )

# models=(
#     # ${baselines[@]}
#     ${cpo_models[@]}
# )

# parameters=(
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
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/writing_prompts/eval \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/writing_prompts/wp${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

# ##############################

################# creative math experiments ######################
# baselines=(
# #     "claude-3-7-sonnet-20250219"
# #     "gpt-4o"
# #     "gemini-2.0-flash"
#     # "meta-llama/Llama-3.1-8B-Instruct"
# )

# cpo_models=(
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua"
# )

# models=(
#     # ${baselines[@]}
#     ${cpo_models[@]}
# )

# parameters=(
#     "-t 0.7 -p 0.95"
#     # "-t 0.9 -p 0.99"
#     # "-t 0.7 -k 50"
#     # "-t 0.8 -p 0.97"
#     # "-t 0.7 -y 0.95"
#     # "-t 0.75 -p 1.0"
#     # "-t 0.85 -p 1.0"
#     # "-t 0.95 -p 1.0"
# )

# for model in "${models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/creative_math/eval \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/creative_math/cm${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done


################## orpo experiments ######################
# cpo_orpo_models=(
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-div"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-orpo-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre"
# )

# parameters=(
#     "-t 0.7 -p 0.95"
#     "-t 0.9 -p 0.99"
#     "-t 0.7 -k 50"
#     "-t 0.8 -p 0.97"
# )


# for model in "${cpo_orpo_models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_multitask_text_raw_heldout_item_lm_data_eval_default.json \
#                 experiments/lm/data/en/test/eval/cpo_en_multitask_text_raw_heldout_task_lm_data_eval_default.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done


####################### short stories experiments ######################
# models=(
#     ${baselines[@]}
#     ${ms11_cpo_models[@]}
# )

# for model in "${models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/en/test/eval/cpo_en_stories_text_raw_test_lm_data_eval_default.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/en_test_stories/en${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024 \
#             -nf 4
#     done
# done

################## ad hoc experiments ######################
# models=(
#     ${baselines[@]}
#     ${ms20_cpo_models[@]}
# )

# parameters=(
#     "-t 0.7 -p 0.95"
#     # "-t 0.9 -p 0.99"
#     # "-t 0.7 -k 50"
#     # "-t 0.8 -p 0.97"
#     # "-t 0.7 -y 0.95"
#     # "-t 0.75 -p 1.0"
#     # "-t 0.85 -p 1.0"
#     # "-t 0.95 -p 1.0"
# )

# for model in "${models[@]}"; do
#     for param in "${parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         echo "Running inference for $model with parameters: $param"
#         python inference_lm.py \
#             -d experiments/lm/data/adhoc/adhoc_lm_data_eval_default.json \
#             -m "$model" \
#             -o "experiments/lm/results/${model_name}/adhoc/adhoc${suffix}" \
#             -b 32 \
#             $param \
#             -g 1024
#     done
# done