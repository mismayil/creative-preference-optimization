#!/bin/bash

####### final human evaluation data ##############
baselines=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "gpt-4o"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20"
)

cpo_nov_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
)

cpo_sur_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
)

cpo_qua_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua"
)

cpo_div_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
)

cpo_cre_models=(
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
)

pids=(
    'cpo_en_multitask_text_raw-train-648b9f7a3343'
    'cpo_en_multitask_text_raw-train-0416801c1cf0'
    'cpo_en_multitask_text_raw-train-bef6b3094023'
    'cpo_en_multitask_text_raw-train-2a969e9c1608'
)

################### novelty preparation ######################
input_paths=()

for model in "${baselines[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/novelty/final_eval_data_sent_comp_nov_base.json \
        -nr 4 \
        -m novelty \
        -pids "${pids[@]}"

input_paths=()

for model in "${cpo_nov_models[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/novelty/final_eval_data_sent_comp_nov_cpo.json \
        -nr 4 \
        -m novelty \
        -pids "${pids[@]}"
##############################################

################### surprise preparation ######################
input_paths=()

for model in "${baselines[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/surprise/final_eval_data_sent_comp_sur_base.json \
        -nr 4 \
        -m perplexity \
        -pids "${pids[@]}"

input_paths=()

for model in "${cpo_sur_models[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/surprise/final_eval_data_sent_comp_sur_cpo.json \
        -nr 4 \
        -m perplexity \
        -pids "${pids[@]}"
##############################################

################### quality preparation ######################
input_paths=()

for model in "${baselines[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/quality/final_eval_data_sent_comp_qua_base.json \
        -nr 4 \
        -m rewards.score \
        -pids "${pids[@]}"

input_paths=()

for model in "${cpo_qua_models[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/quality/final_eval_data_sent_comp_qua_cpo.json \
        -nr 4 \
        -m rewards.score \
        -pids "${pids[@]}"
##############################################

################### diversity preparation ######################
input_paths=()

for model in "${baselines[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/diversity/final_eval_data_sent_comp_div_base.json \
        -nr 4 \
        -m inv_homogen \
        -pids "${pids[@]}"

input_paths=()

for model in "${cpo_div_models[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/diversity/final_eval_data_sent_comp_div_cpo.json \
        -nr 4 \
        -m inv_homogen \
        -pids "${pids[@]}"
##############################################

################### creativity preparation ######################
input_paths=()

for model in "${baselines[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/creativity/final_eval_data_sent_comp_cre_base.json \
        -nr 4 \
        -m novelty perplexity rewards.score inv_homogen \
        -pids "${pids[@]}"

input_paths=()

for model in "${cpo_cre_models[@]}"; do
    model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
done

python -m crpo.prepare_human_eval_data \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/creativity/final_eval_data_sent_comp_cre_cpo.json \
        -nr 4 \
        -m novelty perplexity rewards.score inv_homogen \
        -pids "${pids[@]}"
##############################################