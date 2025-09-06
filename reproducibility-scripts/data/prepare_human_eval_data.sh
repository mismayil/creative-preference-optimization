#!/bin/bash

baselines=(
    "claude-3-7-sonnet-20250219"
    "gpt-4o"
    "gemini-2.0-flash"
    "meta-llama/Llama-3.1-8B-Instruct"
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
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-full-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms30-nov"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-er"
    # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-nov-nbs"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-qua"
    "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre-qua"
)

human_eval_parameters=(
    "-t 0.7 -p 0.95"
    "-t 0.9 -p 0.99"
    "-t 0.7 -k 50"
    "-t 0.8 -p 0.97"
)

# input_paths=()

# for model in "${baselines[@]}"; do
#     for param in "${human_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_scr$suffix")
#     done
# done

# for model in "${cpo_models[@]}"; do
#     for param in "${human_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en$suffix")
#     done
# done

# for model in "${extra_cpo_models[@]}"; do
#     for param in "${human_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en$suffix")
#     done
# done

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/human_eval/human_eval_data_v5.json \
#         -t "Alternate Uses of Objects Task" "Hypothesis Generation" "Metaphors" \
#         -np 1 \
#         -fs "heldout_item_lm_data" "heldout_task_lm_data"

# ####### joke eval data ##############
# input_paths=()

# for model in "${baselines[@]}"; do
#     for param in "${human_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/jokes/jokes$suffix")
#     done
# done

# for model in "${cpo_models[@]}"; do
#     for param in "${human_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/jokes/jokes$suffix")
#     done
# done

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/jokes_eval/jokes_eval_data_v1.json

# anova_eval_parameters=(
#     "-t 0.7 -p 0.95"
#     "-t 0.9 -p 0.99"
#     "-t 0.7 -k 50"
#     "-t 0.8 -p 0.97"
#     "-t 0.7 -y 0.95"
#     "-t 0.75 -p 1.0"
#     "-t 0.85 -p 1.0"
#     "-t 0.95 -p 1.0"
# )

# input_paths=()

# for model in "${baselines[@]}"; do
#     for param in "${anova_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en_scr$suffix")
#     done
# done

# for model in "${cpo_models[@]}"; do
#     for param in "${anova_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en$suffix")
#     done
# done

# for model in "${extra_cpo_models[@]}"; do
#     for param in "${anova_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/en$suffix")
#     done
# done

### anova analysis data
# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/anova_eval/anova_eval_data_v4.json \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" \
#         -np 1 \
#         -ei 'experiments/lm/results/claude-3-7-sonnet-20250219/en_csr_t1.0_p1.0' \
#             'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-cre/en_csr_t1.0_p1.0' \
#             'experiments/lm/results/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-sur/en_csr_t1.0_p1.0' \
#             'experiments/lm/results/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30/en_csr_t1.0_p1.0'

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/anova_eval/anova_eval_data_v5.json \
#         -t "Real-Life Creative Problem Solving" "Alternate Uses of Objects Task" "Design Solutions" "Hypothesis Generation" "Metaphors" "Poems" "Associations" \

####### writing prompts eval data ##############
# input_paths=()

# for model in "${baselines[@]}"; do
#     for param in "${human_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/writing_prompts/wp$suffix")
#     done
# done

# for model in "${cpo_models[@]}"; do
#     for param in "${human_eval_parameters[@]}"; do
#         model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#         suffix=$(echo "$param" | tr -d ' ' | tr '-' '_')
#         input_paths+=("experiments/lm/results/$model_name/writing_prompts/wp$suffix")
#     done
# done

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/wp_eval/wp_eval_data_v1.json

###################### internal evaluation ##############################
# internal_eval_models=(
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms11-mult-cre"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd10-mm20-ms20-mult-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-mult-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-mult-nov-qua"
# )

# input_paths=()

# for model in "${internal_eval_models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths+=("experiments/lm/results/$model_name/en_heldout16")
# done

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/internal_eval/internal_eval_data_hypo_gen.json \
#         -nr 1 \
#         -m novelty \
#         -pid cpo_en_multitask_text_raw-heldout_item-19778154b169

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/internal_eval/internal_eval_data_design_sol.json \
#         -nr 1 \
#         -m novelty \
#         -pid cpo_en_multitask_text_raw-heldout_item-fbf6bc5199c8

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/internal_eval/internal_eval_data_aut.json \
#         -nr 1 \
#         -m novelty \
#         -pid cpo_en_multitask_text_raw-heldout_item-eda81e9d21cf

# internal_eval_models=(
#     # "claude-3-7-sonnet-20250219"
#     # "gpt-4o"
#     # "gemini-2.0-flash"
#     # "meta-llama/Llama-3.1-8B-Instruct"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-div"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-div-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-sur-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm30-ms30-cre-qua"
# )

# internal_eval_models=(
#     # "claude-3-7-sonnet-20250219"
#     # "gpt-4o"
#     # "gemini-2.0-flash"
#     # "meta-llama/Llama-3.1-8B-Instruct"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-sft-llama-3.1-8b-instruct-lora-fa-ms30"
#     # "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-nov-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-sur-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-div-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-cre-qua"
#     "/mnt/scratch/home/ismayilz/project-cpo/models/cpo-dpo-sft-ms30-llama-3.1-8b-instruct-lora-fa-msd5-mm10-ms20-qua"
# )

# input_paths=()

# for model in "${internal_eval_models[@]}"; do
#     model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
#     input_paths+=("experiments/lm/results/$model_name/en_sent_comp_sr/cpo_en_sent_comp_text_test_eval_data_sr.json")
# done

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/internal_eval/ms30_sr/internal_eval_data_sent_comp_ms30_sr_nov.json \
#         -nr 1 \
#         -m novelty \
#         -pid cpo_en_multitask_text_raw-train-ca08119f4c5d

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/internal_eval/ms30_sr/internal_eval_data_sent_comp_ms30_sr_sur.json \
#         -nr 1 \
#         -m perplexity \
#         -pid cpo_en_multitask_text_raw-train-ca08119f4c5d

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/internal_eval/ms30_sr/internal_eval_data_sent_comp_ms30_sr_div.json \
#         -nr 1 \
#         -m inv_homogen \
#         -pid cpo_en_multitask_text_raw-train-ca08119f4c5d

# python prepare_human_eval_data.py \
#         -i "${input_paths[@]}" \
#         -o experiments/data/internal_eval/ms30_sr/internal_eval_data_sent_comp_ms30_sr_qua.json \
#         -nr 1 \
#         -m rewards.score \
#         -pid cpo_en_multitask_text_raw-train-ca08119f4c5d

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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
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

python prepare_human_eval_data.py \
        -i "${input_paths[@]}" \
        -o experiments/data/final_human_eval/creativity/final_eval_data_sent_comp_cre_cpo.json \
        -nr 4 \
        -m novelty perplexity rewards.score inv_homogen \
        -pids "${pids[@]}"
##############################################