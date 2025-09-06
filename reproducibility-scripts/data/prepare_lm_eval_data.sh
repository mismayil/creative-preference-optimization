#!/bin/bash

# # EN LM eval data
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/en/test/raw \
#         -o experiments/lm/data/en/test/eval

# # multilingual LM eval data
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/multiling/test/raw \
#         -o experiments/lm/data/multiling/test/eval

# EN LM train data
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/en/train/raw/cpo_en_multitask_text_raw_train_lm_data.json \
#         -o experiments/lm/data/en/train/eval

# # EN LM eval data using short response template
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/en/test/raw \
#         -o experiments/lm/data/en/test/eval_sr \
#         -t short_response

# EN LM eval data using constrained response template
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/en/test/raw \
#         -o experiments/lm/data/en/test/eval_cr \
#         -t constrained_response

# # EN LM eval data using specific constrained response template
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/en/test/raw \
#         -o experiments/lm/data/en/test/eval_scr \
#         -t scr

# multilingual LM eval data using short response template
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/multiling/test/raw \
#         -o experiments/lm/data/multiling/test/eval_sr \
#         -t short_response

# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/jokes/raw \
#         -o experiments/lm/data/jokes/eval

# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/writing_prompts/raw \
#         -o experiments/lm/data/writing_prompts/eval

# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/creative_math/raw \
#         -o experiments/lm/data/creative_math/eval \
#         -t cmath

# AUT LM eval data using specific constrained response template
# python prepare_lm_eval_data.py \
#         -d experiments/lm/data/en/test/raw/cpo_en_aut_text_raw_lm_data.json \
#         -o experiments/lm/data/en/test/eval_scr \
#         -t aut_bs_scr

# AUT LM eval data
python prepare_lm_eval_data.py \
        -d experiments/lm/data/en/test/raw/cpo_en_aut_text_raw_lm_data.json \
        -o experiments/lm/data/en/test/eval \
        -t default