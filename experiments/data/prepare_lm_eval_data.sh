#!/bin/bash

# # EN LM eval data
# python -m crpo.prepare_lm_eval_data \
#         -d experiments/lm/data/en/MuCE_full_agreement_heldout_item_lm_data.json \
#         -o experiments/lm/data/en/eval

# python -m crpo.prepare_lm_eval_data \
#         -d experiments/lm/data/en/MuCE_full_agreement_heldout_task_lm_data.json \
#         -o experiments/lm/data/en/eval

# # EN LM eval data using specific constrained response template
# python -m crpo.prepare_lm_eval_data \
#         -d experiments/lm/data/en/MuCE_full_agreement_heldout_item_lm_data.json \
#         -o experiments/lm/data/en/eval_scr \
#         -t scr

# python -m crpo.prepare_lm_eval_data \
#         -d experiments/lm/data/en/MuCE_full_agreement_heldout_task_lm_data.json \
#         -o experiments/lm/data/en/eval_scr \
#         -t scr

# AUT LM eval data using specific constrained response template
# python -m crpo.prepare_lm_eval_data \
#         -d experiments/lm/data/en/cpo_en_aut_text_raw_lm_data.json \
#         -o experiments/lm/data/en/eval_scr \
#         -t aut_bs_scr

# AUT LM eval data
# python -m crpo.prepare_lm_eval_data \
#         -d experiments/lm/data/en/raw/cpo_en_aut_text_raw_lm_data.json \
#         -o experiments/lm/data/en/eval \
#         -t default