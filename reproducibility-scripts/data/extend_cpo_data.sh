#!/bin/bash

# # extend EN FA data with diversity, novelty, and surprise
# python extend_cpo_data.py --config configs/cpo_en_fa_data_ext_config.json \

# extend EN FA data with novelty including rejected
# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_full_nov \
#         -m novelty \
#         -ir \

# extend EN FA ms30 data with novelty
# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_ms30 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_ms30_nov \
#         -m novelty

# # extend EN FA ms11 data with novelty
# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_ms11 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_ms11_nov \
#         -m novelty

# # extend EN FA data with novelty computed from external references
# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_nov_er \
#         -m novelty \
#         -er experiments/lm/results/gemma-2-27b-it/en/train/cpo_en_multitask_text_raw_train_lm_data_eval_default_gemma-2-27b-it_696756a88a28.json \
#             experiments/lm/results/gemma-2-27b-it/en/train/cpo_en_multitask_text_raw_train_lm_data_eval_default_gemma-2-27b-it_bc307072fd2d.json

# extend multilingual FA data with diversity, novelty, and surprise
# python extend_cpo_data.py --config configs/cpo_multiling_fa_data_ext_config.json

# # extend EN FA data with quality
# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_div_nov_sur \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_div_nov_sur_qua \
#         -m quality

# extend EN FA data with novelty normalized by sum
# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_nov_nbs \
#         -m novelty \
#         -nbs \

# extend EN PA data with diversity, novelty, and surprise
# python extend_cpo_data.py --config configs/cpo_en_pa_data_ext_config.json \

# extend EN FA data with novelty score using all words in a text
# python extend_cpo_data.py --config configs/cpo_en_fa_data_ext_config2.json

# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_ms20 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm10_ms20_div_nov_sur_qua \
#         --config configs/cpo_en_fa_data_ext_config2.json

# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd10_mm20_ms20 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd10_mm20_ms20_div_nov_sur_qua \
#         --config configs/cpo_en_fa_data_ext_config2.json

# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm30_ms30 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_mm30_ms30_div_nov_sur_qua \
#         --config configs/cpo_en_fa_data_ext_config2.json

# python extend_cpo_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_ms30 \
#         -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_ms30_div_nov_sur_qua \
#         --config configs/cpo_en_fa_data_ext_config2.json

python extend_cpo_data.py \
        -i CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_ms40 \
        -o CNCL-Penn-State/cpo_en_multitask_text_fa_msd5_ms40_div_nov_sur_qua \
        --config configs/cpo_en_fa_data_ext_config2.json