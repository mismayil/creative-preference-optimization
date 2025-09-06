#!/bin/bash

# EN LM data
python prepare_lm_data.py \
        -i CNCL-Penn-State/cpo_en_multitask_text_raw \
        -o experiments/lm/data/en

# multilingual LM data
python prepare_lm_data.py \
        -i CNCL-Penn-State/cpo_multiling_multitask_text_raw \
        -o experiments/lm/data/multiling/test