#!/bin/bash

# # EN complete SFT data, sft min score 30
# python prepare_sft_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_raw \
#         -o CNCL-Penn-State/cpo_en_multitask_text_sft_ms30 \
#         -it "Sentence Completion" \
#         --sft-min-score 30 \

# # EN partial agreement SFT data, sft min score 30
# python prepare_sft_data.py \
#         -i CNCL-Penn-State/cpo_en_multitask_text_pa_raw \
#         -o CNCL-Penn-State/cpo_en_multitask_text_pa_sft_ms30 \
#         -it "Sentence Completion" \
#         --sft-min-score 30 \

# EN full agreement SFT data, sft min score 30
python prepare_sft_data.py \
        -i CNCL-Penn-State/cpo_en_multitask_text_fa_raw \
        -o CNCL-Penn-State/cpo_en_multitask_text_fa_sft_ms30 \
        -it "Sentence Completion" \
        --sft-min-score 30

# # multilingual complete SFT data, sft min score 30
# python prepare_sft_data.py \
#         -i CNCL-Penn-State/cpo_multiling_multitask_text_raw \
#         -o CNCL-Penn-State/cpo_multiling_multitask_text_sft_ms30 \
#         -it "Sentence Completion" \
#         --sft-min-score 30 \

# # multilingual partial agreement SFT data, sft min score 30
# python prepare_sft_data.py \
#         -i CNCL-Penn-State/cpo_multiling_multitask_text_pa_raw \
#         -o CNCL-Penn-State/cpo_multiling_multitask_text_pa_sft_ms30 \
#         -it "Sentence Completion" \
#         --sft-min-score 30 \

# # multilingual full agreement SFT data, sft min score 30
# python prepare_sft_data.py \
#         -i CNCL-Penn-State/cpo_multiling_multitask_text_fa_raw \
#         -o CNCL-Penn-State/cpo_multiling_multitask_text_fa_sft_ms30 \
#         -it "Sentence Completion" \
#         --sft-min-score 30 \