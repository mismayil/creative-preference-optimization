#!/bin/bash

# EN full agreement SFT data, sft min score 30
python -m crpo.prepare_sft_data \
        -i CNCL-Penn-State/MuCE \
        -o CNCL-Penn-State/MuCE-SFT \
        --sft-min-score 30 \
        -c "full_agreement"