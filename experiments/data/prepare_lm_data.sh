#!/bin/bash

# EN LM data
python -m crpo.prepare_lm_data \
        -i CNCL-Penn-State/MuCE \
        -o experiments/lm/data/en \
        -c full_agreement