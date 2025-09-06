#!/bin/bash

# en text data from FullAgreement data
python -m crpo.prepare_raw_data \
        -i CNCL-Penn-State/MultitaskDataset_FullAgreement \
        -o CNCL-Penn-State/MuCE \
        -m text \
        -l english \
        -c full_agreement