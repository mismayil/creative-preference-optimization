#!/bin/bash

# en text data from Complete data
python prepare_raw_data.py \
        -i CNCL-Penn-State/MultitaskDataset_Complete \
        -o CNCL-Penn-State/cpo_en_multitask_text_raw \
        -m text \
        -l english

# en text data from PartialAgreement data
python prepare_raw_data.py \
        -i CNCL-Penn-State/MultitaskDataset_PartialAgreement \
        -o CNCL-Penn-State/cpo_en_multitask_text_pa_raw \
        -m text \
        -l english

# en text data from FullAgreement data
python prepare_raw_data.py \
        -i CNCL-Penn-State/MultitaskDataset_FullAgreement \
        -o CNCL-Penn-State/cpo_en_multitask_text_fa_raw \
        -m text \
        -l english

# multilingual text data from Complete data
python prepare_raw_data.py \
        -i CNCL-Penn-State/MultitaskDataset_Complete \
        -o CNCL-Penn-State/cpo_multiling_multitask_text_raw \
        -m text

# multilingual text data from PartialAgreement data
python prepare_raw_data.py \
        -i CNCL-Penn-State/MultitaskDataset_PartialAgreement \
        -o CNCL-Penn-State/cpo_multiling_multitask_text_pa_raw \
        -m text

# multilingual text data from FullAgreement data
python prepare_raw_data.py \
        -i CNCL-Penn-State/MultitaskDataset_FullAgreement \
        -o CNCL-Penn-State/cpo_multiling_multitask_text_fa_raw \
        -m text