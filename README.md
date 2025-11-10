# Creative Preference Optimization

![framework figure](./static/images/crpo.png)

[![Paper](https://img.shields.io/badge/Paper-ACL%20anthology-b31b1b.svg)](https://aclanthology.org/2025.findings-emnlp.509/)
[![Project](https://img.shields.io/badge/Project%20Page-blue.svg)](https://mete.is/creative-preference-optimization)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

This repo contains the code for the creative preference optimization (CrPO) framework. Models & Data can be found in [HuggingFace](https://huggingface.co/collections/CNCL-Penn-State/crpo-67d0b11ff358430823dbb3df).

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Reproducibility
`experiments` directory contains scripts to reproduce the SFT, DPO training and evaluation.

## Citation
```
@inproceedings{ismayilzada-etal-2025-creative,
    title = "Creative Preference Optimization",
    author = "Ismayilzada, Mete  and
      Laverghetta Jr., Antonio  and
      Luchini, Simone A.  and
      Patel, Reet  and
      Bosselut, Antoine  and
      Plas, Lonneke Van Der  and
      Beaty, Roger E.",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.509/",
    doi = "10.18653/v1/2025.findings-emnlp.509",
    pages = "9580--9609",
    ISBN = "979-8-89176-335-7"
}
```