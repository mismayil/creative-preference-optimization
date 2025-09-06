#!/bin/bash

cd /mnt/scratch/home/ismayilz/project-cpo/MUSE/CPO
CONDA_ENV=cpo
CONDA=/home/ismayilz/.conda/condabin/conda

${CONDA} run -n ${CONDA_ENV} --live-stream bash "$@"
