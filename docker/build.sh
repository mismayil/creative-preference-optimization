#!/bin/sh

docker build -f ./docker/Dockerfile --build-arg DUMMY=${1} -t registry.rcp.epfl.ch/nlp/ismayilz/cpo --secret id=my_env,src=/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/ismayilz/.runai_env .