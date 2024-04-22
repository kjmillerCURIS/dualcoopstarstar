#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=1:59:59
#$ -j y
#$ -m ea
#$ -N tablify

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python tablify_dc_cooc_multistagecorrection_experiments.py

