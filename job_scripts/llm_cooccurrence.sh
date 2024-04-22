#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=1:59:59
#$ -j y
#$ -m ea
#$ -N llm_cooccurrence

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/llm_cooccurrence.py

