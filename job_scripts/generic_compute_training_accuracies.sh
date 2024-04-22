#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=23:59:59
#$ -j y
#$ -m ea
#$ -M nivek@bu.edu
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l gpu_memory=24G

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python compute_training_accuracies.py ${JOB_DIR} ${TITLE} ${CORRECTION_LEVEL}

