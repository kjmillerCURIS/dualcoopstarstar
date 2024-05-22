#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=5:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/do_MWIS.py ${DATASET_NAME} ${CONFLICT_THRESHOLD} ${SCORE_TYPE}

