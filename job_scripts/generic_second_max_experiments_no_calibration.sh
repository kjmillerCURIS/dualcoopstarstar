#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=2:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/second_max_experiments.py ${DATASET_NAME} ${MODEL_TYPE} 0

