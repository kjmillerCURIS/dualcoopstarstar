#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=5:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/logistic_regression_experiment.py ${DATASET_NAME} ${INPUT_TYPE} ${STANDARDIZE} ${BALANCE} ${L1} ${C} ${MINICLASS}

