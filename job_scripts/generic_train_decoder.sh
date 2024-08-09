#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=11:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/train_decoder.py ${DATASET_NAME} ${INPUT_TYPE} ${NUM_HIDDEN_LAYERS} ${HIDDEN_LAYER_SIZE} ${USE_DROPOUT} ${USE_BATCHNORM} ${LR}

