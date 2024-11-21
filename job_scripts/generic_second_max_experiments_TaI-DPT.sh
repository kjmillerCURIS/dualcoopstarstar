#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=2:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
#second_max_experiments(dataset_name, model_type, do_calibration, use_tagclip=0, tagclip_use_log=0, tagclip_use_rowcalibbase=0, use_taidpt=0, taidpt_is_actually_comc=0, taidpt_or_comc_use_rowcalibbase=0, taidpt_us_strong=0, taidpt_them_strong=0, taidpt_seed=None):
python cooccurrence_correction_experiments/second_max_experiments.py ${DATASET_NAME} RN50 1 0 0 0 1 0 ${USE_ROWCALIBBASE} ${US_STRONG} ${THEM_STRONG} ${SEED}

