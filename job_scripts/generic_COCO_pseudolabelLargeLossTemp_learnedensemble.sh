#!/bin/bash -l

#$ -N COCO_01

#$ -m bea

#$ -M nivek@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Request 4 CPUs
#$ -pe omp 2

# Request 2 GPU
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=7.5

#$ -l gpu_memory=48G


#$ -l h_rt=48:00:00

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load miniconda
module load cuda/11.1
module load gcc
conda activate dualcoopstarstar
#conda install -c conda-forge opencv

cd ~/data/dualcoopstarstar

# custom config
DATA=~/data/vislang-domain-exploration-data/dualcoopstarstar-data
TRAINER=Caption_tri_wta_soft_pseudolabelLargeLossTemp_learnedensemble

DATASET=coco2014_partial
CFG=rn101  # config file
CTP=end  # class token position (end or middle)
NCTX=21  # number of context tokens
CSC=True  # class-specific context (False or True)
MOD=pos_norm #yes this is used now
#run_ID=coco2014_partial_tricoop_wta_soft_448_CSC_p0_1-pos200-ctx21_norm
#partial_prob=0.5

#for SEED in 1 3 5
for SEED in 1
do
    DIR=${DATA}/output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    echo "Run this job and save the output to ${DIR}"
    python train_caption_pseudolabelLargeLossTemp_learnedensemble.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --mode ${MOD} \
    TRAINER.Caption.N_CTX ${NCTX} \
    TRAINER.Caption.CSC ${CSC} \
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.Caption.USE_BIAS ${USE_BIAS} \
    TRAIN.PSEUDOLABEL_OBSERVATION_METHOD ${PSEUDOLABEL_OBSERVATION_METHOD} \
    TRAIN.DELTA_REL ${DELTA_REL} \
    TRAIN.MAX_EPOCH_FOR_DELTA_REL ${MAX_EPOCH_FOR_DELTA_REL} \
    llm_name ${llm_name} \
    TRAINER.Caption.THREE_SEPARATE_ENSEMBLES ${THREE_SEPARATE_ENSEMBLES} \
    TRAIN.INIT_WITH_ORIG_CLASSNAMES_ONLY ${ORIG_CLASSNAMES_INIT}
done

# VOC
# bash main_dual.sh voc2007_partial rn101 end 16 True voc2007_partial_dualcoop_448_CSC_p0_5 0.5 0

# COCO
# bash main_dual.sh coco2014_partial rn101 end 16 True coco2014_partial_dualcoop_448_CSC_p0_5 0.5 1

# NUSWIDE
# bash main_dual.sh nuswide_partial rn101_nus end 16 True nuswide_partial_dualcoop_448_CSC_p0_5 0.5 2
