#!/bin/bash -l

#$ -N random_chance_p05_COCO_dualcoopstarstar_fullpartial

#$ -m bea

#$ -M nivek@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 2

#$ -l h_rt=05:59:59

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
TRAINER=DualCoOpStarStar_FullPartial

DATASET=coco2014_partial
CFG=rn101  # config file
CTP=end  # class token position (end or middle)
NCTX=21  # number of context tokens
CSC=True  # class-specific context (False or True)
MOD=pos_norm #(not really used anymore)
#run_ID=coco2014_partial_tricoop_wta_soft_448_CSC_p0_1-pos200-ctx21_norm
partial_prob=0.5

run_ID=random_chance_p05_COCO_dualcoopstarstar_fullpartial


#for SEED in 1 3 5
for SEED in 1
do
    DIR=${DATA}/output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    echo "Run this job and save the output to ${DIR}"
    python train_dualcoopstarstar_fullpartial.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --mode ${MOD} \
    --compute-random-chance \
    DATASET.partial_prob ${partial_prob} \
    TRAINER.DualCoOpStarStar.N_CTX ${NCTX} \
    TRAINER.DualCoOpStarStar.CSC ${CSC} \
    TRAINER.DualCoOpStarStar.CLASS_TOKEN_POSITION ${CTP}
done

# VOC
# bash main_dual.sh voc2007_partial rn101 end 16 True voc2007_partial_dualcoop_448_CSC_p0_5 0.5 0

# COCO
# bash main_dual.sh coco2014_partial rn101 end 16 True coco2014_partial_dualcoop_448_CSC_p0_5 0.5 1

# NUSWIDE
# bash main_dual.sh nuswide_partial rn101_nus end 16 True nuswide_partial_dualcoop_448_CSC_p0_5 0.5 2
