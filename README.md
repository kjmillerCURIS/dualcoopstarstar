conda create -f environment.yml


That will create a conda environment called "dualcoopstarstar".


Some interesting codes:

* trainers/Caption_tri_wta_soft.py - this is DualCoOp++ with full or partial supervision

* trainers/Caption_tri_wta_soft_pseudolabel.py - DualCoOp++ with unsupervised learning using CDUL (cfg.TRAIN.PSEUDOLABEL_UPDATE_STEPSIZE controls step-size of pseudolabel update - setting it to 0 keeps pseudolabels frozen to their initial values)

* trainers/Caption_tri_wta_soft_pseudolabelLargeLossTemp.py - DualCoOp++ with unsupervised learning using LargeLoss (cfg.TRAIN.MAX_EPOCH_FOR_DELTA_REL and cfg.TRAIN.DELTA_REL control the schedule for the percentile for flipping pseudolabels. Setting both to 0 keeps pseudolabels frozen to their initial values. Also, LargeLoss published their code at https://github.com/snucml/LargeLossMatters, so you can take a look there if mine's too confusing.)

* trainers/Caption_tri_wta_soft_pseudolabel_learnedensemble.py - CDUL with learned ensemble of synonyms

* trainers/Caption_tri_wta_soft_pseudolabelLargeLossTemp_learnedensemble.py - LargeLoss with learned ensemble of synonyms


Scripts:

* job_scripts/baseline_COCO_05.sh - train DualCoOp++ with half of the labels supervised

* job_scripts/zsclip_COCO.sh - zero-shot CLIP (i.e. vanilla CLIP), with softmax which seems to help things

* job_scripts/zsclip_use_cossim_COCO.sh - zero-shot CLIP (i.e. vanilla CLIP), just directly using cosine similarities as the scores

* job_scripts/zsclip_llmensemble_COCO.sh - zero-shot CLIP (i.e. vanilla CLIP), with softmax, using ensemble of synonyms

* job_scripts/submit_pseudolabelLargeLossTemp_runs.py, job_scripts/submit_pseudolabel_runs.py, job_scripts/submit_pseudo_llmensemble_runs.py, job_scripts/submit_zsclip_llmensemble_runs.py - does hyperparameter sweeps. submit_pseudo_llmensemble_runs.py handles both CDUL and LargeLoss for the learned-synonyms-ensemble idea.

* Test mAP (i.e. the "validation" set of MSCOCO) is reported in the .o files, and also saved in output directory (more on that later). I still need to catch up on some of the plotting/table scripts. I think there are tensorboard environments too, but not sure...


Data:

* All my data and outputs are in /usr3/graduate/nivek/data/vislang-domain-exploration-data/dualcoopstarstar-data

* Each directory in dualcoopstarstar-data/output is one experiment

* Each directory in dualcoopstarstar-data/mscoco_2014 is where the dataset is
