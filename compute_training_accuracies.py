import os
import sys
import glob
import pickle
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from yacs.config import CfgNode as CN
from dassl.utils import set_random_seed
import clip
from trainers.Caption_tri_wta_soft_pseudolabel import load_clip_to_cpu
from trainers.fixed_prompt_utils import FIXED_PROMPTS_DICT
from dassl.evaluation import build_evaluator
import datasets.coco2014_partial
from dassl.config import get_cfg_default
from dassl.data import DataManager
from dassl.engine import build_trainer
import trainers.Caption_tri_wta_soft_pseudolabel
import trainers.Caption_tri_wta_soft_pseudolabel_ternary_cooccurrence
import trainers.Caption_tri_wta_soft_pseudolabel_ternary_cooccurrence_correctinitial
import trainers.Caption_tri_wta_soft_pseudolabel_ternary_cooccurrence_multistagecorrection
sys.path.append('cooccurrence_correction_experiments')
from compute_mAP import compute_mAP_main


RANDOM_SEED = 42
PLOT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/plots_and_tables')
CONFIG_FILE = os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_tri_wta_soft_pseudolabel/rn101.yaml')
DATASET_CONFIG_FILE = os.path.expanduser('~/data/dualcoopstarstar/configs/datasets/coco2014_partial.yaml')
DATA_ROOT = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data')
CTP = 'end'  # class token position (end or middle)
NCTX = 21  # number of context tokens
CSC = True  # class-specific context (False or True)
MOD = 'pos_norm' #yes this is used now


def load_cfg(job_dir):
    #use folder after job_dir to get config file name
    #remember to add all the stuff that's not in the config file
    #also, remember to make all the labels observed

    cfg = get_cfg_default()

    #extend_cfg stuff:
    cfg.TRAINER.Caption = CN()
    cfg.TRAINER.Caption.N_CTX = 16  # number of context vectors
    cfg.TRAINER.Caption.CSC = False  # class-specific context
    cfg.TRAINER.Caption.CTX_INIT = ""  # initialization words
    cfg.TRAINER.Caption.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.Caption.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.Caption.GL_merge_rate = 0.5
    cfg.TRAINER.Caption.USE_BIAS = 0
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SAMPLE = 0 # Sample some of all datas, 0 for no sampling i.e. using all
    cfg.DATASET.partial_prob = 1.0
    cfg.TRAIN.IF_LEARN_SCALE = False
    cfg.TRAIN.IF_LEARN_spatial_SCALE = False
    cfg.TRAIN.spatial_SCALE_text = 50
    cfg.TRAIN.spatial_SCALE_image = 50
    cfg.TRAIN.IF_ablation = False
    cfg.TRAIN.Caption_num = 0
    cfg.TRAIN.PSEUDOLABEL_INIT_METHOD = 'global_only'
    cfg.TRAIN.PSEUDOLABEL_INIT_PROMPT_KEY = 'ensemble_80'
    cfg.TRAIN.DO_ADJUST_LOGITS = 0
    cfg.TRAIN.ADJUST_LOGITS_MIN_BIAS = -2.0
    cfg.TRAIN.ADJUST_LOGITS_MAX_BIAS = 5.0
    cfg.TRAIN.ADJUST_LOGITS_TARGET = 2.89
    cfg.TRAIN.ADJUST_LOGITS_EPSILON = 0.05
    cfg.TRAIN.ADJUST_LOGITS_MAXITER = 10
    cfg.TRAIN.PSEUDOLABEL_UPDATE_MODE = 'gaussian_grad'
    cfg.TRAIN.PSEUDOLABEL_UPDATE_GAUSSIAN_BANDWIDTH = 0.2 #options will be 0.1, 0.2, 0.4
    cfg.TRAIN.PSEUDOLABEL_UPDATE_STEPSIZE = 0.25 #options will be 0.0625, 0.125, 0.25, 0.5, 1.0
    cfg.TRAIN.PSEUDOLABEL_UPDATE_FREQ = 1
    cfg.TRAIN.SKIP_PSEUDOLABEL_UPDATE_IN_CODE = 0
    cfg.TRAIN.TERNARY_COOCCURRENCE_LOSS_TYPE = 'prob_stopgrad_logit' #this kind of stuff is irrelevant because we're not retraining the models, just need to give them something to make them happy. Now, if it was *synonym* stuff then we'd need to pay attention!
    cfg.TRAIN.TERNARY_COOCCURRENCE_MAT_NAME = 'gt_epsilon0.25_zeta0.25'
    cfg.TRAIN.TERNARY_COOCCURRENCE_ALPHA = 0.25
    cfg.TRAIN.TERNARY_COOCCURRENCE_BETA_OVER_ALPHA = 0.5
    cfg.TRAIN.TERNARY_COOCCURRENCE_LOSS_OFF_DURATION = 0 #obsolete
    cfg.TRAIN.MULTISTAGECORRECTION_MU = 0.0
    cfg.TRAIN.MULTISTAGECORRECTION_EPOCHS = '_3_'
    cfg.TRAIN.LOSSFUNC = 'crossent'
    cfg.TEST.EVALUATOR = 'MLClassificationDualCoOpStarStar'
    cfg.TEST.EVALUATOR_ACT = 'default'
    cfg.TEST.SAVE_PREDS = ""
    cfg.INPUT.random_resized_crop_scale = (0.8, 1.0)
    cfg.INPUT.cutout_proportion = 0.4
    cfg.INPUT.TRANSFORMS_TEST = ("resize", "center_crop", "normalize")
    cfg.TRAIN.CHECKPOINT_FREQ = 1
    cfg.COMPUTE_RANDOM_CHANCE = 0
    cfg.COMPUTE_ZSCLIP = 0
    cfg.ZSCLIP_USE_COSSIM = 0
    cfg.EVAL_TRAINING_PSEUDOLABELS = 0

    #config file stuff:
    cfg.merge_from_file(DATASET_CONFIG_FILE)
    cfg.merge_from_file(CONFIG_FILE) #just use one, they're all the same

    #args stuff:
    cfg.DATASET.ROOT = DATA_ROOT
    cfg.SEED = RANDOM_SEED
    cfg.OUTPUT_DIR = os.path.dirname(get_model_dir(job_dir))
    cfg.MODE = MOD
    cfg.TRAINER.NAME = get_trainer_name(job_dir)
    cfg.TRAINER.Caption.N_CTX = NCTX
    cfg.TRAINER.Caption.CSC = CSC
    cfg.TRAINER.Caption.CLASS_TOKEN_POSITION = CTP
    cfg.TRAINER.Caption.USE_BIAS = 0
    cfg.TRAIN.DO_ADJUST_LOGITS = 0
    cfg.TRAIN.PSEUDOLABEL_UPDATE_GAUSSIAN_BANDWIDTH = 0.0
    cfg.TRAIN.PSEUDOLABEL_UPDATE_STEPSIZE = 0.0
    cfg.TRAIN.SKIP_PSEUDOLABEL_UPDATE_IN_CODE = 1

    cfg.freeze()

    return cfg


def get_trainer_name(job_dir):
    trainer_names = sorted(glob.glob(os.path.join(job_dir, '*')))
    assert(len(trainer_names) == 1)
    return os.path.basename(trainer_names[0])


def get_model_dir(job_dir):
    model_dirs = sorted(glob.glob(os.path.join(job_dir, '*/rn101/nctx21_cscTrue_ctpend/seed1')))
    print(model_dirs)
    assert(len(model_dirs) == 1)
    return model_dirs[0]


#this will give a generator that gives back trainers which are ready to do inference
def load_models(job_dir):
    model_dir = get_model_dir(job_dir)
    cfg = load_cfg(job_dir)
    for epoch in range(1, 51):
        trainer = build_trainer(cfg)
        trainer.load_model(model_dir, epoch=epoch)
        trainer.set_model_mode('eval')
        yield trainer, epoch


#this will get a random-seeded dataloader
#you should call this for every checkpoint
def get_dataloader(job_dir):
    set_random_seed(RANDOM_SEED)
    cfg = load_cfg(job_dir)
    dm = DataManager(cfg)
    return dm.train_loader_x_complete


def get_clip_model(cfg):
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
        # CLIP's default precision is fp16
        clip_model.float()

    clip_model.cuda()
    return clip_model


def softlogits2siglogits(softlogits):
    #how to turn softmax logits into sigmoid logits?
    #probs[i] = exp(-softlogits[i]) / sum_j exp(-softlogits[j])
    #probs[i] = 1 / (1 + exp(-siglogits[i]))
    #...
    #siglogits[i] = log(exp(softlogits[i]) / (sum_j exp(softlogits[j]) - exp(softlogits[i])))
    #             = softlogits[i] - log(sum_j exp(softlogits[j]) - exp(softlogits[i]))
    #best way to make it numerically stable is probably just to subtract the max of softlogits
    softlogits = softlogits - torch.max(softlogits, 1, keepdim=True)[0]
    sumexps = torch.sum(torch.exp(softlogits), 1, keepdim=True)
    siglogits = softlogits - torch.log(sumexps - torch.exp(softlogits))
    return siglogits


def compute_text_embeddings(classnames, clip_model, cfg):
    templates = FIXED_PROMPTS_DICT[cfg.TRAIN.PSEUDOLABEL_INIT_PROMPT_KEY]
    texts = [[template.format(classname) for template in templates] for classname in classnames]
    texts = [clip.tokenize(texts_sub) for texts_sub in texts]
    with torch.no_grad():
        texts = torch.cat(texts).cuda()
        fixed_text_embeddings = clip_model.encode_text(texts)
        fixed_text_embeddings = torch.reshape(fixed_text_embeddings, (len(classnames), len(templates), -1))
        fixed_text_embeddings = fixed_text_embeddings / fixed_text_embeddings.norm(dim=-1, keepdim=True)
        fixed_text_embeddings = torch.mean(fixed_text_embeddings, dim=1)
        fixed_text_embeddings = fixed_text_embeddings / fixed_text_embeddings.norm(dim=-1, keepdim=True)

    return fixed_text_embeddings


#this assumes that we're always using the same pseudolabeler strategy
def compute_pseudolabeler_accuracy(dataloader, job_dir):
    cfg = load_cfg(job_dir)
    evaluator = build_evaluator(cfg)
    evaluator.reset()
    clip_model = get_clip_model(cfg)
    dm = DataManager(cfg)
    text_embeddings = compute_text_embeddings(dm.dataset.classnames, clip_model, cfg)
    for batch in tqdm(dataloader):
        images = batch['img'].cuda()
        labels = batch['label'].cuda()
        with torch.no_grad():
            image_embeddings = clip_model.encode_image(images)
            assert(len(image_embeddings.shape) == 2)
            assert(image_embeddings.shape[0] == images.shape[0])
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            softlogits = clip_model.logit_scale.exp() * image_embeddings @ text_embeddings.t()
            logits = softlogits2siglogits(softlogits)

        evaluator.process({'default' : logits}, labels)

    results = evaluator.evaluate()
    return results['mAP']


def compute_one_train_accuracy(trainer, dataloader):
    trainer.evaluator.reset()
    for batch in tqdm(dataloader):
        images = batch['img'].cuda()
        labels = batch['label'].cuda()
        with torch.no_grad():
            _, logits, _, _ = trainer.model_inference(images)

        trainer.evaluator.process({'default' : logits}, labels)

    results = trainer.evaluator.evaluate()
    return results['mAP']


def get_plot_filename(job_dir):
    job_id = os.path.basename(job_dir)
    plot_base = os.path.join(PLOT_DIR, job_id + '-train_accs')
    return plot_base + '.png', plot_base + '.pkl'


#ommitted stuff can be None
def plot_and_save(pseudo_mAP, corrected_pseudo_mAP, secondcorrected_pseudo_mAP, secondcorrected_epoch, train_mAPs, train_epochs, title, plot_filename, plot_pkl_filename):
    d = {'pseudo_mAP' : pseudo_mAP, 'train_mAPs' : train_mAPs, 'train_epochs' : train_epochs, 'title' : title}

    #allow to store None in dict
    #it's easier that way
    d['corrected_pseudo_mAP'] = corrected_pseudo_mAP
    d['secondcorrected_pseudo_mAP'] = secondcorrected_pseudo_mAP
    d['secondcorrected_epoch'] = secondcorrected_epoch

    with open(plot_pkl_filename, 'wb') as f:
        pickle.dump(d, f)

    just_plot(pseudo_mAP, corrected_pseudo_mAP, secondcorrected_pseudo_mAP, secondcorrected_epoch, train_mAPs, train_epochs, title, plot_filename)


def just_plot(pseudo_mAP, corrected_pseudo_mAP, secondcorrected_pseudo_mAP, secondcorrected_epoch, train_mAPs, train_epochs, title, plot_filename):
    plt.clf()
    max_epoch = max(train_epochs)
    if secondcorrected_epoch is not None:
        max_epoch = max(max_epoch, secondcorrected_epoch + 1)

    plt.plot([0, max_epoch], [pseudo_mAP, pseudo_mAP], linestyle='dashed', color='k', label='pseudolabeler')
    plt.plot(train_epochs, train_mAPs, linestyle='solid', color='r', label='model')
    if corrected_pseudo_mAP is not None:
        plt.plot([0, max_epoch], [corrected_pseudo_mAP, corrected_pseudo_mAP], linestyle='dashed', color='b', label='pseudolabeler corrected')

    if secondcorrected_pseudo_mAP is not None:
        plt.plot([secondcorrected_epoch, max_epoch], [secondcorrected_pseudo_mAP, secondcorrected_pseudo_mAP], linestyle='dashed', color='g', label='pseudolabeler corrected 2nd stage (epoch %d)'%(secondcorrected_epoch))

    plt.grid(True, which='major', axis='both', color='grey', linestyle='-', linewidth=0.5)
    plt.minorticks_on()  # Enables minor ticks
    plt.grid(True, which='minor', axis='both', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.ylim((45, 70))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('train mAP')
    plt.title(title)
    plt.savefig(plot_filename, dpi=300)
    plt.clf()


def compute_stored_accuracy(stored_filename, return_epoch=False):
    return compute_mAP_main(stored_filename, return_epoch=return_epoch)


def compute_init_pseudo_accuracy(job_dir):
    print(job_dir)
    model_dir = get_model_dir(job_dir)
    stored_filename = os.path.join(model_dir, 'pseudolabels.pth.tar-init')
    return compute_stored_accuracy(stored_filename)


def compute_corrected_pseudo_accuracy(job_dir):
    model_dir = get_model_dir(job_dir)
    stored_filename = os.path.join(model_dir, 'pseudolabels.pth.tar-init_corrected')
    return compute_stored_accuracy(stored_filename)


def compute_secondcorrected_pseudo_accuracy(job_dir):
    model_dir = get_model_dir(job_dir)
    stored_filenames = sorted(glob.glob(os.path.join(model_dir, 'pseudolabels.pth.tar-corrected_epoch*')))
    assert(len(stored_filenames) == 1)
    stored_filename = stored_filenames[0]
    return compute_stored_accuracy(stored_filename, return_epoch=True)


def compute_training_accuracies(job_dir, title, correction_level):
    correction_level = int(correction_level)
    assert(correction_level in [0,1,2])

    plot_filename, plot_pkl_filename = get_plot_filename(job_dir)
    corrected_pseudo_mAP = None
    secondcorrected_pseudo_mAP = None
    secondcorrected_epoch = None
    if correction_level == 0:
        dataloader = get_dataloader(job_dir)
        pseudo_mAP = compute_pseudolabeler_accuracy(dataloader, job_dir)
    elif correction_level == 1:
        pseudo_mAP = compute_init_pseudo_accuracy(job_dir)
        corrected_pseudo_mAP = compute_corrected_pseudo_accuracy(job_dir)
    elif correction_level == 2:
        pseudo_mAP = compute_init_pseudo_accuracy(job_dir)
        corrected_pseudo_mAP = compute_corrected_pseudo_accuracy(job_dir)
        secondcorrected_pseudo_mAP, secondcorrected_epoch = compute_secondcorrected_pseudo_accuracy(job_dir)
    else:
        assert(False)

    train_mAPs = []
    train_epochs = []
    model_genny = load_models(job_dir)
    for trainer, epoch in tqdm(model_genny):
        dataloader = get_dataloader(job_dir)
        train_mAP = compute_one_train_accuracy(trainer, dataloader)
        train_mAPs.append(train_mAP)
        train_epochs.append(epoch)
        plot_and_save(pseudo_mAP, corrected_pseudo_mAP, secondcorrected_pseudo_mAP, secondcorrected_epoch, train_mAPs, train_epochs, title, plot_filename, plot_pkl_filename)


def usage():
    print('Usage: python compute_training_accuracies.py <job_dir> <title> <correction_level>')


if __name__ == '__main__':
    compute_training_accuracies(*(sys.argv[1:]))
