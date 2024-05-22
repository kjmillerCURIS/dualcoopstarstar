import os
import sys
import pickle
import torch
from tqdm import tqdm
from harvest_training_gts import get_data_manager, get_cfg
sys.path.append('.')
import clip
from trainers.Caption_tri_wta_soft_pseudolabel import load_clip_to_cpu
from trainers.fixed_prompt_utils import FIXED_PROMPTS_DICT
sys.path.pop()


PSEUDOLABEL_COSSIMS_FILENAME_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training_init_pseudolabel_cossims.pkl'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/nuswide_training_init_pseudolabel_cossims.pkl'), 'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/voc2007_training_init_pseudolabel_cossims.pkl')}


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


#should return npy array on cpu
def compute_pseudolabel_cossims(images, clip_model, text_embeddings):
    with torch.no_grad():
        images = images.cuda()
        image_embeddings = clip_model.encode_image(images)
        assert(len(image_embeddings.shape) == 2)
        assert(image_embeddings.shape[0] == images.shape[0])
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        cossims = image_embeddings @ text_embeddings.t()

    return cossims.cpu().numpy()


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


def get_dataloader(dm):
    return dm.train_loader_x_complete


#should return images, impaths
def parse_batch(batch, dm):
    images = batch["img"]
    idx = batch["idx"]
    impaths = [dm.dataset.train_x[t].impath for t in idx]
    return images, impaths


def compute_initial_cossims(dataset_name):
    dm = get_data_manager(dataset_name)
    cfg = get_cfg(dataset_name)
    clip_model = get_clip_model(cfg)
    classnames = dm.dataset.classnames
    text_embeddings = compute_text_embeddings(classnames, clip_model, cfg)
    dataloader = get_dataloader(dm)
    pseudolabel_cossims = {}
    for batch in tqdm(dataloader):
        images, impaths = parse_batch(batch, dm)
        cossims = compute_pseudolabel_cossims(images, clip_model, text_embeddings)
        for impath, vec in zip(impaths, cossims):
            assert(impath not in pseudolabel_cossims)
            pseudolabel_cossims[impath] = vec

    with open(PSEUDOLABEL_COSSIMS_FILENAME_DICT[dataset_name], 'wb') as f:
        pickle.dump(pseudolabel_cossims, f)


def usage():
    print('Usage: python compute_initial_cossims.py <dataset_name>')


if __name__ == '__main__':
    compute_initial_cossims(*(sys.argv[1:]))
