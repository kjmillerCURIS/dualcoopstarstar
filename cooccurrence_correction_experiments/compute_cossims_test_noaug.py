import os
import sys
import numpy as np
import pickle
import torch
from tqdm import tqdm
from harvest_training_gts import get_data_manager, get_cfg, TESTING_GTS_FILENAME_DICT
sys.path.append('.')
import clip
from trainers.Caption_tri_wta_soft_pseudolabel import load_clip_to_cpu
from trainers.fixed_prompt_utils import FIXED_PROMPTS_DICT
sys.path.pop()


BASE_DIR_DICT = {'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/VOC2007_test'), 'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/COCO2014_test'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/nuswide_test'), 'nuswideTheirVersion_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/nuswideTheirVersion_test')}

def get_input_filename(dataset_name, model_type, single_probe_type):
    return os.path.join(BASE_DIR_DICT[dataset_name], '%s_test_baseline_zsclip_cossims_%s_%s.pkl'%(dataset_name.split('_')[0], model_type, single_probe_type))


def get_clip_model(cfg):
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
        # CLIP's default precision is fp16
        clip_model.float()

    clip_model.cuda()
    return clip_model


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


def compute_text_embeddings(classnames, clip_model, cfg, single_probe_type):
    templates = FIXED_PROMPTS_DICT[single_probe_type]
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
    return dm.test_loader


#should return images, impaths
def parse_batch(batch, dm):
    images = batch["img"]
    idx = batch["idx"]
    impaths = [dm.dataset.test[t].impath for t in idx]
    return images, impaths


def compute_cossims_test_noaug(dataset_name, model_type, single_probe_type):
    input_filename = get_input_filename(dataset_name, model_type, single_probe_type)
    with open(TESTING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    cfg = get_cfg(dataset_name, model_type=model_type)
    dm = get_data_manager(dataset_name, cfg=cfg)
    clip_model = get_clip_model(cfg)
    classnames = dm.dataset.classnames
    text_embeddings = compute_text_embeddings(classnames, clip_model, cfg, single_probe_type)
    dataloader = get_dataloader(dm)
    pseudolabel_cossims = {}
    for batch in tqdm(dataloader):
        images, impaths = parse_batch(batch, dm)
        cossims = compute_pseudolabel_cossims(images, clip_model, text_embeddings)
        for impath, vec in zip(impaths, cossims):
            assert(impath not in pseudolabel_cossims)
            pseudolabel_cossims[impath] = vec

    impaths = sorted(gts.keys())
    input_dict = {'model_type' : model_type, 'single_probe_type' : single_probe_type, 'dataset_name' : dataset_name, 'classnames' : classnames, 'cossims' : np.array([pseudolabel_cossims[impath] for impath in impaths]), 'gts' : np.array([gts[impath] for impath in impaths]), 'impaths' : impaths, 'logit_scale' : clip_model.logit_scale.item()}
    with open(input_filename, 'wb') as f:
        pickle.dump(input_dict, f)


def usage():
    print('Usage: python compute_cossims_test_noaug.py <dataset_name> <model_type> <single_probe_type>')


if __name__ == '__main__':
    compute_cossims_test_noaug(*(sys.argv[1:]))
