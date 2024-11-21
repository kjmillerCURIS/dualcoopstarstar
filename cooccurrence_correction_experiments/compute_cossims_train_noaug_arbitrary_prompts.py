import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from harvest_training_gts import get_data_manager, get_cfg, TRAINING_GTS_FILENAME_DICT
sys.path.append('.')
import clip
from trainers.Caption_tri_wta_soft_pseudolabel import load_clip_to_cpu
from trainers.fixed_prompt_utils import FIXED_PROMPTS_DICT
sys.path.pop()


BASE_DIR_DICT = {'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/nuswide')}
PROMPTS_FILENAME_DICT = {'nuswide_partial' : os.path.join(BASE_DIR_DICT['nuswide_partial'], 'nus_wide_cooccurrence_triplets.txt')}


def load_prompts(dataset_name):
    prompts_filename = PROMPTS_FILENAME_DICT[dataset_name]
    f = open(prompts_filename, 'r')
    lines = f.readlines()
    f.close()
    return [line.rstrip('\n') for line in lines]


def get_input_filenames(dataset_name, model_type):
    input_pkl_filename = os.path.join(BASE_DIR_DICT[dataset_name], 'compound_cossims_%s_%s.pkl'%(dataset_name.split('_')[0], model_type))
    input_csv_filename = os.path.join(BASE_DIR_DICT[dataset_name], 'compound_cossims_%s_%s.csv'%(dataset_name.split('_')[0], model_type))
    return input_pkl_filename, input_csv_filename


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


def compute_text_embeddings(prompts, clip_model, cfg):
    texts = clip.tokenize(prompts)
    with torch.no_grad():
        texts = texts.cuda()
        fixed_text_embeddings = clip_model.encode_text(texts)
        fixed_text_embeddings = fixed_text_embeddings / fixed_text_embeddings.norm(dim=-1, keepdim=True)

    return fixed_text_embeddings


def get_dataloader(dm):
    return dm.train_loader_x_complete_noaug


#should return images, impaths
def parse_batch(batch, dm):
    images = batch["img"]
    idx = batch["idx"]
    impaths = [dm.dataset.train_x[t].impath for t in idx]
    return images, impaths


def compute_cossims_train_noaug_arbitrary_prompts(dataset_name, model_type):
    prompts = load_prompts(dataset_name)
    input_pkl_filename, input_csv_filename = get_input_filenames(dataset_name, model_type)
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    cfg = get_cfg(dataset_name, model_type=model_type)
    dm = get_data_manager(dataset_name, cfg=cfg)
    clip_model = get_clip_model(cfg)
    classnames = dm.dataset.classnames
    text_embeddings = compute_text_embeddings(prompts, clip_model, cfg)
    dataloader = get_dataloader(dm)
    pseudolabel_cossims = {}
    for batch in tqdm(dataloader):
        images, impaths = parse_batch(batch, dm)
        cossims = compute_pseudolabel_cossims(images, clip_model, text_embeddings)
        for impath, vec in zip(impaths, cossims):
            assert(impath not in pseudolabel_cossims)
            pseudolabel_cossims[impath] = vec

    impaths = sorted(gts.keys())
    cossims = np.array([pseudolabel_cossims[impath] for impath in impaths])

    gts = np.array([gts[impath] for impath in impaths])
    input_dict = {'impaths' : impaths, 'model_type' : model_type, 'compoundprompts' : prompts, 'gts' : gts, 'cossims' : cossims, 'dataset_name' : dataset_name}
    with open(input_pkl_filename, 'wb') as f:
        pickle.dump(input_dict, f)

    df = pd.DataFrame(input_dict['cossims'], columns=input_dict['compoundprompts'])
    df.to_csv(input_csv_filename, index=False)


def usage():
    print('Usage: python compute_cossims_train_noaug_arbitrary_prompts.py <dataset_name> <model_type>')


if __name__ == '__main__':
    compute_cossims_train_noaug_arbitrary_prompts(*(sys.argv[1:]))