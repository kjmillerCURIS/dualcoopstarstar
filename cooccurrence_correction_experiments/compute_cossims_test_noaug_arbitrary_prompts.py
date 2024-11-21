import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from harvest_training_gts import get_data_manager, get_cfg, TESTING_GTS_FILENAME_DICT
sys.path.append('.')
import clip
from trainers.Caption_tri_wta_soft_pseudolabel import load_clip_to_cpu
sys.path.pop()


LLM_GENERATES_EVERYTHING = False
PAIR_ONLY = False
ALL_PAIRS = False
CONJUNCTION_OR = False
CONJUNCTION_WITH = False
CONJUNCTION_NEXT_TO = False
CONJUNCTION_NOT = False
ALL_CONJUNCTIONS = True
assert(int(LLM_GENERATES_EVERYTHING) + int(PAIR_ONLY) + int(ALL_PAIRS) + int(CONJUNCTION_OR) + int(CONJUNCTION_WITH) + int(CONJUNCTION_NEXT_TO) + int(CONJUNCTION_NOT) + int(ALL_CONJUNCTIONS) <= 1)
ABLATION_SUFFIX = ''
if LLM_GENERATES_EVERYTHING:
    ABLATION_SUFFIX = '/LLM_GENERATES_EVERYTHING'
if PAIR_ONLY:
    ABLATION_SUFFIX = '/PAIR_ONLY'
if ALL_PAIRS:
    ABLATION_SUFFIX = '/ALL_PAIRS'
if CONJUNCTION_OR:
    ABLATION_SUFFIX = '/CONJUNCTION_OR'
if CONJUNCTION_WITH:
    ABLATION_SUFFIX = '/CONJUNCTION_WITH'
if CONJUNCTION_NEXT_TO:
    ABLATION_SUFFIX = '/CONJUNCTION_NEXT_TO'
if CONJUNCTION_NOT:
    ABLATION_SUFFIX = '/CONJUNCTION_NOT'
if ALL_CONJUNCTIONS:
    ABLATION_SUFFIX = '/ALL_CONJUNCTIONS'


BASE_DIR_DICT = {'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/VOC2007_test' + ABLATION_SUFFIX), 'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/COCO2014_test' + ABLATION_SUFFIX), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/nuswide_test'), 'nuswideTheirVersion_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/nuswideTheirVersion_test' + ABLATION_SUFFIX)}
PROMPTS_FILENAME_DICT = {'VOC2007_partial' : os.path.join(BASE_DIR_DICT['VOC2007_partial'], 'VOC2007_simple_single_and_compound_prompts.txt'), 'COCO2014_partial' : os.path.join(BASE_DIR_DICT['COCO2014_partial'], 'COCO2014_simple_single_and_compound_prompts.txt'), 'nuswide_partial' : os.path.join(BASE_DIR_DICT['nuswide_partial'], 'nuswide_simple_single_and_compound_prompts.txt'), 'nuswideTheirVersion_partial' : os.path.join(BASE_DIR_DICT['nuswideTheirVersion_partial'], 'nuswideTheirVersion_simple_single_and_compound_prompts.txt')}


def load_prompts(dataset_name):
    prompts_filename = PROMPTS_FILENAME_DICT[dataset_name]
    f = open(prompts_filename, 'r')
    lines = f.readlines()
    f.close()
    return [line.rstrip('\n') for line in lines]


def get_input_filenames(dataset_name, model_type, is_taidpt=False):
    suffix = {False : '', True : '_for_TaI-DPT'}[is_taidpt]
    input_pkl_filename = os.path.join(BASE_DIR_DICT[dataset_name], '%s_test_simple_single_and_compound_cossims_%s%s.pkl'%(dataset_name.split('_')[0], model_type, suffix))
    input_csv_filename = os.path.join(BASE_DIR_DICT[dataset_name], '%s_test_simple_single_and_compound_cossims_%s%s.csv'%(dataset_name.split('_')[0], model_type, suffix))
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
    return dm.test_loader


#should return images, impaths
def parse_batch(batch, dm):
    images = batch["img"]
    idx = batch["idx"]
    impaths = [dm.dataset.test[t].impath for t in idx]
    return images, impaths


def compute_cossims_test_noaug_arbitrary_prompts(dataset_name, model_type, is_taidpt=False):
    is_taidpt = int(is_taidpt)
    prompts = load_prompts(dataset_name)
    input_pkl_filename, input_csv_filename = get_input_filenames(dataset_name, model_type, is_taidpt=is_taidpt)
    with open(TESTING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    cfg = get_cfg(dataset_name, model_type=model_type, is_taidpt=is_taidpt)
    dm = get_data_manager(dataset_name, cfg=cfg, is_taidpt=is_taidpt)
    clip_model = get_clip_model(cfg)
    classnames = dm.dataset.classnames
    text_embeddings = compute_text_embeddings(prompts, clip_model, cfg)
    dataloader = get_dataloader(dm)
    pseudolabel_cossims = {}
    already_printed = False
    for batch in tqdm(dataloader):
        images, impaths = parse_batch(batch, dm)
        if not already_printed:
            print(images.shape)
            already_printed = True

        cossims = compute_pseudolabel_cossims(images, clip_model, text_embeddings)
        for impath, vec in zip(impaths, cossims):
            assert(impath not in pseudolabel_cossims)
            pseudolabel_cossims[impath] = vec

    impaths = sorted(gts.keys())
    cossims = np.array([pseudolabel_cossims[impath] for impath in impaths])

    gts = np.array([gts[impath] for impath in impaths])
    input_dict = {'impaths' : impaths, 'model_type' : model_type, 'simple_single_and_compound_prompts' : prompts, 'gts' : gts, 'cossims' : cossims, 'dataset_name' : dataset_name}
    with open(input_pkl_filename, 'wb') as f:
        pickle.dump(input_dict, f)

    df = pd.DataFrame(input_dict['cossims'], columns=input_dict['simple_single_and_compound_prompts'])
    df.to_csv(input_csv_filename, index=False)


def usage():
    print('Usage: python compute_cossims_test_noaug_arbitrary_prompts.py <dataset_name> <model_type>')


if __name__ == '__main__':
    compute_cossims_test_noaug_arbitrary_prompts(*(sys.argv[1:]))
