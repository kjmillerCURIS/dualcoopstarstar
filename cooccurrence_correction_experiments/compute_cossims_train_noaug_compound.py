import os
import sys
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import torch
from tqdm import tqdm
from harvest_training_gts import get_data_manager, get_cfg, TRAINING_GTS_FILENAME_DICT
from compute_cossims_train_noaug import compute_text_embeddings as compute_single_text_embeddings
from run_compound_calibration_experiments import get_compound_input_filename
sys.path.append('.')
import clip
from trainers.Caption_tri_wta_soft_pseudolabel import load_clip_to_cpu
from trainers.fixed_prompt_utils import FIXED_PROMPTS_DICT
sys.path.pop()


COMPOUND_TOP_PROP = 0.2


def get_clip_model(cfg):
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
        # CLIP's default precision is fp16
        clip_model.float()

    clip_model.cuda()
    return clip_model


#should return npy array on cpu
def compute_single_cossims(images, clip_model, single_text_embeddings):
    with torch.no_grad():
        images = images.cuda()
        image_embeddings = clip_model.encode_image(images)
        assert(len(image_embeddings.shape) == 2)
        assert(image_embeddings.shape[0] == images.shape[0])
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        cossims = image_embeddings @ single_text_embeddings.t()

    return cossims.cpu().numpy(), image_embeddings


#yes, this will normalize the text embeddings 
def compute_cached_text_embeddings(queries, clip_model, cache):
    necessary_queries = [query for query in queries if query not in cache]
    if len(necessary_queries) > 0:
        texts = clip.tokenize(necessary_queries)
        with torch.no_grad():
            texts = texts.cuda()
            text_embeddings = clip_model.encode_text(texts)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        for query, text_embedding in zip(necessary_queries, text_embeddings):
            assert(query not in cache)
            cache[query] = text_embedding

    return torch.stack([cache[query] for query in queries])


def make_compound_query(classname_i, classname_j, compound_probe_type):
    if compound_probe_type == 'a_photo_of_a_i_and_a_j':
        template = 'a photo of a {classname_i} and a {classname_j}.'
    else:
        assert(False)

    return template.format(classname_i=classname_i, classname_j=classname_j)


def get_dataloader(dm):
    return dm.train_loader_x_complete_noaug


#should return images, impaths
def parse_batch(batch, dm):
    images = batch["img"]
    idx = batch["idx"]
    impaths = [dm.dataset.train_x[t].impath for t in idx]
    return images, impaths


def compute_cossims_train_noaug_compound(dataset_name, model_type, single_probe_type, compound_probe_type):
    input_filename = get_compound_input_filename(dataset_name, model_type, single_probe_type, compound_probe_type)
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    cfg = get_cfg(dataset_name, model_type=model_type)
    dm = get_data_manager(dataset_name, cfg=cfg)
    clip_model = get_clip_model(cfg)
    classnames = dm.dataset.classnames
    single_text_embeddings = compute_single_text_embeddings(classnames, clip_model, cfg, single_probe_type)
    dataloader = get_dataloader(dm)
    all_single_cossims = {}
    all_compound_cossims = {}
    cache = {}
    for batch in tqdm(dataloader):
        images, impaths = parse_batch(batch, dm)
        single_cossims, image_embeddings = compute_single_cossims(images, clip_model, single_text_embeddings)
        for impath, single_cossims_one, image_embedding in zip(impaths, single_cossims, image_embeddings):
            assert(impath not in all_single_cossims)
            assert(impath not in all_compound_cossims)
            assert(len(image_embedding.shape) == 1)
            all_single_cossims[impath] = single_cossims_one
            threshold = np.percentile(single_cossims_one, 100 * (1 - COMPOUND_TOP_PROP))
            i_list = []
            j_list = []
            queries = []
            for i in range(len(single_cossims_one)):
                for j in range(len(single_cossims_one)):
                    if i == j or single_cossims_one[i] < threshold or single_cossims_one[j] < threshold:
                        continue

                    i_list.append(i)
                    j_list.append(j)
                    query = make_compound_query(classnames[i], classnames[j], compound_probe_type)
                    queries.append(query)

            compound_text_embeddings = compute_cached_text_embeddings(queries, clip_model, cache)
            with torch.no_grad():
                compound_cossims_one = torch.squeeze(torch.unsqueeze(image_embedding, 0) @ compound_text_embeddings.t()).cpu().numpy()

            assert(np.amin(single_cossims_one) > -1.0)
            assert(np.amin(compound_cossims_one) > -1.0)
            compound_cossims_one = csr_matrix((compound_cossims_one + 1, (np.array(i_list), np.array(j_list))), shape=(len(single_cossims_one), len(single_cossims_one)))
            all_compound_cossims[impath] = compound_cossims_one

    impaths = sorted(gts.keys())
    input_dict = {'model_type' : model_type, 'single_probe_type' : single_probe_type, 'compound_probe_type' : compound_probe_type, 'dataset_name' : dataset_name, 'classnames' : classnames, 'single_cossims' : np.array([all_single_cossims[impath] for impath in impaths]), 'compound_cossims' : [all_compound_cossims[impath] for impath in impaths], 'gts' : np.array([gts[impath] for impath in impaths]), 'impaths' : impaths, 'logit_scale' : clip_model.logit_scale.item()}
    with open(input_filename, 'wb') as f:
        pickle.dump(input_dict, f)


def usage():
    print('Usage: python compute_cossims_train_noaug_compound.py <dataset_name> <model_type> <single_probe_type> <compound_probe_type>')


if __name__ == '__main__':
    compute_cossims_train_noaug_compound(*(sys.argv[1:]))
