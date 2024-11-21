import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import torch
from tqdm import tqdm
from second_max_experiments import COCO_VENKATESH_CLASSNAMES_RENAMER, stringmatch_to_compoundprompt


MY_CUDA = 'cuda'
MODEL_TYPE_LIST = ['constant_f', #theta_1^t * 0.1 + theta_0^t
                    'constant', #one constant score per prompt, no bias terms. Sanity check that each prompt's FVU should be 1
                    'additive', #theta_1^t * (a_0^i + y_i^t * a_1^i + a_0^j + y_j^t * a_1^j) + theta_0^t
                    'OR_only', #theta_1^t * max(a_0^i + y_i^t * a_1^i, a_0^j + y_j^t * a_1^j) + theta_0^t
                    'AND_only', #theta_1^t * min(a_0^i + y_i^t * a_1^i, a_0^j + y_j^t * a_1^j) + theta_0^t
                    'OR_with_AND_bonus', #theta_1^t * (max(a_0^i + y_i^t * a_1^i, a_0^j + y_j^t * a_1^j) + delta * min(a_0^i + y_i^t * a_1^i, a_0^j + y_j^t * a_1^j)) + theta_0^t
                    'lookup_table', #theta_1^t * LUT[y_i^t, y_j^t] + theta_0^t
                    'OR_with_AND_bonus_multidelta'] #theta_1^t * (max(a_0^i + y_i^t * a_1^i, a_0^j + y_j^t * a_1^j) + multidelta_ij * min(a_0^i + y_i^t * a_1^i, a_0^j + y_j^t * a_1^j)) + theta_0^t
INPUT_FILENAME = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/%s_test/%s_test_simple_single_and_compound_cossims_%s.pkl')
PAIR_PROMPTS_START_INDEX = {'COCO2014_partial' : 80, 'VOC2007_partial' : 20, 'nuswideTheirVersion_partial' : 81}
PAIR_PROMPTS_END_INDEX = {'COCO2014_partial' : 754, 'VOC2007_partial' : 50, 'nuswideTheirVersion_partial' : 538}

RESULT_DICT_FILENAME = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments/%s_test/%s_test_noise_model_results_%s.pkl')


#return scores, gts, dataset_info
#the data source will be hardcoded
#Note: scores is only for formulaic pair prompts. gts is for classes.
def load_data(dataset_name, clip_model_type):
    with open(INPUT_FILENAME % (dataset_name.split('_')[0], dataset_name.split('_')[0], clip_model_type), 'rb') as f:
        d = pickle.load(f)

    start_index, end_index = PAIR_PROMPTS_START_INDEX[dataset_name], PAIR_PROMPTS_END_INDEX[dataset_name]
    dataset_info = {}
    allprompts = d['simple_single_and_compound_prompts']
    print((allprompts[start_index-1], allprompts[start_index], allprompts[start_index+1]))
    print((allprompts[end_index-2], allprompts[end_index-1], allprompts[end_index]))
    pairprompts = allprompts[start_index:end_index]
    dataset_info['pairprompts'] = pairprompts
    dataset_info['num_prompts'] = len(pairprompts)
    scores = torch.tensor(d['cossims'][:,start_index:end_index], device=MY_CUDA)
    gts = torch.tensor(d['gts'].astype('int32'), device=MY_CUDA)
    dataset_info['num_images'] = gts.shape[0]
    dataset_info['num_classes'] = gts.shape[1]
    assert(scores.shape == (dataset_info['num_images'], dataset_info['num_prompts']))
    classnames = allprompts[:dataset_info['num_classes']]
    if dataset_name == 'COCO2014_partial':
        classnames = [(COCO_VENKATESH_CLASSNAMES_RENAMER[c] if c in COCO_VENKATESH_CLASSNAMES_RENAMER else c) for c in classnames]

    dataset_info['classnames'] = classnames
    matches_list = [stringmatch_to_compoundprompt(classnames, p) for p in pairprompts]
    assert(all([len(matches) == 2 for matches in matches_list]))
    dataset_info['class_pairs'] = torch.tensor(np.array([[classnames.index(m) for m in matches] for matches in matches_list]), device=MY_CUDA)
    assert(np.amin(dataset_info['class_pairs'].cpu().numpy()) >= 0)
    assert(np.amax(dataset_info['class_pairs'].cpu().numpy()) < dataset_info['num_classes'])
    print(np.unique(dataset_info['class_pairs'].cpu().numpy()).shape)
    return scores, gts, dataset_info


#return model_params, model_info
#Note: model_type does NOT mean what it means in most other scripts :)
def initialize_model(model_type, dataset_info, scores, gts):
    scores, gts = scores.cpu().numpy(), gts.cpu().numpy()
    num_prompts, num_images, num_classes = dataset_info['num_prompts'], dataset_info['num_images'], dataset_info['num_classes']
    model_info = {'model_type' : model_type}
    model_params = {}
    if model_type == 'constant': #sanity check - this should get mFVU=1, and each prompt's FVU should be 1
        model_params['prompt_constants'] = 0.5 * np.mean(scores, axis=0) + 0.5 * np.amax(scores, axis=0) #quick-n-dirty guess at the balanced mean
        model_info['bounds'] = [(None, None) for _ in range(num_prompts)]
        return model_params, model_info

    assert(model_type in ['constant_f', 'additive', 'OR_only', 'AND_only', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table'])
    class_pairs = dataset_info['class_pairs'].cpu().numpy()
    prompt_gts_00 = (1-gts[:,class_pairs[:,0]]) * (1-gts[:,class_pairs[:,1]])
    prompt_gts_11 = gts[:,class_pairs[:,0]] * gts[:,class_pairs[:,1]]
    model_params['per_image_biases_0'] = np.sum(scores * prompt_gts_00, axis=1) / np.sum(prompt_gts_00, axis=1)
    model_params['per_image_biases_1'] = np.sum(scores * prompt_gts_11, axis=1) / np.maximum(np.sum(prompt_gts_11, axis=1), 1e-5)
    model_info['per_image_biases_0_indices'] = range(num_images)
    model_info['per_image_biases_1_indices'] = range(num_images, 2 * num_images)
    model_info['len'] = 2 * num_images
    bounds = [(None, None) for _ in range(num_images)] + [(0, None) for _ in range(num_images)]
    if model_type == 'constant_f': #special case
        model_info['bounds'] = bounds
        return model_params, model_info
    elif model_type == 'lookup_table': #special case
        model_params['LUTs'] = np.ones((num_prompts, 2, 2))
        model_params['LUTs'][:,0,0] = 0.1
        model_params['LUTs'][:,1,1] = 1.1
        model_info['LUTs_indices'] = range(2 * num_images, 2 * num_images + 4 * num_prompts)
        model_info['len'] = 2 * num_images + 4 * num_prompts
        bounds.extend([(None, None) for _ in range(4 * num_prompts)])
        model_info['bounds'] = bounds
        return model_params, model_info

    model_params['class_strengths_0'] = 0.1 * np.ones(dataset_info['num_classes'])
    model_params['class_strengths_1'] = np.ones(dataset_info['num_classes'])
    model_info['class_strengths_0_indices'] = range(2 * num_images, 2 * num_images + num_classes)
    model_info['class_strengths_1_indices'] = range(2 * num_images + num_classes, 2 * num_images + 2 * num_classes)
    model_info['len'] = 2 * num_images + 2 * num_classes
    bounds.extend([(0, None) for _ in range(2 * num_classes)])
    if model_type == 'OR_with_AND_bonus':
        model_params['delta'] = 0.5
        model_info['delta_index'] = 2 * num_images + 2 * num_classes
        model_info['len'] = 2 * num_images + 2 * num_classes + 1
        bounds.append((0, None))
    elif model_type == 'OR_with_AND_bonus_multidelta':
        model_params['multidelta'] = 0.5 * np.ones(dataset_info['num_prompts'])
        model_info['multidelta_indices'] = range(2 * num_images + 2 * num_classes, 2 * num_images + 2 * num_classes + num_prompts)
        model_info['len'] = 2 * num_images + 2 * num_classes + num_prompts
        bounds.extend([(0, None) for _ in range(num_prompts)])

    model_info['bounds'] = bounds
    return model_params, model_info


#return x, bounds
def pack_model(model_params, model_info, dataset_info):
    model_type = model_info['model_type']
    if model_type == 'constant':
        return model_params['prompt_constants'], model_info['bounds']

    assert(model_type in ['constant_f', 'additive', 'OR_only', 'AND_only', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table'])
    x = np.zeros(model_info['len'])
    x[model_info['per_image_biases_0_indices']] = model_params['per_image_biases_0']
    x[model_info['per_image_biases_1_indices']] = model_params['per_image_biases_1']
    if model_type == 'constant_f':
        return x, model_info['bounds']
    elif model_type == 'lookup_table':
        x[model_info['LUTs_indices']] = model_params['LUTs'].flatten()
        return x, model_info['bounds']

    x[model_info['class_strengths_0_indices']] = model_params['class_strengths_0']
    x[model_info['class_strengths_1_indices']] = model_params['class_strengths_1']
    if model_type == 'OR_with_AND_bonus':
        x[model_info['delta_index']] = model_params['delta']
    elif model_type == 'OR_with_AND_bonus_multidelta':
        x[model_info['multidelta_indices']] = model_params['multidelta']

    return x, model_info['bounds']


def pack_grad(model_params, model_info, dataset_info):
    model_type = model_info['model_type']
    if model_type == 'constant':
        return model_params['prompt_constants'].grad

    assert(model_type in ['constant_f', 'additive', 'OR_only', 'AND_only', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table'])
    my_grad = torch.zeros(model_info['len'], dtype=torch.float64, device=MY_CUDA)
    my_grad[model_info['per_image_biases_0_indices']] = model_params['per_image_biases_0'].grad
    my_grad[model_info['per_image_biases_1_indices']] = model_params['per_image_biases_1'].grad
    if model_type == 'constant_f':
        return my_grad
    elif model_type == 'lookup_table':
        my_grad[model_info['LUTs_indices']] = model_params['LUTs'].grad.flatten()
        return my_grad

    my_grad[model_info['class_strengths_0_indices']] = model_params['class_strengths_0'].grad
    my_grad[model_info['class_strengths_1_indices']] = model_params['class_strengths_1'].grad
    if model_type == 'OR_with_AND_bonus':
        my_grad[model_info['delta_index']] = model_params['delta'].grad
    elif model_type == 'OR_with_AND_bonus_multidelta':
        my_grad[model_info['multidelta_indices']] = model_params['multidelta'].grad

    return my_grad


#return model_params
def unpack_model(x, model_info, dataset_info):
    num_prompts = dataset_info['num_prompts']
    model_type = model_info['model_type']
    if model_type == 'constant':
        return {'prompt_constants' : torch.tensor(x, requires_grad=True, device=MY_CUDA)}

    assert(model_type in ['constant_f', 'additive', 'OR_only', 'AND_only', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table'])
    model_params = {}
    model_params['per_image_biases_0'] = torch.tensor(x[model_info['per_image_biases_0_indices']], requires_grad=True, device=MY_CUDA)
    model_params['per_image_biases_1'] = torch.tensor(x[model_info['per_image_biases_1_indices']], requires_grad=True, device=MY_CUDA)
    if model_type == 'constant_f':
        return model_params
    elif model_type == 'lookup_table':
        model_params['LUTs'] = torch.tensor(np.reshape(x[model_info['LUTs_indices']], (num_prompts, 2, 2)), requires_grad=True, device=MY_CUDA)
        return model_params

    model_params['class_strengths_0'] = torch.tensor(x[model_info['class_strengths_0_indices']], requires_grad=True, device=MY_CUDA)
    model_params['class_strengths_1'] = torch.tensor(x[model_info['class_strengths_1_indices']], requires_grad=True, device=MY_CUDA)
    if model_type == 'OR_with_AND_bonus':
        model_params['delta'] = torch.tensor(x[model_info['delta_index']], requires_grad=True, device=MY_CUDA)
    elif model_type == 'OR_with_AND_bonus_multidelta':
        model_params['multidelta'] = torch.tensor(x[model_info['multidelta_indices']], requires_grad=True, device=MY_CUDA)

    return model_params


#what we're expecting from dataset_info:
#-class_pairs should be shape (num_prompts, 2) and each value should be a class index
def compute_f_vals(gts, model_params, model_type, dataset_info):
    num_prompts, num_images, num_classes = dataset_info['num_prompts'], dataset_info['num_images'], dataset_info['num_classes']
    assert(model_type in ['constant_f', 'additive', 'OR_only', 'AND_only', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table'])
    class_pairs = dataset_info['class_pairs']
    if model_type in ['constant_f', 'lookup_table']:
        agts = gts #just want 0 and 1
    else:
        agts = model_params['class_strengths_0'][None,:] + model_params['class_strengths_1'][None,:] * gts #want 0 and 1 mapped by class strengths

    aiyti_arr = agts[:,class_pairs[:,0]]
    ajytj_arr = agts[:,class_pairs[:,1]]
    assert(aiyti_arr.shape == (num_images, num_prompts))
    assert(ajytj_arr.shape == (num_images, num_prompts))
    if model_type == 'additive':
        return aiyti_arr + ajytj_arr
    elif model_type == 'OR_only':
        return torch.maximum(aiyti_arr, ajytj_arr)
    elif model_type == 'AND_only':
        return torch.minimum(aiyti_arr, ajytj_arr)
    elif model_type == 'OR_with_AND_bonus':
        return torch.maximum(aiyti_arr, ajytj_arr) + model_params['delta'] * torch.minimum(aiyti_arr, ajytj_arr)
    elif model_type == 'OR_with_AND_bonus_multidelta':
        return torch.maximum(aiyti_arr, ajytj_arr) + model_params['multidelta'][None,:] * torch.minimum(aiyti_arr, ajytj_arr)
    elif model_type == 'lookup_table':
        f_vals = model_params['LUTs'][torch.arange(num_prompts, dtype=torch.int64)[None,:].repeat(num_images, 1), aiyti_arr.long(), ajytj_arr.long()]
        assert(f_vals.shape == (num_images, num_prompts))
        return f_vals
    elif model_type == 'constant_f':
        return 0.1
    else:
        assert(False)


#return pred_scores
def predict(gts, model_params, model_info, dataset_info):
    num_prompts, num_images, num_classes = dataset_info['num_prompts'], dataset_info['num_images'], dataset_info['num_classes']
    model_type = model_info['model_type']
    if model_type == 'constant':
        return model_params['prompt_constants'][None,:].repeat(dataset_info['num_images'], 1)

    assert(model_type in ['constant_f', 'additive', 'OR_only', 'AND_only', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table'])
    f_vals = compute_f_vals(gts, model_params, model_type, dataset_info)
    assert(model_type == 'constant_f' or f_vals.shape == (num_images, num_prompts))
    pred_scores = model_params['per_image_biases_1'][:,None] * f_vals + model_params['per_image_biases_0'][:,None]
    return pred_scores


#return loss (scalar)
def compute_loss(pred_scores, scores, gts, dataset_info):
    num_prompts, num_images, num_classes = dataset_info['num_prompts'], dataset_info['num_images'], dataset_info['num_classes']
    sqdiffs = torch.square(pred_scores - scores)
    class_pairs = dataset_info['class_pairs']
    agts = gts #just want 0 and 1
    yti_arr = agts[:,class_pairs[:,0]]
    ytj_arr = agts[:,class_pairs[:,1]]
    loss = 0.0
    for i_val in [0,1]:
        for j_val in [0,1]:
            mask = ((yti_arr == i_val) & (ytj_arr == j_val)).int()
            MSEs = torch.sum(mask * sqdiffs, dim=0) / torch.maximum(torch.sum(mask, dim=0), torch.tensor(1e-5, device=MY_CUDA))
            loss += 0.25 * num_images * torch.sum(MSEs)

    loss = loss
    return loss


#plug this into minimize()
def opt_fn(x, model_info, dataset_info, scores, gts):
    model_params = unpack_model(x, model_info, dataset_info)
    pred_scores = predict(gts, model_params, model_info, dataset_info)
    loss = compute_loss(pred_scores, scores, gts, dataset_info)
    if model_info['iter'] % 10 == 0:
        print('%d: %f'%(model_info['iter'], loss.item()))

    model_info['iter'] += 1
    loss.backward()
    my_grad = pack_grad(model_params, model_info, dataset_info)
    return loss.item(), my_grad.cpu().numpy()


#return model_params, model_info
def fit_model(scores, gts, dataset_info, model_type):
    model_params, model_info = initialize_model(model_type, dataset_info, scores, gts)
    x0, bounds = pack_model(model_params, model_info, dataset_info)
    model_info['iter'] = 0
    res = minimize(opt_fn, x0, args=(model_info, dataset_info, scores, gts), bounds=bounds, jac=True)
    print(res.success)
    print(res.message)
    x = res.x
    model_params = unpack_model(x, model_info, dataset_info)
    return model_params, model_info


#return eval_dict
def evaluate(pred_scores, scores, gts, dataset_info):
    pred_scores, scores, gts = pred_scores.detach().cpu().numpy(), scores.cpu().numpy(), gts.cpu().numpy()
    num_prompts, num_images, num_classes = dataset_info['num_prompts'], dataset_info['num_images'], dataset_info['num_classes']
    sqdiffs = np.square(pred_scores - scores)
    sqscores = np.square(scores)
    class_pairs = dataset_info['class_pairs'].cpu().numpy()
    agts = gts.astype('int32') #just want 0 and 1
    yti_arr = agts[:,class_pairs[:,0]]
    ytj_arr = agts[:,class_pairs[:,1]]
    vars_partA = np.zeros(num_prompts) #E[X^2]
    vars_partB = np.zeros(num_prompts) #E[X]
    MSEs_by_binpair = {}
    for i_val in [0,1]:
        for j_val in [0,1]:
            mask = ((yti_arr == i_val) & (ytj_arr == j_val)).astype('int32')
            MSEs_by_binpair[(i_val, j_val)] = np.sum(mask * sqdiffs, axis=0) / np.maximum(np.sum(mask, axis=0), 1e-5)
            vars_partA += 0.25 * np.sum(mask * sqscores, axis=0) / np.maximum(np.sum(mask, axis=0), 1e-5)
            vars_partB += 0.25 * np.sum(mask * scores, axis=0) / np.maximum(np.sum(mask, axis=0), 1e-5)

    my_vars = vars_partA - vars_partB ** 2
    MSEs = 0.25 * sum([MSEs_by_binpair[k] for k in [(0,0),(0,1),(1,0),(1,1)]])
    FVUs = MSEs / my_vars
    mFVU = np.mean(FVUs)
    return {'mFVU' : mFVU, 'FVUs' : FVUs, 'MSEs' : MSEs, 'my_vars' : my_vars, 'MSEs_by_binpair' : MSEs_by_binpair}


#return eval_dict, model_params, model_info
def run_one_experiment(scores, gts, dataset_info, model_type):
    model_params, model_info = fit_model(scores, gts, dataset_info, model_type)
    pred_scores = predict(gts, model_params, model_info, dataset_info)
    eval_dict = evaluate(pred_scores, scores, gts, dataset_info)
    print('model_type=%s, mFVU=%f'%(model_type, eval_dict['mFVU']))
    return eval_dict, model_params, model_info


def noise_model(dataset_name, clip_model_type):
    scores, gts, dataset_info = load_data(dataset_name, clip_model_type)
    result_dict = {'data' : {'scores' : scores.cpu().numpy(), 'gts' : gts.cpu().numpy(), 'dataset_info' : {k : (dataset_info[k].cpu().numpy() if k == 'class_pairs' else dataset_info[k]) for k in sorted(dataset_info.keys())}}}
    result_dict['experiments'] = {}
    for model_type in tqdm(MODEL_TYPE_LIST):
        eval_dict, model_params, model_info = run_one_experiment(scores, gts, dataset_info, model_type)
        model_params = {k : model_params[k].detach().cpu().numpy() for k in sorted(model_params.keys())}
        result_dict['experiments'][model_type] = {'eval_dict' : eval_dict, 'model_params' : model_params, 'model_info' : model_info}
        with open(RESULT_DICT_FILENAME % (dataset_name.split('_')[0], dataset_name.split('_')[0], clip_model_type), 'wb') as f:
            pickle.dump(result_dict, f)

    with open(RESULT_DICT_FILENAME % (dataset_name.split('_')[0], dataset_name.split('_')[0], clip_model_type), 'wb') as f:
        pickle.dump(result_dict, f)


def usage():
    print('Usage: python noise_model.py <dataset_name> <clip_model_type>')


if __name__ == '__main__':
    noise_model(*(sys.argv[1:]))
