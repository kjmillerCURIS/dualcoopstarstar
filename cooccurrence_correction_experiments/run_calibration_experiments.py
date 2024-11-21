import os
import sys
import numpy as np
import pickle
from itertools import product
from tqdm import tqdm
from compute_mAP import average_precision


#to be run after compute_zsclip_cossims.py
#TABLE_FILENAME = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/result_table_tables/result_table_table_train_noaug_%s.csv')
TABLE_FILENAME = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/result_table_tables/result_table_table_withmeansub_train_noaug_%s.csv')
INPUT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/result_table_inputs')
OUTPUT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/result_table_outputs')
MODEL_TYPE_LIST = ['RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14', 'ViT-L14336px']
#SINGLE_PROBE_TYPE_LIST = ['classname_only', 'single_standard', 'ensemble_80']
SINGLE_PROBE_TYPE_LIST = ['ensemble_80']
#CALIBRATION_TYPE_LIST = ['no_calibration', 'softmax', 'standardize', 'standardize_otherclasses', 'standardize_outlier3SD']
CALIBRATION_TYPE_LIST = ['no_calibration', 'standardize', 'mean_subtract']


def get_input_filename(dataset_name, model_type, single_probe_type):
    return os.path.join(INPUT_DIR, 'result_table_input_train_noaug_%s_%s_%s.pkl'%(dataset_name.split('_')[0], model_type, single_probe_type))


def get_output_filename(dataset_name, model_type, single_probe_type, calibration_type):
    return os.path.join(OUTPUT_DIR, 'result_table_output_withmeansubtract_train_noaug_%s_%s_%s_%s.pkl'%(dataset_name.split('_')[0], model_type, single_probe_type, calibration_type))


#sorry, too lazy to vectorize it
#input_dict only needed in case we need clip model temperature
def calibrate_one_datapoint(cossims, calibration_type, input_dict):
    assert(len(cossims.shape) == 1)
    if calibration_type == 'no_calibration':
        return cossims
    elif calibration_type == 'softmax':
        softlogits = np.exp(input_dict['logit_scale']) * cossims
        softlogits = softlogits - np.amax(softlogits)
        sumexps = np.sum(np.exp(softlogits))
#        if np.amin(sumexps - np.exp(softlogits)) == 0:
#            print(cossims)
#            print(np.exp(input_dict['logit_scale']))
#            import pdb
#            pdb.set_trace()

        siglogits = softlogits - np.log(sumexps - np.exp(softlogits))
        return siglogits
    elif calibration_type == 'standardize':
        return (cossims - np.mean(cossims)) / np.std(cossims, ddof=1)
    elif calibration_type == 'standardize_otherclasses':
        A = np.tile(cossims[:,np.newaxis], (1, cossims.shape[0]))
        np.fill_diagonal(A, np.nan)
        means = np.nanmean(A, axis=0)
        sds = np.nanstd(A, axis=0, ddof=1)
#        if np.amin(sds) == 0:
#            print(A)
#            import pdb
#            pdb.set_trace()

        return (cossims - means) / sds
    elif calibration_type == 'standardize_outlier3SD':
        init_mean = np.mean(cossims)
        init_sd = np.std(cossims, ddof=1)
        threshold = init_mean + 3 * init_sd
        mean = np.mean(cossims[cossims < threshold])
        sd = np.std(cossims[cossims < threshold], ddof=1)
        return (cossims - mean) / sd
    elif calibration_type == 'mean_subtract':
        return cossims - np.mean(cossims)
    else:
        assert(False)


def do_eval(scores, gts):
    class_APs = np.array([100.0 * average_precision(scores[:,i], gts[:,i]) for i in range(gts.shape[1])])
    mAP = np.mean(class_APs)
    return {'class_APs' : class_APs, 'mAP' : mAP}


def run_one_calibration_experiment(dataset_name, model_type, single_probe_type, calibration_type):
    input_filename = get_input_filename(dataset_name, model_type, single_probe_type)
    if not os.path.exists(input_filename):
        print('input file "%s" does not exist, skipping!'%(input_filename))
        return 'N/A'

    output_filename = get_output_filename(dataset_name, model_type, single_probe_type, calibration_type)
    with open(input_filename, 'rb') as f:
        input_dict = pickle.load(f)

    scores = []
    for cossims_one in input_dict['cossims']:
        scores_one = calibrate_one_datapoint(cossims_one, calibration_type, input_dict)
        scores.append(scores_one)

    scores = np.array(scores)
    output_dict = {'input_dict' : input_dict, 'dataset_name' : dataset_name, 'model_type' : model_type, 'single_probe_type' : single_probe_type, 'calibration_type' : calibration_type, 'scores' : scores}
    eval_dict = do_eval(scores, input_dict['gts'])
    output_dict['eval_dict'] = eval_dict
    with open(output_filename, 'wb') as f:
        pickle.dump(output_dict, f)

    return eval_dict['mAP']


def run_calibration_experiments(dataset_name):
    f = open(TABLE_FILENAME % (dataset_name.split('_')[0]), 'w')
    f.write('model,probe (single),calibration,train_mAP\n')
    for model_type, single_probe_type, calibration_type in tqdm(product(MODEL_TYPE_LIST, SINGLE_PROBE_TYPE_LIST, CALIBRATION_TYPE_LIST)):
        mAP = run_one_calibration_experiment(dataset_name, model_type, single_probe_type, calibration_type)
        line = '%s,%s,%s,%s\n'%(model_type, single_probe_type, calibration_type, str(mAP))
        f.write(line)
        print(line)

    f.close()


def usage():
    print('Usage: python run_calibration_experiments.py <dataset_name>')


if __name__ == '__main__':
    run_calibration_experiments(*(sys.argv[1:]))
