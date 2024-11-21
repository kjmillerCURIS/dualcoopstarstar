import os
import sys
import numpy as np
import pickle
from itertools import product
from tqdm import tqdm
from compute_mAP import average_precision


#to be run after compute_zsclip_cossims.py
TABLE_FILENAME = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/result_table_tables_compound/result_table_table_compound_train_noaug_%s.csv')
INPUT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/result_table_inputs_compound')
OUTPUT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/result_table_outputs_compound')
MODEL_TYPE_LIST = ['RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14', 'ViT-L14336px']
SINGLE_PROBE_TYPE_LIST = ['ensemble_80']
COMPOUND_PROBE_TYPE_LIST = ['a_photo_of_a_i_and_a_j']
COMPOUNDING_STRATEGY_LIST = ['no_compounding', 'max_compounding']
CALIBRATION_TYPE_LIST = ['no_calibration', 'standardize', 'standardize_using_single_stats']


def get_compound_input_filename(dataset_name, model_type, single_probe_type, compound_probe_type):
    os.makedirs(INPUT_DIR, exist_ok=True)
    return os.path.join(INPUT_DIR, 'result_table_input_compound_train_noaug-%s-%s-%s-%s.pkl'%(dataset_name.split('_')[0], model_type, single_probe_type, compound_probe_type))


def get_compound_output_filename(dataset_name, model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, 'result_table_output_compound_train_noaug-%s-%s-%s-%s-%s-%s.pkl'%(dataset_name.split('_')[0], model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type))


def compound_one_datapoint(single_cossims, compound_cossims, compounding_strategy):
    if compounding_strategy == 'no_compounding':
        return single_cossims
    elif compounding_strategy == 'max_compounding':
        compound_cossims = compound_cossims.toarray() - 1
        final_cossims = np.maximum(single_cossims, np.maximum(np.amax(compound_cossims, axis=0), np.amax(compound_cossims, axis=1)))
        assert(np.all(final_cossims >= single_cossims))
        assert(np.amin(final_cossims) > -1.0)
        return final_cossims
    else:
        assert(False)


#sorry, too lazy to vectorize it
def calibrate_one_datapoint(final_cossims, single_cossims, calibration_type):
    assert(len(final_cossims.shape) == 1)
    if calibration_type == 'no_calibration':
        return final_cossims
    elif calibration_type == 'standardize':
        return (final_cossims - np.mean(final_cossims)) / np.std(final_cossims, ddof=1)
    elif calibration_type == 'standardize_using_single_stats':
        return (final_cossims - np.mean(single_cossims)) / np.std(single_cossims, ddof=1)
    else:
        assert(False)


def do_eval(scores, gts):
    class_APs = np.array([100.0 * average_precision(scores[:,i], gts[:,i]) for i in range(gts.shape[1])])
    mAP = np.mean(class_APs)
    return {'class_APs' : class_APs, 'mAP' : mAP}


def run_one_compound_calibration_experiment(dataset_name, model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type):
    input_filename = get_compound_input_filename(dataset_name, model_type, single_probe_type, compound_probe_type)
    output_filename = get_compound_output_filename(dataset_name, model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type)
    with open(input_filename, 'rb') as f:
        input_dict = pickle.load(f)

    final_cossims = []
    scores = []
    for single_cossims_one, compound_cossims_one in zip(input_dict['single_cossims'],input_dict['compound_cossims']):
        final_cossims_one = compound_one_datapoint(single_cossims_one, compound_cossims_one, compounding_strategy)
        final_cossims.append(final_cossims_one)
        scores_one = calibrate_one_datapoint(final_cossims_one, single_cossims_one, calibration_type)
        scores.append(scores_one)

    final_cossims = np.array(final_cossims)
    scores = np.array(scores)
    output_dict = {'input_dict' : input_dict, 'dataset_name' : dataset_name, 'model_type' : model_type, 'single_probe_type' : single_probe_type, 'compound_probe_type' : compound_probe_type, 'compounding_strategy' : compounding_strategy, 'calibration_type' : calibration_type, 'final_cossims' : final_cossims, 'scores' : scores}
    eval_dict = do_eval(scores, input_dict['gts'])
    output_dict['eval_dict'] = eval_dict
    with open(output_filename, 'wb') as f:
        pickle.dump(output_dict, f)

    return eval_dict['mAP']


def run_compound_calibration_experiments(dataset_name):
    table_filename = TABLE_FILENAME % (dataset_name.split('_')[0])
    os.makedirs(os.path.dirname(table_filename), exist_ok=True)
    f = open(table_filename, 'w')
    f.write('model,single_probe_type,compound_probe_type,compounding_strategy,calibration_type,train_mAP\n')
    for model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type in tqdm(product(MODEL_TYPE_LIST, SINGLE_PROBE_TYPE_LIST, COMPOUND_PROBE_TYPE_LIST, COMPOUNDING_STRATEGY_LIST, CALIBRATION_TYPE_LIST)):
        if compounding_strategy == 'no_compounding' and calibration_type == 'standardize_using_single_stats':
            continue

        mAP = run_one_compound_calibration_experiment(dataset_name, model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type)
        line = '%s,%s,%s,%s,%s,%s\n'%(model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type, str(mAP))
        f.write(line)
        print(line)

    f.close()


def usage():
    print('Usage: python run_compound_calibration_experiments.py <dataset_name>')


if __name__ == '__main__':
    run_compound_calibration_experiments(*(sys.argv[1:]))
