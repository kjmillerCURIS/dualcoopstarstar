import os
import sys
import pickle
from tqdm import tqdm
from logistic_regression_experiment import make_output_filename, OUT_PARENT_DIR


#def make_output_filename(dataset_name, input_type, standardize, balance, C):


def tablify_logistic_regression():
    for dataset_name in ['COCO2014_partial', 'nuswide_partial']:
        table_filename = os.path.join(OUT_PARENT_DIR, '..', 'logistic_regression_table_%s.csv'%(dataset_name.split('_')[0]))
        f = open(table_filename, 'w')
        f.write('method,input_type,standardize_input,balance_negatives_and_positives,C (weight of MLE loss),train_mAP,test_mAP\n')
        for input_type in tqdm(['cossims', 'probs', 'log_probs', 'logits']):
            for standardize in [0, 1]:
                for balance in [0, 1]:
                    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]:
                        output_filename = make_output_filename(dataset_name, input_type, standardize, balance, C)
                        with open(output_filename, 'rb') as f_in:
                            output = pickle.load(f_in)

                        f.write('logistic_regression,%s,%s,%s,%s,%f,%f\n'%(input_type, ['no', 'yes'][standardize], ['no', 'yes'][balance], str(C), output['eval']['train_mAP'], output['eval']['test_mAP']))

        f.close()


if __name__ == '__main__':
    tablify_logistic_regression()
