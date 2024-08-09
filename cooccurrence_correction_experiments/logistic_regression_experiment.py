import os
import sys
import copy
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, TESTING_GTS_FILENAME_DICT
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from compute_initial_cossims import PSEUDOLABEL_COSSIMS_FILENAME_DICT
from initial_testing_scores_paths import PSEUDOLABEL_TESTING_LOGITS_FILENAME_DICT, PSEUDOLABEL_TESTING_COSSIMS_FILENAME_DICT
from compute_mAP import average_precision


MINICLASS_NUM_CLASSES = 5
OUT_PARENT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/logistic_regression')


#return a dict
def evaluate(X_train, y_train, X_test, y_test, my_clf):
    train_pred_probs = np.hstack([my_clf[i].predict_proba(X_train)[:,1:] for i in range(len(my_clf))])
    test_pred_probs = np.hstack([my_clf[i].predict_proba(X_test)[:,1:] for i in range(len(my_clf))])
    assert(train_pred_probs.shape == X_train.shape)
    assert(test_pred_probs.shape == X_test.shape)
    train_class_APs = np.array([100.0 * average_precision(train_pred_probs[:,i], y_train[:,i]) for i in range(y_train.shape[1])])
    train_mAP = np.mean(train_class_APs)
    test_class_APs = np.array([100.0 * average_precision(test_pred_probs[:,i], y_test[:,i]) for i in range(y_test.shape[1])])
    test_mAP = np.mean(test_class_APs)
    input_train_class_APs = np.array([100.0 * average_precision(X_train[:,i], y_train[:,i]) for i in range(y_train.shape[1])])
    input_train_mAP = np.mean(input_train_class_APs)
    input_test_class_APs = np.array([100.0 * average_precision(X_test[:,i], y_test[:,i]) for i in range(y_test.shape[1])])
    input_test_mAP = np.mean(input_test_class_APs)
    return {'train_pred_probs' : train_pred_probs, 'test_pred_probs' : test_pred_probs, 'train_class_APs' : train_class_APs, 'test_class_APs' : test_class_APs, 'train_mAP' : train_mAP, 'test_mAP' : test_mAP, 'input_train_class_APs' : input_train_class_APs, 'input_test_class_APs' : input_test_class_APs, 'input_train_mAP' : input_train_mAP, 'input_test_mAP' : input_test_mAP}


#returns model as pipeline
def fit_model(X_train, y_train, standardize, balance, L1, C):
    assert(X_train.shape == y_train.shape)
    my_clf = []
    for i in tqdm(range(y_train.shape[1])):
#        penalty = ('l1' if L1 else 'l2')
#        solver = ('saga' if L1 else 'lbfgs')
#        one_clf = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=1000000, class_weight=('balanced' if balance else None))
#        if standardize:
#            one_clf = Pipeline([('scaler', StandardScaler()), ('clf', one_clf)])
#        else:
#            one_clf = Pipeline([('clf', one_clf)])
#
#        one_clf.fit(X_train, y_train[:,i])

        one_clf = LogisticRegression(C=C, max_iter=1000, class_weight=('balanced' if balance else None))
        if standardize:
            one_clf = Pipeline([('scaler', StandardScaler()), ('clf', one_clf)])
        else:
            one_clf = Pipeline([('clf', one_clf)])

        one_clf.fit(X_train, y_train[:,i])
        if L1:
            coef = copy.deepcopy(one_clf.named_steps['clf'].coef_)
            intercept = copy.deepcopy(one_clf.named_steps['clf'].intercept_)
            one_clf = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=100000, warm_start=True, class_weight=('balanced' if balance else None))
            if standardize:
                one_clf = Pipeline([('scaler', StandardScaler()), ('clf', one_clf)])
            else:
                one_clf = Pipeline([('clf', one_clf)])

            one_clf.fit(X_train, y_train[:,i])
            one_clf.named_steps['clf'].coef_ = coef
            one_clf.named_steps['clf'].intercept_ = intercept
            one_clf.fit(X_train, y_train[:,i])

        my_clf.append(one_clf)

    return my_clf


def load_data_miniclass(dataset_name, input_type):
    assert(input_type == 'cossims')

    #load y and X
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        y_train = pickle.load(f)

    y_train = np.array([y_train[impath] for impath in sorted(y_train.keys())])
    with open(TESTING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        y_test = pickle.load(f)

    y_test = np.array([y_test[impath] for impath in sorted(y_test.keys())])
    train_scores_filename_dict = PSEUDOLABEL_COSSIMS_FILENAME_DICT
    test_scores_filename_dict = PSEUDOLABEL_TESTING_COSSIMS_FILENAME_DICT
    with open(train_scores_filename_dict[dataset_name], 'rb') as f:
        X_train = pickle.load(f)

    with open(test_scores_filename_dict[dataset_name], 'rb') as f:
        X_test = pickle.load(f)

    X_train = np.array([X_train[impath] for impath in sorted(X_train.keys())])
    X_test = np.array([X_test[impath] for impath in sorted(X_test.keys())])

    #select classes and examples
    freqs = np.sum(y_train, axis=0)
    class_indices = np.argsort(-freqs)[:MINICLASS_NUM_CLASSES]
    y_train = y_train[:,class_indices]
    y_test = y_test[:,class_indices]
    X_train = X_train[:,class_indices]
    X_test = X_test[:,class_indices]
    train_example_mask = (np.amax(y_train, axis=1) > 0)
    test_example_mask = (np.amax(y_test, axis=1) > 0)
    y_train = y_train[train_example_mask, :]
    y_test = y_test[test_example_mask, :]
    X_train = X_train[train_example_mask, :]
    X_test = X_test[test_example_mask, :]

    return X_train, y_train, X_test, y_test, class_indices, train_example_mask, test_example_mask


#return X_train, y_train, X_test, y_test
def load_data(dataset_name, input_type, miniclass):
    assert(input_type in ['cossims', 'probs', 'log_probs', 'logits'])
    if miniclass:
        return load_data_miniclass(dataset_name, input_type)

    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        y_train = pickle.load(f)

    y_train = np.array([y_train[impath] for impath in sorted(y_train.keys())])
    with open(TESTING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        y_test = pickle.load(f)

    y_test = np.array([y_test[impath] for impath in sorted(y_test.keys())])
    train_scores_filename_dict = (PSEUDOLABEL_LOGITS_FILENAME_DICT if input_type in ['probs', 'log_probs', 'logits'] else PSEUDOLABEL_COSSIMS_FILENAME_DICT)
    test_scores_filename_dict = (PSEUDOLABEL_TESTING_LOGITS_FILENAME_DICT if input_type in ['probs', 'log_probs', 'logits'] else PSEUDOLABEL_TESTING_COSSIMS_FILENAME_DICT)
    with open(train_scores_filename_dict[dataset_name], 'rb') as f:
        X_train = pickle.load(f)

    with open(test_scores_filename_dict[dataset_name], 'rb') as f:
        X_test = pickle.load(f)

    if input_type == 'probs':
        X_train = {impath : 1. / (1. + np.exp(-X_train[impath])) for impath in sorted(X_train.keys())}
        X_test = {impath : 1. / (1. + np.exp(-X_test[impath])) for impath in sorted(X_test.keys())}
    elif input_type == 'log_probs':
        X_train = {impath : -np.log(1. + np.exp(-X_train[impath])) for impath in sorted(X_train.keys())}
        X_test = {impath : -np.log(1. + np.exp(-X_test[impath])) for impath in sorted(X_test.keys())}

    X_train = np.array([X_train[impath] for impath in sorted(X_train.keys())])
    X_test = np.array([X_test[impath] for impath in sorted(X_test.keys())])

    return X_train, y_train, X_test, y_test, np.arange(y_train.shape[1]), np.ones(y_train.shape[0]).astype('bool'), np.ones(y_test.shape[0]).astype('bool')


def make_output_filename(dataset_name, input_type, standardize, balance, L1, C, miniclass):
    os.makedirs(OUT_PARENT_DIR, exist_ok=True)
    return os.path.join(OUT_PARENT_DIR, 'logistic_regression_%s_%s_standardize%d_balance%d_L1%d_C%s_miniclass%d.pkl'%(dataset_name.split('_')[0], input_type, standardize, balance, L1, str(C), miniclass))


def logistic_regression_experiment(dataset_name, input_type, standardize, balance, L1, C, miniclass):
    standardize = int(standardize)
    balance = int(balance)
    L1 = int(L1)
    C = float(C)
    miniclass = int(miniclass)
    if miniclass: #just some checks against intended use, not that it necessarily needs to be this way
        assert(not standardize)
        assert(not balance)
        assert(not L1)

    info_dict = {'dataset_name' : dataset_name, 'input_type' : input_type, 'standardize' : standardize, 'balance' : balance, 'L1' : L1, 'C' : C, 'miniclass' : miniclass}
    output_filename = make_output_filename(dataset_name, input_type, standardize, balance, L1, C, miniclass)
    X_train, y_train, X_test, y_test, class_indices, train_example_mask, test_example_mask = load_data(dataset_name, input_type, miniclass)
    data_dict = {'X_train' : X_train, 'y_train' : y_train, 'X_test' : X_test, 'y_test' : y_test, 'class_indices' : class_indices, 'train_example_mask' : train_example_mask, 'test_example_mask' : test_example_mask}
    my_clf = fit_model(X_train, y_train, standardize, balance, L1, C)
    eval_dict = evaluate(X_train, y_train, X_test, y_test, my_clf)
    output_dict = {'info' : info_dict, 'data' : data_dict, 'model' : my_clf, 'eval' : eval_dict}
    print('dataset_name=%s, input_type=%s, standardize=%d, balance=%d, L1=%d, C=%s, miniclass=%d, train_mAP=%f, test_mAP=%f, input_train_mAP=%f, input_test_mAP=%f'%(dataset_name, input_type, standardize, balance, L1, str(C), miniclass, eval_dict['train_mAP'], eval_dict['test_mAP'], eval_dict['input_train_mAP'], eval_dict['input_test_mAP']))
    with open(output_filename, 'wb') as f:
        pickle.dump(output_dict, f)


def usage():
    print('Usage: python logistic_regression_experiment.py <dataset_name> <input_type> <standardize> <balance> <L1> <C> <miniclass>')


if __name__ == '__main__':
    logistic_regression_experiment(*(sys.argv[1:]))
