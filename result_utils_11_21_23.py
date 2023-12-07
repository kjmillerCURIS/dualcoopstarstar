import os
import sys
import glob
import pickle


PROMPT_MODE_DICT = {'pos_only_fixed_prompt' : 'posonly_fixed', 'pos_and_neg_learnable_prompt' : 'posneglearn'}
OUTPUT_BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output')
RESULTS_DIR_SUFFIX = 'Caption_tri_wta_soft_pseudolabel/rn101/nctx21_cscTrue_ctpend/seed1/results'


def get_epoch(results_filename):
    return int(os.path.splitext(os.path.basename(results_filename))[0].split('-')[-1])


def extract_random_chance_mAPs():
    results_dir = os.path.join(OUTPUT_BASE_DIR, 'random_chance_p05_COCO_dualcoopstarstar_fullpartial/DualCoOpStarStar_FullPartial/rn101/nctx21_cscTrue_ctpend/seed1/results')
    results_filename = os.path.join(results_dir, 'results-001.pkl')
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    return results['mAP']


def extract_zsclip_mAPs():
    results_dir = os.path.join(OUTPUT_BASE_DIR, 'zsclip_ensemble80_COCO/Caption_tri_wta_soft_pseudolabel/rn101/nctx21_cscTrue_ctpend/seed1/results')
    results_filename = os.path.join(results_dir, 'results-001.pkl')
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    return results['mAP']


#returns {'hugarian' : {'epoch' : [], 'mAP' : []}, 'max' : {'epoch' : [], 'mAP' : []}}
#or just {'epoch' : [], 'mAP' : []} if is_baseline=True
def extract_mAPs_given_runID(runID, results_dir_suffix=RESULTS_DIR_SUFFIX, discard_last_point=False):
    results_dir = os.path.join(OUTPUT_BASE_DIR, runID, results_dir_suffix)
    results_filenames = sorted(glob.glob(os.path.join(results_dir, 'results-*.pkl')))
    results_filenames = [s for s in results_filenames if os.path.splitext(os.path.basename(s))[0].split('-')[-1] != 'after_train']
    if len(results_filenames) == 0:
        return None

    print(results_dir)
    epochs, results_filenames = zip(*sorted([(get_epoch(results_filename), results_filename) for results_filename in results_filenames], key = lambda p : p[0]))
    if discard_last_point:
        epochs = epochs[:-1]
        results_filenames = results_filenames[:-1]

    ret = {'epoch' : epochs, 'mAP' : []}
    for results_filename in results_filenames:
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)

        ret['mAP'].append(results['mAP'])

    return ret


def extract_mAPs_given_runID_trainpseudoeval(runID, results_dir_suffix=RESULTS_DIR_SUFFIX):
    results_filename = os.path.join(OUTPUT_BASE_DIR, runID, results_dir_suffix, '../training_pseudolabel_results.pkl')
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    key_list = sorted(results.keys(), key=lambda k: (0 if k == 'init' else k))
    epochs = [(0 if k == 'init' else k) for k in key_list]
    mAPs = [results[k]['mAP'] for k in key_list]
    return {'epoch' : epochs, 'mAP' : mAPs}


#def make_runID(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable):
#    return 'dualcoopstarstar_fullpartial_coco_p05_seed1_%s_enc%d_dec%d_spatl%d_spatu%d_templearn%d_mean_except_class'%(PROMPT_MODE_DICT[prompt_mode], num_encoder_layers, num_decoder_layers, spatial_lower, spatial_upper, temperature_is_learnable)


def make_runID(do_adjust_logits, use_bias, bandwidth, stepsize):
    return 'pseudolabel_coco_seed1_adjust%d_bias%d_bandwidth%.5f_stepsize%.5f'%(do_adjust_logits, use_bias, bandwidth, stepsize)


def extract_mAPs_given_hparams(do_adjust_logits, use_bias, bandwidth, stepsize):
    runID = make_runID(do_adjust_logits, use_bias, bandwidth, stepsize)
    return extract_mAPs_given_runID(runID)


def extract_mAPs_given_hparams_trainpseudoeval(do_adjust_logits, use_bias, bandwidth, stepsize):
    runID = make_runID(do_adjust_logits, use_bias, bandwidth, stepsize)
    return extract_mAPs_given_runID_trainpseudoeval(runID)
