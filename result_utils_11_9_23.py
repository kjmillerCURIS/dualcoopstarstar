import os
import sys
import glob
import pickle


PROMPT_MODE_DICT = {'pos_only_fixed_prompt' : 'posonly_fixed', 'pos_and_neg_learnable_prompt' : 'posneglearn'}
OUTPUT_BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output')
RESULTS_DIR_SUFFIX = 'DualCoOpStarStar_FullPartial/rn101/nctx21_cscTrue_ctpend/seed1/results'
DISCARD_LAST_POINT = True #do this because I think my code might be retesting the "best" model and accidentally rewriting the results-50.pkl file


def get_epoch(results_filename):
    return int(os.path.splitext(os.path.basename(results_filename))[0].split('-')[-1])


def extract_random_chance_mAPs():
    results_dir = os.path.join(OUTPUT_BASE_DIR, 'random_chance_p05_COCO_dualcoopstarstar_fullpartial/DualCoOpStarStar_FullPartial/rn101/nctx21_cscTrue_ctpend/seed1/results')
    results_filename = os.path.join(results_dir, 'results-001.pkl')
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    return results['results_all']



#returns {'hugarian' : {'epoch' : [], 'mAP' : []}, 'max' : {'epoch' : [], 'mAP' : []}}
#or just {'epoch' : [], 'mAP' : []} if is_baseline=True
def extract_mAPs_given_runID(runID, is_baseline=False, results_dir_suffix=RESULTS_DIR_SUFFIX):
    results_dir = os.path.join(OUTPUT_BASE_DIR, runID, results_dir_suffix)
    print(results_dir)
    results_filenames = sorted(glob.glob(os.path.join(results_dir, 'results-*.pkl')))
    epochs, results_filenames = zip(*sorted([(get_epoch(results_filename), results_filename) for results_filename in results_filenames], key = lambda p : p[0]))
    if DISCARD_LAST_POINT:
        epochs = epochs[:-1]
        results_filenames = results_filenames[:-1]

    if is_baseline:
        ret = {'epoch' : epochs, 'mAP' : []}
    else:
        ret = {'hungarian' : {'epoch' : epochs, 'mAP' : []}, 'max' : {'epoch' : epochs, 'mAP' : []}}

    for results_filename in results_filenames:
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)

        if is_baseline:
            ret['mAP'].append(results['mAP'])
        else:
            for meow in ['hungarian', 'max']:
                ret[meow]['mAP'].append(results['results_all'][meow])

    return ret


def make_runID(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable):
    return 'dualcoopstarstar_fullpartial_coco_p05_seed1_%s_enc%d_dec%d_spatl%d_spatu%d_templearn%d_mean_except_class'%(PROMPT_MODE_DICT[prompt_mode], num_encoder_layers, num_decoder_layers, spatial_lower, spatial_upper, temperature_is_learnable)


def extract_mAPs_given_hparams(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable):
    runID = make_runID(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable)
    return extract_mAPs_given_runID(runID)
