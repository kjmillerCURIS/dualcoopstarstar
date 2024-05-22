import os
import sys
import torch
from tqdm import tqdm
from do_MWIS import get_results_filename


CONFLICT_THRESHOLD_LIST = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2]
SCORE_TYPE_LIST = ['rank_with_gtmargs', 'rank_without_gtmargs', 'binary_top1perc', 'binary_top5perc', 'binary_top2perc', 'prob', 'neglogcompprob', 'adaptivelogprob_onemin', 'adaptivelogprob_minperclass']
TABLE_FILENAME_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/%s_MWIS_results.csv'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}


def tablify_MWIS_results(dataset_name):
    table_filename = TABLE_FILENAME_DICT[dataset_name]
    f = open(table_filename, 'w')
    f.write('"conflict threshold (if Pr(i,j) / (Pr(i)*Pr(j)) < threshold ==> conflict)","score type","classes outside of top-20?","mAP"\n')
    for conflict_threshold in CONFLICT_THRESHOLD_LIST:
        for score_type in SCORE_TYPE_LIST:
            for zoo in [False, True]:
                items = [str(conflict_threshold), score_type]
                items.append({False : 'keep them', True : 'eliminate them'}[zoo])
                results_filename = get_results_filename(dataset_name, conflict_threshold, score_type)
                checkpoint = torch.load(results_filename)
                items.append(str(100.0 * checkpoint[{False : 'mAP_zooNO', True : 'mAP_zooYES'}[zoo]]) + '%')
                items = ['"' + x + '"' for x in items]
                f.write(','.join(items) + '\n')

    f.close()


def usage():
    print('Usage: python tablify_MWIS_results.py <dataset_name>')


if __name__ == '__main__':
    tablify_MWIS_results(*(sys.argv[1:]))
