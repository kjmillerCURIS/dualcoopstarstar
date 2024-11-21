import os
import sys
from second_max_experiments import second_max_experiments


DATASET_NAMES = ['COCO2014_partial', 'VOC2007_partial', 'nuswide_partial']
MODEL_TYPES = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14', 'ViT-L14336px']


def bulk_second_max_experiments():
    for dataset_name in DATASET_NAMES:
        for model_type in MODEL_TYPES:
            second_max_experiments(dataset_name, model_type, 1)


if __name__ == '__main__':
    bulk_second_max_experiments()
