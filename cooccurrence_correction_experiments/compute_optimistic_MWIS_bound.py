import os
import sys
import numpy as np
import pickle
from tqdm import tqdm


CURRENT_MAPS_DICT = {}
CURRENT_MAP_NAMES = ['prob_stopgrad_logit', 'correlation']
