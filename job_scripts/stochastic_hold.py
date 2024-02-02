import os
import sys
import random

def stochastic_hold(mode, p):
    p = float(p)
    if mode == 'qhold':
        lines = os.popen('qstat -u nivek | grep " nivek " | grep " qw "').readlines()
    elif mode == 'qrls':
        lines = os.popen('qstat -u nivek | grep " nivek " | grep " hqw "').readlines()
    else:
        assert(False)

    ids = [line.strip().split(' ')[0] for line in lines]
    print(ids)
    for my_id in ids:
        if random.uniform(0,1) < p:
            os.system(mode + ' -h u ' + my_id)


if __name__ == '__main__':
    stochastic_hold(*(sys.argv[1:]))
