from hickle import *
import random
from IPython import embed
import numpy as np

def augmentation(a):
    keys = a.keys()
    b = dict()
    for key in keys: b[key] = list(a[key])

    for i in xrange(9566):
        for _ in xrange(10):
            correct_index = b['ground_truth_train'][i]
            random_index = random.randrange(0,9566)
            new_answer = b['zaj_train'][random_index]
            new_answer[correct_index] = b['zaj_train'][i][correct_index]
            b['zq_train'].append(b['zq_train'][i])
            b['zsl_train'].append(b['zsl_train'][i])
            b['ground_truth_train'].append(b['ground_truth_train'][i])
            b['zaj_train'].append(new_answer)

    for key in keys:
        print '====================================='
        b[key] = np.array(b[key])
        print 'key >> ', key
        print 'shape >> ', b[key].shape

    return b



