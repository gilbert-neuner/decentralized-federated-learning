import numpy as np
from numpy import linalg as LA

def confusion_matrix(rule, actual, predicted):
    out = np.zeros([2, 2])
    for i in range(len(actual)):
        if rule(actual[i]) and rule(predicted[i]):
            out[0, 0] += 1
        elif rule(actual[i]) and not rule(predicted[i]):
            out[0, 1] += 1
        elif not rule(actual[i]) and rule(predicted[i]):
            out[1, 0] += 1
        else:
            out[1, 1] += 1
    return out

def F1(rule, actual, predicted):
    conf_mtx = confusion_matrix(rule, actual, predicted)
    return 2 * conf_mtx[0, 0] / (2 * conf_mtx[0, 0] + conf_mtx[0, 1] + conf_mtx[1, 0])

# true - hat / true
def rel_norm(actual, predicted):
    return LA.norm(actual - predicted) / LA.norm(actual)
    