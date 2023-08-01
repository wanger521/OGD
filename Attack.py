import numpy as np
from Config import regular, byzantine

"""
Different Byzantine attacks, include:
Gaussian attacks, sign-flipping attacks, sample-duplicating attacks

@:param workerGrad : the set of workers' gradients
"""

def gaussian_attacks(messages, regular=regular, byzantine = byzantine):
    # class_x = np.size(messages, 1)
    # dimension_y = np.size(messages, 2)
    shape1 = messages[0].shape
    for id in byzantine:
        messages[id] = np.random.normal(loc=0, scale=200,  size=shape1) # 200 for mnist,artifial2,artificial6,500 for artifcial 4, 5
    return messages, '-gs'


def sign_flipping_attacks(messages, regular=regular, byzantine=byzantine):
    for id in byzantine:
        messages[id] = - 1 * messages[regular[0]]  # 之前是-1 for mnist,artificial6, -1.5 for artificial2,-3 for artificial 4,5
    return messages, '-sf'


def sample_duplicating_attacks(messages, regular=regular, byzantine = byzantine):
    for id in byzantine:
        messages[id] = messages[regular[0]]
    return messages, '-hd'