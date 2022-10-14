import numpy as np
from Config import regular, byzantine

"""
Different Byzantine attacks, include:
Gaussian attacks, sign-flipping attacks, sample-duplicating attacks

@:param workerGrad : the set of workers' gradients
"""

def gaussian_attacks(messages):
    class_x = np.size(messages, 1)
    dimension_y = np.size(messages, 2)
    for id in byzantine:
        messages[id] = np.random.normal(loc=0, scale=200,  size=(class_x, dimension_y)) # 200
    return messages, '-gs'


def sign_flipping_attacks(messages):
    for id in byzantine:
        messages[id] = - 1 * messages[regular[0]]
    return messages, '-sf'


def sample_duplicating_attacks(messages):
    for id in byzantine:
        messages[id] = messages[regular[0]]
    return messages, '-hd'
