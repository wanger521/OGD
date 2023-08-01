
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import platform
import os
import numpy as np


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = X.astype("float")
        Y = np.array(Y)
        return X, Y


def getCifar10Data(ROOT):
    """
    Get the images and labels from cifar10 dataset

    :param grootPath:
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    #Xtr = Xtr.reshape((50000, 3072))
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    #Xte = Xte.reshape((10000, 3072))
    return Xtr, Ytr, Xte, Yte



def data_redistribute_cifar10(image, label):
    """
    Rearrange the samples in the order of label

    :param image: image, shape(10, 784)
    :param label: label, scalar
    """
    number_sample = len(label)
    im = [[] for _ in range(10)]
    la = [[] for _ in range(10)]
    for i in range(number_sample):
        im[label[i]].append(image[i])
        la[label[i]].append(label[i])
    data_image = []
    data_label = []
    for i in range(10):
        for j in range(len(la[i])):
            data_image.append(im[i][j])
            data_label.append(la[i][j])

    return data_image, data_label



