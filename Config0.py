import random
import numpy as np
np.random.seed(32)
random.seed(32)

# configuration
optConfig = {
    'nodeSize': 30,
    'byzantineSize': 0, 

    'iterations': 10000,
    'epoch': 10,
    'decayWeight': 0,

    'GeoMedMaxIter': 80,
    'GeoMedTol': 1e-5,

    'ResamplingTime': 30,
    'ResamplingSize': 2,

    'batchSize': 32,

}

SgdConfig = optConfig.copy()
SgdConfig['learningStep'] = 0.1

TrimSgdConfig = optConfig.copy()
TrimSgdConfig['learningStep'] = 0.1
TrimSgdConfig['trimmed_range'] = TrimSgdConfig['byzantineSize']/TrimSgdConfig['nodeSize']

KrumConfig = optConfig.copy()
KrumConfig['learningStep'] = 0.1

MutiKrumConfig = optConfig.copy()
MutiKrumConfig['learningStep'] = 0.1
MutiKrumConfig['M'] = 5

ByrdSgdConfig = optConfig.copy()
ByrdSgdConfig['learningStep'] = 0.1

CMedianConfig = optConfig.copy()
CMedianConfig['learningStep'] = 0.001

CenterClipConfig = optConfig.copy()
CenterClipConfig['learningStep'] = 0.1
CenterClipConfig['tau'] = 0.02
CenterClipConfig['iner_iter'] = 5

FABAConfig = optConfig.copy()
FABAConfig['learningStep'] = 0.1

PhocasConfig = optConfig.copy()
PhocasConfig['learningStep'] = 0.1
PhocasConfig['trimmed_range'] = 0.2

ZenoConfig = optConfig.copy()
ZenoConfig['learningStep'] = 0.1
ZenoConfig['rho_ratio'] = 200 #20
ZenoConfig['remove_size'] = ZenoConfig['byzantineSize']+2
ZenoConfig['zeno_batch'] = 32
ZenoConfig['gamma'] = 0.05

# MNIST dataset
mnistConfig = {
    'trainNum': 60000,
    'testNum': 10000,
    'dimension': 784,
    'classes': 10,
}

# Randomly generate Byzantine workers
byzantine = random.sample(range(optConfig['nodeSize']), optConfig['byzantineSize'])
regular = list(set(range(optConfig['nodeSize'])).difference(byzantine))

