import random
import numpy as np
#np.random.seed(32)
#random.seed(32)

# configuration
optConfig = {
    'nodeSize': 30, #30,
    'byzantineSize': 0, # 之前是5， if without byzantine attack, set byzantineSize=0

    'iterations': 10000,#10000 mnist
    'epoch': 10,
    'decayWeight': 0,

    'GeoMedMaxIter': 80,
    'GeoMedTol': 1e-5,

    'ResamplingTime': 30,
    'ResamplingSize': 2,

    'batchSize': 32# 32,

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

Byrd3SgdConfig = optConfig.copy()
Byrd3SgdConfig['learningStep'] = 0.2
Byrd3SgdConfig['q'] = Byrd3SgdConfig['byzantineSize']+2

RsByrdSgdConfig = optConfig.copy()
RsByrdSgdConfig['learningStep'] = 0.4

ByrdSagaConfig = optConfig.copy()
ByrdSagaConfig['learningStep'] = 0.1

RsaConfig = optConfig.copy()
RsaConfig['learningStep'] = 1
RsaConfig['penaltyPara'] = 0.005

CMedianConfig = optConfig.copy()
CMedianConfig['learningStep'] = 0.001

# CenterClipConfig = optConfig.copy()
# CenterClipConfig['learningStep'] = 0.1
# CenterClipConfig['tau'] = 0.009  # while iid 0.02 ; noniid
# CenterClipConfig['iner_iter'] = 3  # 5

# for artificial data
CenterClipConfig = optConfig.copy()
CenterClipConfig['learningStep'] = 0.1
CenterClipConfig['tau'] = 0.02#0.02 artificial constant step # while iid 0.02 ; noniid
CenterClipConfig['iner_iter'] = 1  # 5
CenterClipConfig['get_true'] = False

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

OursConfig = optConfig.copy()
OursConfig['learningStep'] = 0.05
OursConfig['trimmed_range'] = 0.2
OursConfig['tau'] =   6 #=6 : when byzantine 5
OursConfig['delta'] = 3.5 #3.5 : when byzantine 5
OursConfig['iner_iter'] = 1

OursMeanConfig = optConfig.copy()
OursMeanConfig['learningStep'] = 0.05
OursMeanConfig['trimmed_range'] = 0.2
OursMeanConfig['tau'] =  6 #=6 : when byzantine 5
OursMeanConfig['delta'] = 3.5 #3.5 : when byzantine 5
OursMeanConfig['iner_iter'] = 1

# MNIST dataset
mnistConfig = {
    'trainNum': 60000,
    'testNum': 10000,
    'dimension': 784,
    'classes': 10,
}

artificialConfig = {
    'trainNum': 60000,
    'testNum': 100,
    'dimension': 10
}

cifar10Config = {
    'trainNum': 50000,
    'testNum': 10000,
    'dimension': 3072,
    'classes': 10,
}

# Randomly generate Byzantine workers
byzantine = random.sample(range(optConfig['nodeSize']), optConfig['byzantineSize'])
regular = list(set(range(optConfig['nodeSize'])).difference(byzantine))

