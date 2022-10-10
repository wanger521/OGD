import math
import math

import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

from numpy import ndarray

from LoadMnist import getData, data_redistribute
import Config
import Config0
from FatherModel import Softmax, get_accuracy, get_vars
from Attack import gaussian_attacks, sign_flipping_attacks, sample_duplicating_attacks
import logging.config
import os
from Models.GeoMed.GeoMed import GeoMedWorkerSoftmax, GeoMedServerSoftmax
from Models.Mean.Mean import MeanWorkerSoftmax, MeanServerSoftmax
from Models.trimmed_mean.trimmed_mean import TrimMeanWorkerSoftmax, TrimMeanServerSoftmax
from Models.Krum.Krum import KrumWorkerSoftmax, KrumServerSoftmax
from Models.Krum.MutiKrum import MutiKrumWorkerSoftmax, MutiKrumServerSoftmax
from Models.CMedian.CMedian import CMedianWorkerSoftmax, CMedianServerSoftmax
from Models.CenterClip.CenterClip import CenterClipWorkerSoftmax, CenterClipServerSoftmax
from Models.FABA.FABA import FABAWorkerSoftmax, FABAServerSoftmax
from Models.Phocas.Phocas import PhocasWorkerSoftmax, PhocasServerSoftmax
from Models.Zeno.Zeno import ZenoWorkerSoftmax, ZenoServerSoftmax
from Models.Ours.Ours import OursWorkerSoftmax, OursServerSoftmax
from Models.Ours.OursMean import OursMeanWorkerSoftmax, OursMeanServerSoftmax

logging.config.fileConfig(fname='Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger("infoLogger")


def personilized(model_name, attack):
    if attack == None:
        if model_name == 'Mean':
            WorkerSoftmax = MeanWorkerSoftmax
            ServerSoftmax = MeanServerSoftmax
            conf = Config0.SgdConfig.copy()
        elif model_name == 'GeoMed':
            WorkerSoftmax = GeoMedWorkerSoftmax
            ServerSoftmax = GeoMedServerSoftmax
            conf = Config0.ByrdSgdConfig.copy()
        elif model_name == 'trimmed-mean':
            WorkerSoftmax = TrimMeanWorkerSoftmax
            ServerSoftmax = TrimMeanServerSoftmax
            conf = Config0.TrimSgdConfig.copy()
        elif model_name == 'Krum':
            WorkerSoftmax = KrumWorkerSoftmax
            ServerSoftmax = KrumServerSoftmax
            conf = Config0.KrumConfig.copy()
        elif model_name == 'MutiKrum':
            WorkerSoftmax = MutiKrumWorkerSoftmax
            ServerSoftmax = MutiKrumServerSoftmax
            conf = Config0.MutiKrumConfig.copy()
        elif model_name == 'CMedian':
            WorkerSoftmax = CMedianWorkerSoftmax
            ServerSoftmax = CMedianServerSoftmax
            conf = Config0.CMedianConfig.copy()
        elif model_name == 'CenterClip':
            WorkerSoftmax = CenterClipWorkerSoftmax
            ServerSoftmax = CenterClipServerSoftmax
            conf = Config0.CenterClipConfig.copy()
        elif model_name == 'FABA':
            WorkerSoftmax = FABAWorkerSoftmax
            ServerSoftmax = FABAServerSoftmax
            conf = Config0.FABAConfig.copy()
        elif model_name == 'Phocas':
            WorkerSoftmax = PhocasWorkerSoftmax
            ServerSoftmax = PhocasServerSoftmax
            conf = Config0.PhocasConfig.copy()
        elif model_name == 'Zeno':
            WorkerSoftmax = ZenoWorkerSoftmax
            ServerSoftmax = ZenoServerSoftmax
            conf = Config0.ZenoConfig.copy()
        elif model_name == 'Ours':
            WorkerSoftmax = OursWorkerSoftmax
            ServerSoftmax = OursServerSoftmax
            conf = Config0.OursConfig.copy()
        elif model_name == '0-Ours':
            WorkerSoftmax = OursMeanWorkerSoftmax
            ServerSoftmax = OursMeanServerSoftmax
            conf = Config0.OursMeanConfig.copy()
        else:
            WorkerSoftmax = MeanWorkerSoftmax
            ServerSoftmax = MeanServerSoftmax
            conf = Config0.SgdConfig.copy()
    else:
        if model_name == 'Mean':
            WorkerSoftmax = MeanWorkerSoftmax
            ServerSoftmax = MeanServerSoftmax
            conf = Config.SgdConfig.copy()
        elif model_name == 'GeoMed':
            WorkerSoftmax = GeoMedWorkerSoftmax
            ServerSoftmax = GeoMedServerSoftmax
            conf = Config.ByrdSgdConfig.copy()
        elif model_name == 'trimmed-mean':
            WorkerSoftmax = TrimMeanWorkerSoftmax
            ServerSoftmax = TrimMeanServerSoftmax
            conf = Config.TrimSgdConfig.copy()
        elif model_name == 'Krum':
            WorkerSoftmax = KrumWorkerSoftmax
            ServerSoftmax = KrumServerSoftmax
            conf = Config.KrumConfig.copy()
        elif model_name == 'MutiKrum':
            WorkerSoftmax = MutiKrumWorkerSoftmax
            ServerSoftmax = MutiKrumServerSoftmax
            conf = Config.MutiKrumConfig.copy()
        elif model_name == 'CMedian':
            WorkerSoftmax = CMedianWorkerSoftmax
            ServerSoftmax = CMedianServerSoftmax
            conf = Config.CMedianConfig.copy()
        elif model_name == 'CenterClip':
            WorkerSoftmax = CenterClipWorkerSoftmax
            ServerSoftmax = CenterClipServerSoftmax
            conf = Config.CenterClipConfig.copy()
        elif model_name == 'FABA':
            WorkerSoftmax = FABAWorkerSoftmax
            ServerSoftmax = FABAServerSoftmax
            conf = Config.FABAConfig.copy()
        elif model_name == 'Phocas':
            WorkerSoftmax = PhocasWorkerSoftmax
            ServerSoftmax = PhocasServerSoftmax
            conf = Config.PhocasConfig.copy()
        elif model_name == 'Zeno':
            WorkerSoftmax = ZenoWorkerSoftmax
            ServerSoftmax = ZenoServerSoftmax
            conf = Config.ZenoConfig.copy()
        elif model_name == 'Ours':
            WorkerSoftmax = OursWorkerSoftmax
            ServerSoftmax = OursServerSoftmax
            conf = Config.OursConfig.copy()
        elif model_name == '0-Ours':
            WorkerSoftmax = OursMeanWorkerSoftmax
            ServerSoftmax = OursMeanServerSoftmax
            conf = Config.OursMeanConfig.copy()
        else:
            WorkerSoftmax = MeanWorkerSoftmax
            ServerSoftmax = MeanServerSoftmax
            conf = Config.SgdConfig.copy()
    return WorkerSoftmax, ServerSoftmax, conf


def model_sgd(setting,descent_way, attack, w_best, eta, model_name, m, groot):
    # print Byzantine and honest node
    if attack == None:
        byzantine = Config0.byzantine
        regular = Config0.regular
    else:
        byzantine = Config.byzantine
        regular = Config.regular
    print(byzantine)
    print(regular)

    # Get the mnist training data
    image, label = getData('datasets\\MNIST\\train-images.idx3-ubyte',
                           'datasets\\MNIST\\train-labels.idx1-ubyte')

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image, label = data_redistribute(image, label)

    # Get the testing data
    image_test, label_test = getData('datasets\\MNIST\\t10k-images.idx3-ubyte',
                                     'datasets\\MNIST\\t10k-labels.idx1-ubyte')
    # Load the configurations
    WorkerSoftmax, ServerSoftmax, conf = personilized(model_name, attack)
    data_config = Config.mnistConfig
    num_data = int(data_config['trainNum'] / conf['nodeSize'])
    data_dimension = data_config['dimension']
    data_classes = data_config['classes']
    nodesize = conf['nodeSize']
    last_str = ''

    # Parameter initialization
    serverPara = np.zeros((data_classes, data_dimension))
    workerGrad = np.zeros((nodesize, data_classes, data_dimension))
    momentum = np.zeros((nodesize, data_classes, data_dimension))
    messages = np.zeros((nodesize, data_classes, data_dimension))  # If aggregate model, it mean workers variable; If
    # gradient, it mean gradients.
    last_agg = np.zeros((data_classes, data_dimension))

    k = 0
    max_iterations = conf['iterations']

    classification_accuracy = []
    variances = []
    regret_train_ls = []
    regret_train_t = np.array(0, dtype=float)
    acc_last = 0

    logger.info('Start！')
    while k < max_iterations:
        k += 1
        regret_train = []
        # workers compute gradients
        for id in range(nodesize):
            select = np.random.randint(0, num_data - conf['batchSize'])
            worker = WorkerSoftmax(serverPara, id, select, conf)
            image_input = image[id * num_data: (id + 1) * num_data]
            label_input = label[id * num_data: (id + 1) * num_data]
            workerGrad[id] = worker.cal_batch_sto_grad(image_input, label_input)
            if id in regular:
                regret_train.append(worker.cal_t_regret(image_input, label_input, w_best))
            regret_train_t = regret_train_t + np.mean(np.array(regret_train))

        # workers compute momentum
        true_m = 0
        if m <= 1:  # constant momentum size
            true_m = m
        elif 1 < m <= 4:  # diminishing momentum size
            if k < 500:
                true_m = 0.1
            else:
                true_m = m * 50 / k
        else:   # 1/sqrtT momentum size
            true_m = 0.001 * m  # sqrtT momentum
        # according true_m (momentum size) to update momentum for each worker
        momentum = true_m * workerGrad + (1-true_m) * momentum

        # transcoding to true step size based on the given value of eta
        true_eta = 0
        if eta <= 1:  # constant step size, which is set in Config for different aggregation rulse
            true_eta = eta  # conf['learningStep']
        elif 1 < eta <= 4:  # diminishing step size
            if k < 500:
                true_eta = 0.1
            else:
                true_eta = eta * 50 / k
        else:  # 1/sqrtT step size
            true_eta = 0.001 * eta

        # train model, if we aggregate model or gradient, using different train way
        if descent_way == 'model':  # if we aggregate model
            messages = serverPara - true_eta * momentum  # each worker make gradient descent first
            # Byzantine make attack
            if attack is not None:
                messages, last_str = attack(messages)
            # Master node aggregate messages, hence update the global model
            last_agg = serverPara  # update last iteration aggregation results
            server = ServerSoftmax(serverPara, conf, messages, last_agg)
            agg_results = server.train()
            serverPara = agg_results
        elif descent_way == 'gradient':  # if we aggregate gradient
            messages = workerGrad
            if attack is not None:
                messages, last_str = attack(messages)
            server = ServerSoftmax(serverPara, conf, messages, last_agg)
            agg_results = server.train()
            last_agg = agg_results
            serverPara = serverPara - true_eta * agg_results

        # calculate accuracy and regret, and print

        acc = get_accuracy(serverPara, image_test, label_test)
        classification_accuracy.append(acc)
        regret_train_ls.append(regret_train_t)

        var = get_vars(regular, momentum)
        variances.append(var)
        if k % 500 == 0:
            logger.info('the {}th iteration acc: {}, regret: {}, vars: {}'.format(k, acc, regret_train_t, var))

        if k == max_iterations and attack is None and model_name == 'Mean':
            w_best = serverPara.copy()
            acc_last = acc

    # save model
    print(classification_accuracy)
    # Save the experiment results
    output = open('results\\{0}\\{1}\\{2}\\{3}{4}-step{5}-setting-{6}-momentum{7}.pkl'.format(
                                groot, descent_way, model_name, model_name, last_str, str(int(eta * 10)),
                                setting, str(int(m * 10))), 'wb')
    pickle.dump((classification_accuracy, variances, regret_train_ls), output, protocol=pickle.HIGHEST_PROTOCOL)
    print("Run over {0}{1}-step{2}-setting-{3}-momentum{4}.pkl".format(model_name, last_str,
                                                                        str(int(eta * 10)), setting, str(int(m * 10))))
    # return results for find best solution
    if attack is None and model_name == 'Mean':
        return acc_last, w_best


def find_w_best(descent_way, groot):
    filename = 'results\\{0}\\{1}\\{2}\\w_best_1.csv'.format(groot, descent_way, 'Mean')
    filename_eta = 'results\\{0}\\{1}\\{2}\\eta_best.csv'.format(groot, descent_way, 'Mean')
    if os.path.exists(filename) and os.path.exists(filename_eta):
        print("Already find the best solution! The results give u.")
        with open(filename_eta, 'r') as f:
            eta_best = f.read()
        w_best = np.loadtxt(open(filename, 'rb'), delimiter=',', skiprows=0)
        return w_best, eta_best
    acc = 0
    eta_best = 0
    data_config = Config.mnistConfig
    data_dimension = data_config['dimension']
    data_classes = data_config['classes']
    w_best = np.zeros((data_classes, data_dimension))
    w_best.fill(random.random() + 1e-7)
    eta_best_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 2, 3, 5, 20, 50]

    print("Start finding best solution!")
    for eta in eta_best_ls:
        acc_last, w_best_last = model_sgd(setting='iid', descent_way=descent_way, attack=None, w_best=w_best, eta=eta,
              model_name='Mean', m=1, groot=groot)
        print(acc_last)
        if acc_last > acc:
            acc = acc_last
            w_best = w_best_last
            eta_best = eta
    print("eta best is :")
    print(eta_best)
    with open(filename_eta, 'w') as f:
        f.write(str(eta_best))
    np.savetxt(filename, w_best, delimiter=',')
    print("Finding over! Saved over!")
    return w_best, eta_best


if __name__ == '__main__':
    groot = "test"
    descent_way_ls = ['model']  # ['model', 'gradient']
    eta_ls = [0.1, 2, 10]
    setting_ls = ['iid', 'noniid']
    attack_ls = [None, sign_flipping_attacks, gaussian_attacks, sample_duplicating_attacks]
    last_str = ['', '-sf', '-gs', '-hd']
    momentum_ls = [0.1, 1, 2, 10]
    model_name_ls = ['Mean', 'CMedian', 'trimmed-mean', 'GeoMed',  'Krum', 'CenterClip', 'Phocas', 'FABA']
    # ['Mean', 'CMedian', 'trimmed-mean', 'GeoMed',  'Krum', 'CenterClip', 'Phocas', 'FABA']
    # [ 'MutiKrum', 'Zeno', 'Ours', '0-Ours']
    w_best, eta_best = find_w_best(descent_way='model', groot=groot)

    # run
    for descent_way in descent_way_ls:
        for model_name in model_name_ls:
            for setting in setting_ls:
                for attack, last in zip(attack_ls, last_str):
                    for eta in eta_ls:
                        for m in momentum_ls:
                            paths = 'results\\{0}\\{1}\\{2}\\{3}{4}-step{5}-setting-{6}-momentum{7}.pkl'.format(
                                groot, descent_way, model_name, model_name, last, str(int(eta * 10)), setting, str(int(m * 10)))
                            if os.path.exists(paths):
                                print("Run over {0}{1}-step{2}-setting-{3}-momentum{4}.pkl".format(model_name, last,
                                                                                                   str(int(eta * 10)),
                                                                                                   setting,
                                                                                                   str(int(m * 10))))
                            else:
                                model_sgd(setting=setting, descent_way=descent_way, attack=attack, w_best=w_best,
                                          eta=eta, model_name=model_name, m=m, groot=groot)
