import math
import math

import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

from numpy import ndarray

from LoadMnist import getData, data_redistribute
import Config_artificial as Config
import Config0_artificial as Config0
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


def model_sgd(setting, descent_way, attack, eta, model_name, m, groot):
    # print Byzantine and honest node
    if attack == None:
        byzantine = Config0.byzantine
        regular = Config0.regular
    else:
        byzantine = Config.byzantine
        regular = Config.regular
    print(byzantine)
    print(regular)

    # create artificial data
    if setting == 'iid':
        image, label, image_test, label_test, w_best, w_ad_best = create_data()
    else:
        image, label, w_best, w_ad_best = create_noniid_data()

    # Load the configurations
    WorkerSoftmax, ServerSoftmax, conf = personilized(model_name, attack)
    data_config = Config.artificialConfig
    num_data = int(data_config['trainNum'] / conf['nodeSize'])
    data_dimension = data_config['dimension']
    nodesize = conf['nodeSize']
    last_str = ''

    epoch = 0
    max_epoch = conf['epoch']

    final_loss_ls = []
    final_sregret_ls = 0
    final_regret_ls = []
    last_epoch_regret = 0

    logger.info('Start！')
    while epoch < max_epoch:
        epoch += 1
        print('epoch: ', epoch)
        # Parameter initialization
        serverPara = np.random.normal(loc=0, scale=1, size=data_dimension)  # np.zeros((data_dimension,))
        workerGrad = np.random.normal(loc=0, scale=1,
                                      size=(nodesize, data_dimension))  # np.zeros((nodesize, data_dimension))
        momentum = workerGrad  # np.zeros((nodesize, data_dimension))
        last_agg = np.zeros((data_dimension,))
        k = 0
        max_iterations = conf['iterations']
        loss_train = []
        loss_train_ls = []
        loss_train_t = np.array(0, dtype=float)
        regret_train = []
        regret_train_ls = []
        regret_train_t = np.array(0, dtype=float)
        sregret_train = []
        sregret_train_ls = []
        sregret_train_t = np.array(0, dtype=float)
        while k < max_iterations:
            k += 1
            regret_train = []
            loss_train = []
            sregret_train = []
            # workers compute gradients
            for id in range(nodesize):
                # select = np.random.randint(0, num_data - conf['batchSize'])
                select = k-1
                worker = WorkerSoftmax(serverPara, id, select, conf)
                image_input = image[id * num_data: (id + 1) * num_data]
                label_input = label[id * num_data: (id + 1) * num_data]
                batch = conf['batchSize']
                X = np.array(image_input[select: select + batch])
                Y = np.array(label_input[select: select + batch])
                Z = 0
                # Z = np.random.normal(loc=0, scale=1, size=len(Y))  # 模拟环境加入噪声
                #Z = np.random.laplace(loc=0.0, scale=float(1.2), size=len(Y))
                workerGrad[id] = worker.artificial_gradient(X, Y, Z)
                if id in regular:
                    one_loss = worker.artificial_loss(X, Y, Z, serverPara)
                    loss_train.append(one_loss)
                    regret_train.append(worker.artificial_adversarial_regret(X, Y, Z, w_ad_best))
                    sregret_train.append(worker.artificial_stochastic_regret(w_best))
            regret_train_t = regret_train_t + np.mean(np.array(regret_train))
            sregret_train_t = sregret_train_t + np.mean(np.array(sregret_train))
            loss_train_t = np.mean(loss_train)


            # workers compute momentum
            true_m = 0
            if m <= 1:  # constant momentum size
                true_m = m
            elif 1 < m <= 4:  # diminishing momentum size
                if k < 500:
                    true_m = 0.001*m*2
                else:
                    true_m = m / k
            else:  # 1/sqrtT momentum size
                true_m = 0.0001 * m  # sqrtT momentum
            # according true_m (momentum size) to update momentum for each worker
            momentum = (true_m * workerGrad + (1 - true_m) * momentum)

            # transcoding to true step size based on the given value of eta
            true_eta = 0
            if eta <= 1:  # constant step size, which is set in Config for different aggregation rulse
                true_eta = eta  # conf['learningStep']
            elif 1 < eta <= 4:  # diminishing step size
                if k < 500:
                    true_eta = 0.001*eta*2
                else:
                    true_eta = eta / k
            else:  # 1/sqrtT step size
                true_eta = 0.0001 * eta

            # train model, if we aggregate model or gradient, using different train way
            if descent_way == 'model':  # if we aggregate model
                messages = serverPara - true_eta * momentum  # each worker make gradient descent first
                # Byzantine make attack
                if attack is not None:
                    messages, last_str = attack(messages, regular, byzantine)
                # Master node aggregate messages, hence update the global model
                last_agg = serverPara  # update last iteration aggregation results
                server = ServerSoftmax(serverPara, conf, messages, last_agg)
                agg_results = server.train()
                serverPara = agg_results
            elif descent_way == 'gradient':  # if we aggregate gradient
                messages = momentum
                if attack is not None:
                    messages, last_str = attack(messages)
                server = ServerSoftmax(serverPara, conf, messages, last_agg)
                agg_results = server.train()
                last_agg = agg_results
                serverPara = serverPara - true_eta * agg_results

            # calculate loss and regret, and print

            regret_train_ls.append(regret_train_t)
            sregret_train_ls.append(sregret_train_t)
            loss_train_ls.append(loss_train_t)

            if k % 500 == 0:
                logger.info('the {}th iteration loss: {}, regret: {}, sregret: {}'.format(k, loss_train_t, regret_train_t, sregret_train_t))
                #print(regret_train)

        if regret_train_ls[-1] > last_epoch_regret:
            last_epoch_regret = regret_train_ls[-1]
            final_loss_ls = loss_train_ls
            final_regret_ls = regret_train_ls
        final_sregret_ls = final_sregret_ls + np.array(sregret_train_ls)
    final_sregret_ls = final_sregret_ls / epoch
    # Save the experiment results
    output = open('results\\{0}\\{1}\\{2}\\{3}{4}-step{5}-setting-{6}-momentum{7}.pkl'.format(
        groot, descent_way, model_name, model_name, last_str, str(int(eta * 10)),
        setting, str(int(m * 10))), 'wb')
    pickle.dump((final_loss_ls, final_sregret_ls, final_regret_ls), output, protocol=pickle.HIGHEST_PROTOCOL)
    print("Run over {0}{1}-step{2}-setting-{3}-momentum{4}.pkl".format(model_name, last_str,
                                                                       str(int(eta * 10)), setting, str(int(m * 10))))



def create_data():
    file_train_image = 'datasets\\Artificial\\Artificial_train_image.npy'
    file_train_label = 'datasets\\Artificial\\Artificial_train_label.npy'
    file_test_image = 'datasets\\Artificial\\Artificial_test_image.npy'
    file_test_label = 'datasets\\Artificial\\Artificial_test_label.npy'
    file_best_solution = 'datasets\\Artificial\\Artificial_best_solution.npy'
    file_ad_best_solution = 'datasets\\Artificial\\Artificial_ad_best_solution.npy'

    if os.path.exists(file_train_image) and os.path.exists(file_train_label) and os.path.exists(file_test_image) and os. \
            path.exists(file_test_label) and os.path.exists(file_best_solution) and os.path.exists(file_ad_best_solution):
        image_train = np.load(file_train_image)
        label_train = np.load(file_train_label)
        image_test = np.load(file_test_image)
        label_test = np.load(file_test_label)
        best_solution = np.load(file_best_solution)
        w_ad_best = np.load(file_ad_best_solution)
        return image_train, label_train, image_test, label_test, best_solution, w_ad_best
    dimension = Config.artificialConfig['dimension']
    trainNum = Config.artificialConfig['trainNum']
    testNum = Config.artificialConfig['testNum']
    best_solution = np.random.normal(loc=0, scale=1, size=dimension)
    np.save(file_best_solution, best_solution)
    image_train = np.random.normal(loc=0, scale=1, size=(trainNum, dimension))
    # image_train[:, -1] = np.ones(trainNum, )
    label_train = np.dot(best_solution, image_train.T) + np.random.normal(loc=0, scale=0.2, size=trainNum)
    np.save(file_train_image, image_train)
    np.save(file_train_label, label_train)
    image_test = np.random.normal(loc=0, scale=1, size=(testNum, dimension))
    # image_test[:, -1] = np.ones(testNum, )
    label_test = np.dot(best_solution, image_test.T) + np.random.normal(loc=0, scale=0.2, size=testNum)
    np.save(file_test_image, image_test)
    np.save(file_test_label, label_test)
    w_ad_best = np.dot(np.dot(np.linalg.pinv(np.dot(image_train.T, image_train)), image_train.T), label_train)
    np.save(file_ad_best_solution, w_ad_best)
    return image_train, label_train, image_test, label_test, best_solution, w_ad_best

def create_noniid_data():
    file_train_image = 'datasets\\Artificial\\Artificial_train_image_noniid.npy'
    file_train_label = 'datasets\\Artificial\\Artificial_train_label_noniid.npy'
    file_best_solution = 'datasets\\Artificial\\Artificial_best_solution_noniid.npy'
    file_ad_best_solution = 'datasets\\Artificial\\Artificial_ad_best_solution_noniid.npy'

    if os.path.exists(file_train_image) and os.path.exists(file_train_label) and os.path.exists(file_best_solution) and os.path.exists(file_ad_best_solution):
        image_train = np.load(file_train_image)
        label_train = np.load(file_train_label)
        best_solution = np.load(file_best_solution)
        w_ad_best = np.load(file_ad_best_solution)
        return image_train, label_train, best_solution, w_ad_best
    dimension = Config.artificialConfig['dimension']
    trainNum = Config.artificialConfig['trainNum']
    testNum = Config.artificialConfig['testNum']
    best_solution_1 = np.random.normal(loc=0, scale=1, size=dimension)
    np.save(file_best_solution, best_solution_1)
    non_degree = int(Config.optConfig['nodeSize']/3)
    one_degree = int(trainNum/non_degree)
    image_train = np.random.normal(loc=0, scale=1, size=(trainNum, dimension))
    best = 0
    label_train = np.dot(best_solution_1, image_train.T) #+ np.random.normal(loc=0, scale=0.1, size=trainNum)
    for i in range(non_degree):
        best_solution = best_solution_1 + np.random.normal(loc=i*0.2, scale=0.5, size=dimension)
        best += best_solution
        image_train[i*one_degree:(i+1)*one_degree] = np.random.normal(loc=0+i, scale=1, size=(one_degree, dimension))
        label_train[i*one_degree:(i+1)*one_degree] = np.dot(best_solution, image_train[i*one_degree:(i+1)*one_degree].T) \
                                                     + np.random.normal(loc=0, scale=0.2, size=one_degree)
    # image_train[:, -1] = np.ones(trainNum, )
    np.save(file_train_image, image_train)
    np.save(file_train_label, label_train)

    best_solution = best/3
    w_ad_best = np.dot(np.dot(np.linalg.pinv(np.dot(image_train.T, image_train)), image_train.T), label_train)
    np.save(file_ad_best_solution, w_ad_best)
    return image_train, label_train, best_solution, w_ad_best


if __name__ == '__main__':
    groot = "artificial6" # artificial5
    descent_way_ls = ['model']  # ['model', 'gradient']
    eta_ls = [0.005,0.1,4,10]#[10,4]#[0.008, 4, 5]# [0.001, 0.1, 4]
    setting_ls = ['iid', 'noniid']
    attack_ls = [sign_flipping_attacks, gaussian_attacks, sample_duplicating_attacks, None ]#, gaussian_attacks, sample_duplicating_attacks, None
    last_str = ['-sf', '-gs', '-hd', '']
    momentum_ls = [1,0.005,0.1,4,10]#[10,1,4]#[1, 0.008,  4, 5]#[0.001, 0.1, 4]
    model_name_ls = ['Mean', 'CenterClip',  'CMedian',  'trimmed-mean', 'GeoMed', 'Krum', 'Phocas', 'FABA']
    # ['Mean', 'CMedian', 'trimmed-mean', 'GeoMed',  'Krum', 'CenterClip', 'Phocas', 'FABA']
    # [ 'MutiKrum', 'Zeno', 'Ours', '0-Ours']
    # w_best, eta_best = find_w_best(descent_way='model', groot=groot)


    #  run
    for descent_way in descent_way_ls:
        for setting in setting_ls:
            for model_name in model_name_ls:
                for attack, last in zip(attack_ls, last_str):
                    for eta in eta_ls:
                        for m in [1, eta]:
                            paths = 'results\\{0}\\{1}\\{2}\\{3}{4}-step{5}-setting-{6}-momentum{7}.pkl'.format(
                                groot, descent_way, model_name, model_name, last, str(int(eta * 10)), setting, str(int(m * 10)))
                            if os.path.exists(paths):
                                print("Run over {0}{1}-step{2}-setting-{3}-momentum{4}.pkl".format(model_name, last,
                                                                                                   str(int(eta * 10)),
                                                                                                   setting,
                                                                                                   str(int(m * 10))))
                            else:
                                model_sgd(setting=setting, descent_way=descent_way, attack=attack,
                                          eta=eta, model_name=model_name, m=m, groot=groot)
