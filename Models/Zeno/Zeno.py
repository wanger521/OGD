import numpy as np
from FatherModel import Softmax
import heapq
from LoadMnist import getData


class ZenoWorkerSoftmax(Softmax):
    def __init__(self, w, id, select, config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


def get_test_data():
    image_test, label_test = getData('..\\datasets\\MNIST\\t10k-images.idx3-ubyte',
                                     '..\\datasets\\MNIST\\t10k-labels.idx1-ubyte')
    return image_test, label_test


class ZenoServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def cal_loss_score(self, image, label, w_i, select_zeno):
        """
        Compute loss of softmax regression

        :param image: image, shape(784) for mnist
        :param label: label, scalar
        """
        batchsize = self.config['zeno_batch']
        X = np.array(image[select_zeno: select_zeno + batchsize])
        Y = np.array(label[select_zeno: select_zeno + batchsize])
        Y = self.one_hot(Y)
        num_data = X.shape[0]
        t1 = np.dot(w_i, X.T)
        t1 = t1 - np.max(t1, axis=0)
        t = np.exp(t1)
        tmp = t / np.sum(t, axis=0)
        delta = 1e-7  # Avoid log(0) happen
        loss = -np.sum(Y.T * np.log(tmp+delta)) / num_data + self.config['decayWeight'] * np.sum(w_i ** 2) / 2
        return loss

    def get_score(self, image, label, gamma, id, select_zeno):
        loss_1 = self.cal_loss_score(image, label, self.w, select_zeno)
        mu = self.messages[id]
        new_estimator = self.w - gamma * mu
        loss_2 = self.cal_loss_score(image, label, new_estimator, select_zeno)
        rho = gamma / self.config['rho_ratio']
        third = rho * np.square(np.linalg.norm(mu))
        score = loss_1 - loss_2 - third
        return score

    def train(self):
        """
        getting zeno aggregation results
        """
        batchsize = self.config['zeno_batch']
        score_ls = np.zeros(self.config['nodeSize'])
        image_test, label_test = get_test_data()
        select_zeno = np.random.randint(0, len(label_test) - batchsize)
        gamma = self.config['gamma']   # gamma = eta
        for id in range(self.config['nodeSize']):
            score_ls[id] = self.get_score(image_test, label_test, gamma, id, select_zeno)
        score_ls = np.array(score_ls)
        remain = self.messages
        remove_index = heapq.nsmallest(self.config['remove_size'], range(len(score_ls)), score_ls.__getitem__)
        remain = np.delete(remain, remove_index, axis=0)
        agg_results = np.mean(remain, axis=0)
        return agg_results
