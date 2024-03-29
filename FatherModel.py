import numpy as np
import math

class Softmax():

    def __init__(self, w, id , select, config):
        """
        Initialize the solver of softmax regression

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param config: configurations of method, type:dictionary
        """
        self.w = w
        self.id = id
        self.config = config
        self.select = select

    def cal_e(self, x, j):
        """
        Compute e^{w_j x}

        :param x: image, shape(784)
        :param j: row number of parameter w
        """
        theta_j = self.w[j]
        a = np.dot(theta_j, x)
        if a < 700:
            return math.exp(a)
        else:
            return math.exp(700)

    def cal_probability(self, x, j):
        """
        Compute e^{w_j x} / \sum_{i=0}^9 e^{w_i x}

        :param x: image, shape(784)
        :param j: row number of parameter w
        """
        fenzi = self.cal_e(x,j)
        fenmu = sum([self.cal_e(x,i) for i in range(10)])
        return fenzi / fenmu

    def cal_partical_gradient(self, x, y, j):
        """
        Compute the j-th component of gradient

        :param x: image, shape(784)
        :param y: label, scalar
        :param j: row number of parameter w
        """
        first = self.cal_probability(x, j)
        second = int(y == j)
        partical_gradient_j = x * (first - second) + self.config['decayWeight']*self.w[j]
        return partical_gradient_j

    def cal_Sto_gradient(self, image, label):
        """
        Compute stochastic gradient

        :param image: image, shape(784)
        :param label: label, scalar
        """
        x = image[self.select]
        y = label[self.select]
        part_gradient = [self.cal_partical_gradient(x, y, j) for j in range(10)]
        part_gradient = np.array(part_gradient)
        partical_gradient = part_gradient
        return partical_gradient

    def one_hot(self, label):
        """
        Turn the label into the form of one-hot

        :param label: label, scalar
        """
        m = label.shape[0]
        label_onehot = [[1 if j == label[i] else 0 for j in range(10)] for i in range(m)]
        return np.array(label_onehot)

    def cal_batch_sto_grad(self, image, label):
        """
        Compute mini-batch gradient

        :param image: image, shape(784)
        :param label: label, scalar
        :return:
        """
        select = self.select
        batchsize = self.config['batchSize']
        X = np.array(image[select: select + batchsize])
        Y = np.array(label[select: select + batchsize])
        Y = self.one_hot(Y)
        t = np.dot(self.w, X.T)
        t = t - np.max(t, axis=0) #防溢出
        pro = np.exp(t) / np.sum(np.exp(t), axis=0)
        partical_gradient = - np.dot((Y.T - pro), X)/batchsize +self.config['decayWeight']*self.w
        #print(np.linalg.norm(partical_gradient))
        return partical_gradient

    def cal_loss(self, image, label, w):
        """
        Compute loss of softmax regression

        :param image: image, shape(784)
        :param label: label, scalar
        """
        batchsize = self.config['batchSize']
        X = np.array(image[self.select: self.select + batchsize])
        Y = np.array(label[self.select: self.select + batchsize])
        #X = np.array(image[self.select])
        #Y = np.array(label[self.select])
        Y = self.one_hot(Y)
        num_data = X.shape[0]
        t1 = np.dot(w, X.T)
        t1 = t1 - np.max(t1, axis=0)
        t = np.exp(t1)
        tmp = t / np.sum(t, axis=0)
        delta = 1e-7  # Avoid log(0) happen
        loss = -np.sum(Y.T * np.log(tmp+delta)) / num_data + self.config['decayWeight'] * np.sum(w ** 2)/2
        return loss

    def cal_t_regret(self, image, label, w_best):
        regret_front = self.cal_loss(image, label, self.w)
        regret_later = self.cal_loss(image, label, w_best)
        regret_t = regret_front-regret_later
        return regret_t

    '''
    定义回归问题的损失函数和梯度，为了不另外修改其他代码，就放在softmax这个class里了
    '''
    def artificial_gradient(self, X, Y, Z):
        tmp = Y+Z - np.dot(self.w, X.T)
        gradients = np.dot(tmp, -X)
        return gradients

    def artificial_loss(self, X, Y, Z, w):
        tmp = np.square(Y + Z - np.dot(w, X.T))
        loss = np.mean(tmp) / 2
        return loss

    def artificial_adversarial_regret(self, X, Y, Z, w_best):
        loss_t = self.artificial_loss(X, Y, Z, self.w)
        loss_best = self.artificial_loss(X, Y, Z, w_best)
        return loss_t - loss_best

    def artificial_stochastic_regret(self, w_best):
        stoc_regret = np.sum(np.square(w_best - self.w)) / 2
        return stoc_regret


    def get_w(self):
        return self.w

    def get_image(self, image):
        batchsize = self.config['batchSize']
        return np.array(image[self.select: self.select + batchsize])

    def get_label(self, label):
        batchsize = self.config['batchSize']
        return np.array(label[self.select: self.select + batchsize])


def predict(w, test_image, test_label):
    """
    Predict the label of the test_image

    :param w: model parameter, shape(10, 784)
    :param test_image: shape(784)
    :param test_label: scalar
    """
    mat = np.dot(w, test_image.T)
    predict_label = np.argmax(mat)
    # print("label :",test_label , "predict_label:",predict_label)
    return predict_label


def get_accuracy(w, image, label):
    """
    Compute the accuracy of the method

    :param w: model parameter, shape(10, 784)
    :param image: image, shape(784)
    :param label: label, scalar
    """
    number_sample = len(label)
    right = 0
    for i in range(number_sample):
        predict_label = predict(w, image[i], label[i])
        if predict_label == label[i]:
            right += 1
    accuracy = right / number_sample
    # print("the accuracy of training set is :", accuracy)
    return accuracy


def get_vars(regular, W):
    """
    Compute the variation of the regular gradients or regualr model parameters

    :param regular: the set of regular workers
    :param W: the set of regular gradients or regular model parameters
    """

    W_regular = []
    for i in regular:
        W_regular.append(W[i])
    W_regular = np.array(W_regular)

    mean = np.mean(W_regular, axis=0)
    var = 0
    num = W_regular.shape[0]
    for i in range(num):
        var += np.linalg.norm(W_regular[i] - mean) ** 2

    return var / num


def get_learning(alpha, k):
    """
    Compute the decreasing learning step

    :param alpha: coefficient
    :param k: iteration time
    """
    return alpha / math.sqrt(k)


def get_learning_v2(alpha, k):
    return alpha / k
