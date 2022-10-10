import numpy as np
from FatherModel import Softmax


class CenterClipWorkerSoftmax(Softmax):
    def __init__(self, w, id, select, config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class CenterClipServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def clip(self):
        nodesize = self.config['nodeSize']
        tau = self.config['tau']
        iner_iter = self.config['iner_iter']
        inil = self.last_agg
        for iiter in range(iner_iter):
            messages_norm = np.array([np.linalg.norm(self.messages[i] - inil, 2) for i in range(nodesize)])
            messages_min = np.minimum(1, tau/messages_norm)
            messages_scale = np.mean(np.array([(self.messages[i] - inil)*messages_min[i] for i in range(nodesize)]), axis=0)
            inil = inil + messages_scale
        return inil

    def train(self):
        """
        getting centered clipping aggregation results
        """
        agg_results = self.clip()
        return agg_results


