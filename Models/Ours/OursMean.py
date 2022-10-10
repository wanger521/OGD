import numpy as np
from FatherModel import Softmax


class OursMeanWorkerSoftmax(Softmax):
    def __init__(self, w, id, select, config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class OursMeanServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def clip(self, remain):
        nodesize = len(remain)
        tau = self.config['tau']
        iner_iter = self.config['iner_iter']
        inil = self.last_agg
        for iiter in range(iner_iter):
            messages_norm = np.array([np.linalg.norm(remain[i] - inil, 2) for i in range(nodesize)])
            messages_min = np.minimum(1, tau / messages_norm)
            messages_scale = np.mean(np.array([(remain[i][i] - inil) * messages_min[i] for i in range(nodesize)]),
                                     axis=0)
            inil = inil + messages_scale
        return inil

    def train(self):
        """
        getting FABA-CC(-mean) aggregation results
        """
        # remain = self.workerGrad+self.add_guass(self.workerGrad)
        remain = self.messages
        for i in range(self.config['byzantineSize']):
            # simple mean
            meanr = np.mean(remain, axis=0)
            # remove the largest 'byzantine size' model
            # cal 欧式距离
            distances = np.array([np.linalg.norm(model - meanr) for model in remain])
            remove_index = distances.argmax()
            dis_median = np.median(distances)
            if distances[remove_index] > self.config['delta'] * dis_median:
                remain = np.delete(remain, remove_index, axis=0)
            else:
                remain = self.w+self.clip(remain)
                break
        agg_results = np.mean(remain, axis=0)
        return agg_results










