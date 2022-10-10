import numpy as np
from scipy import stats
from FatherModel import Softmax
import heapq


class PhocasWorkerSoftmax(Softmax):
    def __init__(self, w, id, select,config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class PhocasServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def trimmed_mean(self):
        trimmed_range = self.config['byzantineSize']/self.config['nodeSize']
        guess = stats.trim_mean(self.messages, trimmed_range, axis=0) # FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.return np.mean(atmp[sl], axis=axis)
        return guess

    def train(self):
        """
        Update the global model
        """
        trim = self.trimmed_mean()
        remain =self.messages
        distances = np.array([np.linalg.norm(model - trim) for model in remain])
        remove_index = heapq.nlargest(self.config['byzantineSize'], range(len(distances)), distances.__getitem__)
        remain = np.delete(remain, remove_index, axis=0)
        agg_results = np.mean(remain, axis=0)
        return agg_results



