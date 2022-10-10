import numpy as np
from FatherModel import Softmax


class FABAWorkerSoftmax(Softmax):
    def __init__(self, w, id, select, config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class FABAServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def train(self):
        """
        getting FABA aggregation results
        """
        remain = self.messages
        for _ in range(self.config['byzantineSize']):
            meanr = np.mean(remain, axis=0)
            # remove the largest 'byzantine size' model
            distances = np.array([np.linalg.norm(model - meanr) for model in remain])
            remove_index = distances.argmax()
            remain = np.delete(remain, remove_index, axis=0)

        agg_results = np.mean(remain, axis=0)
        return agg_results


