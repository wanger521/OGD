import numpy as np
from FatherModel import Softmax

class KrumWorkerSoftmax(Softmax):
    def __init__(self, w, id, select, config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class KrumServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def train(self):
        """
        getting Krum aggregation results
        """
        nodeSize = self.config['nodeSize']
        byzantineSize = self.config['byzantineSize']
        score = []

        # calculate score for every worker
        for i in range(nodeSize):
            dist = [np.linalg.norm(other - self.messages[i]) ** 2 for other in self.messages]
            dist = np.sort(dist)
            score.append(np.sum(dist[: nodeSize - byzantineSize - 2]))

        index = score.index(min(score))
        agg_results = self.messages[index]
        return agg_results

