import numpy as np
from FatherModel import Softmax


class MutiKrumWorkerSoftmax(Softmax):
    def __init__(self, w, id, select, config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class MutiKrumServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def train(self):
        """
        getting multi Krum aggregation results
        """
        nodeSize = self.config['nodeSize']
        byzantineSize = self.config['byzantineSize']
        multi = self.config['M']
        index = []
        # calculate score for every worker
        left_node = list(np.arange(nodeSize))
        for j in range(multi):
            score = []
            for i in range(nodeSize):
                dist = [np.linalg.norm(self.messages[other] - self.messages[i]) ** 2 for other in left_node]
                dist = np.sort(dist)
                score.append(np.sum(dist[: nodeSize - byzantineSize - 2 - j]))
            for m in index:
                score[m] = max(score)
            index.append(score.index(min(score)))
            left_node.remove(int(index[-1]))

        agg_results = 0
        for s in index:
            agg_results += 1 / multi * (self.messages[s])
        return agg_results
