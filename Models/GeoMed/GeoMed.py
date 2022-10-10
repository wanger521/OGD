import numpy as np
from FatherModel import Softmax


class GeoMedWorkerSoftmax(Softmax):
    def __init__(self, w, id, select, config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class GeoMedServerSoftmax(Softmax):

    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def train(self):
        """
        getting geometric median aggregation results
        """
        # Compute the geometric median of the gradient set using Weiszfeld method
        guess = np.mean(self.messages, axis=0)
        for _ in range(self.config['GeoMedMaxIter']):
            res1 = np.zeros_like(guess)
            res2 = 0
            for i in range(self.messages.shape[0]):
                dist = np.linalg.norm(self.messages[i] - guess, 2)
                res1 += self.messages[i] / dist
                res2 += 1 / dist
            guess_next = res1 / res2
            guess_move = np.linalg.norm(guess - guess_next, 2)
            guess = guess_next

            if guess_move <= self.config['GeoMedTol']:
                break

        agg_results = guess
        return agg_results
