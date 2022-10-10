from FatherModel import Softmax, get_accuracy, get_learning, get_vars
from scipy import stats



class TrimMeanWorkerSoftmax(Softmax):
    def __init__(self, w, id, select,config):
        """
        Initialize the solver for regular worker

        :param w: model parameter, shape(10, 784) for mnist
        :param id: id of worker
        :param select: stochastic selected location in training data
        :param config: configuration of the method, type:dictionary
        """
        super().__init__(w, id, select, config)


class TrimMeanServerSoftmax(Softmax):
    def __init__(self, w, config, messages, last_agg):
        self.w = w
        self.config = config
        self.messages = messages
        self.last_agg = last_agg

    def train(self):
        """
        getting trimmed mean aggregation results
         """
        trimmed_range = self.config['byzantineSize']/self.config['nodeSize']
        agg_results = stats.trim_mean(self.messages, trimmed_range, axis=0)
        return agg_results

