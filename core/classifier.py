import numpy as np

from activation_functions import sigmoid_function, tanh_function, linear_function
from cost_functions import sum_squared_error, exponential_cost
from learning_algorithms import backpropagation, resilient_backpropagation
from neuralnet import NeuralNet
from tools import Instance

import config

class NeuralNetClassifier(object):
    def __init__(self, nodes):
        settings = {
            "cost_function": {
                "sum_squared_error": sum_squared_error,
                "exponential_cost": exponential_cost,
            }[config.neuralnet_cost_function],
            "n_inputs": nodes[0],
            "layers": [(x, sigmoid_function) for x in nodes],
        }
        self.neuralnet = NeuralNet(settings)
        self.save_path = config.neuralnet_save_path
        self.train_fn = {
            "backpropagation": backpropagation,
            "resilient_backpropagation": resilient_backpropagation
        }[config.neuralnet_train_fn]
        self.max_iterations = config.neuralnet_maxiterations
        self.learning_rate = config.neuralnet_learningrate

    def train(self, data):
        inp = [Instance(d[0], d[1]) for d in data]

        backpropagation(
                self.neuralnet,
                inp,
                max_iterations=self.max_iterations,
                learning_rate=self.learning_rate,
                momentum_factor=0.9)
        self.neuralnet.save_to_file(self.save_path)

    def test(self, data):
        #Reference: https://github.com/jorgenkg/python-neural-network/blob/master/backprop/neuralnet.py

        inp = [Instance(d[0], d[1]) for d in data]
        test_data = np.array([obj.features for obj in inp])
        test_targets = np.array([obj.targets for obj in inp])

        input_signals, derivates = self.neuralnet.update(test_data, trace=True)
        out = input_signals[-1]
        error = self.neuralnet.cost_function(out, test_targets)
        
        n_output = self.neuralnet.layers[-1][0]
        confusion_matrix = [[0]*n_output for _ in range(n_output)]
        max_index = lambda x: max(enumerate(x), key=lambda y: y[1])[0]
        for result, target in zip(out, test_targets):
            confusion_matrix[max_index(target)][max_index(result)] += 1
        print "Confusion Matrix:"
        print "\n".join(" ".join("%4d"%x for x in row) for row in confusion_matrix)

        correct = sum(confusion_matrix[i][i] for i in range(n_output))
        total = len(data)
        print "Overall Accuracy:", correct / float(total), 
        print "(%d out of %d)"%(correct, total)

        with open("nn.out.txt", "w") as f:
            for entry, result, target in zip(test_data, out, test_targets):
                print>>f, "%s\t%s\t%s" % tuple(map(str, [entry, result, target]))


