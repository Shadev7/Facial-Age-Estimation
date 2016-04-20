import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm

import config
from utils import max_index

class ScikitClassifier(object):
    def train(self, data):
        self.classifier.fit(
                map(self.get_x, (d[0] for d in data)),
                map(self.get_y, (d[1] for d in data)))
        with open(config.neuralnet_save_path, "w") as f:
            pickle.dump(self.classifier, f)
    
    def get_x(self, x):
        return x

    def get_y(self, y):
        return y

    def test(self, data):
        return self.classifier.predict(map(self.get_x, (d[0] for d in data)))

class ScikitNeuralNetClassifier(ScikitClassifier):
    def __init__(self, nodes):
        self.classifier = MLPClassifier(
                hidden_layer_sizes=nodes[1:-1],
                activation='relu',
                tol=1e-100,
                learning_rate='adaptive',
                max_iter=config.neuralnet_maxiterations,
                learning_rate_init=config.neuralnet_learningrate,
                verbose=True)

class ScikitSvmClassifier(ScikitClassifier):
    def __init__(self):
        self.classifier = svm.SVC(
                #decision_function_shape='ovo',
                verbose=True)
    
    def get_x(self, d):
        return [int(x*10)/10.0 for x in d]

