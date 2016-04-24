import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm

import config
from utils import max_index
from core.featureconverter import GenericFeatureConverter

class GenericClassifier(object):
    def __init__(self, converter):
        self.converter = converter

    def train(self, data):
        pass

    def test(self, data):
        pass

    def get_x(self, data):
        return data[0]

    def get_y(self, data):
        return data[1]

    def preprocess(self, data):
        result = []
        for meta in data:
            res = self.converter.convert_data(meta)
            result.append(res)
            if res is not None and config.verbose:
                print "Processed:", meta, "with %d features"%len(result[-1][0])
        return [d for d in result if d is not None]

class ScikitClassifier(GenericClassifier):
    def __init__(self, converter):
        super(ScikitClassifier, self).__init__(converter)

    def train(self, data):
        data = self.preprocess(data)
        self.classifier.fit(
                map(self.get_x, data),
                map(self.get_y, data))
        with open(config.neuralnet_save_path, "w") as f:
            pickle.dump(self.classifier, f)
    
    def test(self, data):
        data = self.preprocess(data)
        x = map(self.get_x, data)
        y = map(self.get_y, data)
        return self.classifier.predict(x), self.classifier.predict_proba(x), y

class ScikitNeuralNetClassifier(ScikitClassifier):
    def __init__(self, converter, nodes):
        super(ScikitNeuralNetClassifier, self).__init__(converter)
        self.classifier = MLPClassifier(
                hidden_layer_sizes=nodes[1:-1],
                activation='relu',
                tol=1e-100,
                learning_rate='adaptive',
                max_iter=config.neuralnet_maxiterations,
                learning_rate_init=config.neuralnet_learningrate,
                verbose=config.verbose)

class CombinedClassifier(GenericClassifier):
    def __init__(self, *tuples):
        super(CombinedClassifier, self).__init__(None)
        for arg in tuples[0][1]:
            if isinstance(arg, GenericFeatureConverter):
                self.converter = arg
        self.classifiers = [cls(*args) for cls, args in tuples]
    
    def train(self, data):
        data = list(data)
        for c in self.classifiers:
            c.train(data)
    
    def test(self, data):
        data = list(data)
        res = []
        y = []
        for c in self.classifiers:
            predicted, probs, actual = c.test(data)
            res.append(probs)

        probs = [sum(map(list,x),[]) for x in zip(*res)]
        max_prob = [max(x) for x in probs]
        max_prob_index = [max_index(x) % self.classifiers[0].converter.n_output
                for x in probs]

        return max_prob_index, max_prob, actual

class ScikitSvmClassifier(ScikitClassifier):
    def __init__(self):
        self.classifier = svm.SVC(
                #decision_function_shape='ovo',
                verbose=True)
    
    def get_x(self, d):
        return [int(x*10)/10.0 for x in d[0]]

