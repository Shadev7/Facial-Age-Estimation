import sys
import glob
import os
from itertools import chain

import dlib
import cv2
from skimage import io

from core.inputreader import MugshotExtractor, DataTangExtractor
from core.classifier import *
from core.featureconverter import *
from core.model import AgeBucket
import config
from utils import Tee


def train_test(train, test, classifier):
    classifier.train(train.list_data())
    res = classifier.test(test.list_data())
    
    confusion_matrix = [[0]*classifier.converter.n_output 
            for _ in range(classifier.converter.n_output)]

    for index, (predicted, _, correct) in enumerate(zip(*res)):
    	if predicted != correct:
    		# print "incorrect detection:", test.list_data()[index]
    		pass
        confusion_matrix[correct][predicted] += 1

    print "Confusion Matrix:"
    print "\n".join(" ".join("%4d"%x for x in row) for row in confusion_matrix)

    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total = len(res[0])
    print "Overall Accuracy:", correct / float(total) *100, "%", 
    print "(%d out of %d)"%(correct, total)
    return correct, total, correct/float(total)

def run(cls, classifier_param):
    train = DataTangExtractor("data/datatang/train")
    test = DataTangExtractor("data/datatang/test")

    repeat = 2
    total, totalAccuracy = 0, 0
    for i in range(repeat):
        classifier = cls(*classifier_param)

        print "Trial #", i + 1

        correct, len_td, accuracy = train_test(train, test, classifier)
        total += correct
        totalAccuracy += accuracy
    return (total/float(repeat), totalAccuracy/float(repeat), 
                totalAccuracy/float(repeat) * classifier.converter.n_output)

def main():
    stdout = sys.stdout
    f = open('stdout.txt', 'w')
    sys.stdout = Tee(sys.stdout, f)
    classifiers = [
        (ScikitNeuralNetClassifier, [[0, 10, 0]]),
        (ScikitNeuralNetClassifier, [[0, 15, 0]]),
        (ScikitNeuralNetClassifier, [[0, 30, 0]]),
        (ScikitNeuralNetClassifier, [[0, 50, 0]]),
        (ScikitNeuralNetClassifier, [[0, 100,0]]),
        (ScikitNaiveBayesClassifier, []),
        (NearestNeighborsClassifier, [5]),
        (NearestNeighborsClassifier, [10]),
        (NearestNeighborsClassifier, [15]),
    ]
    features = [
        FaceLandmarkFeatureConverter,
        FaceBoundaryFeatureConverter,
        FaceLandmarkBoundaryFeatureConverter,
    ]
    age_groups = [
        (12, 23, 100),
        (7, 15 ,22, 100),
        (5, 12, 18, 100),
        (5, 10, 15, 20, 25,  100),
    ]

    for cls, param in classifiers:
        for ftr in features:
            for ag in age_groups:
                print "Classifier:", cls
                print "Classifier param:", param
                print "Feature:", ftr
                print "Age Group:", ag
                res = run(cls, [ftr(AgeBucket(*ag))] + param)
                print "Avg. Correct:", res[0]
                print "Avg. Accuracy:", res[1]*100, "%"
                print "BTRG:", res[2], "times"
                sys.stdout.flush()
    f.close()
if __name__ == '__main__':
    main()
