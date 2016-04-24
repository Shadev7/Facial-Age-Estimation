import sys
import glob
import os
from itertools import chain

import dlib
import cv2
from skimage import io

from core.inputreader import MugshotExtractor, DataTangExtractor
from core.classifier import ScikitNeuralNetClassifier, ScikitSvmClassifier
from core.classifier import CombinedClassifier
from core.featureconverter import FaceLandmarkFeatureConverter, \
            FaceBoundaryFeatureConverter, FaceLandmarkBoundaryFeatureConverter
from core.model import AgeBucket
import config

def train_test(train, test, classifier):
    classifier.train(train.list_data())
    res = classifier.test(test.list_data())

    confusion_matrix = [[0]*classifier.converter.n_output 
            for _ in range(classifier.converter.n_output)]

    for predicted, _, correct in zip(*res):
        confusion_matrix[correct][predicted] += 1

    print "Confusion Matrix:"
    print "\n".join(" ".join("%4d"%x for x in row) for row in confusion_matrix)

    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total = len(list(test.list_data()))
    print "Overall Accuracy:", correct / float(total), 
    print "(%d out of %d)"%(correct, total)
    return correct, total, correct/float(total)

def main():
    train = DataTangExtractor("data/datatang/train")
    test = DataTangExtractor("data/datatang/test")

    repeat = 5
    total, totalAccuracy = 0, 0
    
    for _ in range(repeat):
        ab = AgeBucket(7, 15, 22, 100)
        fl = FaceLandmarkFeatureConverter(ab)
        fb = FaceBoundaryFeatureConverter(ab)
        combined_params = [
            (ScikitNeuralNetClassifier, [fl, [fl.n_features, 15, fl.n_output]]),
            (ScikitNeuralNetClassifier, [fb, [fb.n_features, 15, fb.n_output]]),
        ]
        #converter = FaceBoundaryFeatureConverter(ab)
        #classifier = ScikitNeuralNetClassifier(converter, 
        #            [converter.n_features, 15, converter.n_output])
        classifier = CombinedClassifier(*combined_params)
        print "Trial #", _ + 1
        correct, len_td, accuracy = train_test(train, test, classifier)
        total += correct
        totalAccuracy += accuracy
    print "Average Correct:", total / float(repeat)
    print "Average Accuracy:", totalAccuracy / float(repeat)
    print "Better than random:", (totalAccuracy / float(repeat)) * len(ab)

if __name__ == '__main__':
    main()

