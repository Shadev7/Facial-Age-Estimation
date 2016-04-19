import sys
import glob
import os

import dlib
import cv2
from skimage import io

sys.path.insert(0, "python-neural-network/backprop")

from core.inputreader import MugshotExtractor, DataTangExtractor
from core.classifier import NeuralNetClassifier
from core.featureconverter import FaceLandmarkFeatureConverter
from utils import max_index

def convert_wrapper(datalist, converter):
    result = []
    for meta in datalist:
        res = converter.convert_data(meta)
        if res is not None:
            result.append(res)

    return result

def train_test1(train, test):
    converter = FaceLandmarkFeatureConverter()

    nodes = [ 1, 50, 1]
    nn = NeuralNetClassifier(nodes)

    train_data = convert_wrapper(train.list_data(), converter)
    nn.train(train_data)

    test_data = convert_wrapper(test.list_data(), converter)
    res = nn.test(test_data)

    bucket = lambda y: max_index([int(x[0]<=y<=x[1]) for x in [(0,10), (11, 30), (31,100)]])

    confusion_matrix = [[0]*3 for _ in range(3)]
    for correct, predicted in zip([d[1][0] for d in test_data], res):
        confusion_matrix[bucket(correct)][bucket(predicted)] += 1

    print "Confusion Matrix:"
    print "\n".join(" ".join("%4d"%x for x in row) for row in confusion_matrix)

    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total = len(test_data)
    print "Overall Accuracy:", correct / float(total), 
    print "(%d out of %d)"%(correct, total)
    for correct, predicted in zip([d[1][0] for d in test_data], res):
        print correct, predicted



def main():
    #e = MugshotExtractor("data/Mugshot-temp")
    train = DataTangExtractor("data/datatang/train")
    test = DataTangExtractor("data/datatang/test")
    train_test1(train, test)
    


if __name__ == '__main__':
    main()

