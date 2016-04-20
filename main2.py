import sys
import glob
import os
from itertools import chain

import dlib
import cv2
from skimage import io

sys.path.insert(0, "python-neural-network/backprop")

from core.inputreader import MugshotExtractor, DataTangExtractor
from core.classifier import ScikitNeuralNetClassifier, ScikitSvmClassifier
from core.featureconverter import FaceLandmarkFeatureConverter
from utils import max_index


class AgeBucket(object):
    def __init__(self, *max_ages):
        res = [0] + sum(map(list, zip(max_ages, [x+1 for x in max_ages])), [])[:-1]
        self.buckets = zip(res[::2], res[1::2])
        self.age_bucket = lambda y: max_index([int(x[0]<=y<=x[1]) 
            for x in self.buckets])

    def __call__(self, age):
        return self.age_bucket(age)

    def __len__(self):
        return len(self.buckets)

    def __repr__(self):
        return repr(self.buckets)
        

def convert_wrapper(datalist, converter, bucket=None):
    result = []
    for meta in datalist:
        res = converter.convert_data(meta)
        if res is not None:
            result.append(res if not bucket else (res[0], bucket(res[1])))
    return result

def train_test1(train, test):
    converter = FaceLandmarkFeatureConverter()
    
    age_bucket = AgeBucket(15, 100)
    nn = ScikitNeuralNetClassifier([converter.n_features, 50, len(age_bucket)])
    #nn = ScikitSvmClassifier()

    train_data = convert_wrapper(train.list_data(), converter, bucket=age_bucket)
    nn.train(train_data)

    test_data = convert_wrapper(test.list_data(), converter, bucket=age_bucket)
    res = nn.test(test_data)

    confusion_matrix = [[0]*len(age_bucket) for _ in range(len(age_bucket))]
    for correct, predicted in zip([d[1] for d in test_data], res):
        confusion_matrix[correct][predicted] += 1

    print "Confusion Matrix:"
    print "\n".join(" ".join("%4d"%x for x in row) for row in confusion_matrix)

    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total = len(test_data)
    print "Overall Accuracy:", correct / float(total), 
    print "(%d out of %d)"%(correct, total)
    for correct, predicted in zip([d[1] for d in test_data], res):
        #print correct, predicted
        pass



def main():
    train = DataTangExtractor("data/datatang/train")
    test = DataTangExtractor("data/datatang/test")
    #train = MugshotExtractor("data/Mugshots/train")
    #test = MugshotExtractor("data/Mugshots/test")
    train_test1(train, test)
    


if __name__ == '__main__':
    main()

