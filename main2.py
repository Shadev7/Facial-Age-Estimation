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

def convert_wrapper(datalist, converter):
    result = []
    for meta in datalist:
        res = converter.convert_data(meta)
        if res is not None:
            result.append(res)

    return result

def train_test1(train, test):
    converter = FaceLandmarkFeatureConverter()

    nodes = [ converter.n_inputs, 50, 50, converter.n_outputs]
    nn = NeuralNetClassifier(nodes)

    nn.train(convert_wrapper(train.list_data(), converter))
    nn.test(convert_wrapper(test.list_data(), converter))

def main():
    #e = MugshotExtractor("data/Mugshot-temp")
    train = DataTangExtractor("data/datatang/train")
    test = DataTangExtractor("data/datatang/test")
    train_test1(train, test)
    


if __name__ == '__main__':
    main()

