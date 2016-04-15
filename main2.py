import sys
import glob
import os

import dlib
import cv2
from skimage import io

from core.inputreader import MugshotExtractor
from FaceLandmark import FaceLandmarkDetector

def main():
   m = MugshotExtractor("data/Mugshot-temp")
   for data in m.list_data():
       print data


if __name__ == '__main__':
    main()

