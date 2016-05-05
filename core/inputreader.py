import re
import os.path
from glob import glob
from itertools import chain
import random

from core.model import FaceMetadata
from core.model import Gender

class GenericFaceDataExtractor(object):
    def __init__(self, dirName):
        self.dirName = dirName

    def list_data(self):
        res = []
        for pattern in self.glob_patterns():
            for path in glob(os.path.join(self.dirName, pattern)):
                res.append(self.extract_metadata(path))
        res = self.subsample(res, percent = 100)
        return res

    def extract_metadata(self, path):
        return FaceMetadata(path, 30, Gender.MALE)

    def subsample(self, res, percent = 100):
        res_sub = []
        randindices = random.sample(range(1, len(res)), len(res)*percent/100)
        for i in randindices:
            res_sub.append(res[i])
        return res_sub

class MugshotExtractor(GenericFaceDataExtractor):
    def glob_patterns(self):
        return ["*.png"]

    def extract_metadata(self, path):
        with open(path[:-3] + "txt") as f:
            contents = [map(lambda x: x.strip(), l.strip().split(":")) for l in f]
            return FaceMetadata(path, **dict(contents))

class DataTangExtractor(GenericFaceDataExtractor):
    def glob_patterns(self):
        return ["*.JPG"]

    def extract_metadata(self, path):
        filename = os.path.basename(path)
        pic_id, age = re.findall(r'(\d{3})A(\d{2})', filename, re.IGNORECASE)[0]
        return FaceMetadata(path, Age=int(age, 10), Gender="M")
