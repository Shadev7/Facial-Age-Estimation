from enum import Enum
import sys

from utils import max_index

Gender = Enum("Gender", "M F")

class FaceMetadata(object):
    """Class to store file name and various parameters about the face image
    such as age, gender etc..
    """
    def __init__(self, path, **kwargs):
        self.path = path
        try:
            self.age = int(kwargs.pop('Age'))
            self.gender = getattr(Gender, kwargs.pop('Gender'))
            self.position = kwargs.pop('Position', None)
            self.others = kwargs
        except Exception, e:
            print>>sys.stderr, "Unable to use:", path, e

    def __repr__(self):
        return "Face [%s] (%02d, %s)"%(self.path, self.age, str(self.gender))

class FacialFeatures(object):
    def __init__(self, **kwargs):
        self.ratios = kwargs.pop('ratios', None)
        self.feature_points = kwargs.pop('feature_points', None)
        self.face_boundary = kwargs.pop('face_boundary', None)
        self.others = kwargs

    
    def __repr__(self):
        return "%s: %s"%(self.__class__.__name__, self.ratios)

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
        


