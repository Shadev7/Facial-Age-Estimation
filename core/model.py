from enum import Enum
import sys

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

