import sys
import math
import os.path
import json

from skimage import io

from core.FaceLandmark import FaceLandmarkDetector
import config

class GenericFeatureConverter(object):
    n_features = 0

    def convert_data(self, obj):
        return None

class CacheMixin(object):
    def with_cache(self, fn, path, args):
        if os.path.isfile(path + ".cached"):
            with open(path + ".cached") as f:
                return json.load(f)
        result = fn(*args)
        with open(path + ".cached", "w") as f:
            json.dump(result, f)
        return result


class FaceLandmarkFeatureConverter(GenericFeatureConverter, CacheMixin):
    fl = FaceLandmarkDetector(config.facelandmarkdetector_path)

    params = ("facial_ind mandibular_ind intercanthal_ind orbital_width_ind " + 
              #"eye_fissure_ind"
              "nasal_ind vermillion_height_ind " +
              "mouth_face_width_ind").split()
    n_features = len(params)
    def convert_data(self, meta):
        try:
            facial_feature = self.with_cache(
                    lambda x: list(self.fl.detect(io.imread(x)))[0].ratios,
                    meta.path,
                    [meta.path])
            res = ([facial_feature[x] for x in self.params], meta.age)
            print "Converted:", meta.path,
            print "%d features => %s"%(len(res[0]), str(meta.age))
            return res
        except Exception, e:
            print e
            print>>sys.stderr, "Unable to use: " + meta.path
            return None

