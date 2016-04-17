import sys
import math
import os.path
import json

from skimage import io

from core.FaceLandmark import FaceLandmarkDetector
import config

class GenericFeatureConverter(object):
    n_inputs = 0
    n_output = 0
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

    age_categories = [(0,10), (11,30), (30, 100)]
    age_categories = [(x, x+4) for x in range(0, 100, 5)]
    age_categories = [(0,10), (11, 100)]

    n_inputs = 7
    n_outputs = len(age_categories)
    fl = FaceLandmarkDetector(config.facelandmarkdetector_path)

    params = ("facial_ind mandibular_ind intercanthal_ind orbital_width_ind " + 
              #"eye_fissure_ind"
              "nasal_ind vermillion_height_ind " +
              "mouth_face_width_ind").split()
    def convert_data(self, meta):
        try:
            #facial_feature = list(self.fl.detect(io.imread(meta.path)))[0]
            facial_feature = self.with_cache(
                    lambda x: list(self.fl.detect(io.imread(x)))[0].ratios,
                    meta.path,
                    [meta.path])
            age_group = [int(x[0]<=meta.age<=x[1]) for x in self.age_categories]
            res = ([math.log(facial_feature[x]) for x in self.params], age_group)
            print "Converted:", meta.path,
            print "%d features => %s"%(len(res[0]), str(age_group))
            return res
        except Exception, e:
            print>>sys.stderr, "Unable to use: " + meta.path
            return None

