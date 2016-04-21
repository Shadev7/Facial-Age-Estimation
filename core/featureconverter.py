import sys
import math
import os.path
import json

from skimage import io

from core.feature import FaceLandmarkDetector
import config

class GenericFeatureConverter(object):
    n_features = 0

    def convert_data(self, obj):
        return None

class CacheMixin(object):
    def with_cache(self, key, fn, path, args):
        json_obj = {}
        if os.path.isfile(path + ".cached"):
            with open(path + ".cached") as f:
                json_obj = json.load(f)

        if key in json_obj:
            return json_obj[key]
        
        json_obj[key] = fn(*args)

        with open(path + ".cached", "w") as f:
            json.dump(json_obj, f)
        return json_obj[key]

class FaceLandmarkFeatureConverter(GenericFeatureConverter, CacheMixin):
    fl = FaceLandmarkDetector(config.facelandmarkdetector_path)
    n_features = 7

    def convert_data(self, meta):
        def get_ratios(fl, path):
            params = ("facial_ind mandibular_ind intercanthal_ind " +
                            "orbital_width_ind nasal_ind vermillion_height_ind " +
                            "mouth_face_width_ind").split()
            ratios = next(fl.detect(io.imread(path))).ratios
            return [ratios[x] for x in params]
        try:
            facial_feature = self.with_cache(
                    "FaceLandmarkFeatureConverter",
                    lambda x: get_ratios(self.fl, x),
                    meta.path,
                    [meta.path])
            res = (facial_feature, meta.age)
            return res
        except Exception, e:
            print>>sys.stderr, "Unable to use:", meta.path
            return None

class FaceBoundaryFeatureConverter(GenericFeatureConverter, CacheMixin):
    fl = FaceLandmarkDetector(config.facelandmarkdetector_path)
    n_features = 14

    def convert_data(self, meta):
        try:
            face_boundary = self.with_cache(
                    "FaceBoundaryFeatureConverter",
                    lambda x: list(self.fl.detect(io.imread(x)))[0].face_boundary,
                    meta.path,
                    [meta.path])
            res = (face_boundary, meta.age)
            return res
        except Exception, e:
            print e
            print>>sys.stderr, "Unable to use: " + meta.path
            return None

class FaceLandmarkBoundaryFeatureConverter(FaceLandmarkFeatureConverter, FaceBoundaryFeatureConverter):
    n_features = FaceLandmarkFeatureConverter.n_features + FaceBoundaryFeatureConverter.n_features
    
    def convert_data(self, meta):
        res1 = FaceLandmarkFeatureConverter.convert_data(self, meta)
        res2 = FaceBoundaryFeatureConverter.convert_data(self, meta)
        if res1 is None or res2 is None:
            return None
        return (res1[0] + res2[0], res1[1])

