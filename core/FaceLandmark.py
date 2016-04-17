import dlib
from skimage import io
import math

from core.model import FacialFeatures

class FaceLandmarkDetector(object):
    def __init__(self, path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)

    def detect(self, image):
        detections = self.detector(image, 1)
        for k,d in enumerate(detections):
            shape = self.predictor(image, d)
            feature_dict = self.parseShape(shape)
            ratios = self.calculate_ratios(feature_dict)
            yield FacialFeatures(ratios=ratios, feature_points=feature_dict,
                    all_points=shape)

    def parseShape(self, shape):
        ranges = [range(0,17), range(17,22), range(22,27), range(28,31),
                range(31,36),  range(36,42), range(42,48), range(48,60), range(61,68)]

        names = "jaw left_brow right_brow nose_unnecessary nose left_eye right_eye lips lips_divider".split()
        s = lambda x: (x.x, x.y)
        return {k: [s(shape.part(i)) for i in indices] 
                        for k, indices in zip(names,ranges)}

    def calculate_ratios(self, feature_dict):
        ratios = {}
        names = "facial_ind mandibular_ind intercanthal_ind orbital_width_ind eye_fissure_ind \
                    nasal_ind vermillion_height_ind mouth_face_width_ind".split()

        third_eye_x = feature_dict["left_brow"][0][0] + feature_dict["right_brow"][-1][0] + \
         feature_dict["left_brow"][-1][0] + feature_dict["right_brow"][0][0]
        third_eye_y = feature_dict["left_brow"][0][1] + feature_dict["right_brow"][-1][1] + \
         feature_dict["left_brow"][-1][1] + feature_dict["right_brow"][0][1]
         

        feature_dict["third_eye"] = [(third_eye_x/4,third_eye_y/4)]
        
        ratios[names[0]] = self.distance_ratio(feature_dict,"third_eye",0,"jaw",8,"jaw",1,"jaw",-2)
        ratios[names[1]] = self.distance_ratio(feature_dict,"lips_divider",1,"jaw",8,"jaw",5,"jaw",11)
        ratios[names[2]] = self.distance_ratio(feature_dict,"left_eye",3,"right_eye",0,"left_eye",0,"right_eye",3)
        ratios[names[3]] = self.distance_ratio(feature_dict,"left_eye",0,"left_eye",3,"left_eye",3,"right_eye",0)
#        ratios[names[4]] = self.distance_ratio(feature_dict,"left_eye",,"left_eye",,"left_eye",0,"left_eye",3)
        ratios[names[5]] = self.distance_ratio(feature_dict,"nose",0,"nose",-1,"third_eye",0,"nose",2)
        ratios[names[6]] = self.distance_ratio(feature_dict,"lips",3,"lips_divider",1,"lips_divider",-2,"lips",9)
        ratios[names[7]] = self.distance_ratio(feature_dict,"lips",0,"lips",6,"jaw",1,"jaw",-2)
        

        return ratios

    def distance_ratio(self, feature_dict, num_f1,num_i1, num_f2,num_i2, den_f1,den_i1, den_f2,den_i2):
    	num = (feature_dict[num_f1][num_i1][0] - feature_dict[num_f2][num_i2][0]) ** 2 \
    			+ (feature_dict[num_f1][num_i1][1] - feature_dict[num_f2][num_i2][1]) ** 2

    	den = (feature_dict[den_f1][den_i1][0] - feature_dict[den_f2][den_i2][0]) ** 2 \
    			+ (feature_dict[den_f1][den_i1][1] - feature_dict[den_f2][den_i2][1]) ** 2
    	return math.sqrt(num * 1.0 / den)
