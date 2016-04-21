import dlib
from skimage import io
import math
import cv2

from core.model import FacialFeatures

class FaceBoundaryDetector(object):
    def __init__(self, path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)

    def detect(self, image):
        detections = self.detector(image, 1)
        for k,d in enumerate(detections):
            shape = self.predictor(image, d)
            feature_dict = self.parseShape(shape)

        return self.calculate_face_boundary_angles(feature_dict)

    
    def parseShape(self, shape):
        ranges = [range(0,17), range(17,22), range(22,27), range(28,31),
                range(31,36),  range(36,42), range(42,48), range(48,60), range(61,68)]

        names = "jaw left_brow right_brow nose_unnecessary nose left_eye right_eye lips lips_divider".split()
        s = lambda x: (x.x, x.y)
        return {k: [s(shape.part(i)) for i in indices] 
                        for k, indices in zip(names,ranges)}

    def calculate_face_boundary_angles(self, feature_dict):
        angles_with_x_axis = self.find_angles(feature_dict["jaw"]) #gives angles that each point makes with the x-axis
        
        angles = []
        for i in range(1,15):
            angles.append(angles_with_x_axis[i+1] + math.pi - angles_with_x_axis[i])
        
        return angles

    def find_angles(self, jaws):
        theta = [0]
        for i in range(0,15):
            theta.append(self.angle_between(jaws[i], jaws[i+1]))
        return theta
    
    def angle_between(self, p1, p2):
        if p2[0]-p1[0] == 0:
            if p2[1]>p1[1]:
                return math.pi/2
            return -math.pi/2
        return math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))
