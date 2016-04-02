from collections import defaultdict

import cv2

class DetectionFeatures(object):
    def __init__(self):
        self.faces = []
        self.age = -1

class Eye(object):
    def __init__(self, img, rects):
        self.img = img
        self.rects = rects

    def annotate(self):
        for x in self.rects:
            cv2.rectangle(self.img, (x[0],x[1]), (x[0] + x[2], x[1] + x[3]),(255,0,0),2)

class Face(object):
    eye_cascade = cv2.CascadeClassifier('haar_models/haarcascade_eye.xml')

    def __init__(self, img, rect):
        self.img = img
        self.rect = rect
        self.eyes = defaultdict(list)

    def detectFeatures(self):
        grey =  cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.eyes = self.eye_cascade.detectMultiScale(grey, 1.3, 5)

    def annotate(self):
        cv2.rectangle(self.img, (self.rect[0],self.rect[1]),
                (self.rect[0] + self.rect[2],self.rect[1] + self.rect[3]),(255,0,0),2)

    def __repr__(self):
        return "Face at rect: " + repr(self.rect) + " with %d eyes detected."%self.eyes

class FaceDetector(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haar_models/haarcascade_frontalface_default.xml')

    def detectFeatures(self, img):
        img = cv2.resize(img, (400, int(400 * float(img.shape[0]) / float(img.shape[1]))))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        result = DetectionFeatures()

        for face_rect in faces:
            face = Face(img, face_rect)
            face.detectFeatures()
            face.annotate()

            result.faces.append(face)

