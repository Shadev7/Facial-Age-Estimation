from FaceLandmark import FaceLandmarkDetector
import sys
from skimage import io
import dlib
import cv2

img = io.imread(sys.argv[1])
f = FaceLandmarkDetector('./shape_predictor_68_face_landmarks.dat')
shape, feature_dict, ratios = f.detect(img)
print(ratios)

#print(feature_dict["jaw_points"][0][0])
cv2.circle(img, feature_dict["nose"][0], 4 , (255,0,0), 3)
cv2.circle(img, feature_dict["nose"][-1], 4 , (0,255,0), 3)
cv2.circle(img, feature_dict["nose"][2], 4 , (0,0,255), 3)
#cv2.circle(img, feature_dict["right_brow"][0], 4 , (0,0,255), 3)


third_eye_x = feature_dict["left_brow"][0][0] + feature_dict["right_brow"][-1][0] + \
         feature_dict["left_brow"][-1][0] + feature_dict["right_brow"][0][0]
third_eye_y = feature_dict["left_brow"][0][1] + feature_dict["right_brow"][-1][1] + \
         feature_dict["left_brow"][-1][1] + feature_dict["right_brow"][0][1]

cv2.circle(img, (third_eye_x/4 ,third_eye_y/4), 4 , (0,0,255), 3)

cv2.imshow("part 0", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

win = dlib.image_window()
win.clear_overlay()
win.set_image(img)
win.add_overlay(shape)
dlib.hit_enter_to_continue()
