from FaceLandmark import FaceLandmarkDetector
import sys
from skimage import io
import dlib
import cv2
import glob
import os

#OLDER code:
# img = io.imread(sys.argv[1])
# f = FaceLandmarkDetector('./shape_predictor_68_face_landmarks.dat')
# fd = f.detect(img)
# shape = fd[0]
# feature_dict = fd[1]
# ratios = fd[2]
allratios =[]
ages= []
genders = []
# for f in glob.glob(os.path.join(sys.argv[1], "*.png")):
for f in glob.glob(os.path.join("../b/*.png")):
    print("Processing file: {}\n".format(f))
    label = f[:-4] + ".txt"
    label_file = open(label, "r")
    # print label_file.readlines()
    labels = label_file.readlines()
    genders.append(labels[0].split()[1])
    ages.append(labels[1].split()[1])
    # print "Age = " + labels[0]
    label_file.close()
    img = io.imread(f)
    f = FaceLandmarkDetector('./shape_predictor_68_face_landmarks.dat')
    fd = f.detect(img)
    shape = fd[0]
    feature_dict = fd[1]
    ratios = fd[2]
    allratios.append(ratios)
print "ages: ", ages
print "gender:", genders
    
model_file = open("train.data", "w")

line = "" 
counter = 0
for ratio in allratios:
	line += ages[counter] + " "
	counter += 1
	for i in ratio:
		line += str(ratio[i]) + " "
	line = line[:-1] + "\n"
model_file.write(line)
model_file.close()

# train((svm_c_trainer_linear)arg1, (vectors)arg2, (array)arg3) 
#print(feature_dict["jaw_points"][0][0])
# cv2.circle(img, feature_dict["nose"][0], 4 , (255,0,0), 3)
# cv2.circle(img, feature_dict["nose"][-1], 4 , (0,255,0), 3)
# cv2.circle(img, feature_dict["nose"][2], 4 , (0,0,255), 3)
#cv2.circle(img, feature_dict["right_brow"][0], 4 , (0,0,255), 3)

# third eye is taken care of FaceLandmark.py, we can remove this later	
# third_eye_x = feature_dict["left_brow"][0][0] + feature_dict["right_brow"][-1][0] + \
#          feature_dict["left_brow"][-1][0] + feature_dict["right_brow"][0][0]
# third_eye_y = feature_dict["left_brow"][0][1] + feature_dict["right_brow"][-1][1] + \
#          feature_dict["left_brow"][-1][1] + feature_dict["right_brow"][0][1]

# cv2.circle(img, (third_eye_x/4 ,third_eye_y/4), 4 , (0,0,255), 3)
# cv2.circle(img, feature_dict["left_eye"][0], 4 , (255,0,255), 3)
# cv2.circle(img, feature_dict["left_eye"][3], 4 , (255,0,255), 3)
# cv2.circle(img, feature_dict["right_eye"][0], 4 , (255,0,255), 3)
# cv2.circle(img, feature_dict["right_eye"][3], 4 , (255,0,255), 3)
# cv2.circle(img, feature_dict["right_eye"][3], 4 , (255,0,255), 3)
# cv2.circle(img, feature_dict["lips"][0], 4 , (140,200,0), 3)
# cv2.circle(img, feature_dict["lips"][6], 4 , (140,200,0), 3)

# cv2.imshow("part 0", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# win = dlib.image_window()
# win.clear_overlay()
# win.set_image(img)
# win.add_overlay(shape)
# dlib.hit_enter_to_continue()
