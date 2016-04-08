import sys
import numpy as np
import cv2
from collections import namedtuple

from FaceDetector import FaceDetector
from CustomFilter import create_eye_left_filter

def main():
#     input image
    img_name = "face1.png"
    #img_name = "face1.png"
    #img_name = "data/28754132@N06/landmark_aligned_face.608.9691370454_849ce3fb06_o.jpg"
    input_img = cv2.imread(img_name)
    height, width, channels = input_img.shape
    #print "input image shape: ", input_img.shape
    aspect_ratio = 1.0 * width / height
    #print "input image aspect_ratio: ", aspect_ratio

    FaceAttrib = namedtuple("FaceAttrib", "n en ex ch al") #as per the face annotation diagram
    
    
#     resize image
    new_width = 400
    new_height = int(new_width / aspect_ratio)
    resized_img = cv2.resize(input_img, (new_width, new_height)) 
    #print "resized image shape: ", resized_img.shape
    #cv2.imshow("Resized Image", resized_img)
 
#     detect face
    face_cascade = cv2.CascadeClassifier('haar_models/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haar_models/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('haar-models/Mouth.xml')
    nose_cascade = cv2.CascadeClassifier('haar-models/Nariz.xml')
    #eye_cascade = cv2.CascadeClassifier('parojosG.xml')
    nose_cascade2 = cv2.CascadeClassifier('haar-models/Nariz_nuevo_20stages.xml')
    
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1, #compensate closeness to the camera
        minNeighbors=5, #show many objects are detected near the current one before it declares the face found
        minSize=(30, 30), #size of each window
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (255, 0, 0), 2) #face rectangle
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = resized_img[y:y + h, x:x + w]
#     detect eyes 
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        i=0
        FaceAttrib.ex = (eyes[0][0] + eyes[0][2]) - eyes[1][0] # distance between eyes extreme ends x-cord
        FaceAttrib.en =  eyes[0][0] - (eyes[1][0] + eyes[1][2]) # distance between eyes near ends x-cord
        FaceAttrib.n =  eyes[0][0] + FaceAttrib.ex/2 #point between eyes y-cord
        print "ex : ", FaceAttrib.ex
        print  "en : ", FaceAttrib.en
        print "Orbital width index :", str(FaceAttrib.ex/FaceAttrib.en)
        print  "n : ", FaceAttrib.n
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            roi_eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            smoothed = cv2.GaussianBlur(roi_eye_gray, (3,3), 30)
            cannied = cv2.Canny(smoothed, 50,100 )
            gftt = cv2.goodFeaturesToTrack (roi_eye_gray, 20, 0.01,10)

	    #cv2.imshow("GFTT", gftt)
            #cv2.imshow("Canny", cannied)
	    p = gftt > 0.01*gftt.max()
            roi_color[ex,ey] = [0, 0, 255]
	    for i in gftt:
	        #print (i[0][0])
		cv2.circle(roi_color, (ex+ int(i[0][0]), ey+int(i[0][1])), 2, (0, 255, 255), 1)
            '''for i in range(gftt.shape[0]):
                for j in range(gftt.shape[1]):
		    print "gftt"
		    print gftt[i,j]
		    cv2.circle(roi_color, (ex+i, ey+j), 2, (0, 255, 0), 1)
                    if(gftt[i,j] > 0.01*gftt.max()):
                        cv2.circle(roi_color, (ex+i, ey+j), 2, (0, 255, 0), 1)'''

            temp_kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(cannied, temp_kernel, iterations=1)
            cv2.imshow("processed", cv2.erode(dilated, temp_kernel, iterations=1))
            #filt = create_eye_left_filter(100)
            for i in range(2,15):
                res = cv2.filter2D(dilated, cv2.CV_32F, create_eye_left_filter(i))
                #cv2.imshow(str(i), res)
            print "Min:", res.min(), "| Max:", res.max()
            res = (res - res.min()) * (255 / (res.max() - res.min()))
            print "Min:", res.min(), "| Max:", res.max()
            print res
            #cv2.waitKey(0)
            #exit()



#     detect nose 
        nose = nose_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        i=0
        #n_x = 0
        n_y = 0
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 100), 2)
            #print "rect: ", ex, ey
            print "w,h: ", nw, nh
            roi_gray2 = roi_gray[ny:ny + nh, nx:nx + nw]
#             cv2.imshow("Detected Eye Corners"+str(i), roi_gray2)
            i=i+1
            n_y = ny + nh +nh

#     detect nose2 
        nose = nose_cascade2.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        i=0
        #n_x = 0
        n_y = 0
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 100, 100), 2)
            #print "rect: ", ex, ey
            print "w,h: ", nw, nh
            roi_gray2 = roi_gray[ny:ny + nh, nx:nx + nw]
#             cv2.imshow("Detected Eye Corners"+str(i), roi_gray2)
            i=i+1
            n_y = ny + nh +nh

#     detect mouth
        roi_gray_m = gray[n_y:y + h, x:x + w]
        roi_color_m = resized_img[n_y:y + h, x:x + w]
        mouth = mouth_cascade.detectMultiScale(
            roi_gray_m,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color_m, (mx, my), (mx + mw, my + mh), (100, 100, 0), 2)
            #print "rect: ", ex, ey
	    #print "w,h: ", ew, eh
            roi_gray2 = roi_gray_m[my:my + mh, mx:mx + mw]
#             cv2.imshow("Detected Eye Corners"+str(i), roi_gray2)
            #detect corner of mouth
            '''dst = cv2.cornerHarris(roi_gray2,2,3,0.04)
            dst = cv2.dilate(dst, None)
            print dst.shape
            p = dst > 0.01*dst.max()
            roi_color[ex,ey] = [0, 0, 255]
            for i in range(dst.shape[0]):
                for j in range(dst.shape[1]):
                    if(dst[i,j] > 0.01*dst.max()):
                        cv2.circle(roi_color, (ex+i, ey+j), 2, (0, 255, 0), 1)'''            

    cv2.imshow("Detected Face and Eyes", resized_img)
    
#     cv2.imshow("Detected Eye Corners", resized_img)
    
#==================================================================================
#==================================================================================    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

if __name__ == '__main__':
    main()
