import sys
import numpy as np
import cv2

from FaceDetector import FaceDetector

def main():
#     input image
    img_name = "test_image.png"
    input_img = cv2.imread(img_name)
    height, width, channels = input_img.shape
    #print "input image shape: ", input_img.shape
    aspect_ratio = 1.0 * width / height
    #print "input image aspect_ratio: ", aspect_ratio
    
#     resize image
    new_width = 400
    new_height = int(new_width / aspect_ratio)
    resized_img = cv2.resize(input_img, (new_width, new_height)) 
    #print "resized image shape: ", resized_img.shape
    #cv2.imshow("Resized Image", resized_img)
 
#     detect face
    face_cascade = cv2.CascadeClassifier('haar_models/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haar_models/haarcascade_eye.xml')
    
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
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            print "rect: ", ex, ey
            roi_gray2 = roi_gray[ey:ey + eh, ex:ex + ew]
#             cv2.imshow("Detected Eye Corners"+str(i), roi_gray2)
            i=i+1
            #detect corner of eyes
            dst = cv2.cornerHarris(roi_gray2,2,3,0.04)
            dst = cv2.dilate(dst, None)
            print dst.shape
            p = dst > 0.01*dst.max()
            roi_color[ex,ey] = [0, 0, 255]
            for i in range(dst.shape[0]):
                for j in range(dst.shape[1]):
                    if(dst[i,j] > 0.01*dst.max()):
                        cv2.circle(roi_color, (ex+i, ey+j), 2, (0, 255, 0), 1)
            
            cv2.imshow("Detected Eye Corners", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()
    #cv2.imshow("Detected Face and Eyes", resized_img)
    
#     cv2.imshow("Detected Eye Corners", resized_img)
    
#==================================================================================
#==================================================================================    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
