import sys

import cv2

from FaceDetector import FaceDetector

def main():
    img = cv2.imread(sys.argv[1])
    fd = FaceDetector()
    fd.detectFeatures(img)
    
    cv2.imshow("img", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



