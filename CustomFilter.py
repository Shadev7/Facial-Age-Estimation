import cv2
import numpy as np

def create_filter(arr):
    kernel = np.zeros((len(arr), len(arr), 1), dtype=np.float32)
    for rowNum, row in enumerate(arr):
        for cellNum, cell in enumerate(row):
            kernel[rowNum, cellNum] = cell
    return kernel

def create_eye_left_filter(size):
    #Assume size is odd
    arr = [[0 for __ in range(size)] for _ in range(size)]
    for i in range(size/2 + 1):
        arr[i][size - i - 1] = 255.0 / size
        arr[size - i - 1][size - i - 1] = 255.0 / size
    #print "\n".join(" ".join(map(str, x)) for x in arr)
    return create_filter(arr)

if __name__ == '__main__':
    create_eye_left_filter(15)


