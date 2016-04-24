import sys

import cv2
import dlib

def draw_lines(img, lines, color=(255,0,255)):
    for part in lines:
        cv2.line(img, part[0], part[1], color)


def display_image(img, shape):
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    if shape:
        win.add_overlay(shape)


max_index = lambda x: max(enumerate(x), key=lambda y: y[1])[0]

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

