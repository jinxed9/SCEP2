#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob


def getAllFrames():
    vidcap = cv2.VideoCapture('project_video.mp4')
    ret, image = vidcap.read()
    count = 0
    while ret:
        ret, image = vidcap.read()
        count += 1
        print(count)
        cv2.imwrite("render/frame%05d.jpg" % count, image)

getAllFrames()
        