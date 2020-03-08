#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob

def renderVideoFFMPEG():
    os.system("ffmpeg -f image2 -r 25 -i ./output_images/frame%01d.jpg -vcodec mpeg4 -y ./output.mp4")


# This doesn't work, out of memory error
def renderVideo():
    imgs = []
    for filename in glob.glob('output_images/frame*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        imgs.append(img)

    out = cv2.VideoWriter('output_video.avi', -1, 15, size)

    for i in range(len(imgs)):
        out.write(imgs[i])
    out.release()

renderVideoFFMPEG()