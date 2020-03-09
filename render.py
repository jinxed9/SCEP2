#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob

def renderVideoFFMPEG():
    os.system("ffmpeg -f image2 -s 1280x720 -r 25 -i ./render/frameOut%05d.jpg -vcodec mpeg4 -y ./output.mp4")
