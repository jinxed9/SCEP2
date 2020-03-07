#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


def doCalibration():
    nx = 9 #Number of inside corners in x
    ny = 5 #Number of inside corners in y

    fname = "calibration1.jpg"
    print(fname)
    img = cv2.imread(r".\camera_cal\calibration1.jpg")    

    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #plt.imshow(gray)
    #plt.show()

    #Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        #Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        plt.show()

#image = mpimg.imread(parent + '/test_images/test5.jpg')

#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image 
#called 'gray', for example, call as plt.imshow(gray, cmap='gray')
#plt.show()

doCalibration()