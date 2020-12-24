import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob

def doCalibration(images):
    objPoints = [] #3D points in real world space
    imgPoints = [] #2D points in image plane

    insideCornerCount = [(9,5),(9,6),(9,6),(9,6),(9,6),
                         (9,6),(9,6),(9,6),(9,6),(9,6),
                         (9,6),(9,6),(9,6),(9,6),(6,5),
                         (7,6),(9,6),(9,6),(9,6),(9,6)]

    for idx, fname in enumerate(images):
        print(fname, end='\r', flush=True)
        img = cv2.imread(images[idx])
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, insideCornerCount[idx], None)

        nx = insideCornerCount[idx][0]
        ny = insideCornerCount[idx][1]

        objP = np.zeros((nx*ny, 3), np.float32)
        objP[:,:2 ] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x, y coordinates

        if ret == True:
            imgPoints.append(corners)
            objPoints.append(objP)

    print('\n')

    #get calibration stuff
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

    return mtx, dist