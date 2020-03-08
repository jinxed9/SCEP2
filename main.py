#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob


def doCalibration(images):
    insideCornerCount = [(9,5),(9,6),(9,6),(9,6),(9,6),
                         (9,6),(9,6),(9,6),(9,6),(9,6),
                         (9,6),(9,6),(9,6),(9,6),(6,5),
                         (7,6),(9,6),(9,6),(9,6),(9,6)]

    for idx, fname in enumerate(images):
        print(fname)
        img = cv2.imread(images[idx])
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        print(insideCornerCount[idx])
        ret, corners = cv2.findChessboardCorners(gray, insideCornerCount[idx], None)

        nx = insideCornerCount[idx][0]
        ny = insideCornerCount[idx][1]

        objP = np.zeros((nx*ny, 3), np.float32)
        objP[:,:2 ] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x, y coordinates

        if ret == True:
            print("Corners found.")
            imgPoints.append(corners)
            objPoints.append(objP)


    #get calibration stuff
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

    return mtx, dist


def warp(img):
    #Define calibration box in source (original) and destination (desired or warped) coordinates

    img_size = (img.shape[1], img.shape[0])

    #Four source coordinates
    src = np.float32(
        [[850, 320],
         [865, 450],
         [535, 210],
         [535, 210]]
    )

    #Four desired coordinates
    dst = np.float32(
        [[870, 240],
         [870, 370],
         [520, 370],
         [520, 240]]
    )

    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)

    # Could compute the inverse also by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)


def doPerspectiveTransoform:





images = glob.glob(r".\camera_cal\calibration*.jpg")    

objPoints = [] #3D points in real world space
imgPoints = [] #2D points in image plane


mtx, dist = doCalibration(images)

doPerspectiveTransoform()
img = cv2.imread(images[4])

#undistort
dst = cv2.undistort(img, mtx, dist, None, mtx)
plt.imshow(dst)
plt.show()