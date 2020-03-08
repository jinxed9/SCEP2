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
        [[200, 675],
         [1100, 675],
         [700, 450],
         [580, 450]]
    )

    #Four desired coordinates
    dst = np.float32(
        [[200, 675],
         [1100, 675],
         [1100, 0],
         [200, 0]]
    )

    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)

    # Could compute the inverse also by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped



def getFrame(frame):
    vidcap = cv2.VideoCapture('project_video.mp4')
    ret, image = vidcap.read()
    count = 0
    while ret:       
        ret,image = vidcap.read()
        print('Read a new frame: ', ret)
        count += 1
        print(count)
        if count == frame:
            cv2.imwrite("frame%d.jpg" % count, image)
            break


def doPerspectiveTransform(img):
    return warp(img)

#----------------- Begin Gradient ----------------------- 

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1


    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
 #------------------- End Gradient -----------------------

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output    

images = glob.glob(r".\camera_cal\calibration*.jpg")    

objPoints = [] #3D points in real world space
imgPoints = [] #2D points in image plane


mtx, dist = doCalibration(images)

img = cv2.imread('frame510.jpg')
#img = mag_thresh(img, sobel_kernel=3, mag_thresh=(30,100))
img = dir_threshold(img, sobel_kernel = 15, thresh=(0.7, 1.3))
#img = doPerspectiveTransform(img)
plt.imshow(img, cmap='gray')
plt.show()


# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi/2))

hls_binary = hls_select(image, thresh=(90, 255))


#undistort
dst = cv2.undistort(img, mtx, dist, None, mtx)
plt.imshow(dst)
plt.show()