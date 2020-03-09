#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
from PIL import Image, ImageDraw


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
        #print(insideCornerCount[idx])
        ret, corners = cv2.findChessboardCorners(gray, insideCornerCount[idx], None)

        nx = insideCornerCount[idx][0]
        ny = insideCornerCount[idx][1]

        objP = np.zeros((nx*ny, 3), np.float32)
        objP[:,:2 ] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x, y coordinates

        if ret == True:
            #print("Corners found.")
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

def unwarp(img):
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
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

    return unwarped

#------------- Video Processing ----------------------------

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

#----------------------------------------------------------------

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

#----------------- Thesholding -----------------------------------

def doThresholding(img):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    #   Convert to grayscale
    gray =cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #   Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)# Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivate to accentuate line away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    #   Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    #   Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    #   Stack each channel to view their individual contributions in green and blue respectively
    #   This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    #   Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    #   Plotting thresholded images
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    #ax1.set_title('Stacked thresholds')
    #ax1.imshow(color_binary)

    #ax2.set_title('Combined S channel and gradient thresholds')
    #ax2.imshow(combined_binary, cmap='gray')
    #plt.show()

    return combined_binary

#------------------ Histogram -------------------
def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram


#-------- Sliding Window ------------------------
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    left_fitx_int = np.around(left_fitx)
    left_fitx_int = left_fitx_int.astype(int)
    right_fitx_int = np.around(right_fitx)
    right_fitx_int = right_fitx_int.astype(int)
    ploty_int = ploty.astype(int)

    lines = np.zeros_like(out_img)
    #lines[ploty_int, left_fitx_int] = [255, 255, 255]
    #lines[ploty_int, right_fitx_int] = [255, 255, 255]

    pts_left = np.vstack((left_fitx_int,ploty_int)).T
    pts_right = np.vstack((right_fitx_int, ploty_int)).T
    
    # Generate top boundry for fill


    pts = []
    pts.append(pts_right)
    pts.append(pts_left)
    #pts.append(np.array(top))
    #pts.append(bottom)
    # Draw the lane onto the warped blank image
    #cv2.fillPoly(lines, np.array(pts,'int32'), (0,255, 0))
    cv2.polylines(lines, np.int_([pts_left]), False, (255, 0, 0), thickness=20)
    cv2.polylines(lines, np.int_([pts_right]), False, (0, 0, 255), thickness=20)

    #plt.imshow(lines)
    #plt.show()
    
    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return lines

#----------------Do Lane Detection --------------

def doLaneDetection(img):
    img = cv2.imread(img)

    # 1. Undistort
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # 2. Thresholding
    color_binary = doThresholding(img)

    # 3. Perspective Transform
    top_down = doPerspectiveTransform(color_binary)

    # 5. Fit a Polynomial
    out_img = fit_polynomial(top_down)
    out_img = unwarp(out_img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stacked = cv2.addWeighted(img, 1, out_img, 0.5, 0)

    return stacked



#--------------- Main ------------------------------
# Steps:
# 1. Do calibration (only need to do once)
images = glob.glob(r".\camera_cal\calibration*.jpg")    
objPoints = [] #3D points in real world space
imgPoints = [] #2D points in image plane
mtx, dist = doCalibration(images)


# Main loop
images = glob.glob(r".\render\frame*.jpg")

count = 0
for img in images:
    processed = doLaneDetection(img)
    count += 1
    cv2.imwrite(".\\render\\frameOut%05d.jpg" % count, cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))


# 6.    Once you've found lines, skip sliding window step
# 7. Measure curvature 1. and 2
# 8. Draw lines, use inverse perspective transform.

