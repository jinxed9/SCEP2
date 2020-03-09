#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from render import renderVideoFFMPEG
from calibration import doCalibration


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

#----------------- Begin Gradient ----------------------- 


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

def measure_curvature_pixels(ploty, left_fit, right_fit):
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    return left_curverad, right_curverad

def fit_polynomial(binary_warped, print_stages=False):
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

    pts_left = np.vstack((left_fitx_int,ploty_int)).T
    pts_right = np.vstack((right_fitx_int, ploty_int)).T

    pts = np.append(pts_left, np.flip(pts_right, axis=0), axis=0)
    

    # Draw the lane onto the warped blank image
    cv2.fillPoly(lines, np.int_([pts]), (0,255, 0))
    cv2.polylines(lines, np.int_([pts_left]), False, (255, 0, 0), thickness=30)
    cv2.polylines(lines, np.int_([pts_right]), False, (0, 0, 255), thickness=30)

    #plt.imshow(lines)
    #plt.show()
    
    if print_stages:
        # Plots the left and right polynomials on the lane lines
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        cv2.polylines(out_img, np.int_([pts_left]), False, (255, 255, 0), thickness=10)
        cv2.polylines(out_img, np.int_([pts_right]), False, (255, 255, 0), thickness=10)
        plt.imsave(".\\output_images\\test%d_top_down.jpg" % count,out_img,cmap='gray')


    # Calculation of vehicle position
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_lane = left_fitx_int[719]
    right_lane = right_fitx_int[719]
    center = 1280/2
    lane_position = (right_lane+left_lane)/2
    vehicle_position = (lane_position-center) * xm_per_pix

    #left_curverad, right_curverad = measure_curvature_pixels(ploty,left_fit, right_fit)
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fit, right_fit)
    #print(left_curverad, right_curverad)
    radius = (left_curverad+right_curverad)/2



    return lines, radius, vehicle_position

#----------------Do Lane Detection --------------

def doLaneDetection(img,print_stages=False):

    #TODO: make sure this works on white lanes
    # Thresholding
    color_binary = doThresholding(img)
    if print_stages:
        plt.imsave(".\\output_images\\test%d_color_binary.jpg" % count,color_binary,cmap='gray')

    # Perspective Transform
    top_down = warp(color_binary)


    #TODO: Add sliding window
    #TODO: Add low pass filter
    # Fit a Polynomial
    out_img, radius, position = fit_polynomial(top_down, print_stages)

    # Reverse Perspective Transform
    out_img = unwarp(out_img)

    cv2.putText(out_img,"Radius: %0dm" %radius,(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(out_img,"Vehicle is %2fm left of center" %position,(10,200),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

    #TODO: Add curvature and center position
    # Draw lanes on image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stacked = cv2.addWeighted(img, 1, out_img, 0.5, 0)

    if print_stages:
        cv2.imwrite(".\\output_images\\test%d_stacked.jpg" % count, cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))

    return stacked



#--------------- Main ------------------------------

processVideo = True

# Do calibration (only need to do once)
images = glob.glob(r".\camera_cal\calibration*.jpg")    
mtx, dist = doCalibration(images)


# Process Images
images = glob.glob(r".\test_images\test*.jpg")
images.append(r".\test_images\straight_lines1.jpg")
images.append(r".\test_images\straight_lines2.jpg")
count = 0
for img in images:
    count += 1
    
    #Undistort
    img = cv2.imread(img)
    undistorted = cv2.undistort(img,mtx, dist, None, mtx)
    cv2.imwrite(".\\output_images\\test%d_undistorted.jpg" % count, undistorted)
    print("Processing image %2d" % count, end='\r', flush=True)
    processed = doLaneDetection(img,print_stages=True)    

print("\nFinished Images")


# Process Video
if processVideo:
    images = glob.glob(r".\render\frame*.jpg")
    count = 0
    for img in images:
        count += 1
        if count > 1252:
            (print("\n# Finished #"))
            break

        # Undistort
        img = cv2.imread(img)
        img = cv2.undistort(img, mtx, dist, None, mtx)

        # Process
        processed = doLaneDetection(img,print_stages=False)
        print("Processing frame %2d" % count, end='\r', flush=True)

        cv2.imwrite(".\\render\\frameOut%05d.jpg" % count, cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    # When finished, render video
    renderVideoFFMPEG()

