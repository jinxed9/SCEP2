#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

#reading in an image
path = os.getcwd()
print("Current Directory", path)
print()

parent = os.path.dirname(path)
print("Parent directory", parent)

image = mpimg.imread(parent + '/test_images/test5.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image 
#called 'gray', for example, call as plt.imshow(gray, cmap='gray')
plt.show()