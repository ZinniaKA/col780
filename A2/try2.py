import numpy as np
import cv2
from numpy.linalg import norm
from matplotlib import pyplot as plt
from numpy.linalg import det
from numpy.linalg import inv
from scipy.linalg import rq
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
from scipy import ndimage, spatial

def harriscornerdetector(Image):
  '''
  Extra Credit Part
  Goal:Compute Harris Features
  Steps Followed
  1.Apply two filters on the entire image to get the derivative image of x axis and y axis
  2.Computed the harris matrix of each pixel in its neighborhood using a gaussian mask and the derivative images.
  Sum over a 5X5 window. Apply 5X5 Guassian mask with .5 standard deviation 
  3.Then compute the harris score using the matrix
  The response is given by the formula 
  R_score=det(M)-alpha*trace(M)^2 for each pixel window
  4.Finally we take the eigenvector corresponding to the first eigenvalue as the orientation of the feature transformed to radian by atan() and atan()+pi  
  
  '''
  
  
  harris = np.zeros(Image.shape[:2])
  orientations = np.zeros(Image.shape[:2])
  #Step 1
  i_x = ndimage.sobel(Image, axis=-1)
# Display the i_x image
#   cv2.imshow('i_x', i_x)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
  i_y = ndimage.sobel(Image, axis=0) 
#   cv2.imshow('i_y', i_y)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
  i_x_sqr = i_x**2
#   cv2.imshow('i_x2', i_x_sqr)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
  i_y_sqr = i_y**2
#   cv2.imshow('i_y2', i_y_sqr)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
  i_x_times_i_y = i_x*i_y
#   cv2.imshow('i_xiy', i_x_times_i_y)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
  #Step 2
  s = 0.9 #Sigma
  G = 31  #Gauss Mask   
  truncate_SD = G/(s*2)
  sumix2 = ndimage.gaussian_filter(i_x_sqr, s, truncate=truncate_SD)
#   cv2.imshow('i_xiy', sumix2)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

  sumiy2 = ndimage.gaussian_filter(i_y_sqr, s, truncate=truncate_SD)
#   cv2.imshow('i_xiy', sumiy2)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

  sumixiy2 = ndimage.gaussian_filter(i_x_times_i_y, s, truncate=truncate_SD)
#   cv2.imshow('i_xiy', sumixiy2)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

  #Step 3
  alpha = 0.01
  det = sumix2*sumiy2 - sumixiy2 **2
  trace = sumix2+sumiy2
  harris = det- alpha*(trace**2)
  orientations = np.degrees(np.arctan2(i_y.flatten(),i_x.flatten()).reshape(orientations.shape)) 
  return harris, orientations


def LocalMaxima(Image):
    '''
    This function takes a numpy array containing the Harris score at
    each pixel and returns an numpy array containing True/False at
    each pixel, depending on whether the pixel is a local maxima 
    Steps adopted
    1.Calculate the local maxima image
    2.And find the maximum pixels in the 7X7 window
    3.Then return true when pixel is the maximum, otherwise false
    '''
    destImage = np.zeros_like(Image, bool)
    harrisImage_max = ndimage.filters.maximum_filter(Image, size=(81,81))
    destImage = (Image == harrisImage_max)
    return destImage


def detectKeypoints(image):
    '''
    This function takes in the  image and returns detected keypoints
    Steps:
    1.Grayscale image used for Harris detection
    2.Call harriscornerdetector() which gives the harris score at each pixel
    position
    3.Compute local maxima in the Harris image
    4.Update the cv2.KeyPoint() class objects with the coordinate, size, angle and response
    '''
    image = image.astype(np.float32)
    image /= 255.
    h, w = image.shape[:2]
    keypoints = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    harris, orientation = harriscornerdetector(gray)
    # print(harris)
    maxi = LocalMaxima(harris)

    for y in range(h):
        for x in range(w):
            if not maxi[y, x]:
                continue

            f = cv2.KeyPoint()
            f.pt = x, y
            f.size = 1
            #f.angle = orientation[y, x]
            #f.response = harris[y, x]
            keypoints.append(f)

    return keypoints

image_path = "data/Images/Field/1.jpg"
image = cv2.imread(image_path)
keypoints = detectKeypoints(image)

print(len(keypoints))
#Visualize keypoints (without plotting on the image)
# for keypoint in keypoints:
#     print("Keypoint coordinates:", keypoint.pt)
# Visualize keypoints
for keypoint in keypoints:
    center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
    cv2.circle(image, center, 3, (0, 0, 255), -1)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()         