import numpy as np
import cv2
from PIL import Image as I
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse
import math

class Keypoint:
    def __init__(self, pt, size,orientation,magnitude):
        self.pt = pt  # (x, y) coordinates
        self.size = size
        self.orientation = orientation
        self.magnitude = magnitude

def convolve(image, kernel):
    """
    Perform 2D convolution between an image and a kernel.
    
    Args:
    - image: 2D numpy array representing the input image
    - kernel: 2D numpy array representing the kernel
    
    Returns:
    - result: 2D numpy array representing the convolved image
    """
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Compute the size of the output image
    result_height = image_height - kernel_height + 1
    result_width = image_width - kernel_width + 1
    
    # Initialize the result array
    result = np.zeros((result_height, result_width))
    
    # Perform convolution
    for i in range(result_height):
        for j in range(result_width):
            # Extract the region of interest from the image
            roi = image[i:i+kernel_height, j:j+kernel_width]
            # Perform element-wise multiplication between the ROI and the kernel
            conv_value = np.sum(np.multiply(roi, kernel))
            # Store the result in the output image
            result[i, j] = conv_value
    
    return result

def harriscornerdetector(Image):
    harris = np.zeros(Image.shape[:2])
    orientations = np.zeros(Image.shape[:2])
    
    # Sobel x-axis kernel
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="int32")
    # Sobel y-axis kernel
    SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="int32")
    # Gaussian kernel
    GAUSS = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]], dtype="float64")
    
    # Step 1: Compute image gradients
    i_x = convolve(Image, SOBEL_X)
    i_y = convolve(Image, SOBEL_Y)
    i_x_sqr = i_x**2
    i_y_sqr = i_y**2
    i_x_times_i_y = i_x * i_y
    # Step 2: Apply Gaussian mask
    s = 0.9  # Sigma
    G=31
    trncate=G/(s*2)
    sumix2 = convolve(i_x_sqr, GAUSS)
    sumiy2 = convolve(i_y_sqr, GAUSS)
    sumixiy2 = convolve(i_x_times_i_y,GAUSS)
    # Step 3: Compute Harris score
    alpha = 0.01
    det = sumix2*sumiy2 - sumixiy2**2
    trace = sumix2 + sumiy2
    harris = det - alpha*(trace**2)
    # magnitude = np.degrees(math.sqrt())
    orientations = np.degrees(np.arctan2(i_y.flatten(),i_x.flatten()).reshape(orientations.shape)) 
    return harris, orientations

def maximum_filter(image, size):
    h, w = image.shape
    result = np.zeros_like(image, dtype=bool)
    pad_width = size // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=np.min(image))
    
    for i in range(h):
        for j in range(w):
            patch = padded_image[i:i+size, j:j+size]
            result[i, j] = image[i, j] == np.max(patch)
    
    return result

def LocalMaxima(Image,size=81):
    destImage = np.zeros_like(Image, dtype=bool)
    h, w = Image.shape
    half_patch = size//2
    for i in range(h):
        for j in range(w):
            patch = Image[max(0, i-half_patch):min(h, i+half_patch +1), max(0, half_patch-3):min(w, j+half_patch +1)]
            destImage[i, j] = Image[i, j] == np.max(patch)
    return destImage

def detectKeypoints(image,threshold=0.001):
    image = image.astype(np.float32)
    image /= 255.
    h, w = image.shape[:2]
    keypoints = []
    grayImage = rgb_to_gray(image)

    harris,orientation = harriscornerdetector(grayImage)
    # print(harris)
    maxi = LocalMaxima(harris)

    for y in range(h):
        for x in range(w):
            if y < maxi.shape[0] and x < maxi.shape[1] and maxi[y, x]:
                if harris[y,x]>threshold:
                    f = Keypoint((x, y), size=1,orientation=orientation[y,x],magnitude=harris[y,x])
                    keypoints.append(f)

    return keypoints

def resize(img,width=None,height=810):
        """
        Resize the i/p image to specified width and height"""
        # with mpimg.imread as img:
        if width is None and height is None:
            return np.array(img)
            
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height

        if width is None:
            width = int(height * aspect_ratio)
        elif height is None:
            height = int(width / aspect_ratio)

        resized_img = img.resize((width, height), I.LANCZOS)
        return resized_img

def rgb_to_gray(rgb):
    """
    Convert an RGB image to grayscale.

    Args:
    - rgb_image: 3D numpy array representing the RGB image

    Returns:
    - gray_image: 2D numpy array representing the grayscale image
    """
    x = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return x


image_path = "data/Images/Field/1.jpg"
image = cv2.imread(image_path)

# Detect keypoints
keypoints = detectKeypoints(image)

print(len(keypoints))
for keypoint in keypoints:
    cv2.circle(image, keypoint.pt, 3, (0, 0, 255), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()