import numpy as np
import cv2
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image as I
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse

class Keypoint:
    def __init__(self, pt, size):
        self.pt = pt  # (x, y) coordinates
        self.size = size

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
    # print("image_")
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
    # cv2.imshow('i_x', i_x)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    i_y = convolve(Image, SOBEL_Y)
    # cv2.imshow('i_y', i_y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    i_x_sqr = i_x**2
    # cv2.imshow('i_x_sqr', i_x_sqr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    i_y_sqr = i_y**2
    # cv2.imshow('i_y_sqr', i_y_sqr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    i_x_times_i_y = i_x * i_y
    # cv2.imshow('i_x', i_x_times_i_y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 2: Apply Gaussian mask
    s = 0.9  # Sigma
    G=31
    trncate=G/(s*2)
    sumix2 = convolve(i_x_sqr, GAUSS)
    # cv2.imshow('i_x', sumix2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    sumiy2 = convolve(i_y_sqr, GAUSS)
    # cv2.imshow('i_x', sumiy2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    sumixiy2 = convolve(i_x_times_i_y,GAUSS)
    # cv2.imshow('i_x', sumixiy2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Step 3: Compute Harris score
    alpha = 0.01
    det = sumix2*sumiy2 - sumixiy2**2
    # print(det)
    trace = sumix2 + sumiy2
    # print(trace)
    harris = det - alpha*(trace**2)
    # print(harris)
    # orientations = np.degrees(np.arctan2(i_y.flatten(), i_x.flatten()).reshape(orientations.shape))

    return harris#, orientations

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

    # resized_image = resize(image)
    # cv2.imshow("gray",resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    grayImage = rgb_to_gray(image)
    # cv2.imshow("gray",grayImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    harris = harriscornerdetector(grayImage)
    print(harris)
    maxi = LocalMaxima(harris)

    for y in range(h):
        for x in range(w):
            if y < maxi.shape[0] and x < maxi.shape[1] and maxi[y, x]:
                if harris[y,x]>threshold:
                    f = Keypoint((x, y), size=1)
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
        #     # Save the resized image
        # resized_path = self.path.replace(".jpg", "_resized.jpg")  # Modify the path as needed
        # resized_img.save(resized_path)
        #     # Read the saved resized image using cv2.imread
        # resized_image: np.ndarray = cv2.imread(resized_path)
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


# Function to process images in a folder
def process_images_in_folder(folder_path):
    # List all image files in the folder
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]

    # Iterate over each image file
    for image_path in image_files:
        # Load the image
        image = cv2.imread(image_path)
        
        # Detect keypoints
        keypoints = detectKeypoints(image)
        
        # Visualize keypoints
        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), 3, (0, 0, 255), -1)
        
        # Display the image with keypoints
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Detect and visualize keypoints in images in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")

    # Parse the arguments
    args = parser.parse_args()

    # Process images in the specified folder
    process_images_in_folder(args.folder_path)