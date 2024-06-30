import cv2
import numpy as np
from PIL import Image as I
import matplotlib.pyplot as plt
import math
import random

print("import success")

class Image:
    def __init__(self,path:str,size:int |None = None) -> None:
        """
        Image constructor
        
        path: path to image
        size: maximum dimension to resize the image"""

        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        if size is not None:
            h, w = self.image.shape[:2]
            if max(w,h)>size:
                if w>h:
                    self.image = self.resize(size,int(h*size /w))
                else:
                    self.image = self.resize(int(w*size /h),size)
        
        self .keypoints = None
        self.features = None
        self.H: np.ndarray = np.eye(3)
        self.component_id: int = 0
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)

    def resize(self,width=None,height=None):
        """
        Resize the i/p image to specified width and height"""
        with I.open(self.path) as img:
            if width is None and height is None:
                return np.array(img)
            
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height

            if width is None:
                width = int(height * aspect_ratio)
            elif height is None:
                height = int(width / aspect_ratio)

            resized_img = img.resize((width, height), I.LANCZOS)
            # Save the resized image
            resized_path = self.path.replace(".jpg", "_resized.jpg")  # Modify the path as needed
            resized_img.save(resized_path)
            # Read the saved resized image using cv2.imread
            resized_image: np.ndarray = cv2.imread(resized_path)
            return resized_image

    def compute_features(self) -> None:
        # """
        # Computethe features and the keypoints of the image using SIFTs"""
        # descriptor = cv2.SIFT_create()
        # keypoints, features = descriptor.detectAndCompute(self.image, None)
        # self.keypoints = keypoints
        # self.features = features
        harris = np.zeros(self.image.shape[:2])
        orientations = np.zeros(self.image.shape[:2])

        i_x = ndimage.filters.sobel(self.image,axis=-1)


    def visualize_keypoints(self) -> None:
        """
        Visualize keypoints on the image"""
        img_with_keypoints = cv2.drawKeypoints(self.image, self.keypoints, None)
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Keypoints")
        plt.show()

    def visualize_features(self) -> None:
        """
        Visualize features on the image"""
        img_with_features = cv2.drawKeypoints(self.image, self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(cv2.cvtColor(img_with_features, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Features")
        plt.show()


image1 = Image("data/Images/Field/1.jpg")
# image2 = Image("data/Images/Field/2.jpg")
# image3 = Image("data/Images/Field/3.jpg")
# image4 = Image("data/Images/Field/4.jpg")
# image5 = Image("data/Images/Field/5.jpg")
# image6 = Image("data/Images/Field/6.jpg")

# Compute features for each image
images = [image1] #, image2, image3, image4, image5, image6]
for image in images:
    image.compute_features()
    image.visualize_keypoints()
    image.visualize_features()
