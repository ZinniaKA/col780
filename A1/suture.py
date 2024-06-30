import cv2
import numpy as np

def count_sutures(image_path):
    # Read the image
    image = cv2.imread(image_path)

   # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    # Enhance contrast using histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Perform adaptive thresholding to separate background and foreground
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Canny edge detection
    edges = cv2.Canny(thresholded, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image (for visualization)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Count the number of contours (sutures)
    num_sutures = len(contours)

    # Display the image with contours (for visualization)
    cv2.imshow("Image with Contours", image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return num_sutures

# Path to the input image
image_path = "data/img1.png"

# Count the number of sutures
num_sutures = count_sutures(image_path)
print("Number of micro-sutures:", num_sutures)
