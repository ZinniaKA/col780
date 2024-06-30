import cv2
import numpy as np

# Load the image
image = cv2.imread('data/img1.png')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # Increased kernel size for stronger blurring
blur2 = cv2.GaussianBlur(blurred, (9, 9),0)  # Increased kernel size for stronger blurring

# Perform edge detection using Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Dilate the edges to connect adjacent edges
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# Find contours in the dilated edges
contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for the contours
mask = np.zeros_like(image)

# Draw contours on the mask
cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Apply the mask to the original image
result = cv2.bitwise_and(image, mask)

# Option 1: Connected components with size filtering
_, labels = cv2.connectedComponents(edges)
num_sutures1 = np.max(labels)  # Adjust based on size filtering

# Option 2: Contour detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_sutures2 = len(contours)  # Adjust based on contour filtering

# Filtering and Refinement:
# Your filtering and refinement code here...

# Print the number of sutures
print("Number of sutures (Option 1):", num_sutures1)
print("Number of sutures (Option 2):", num_sutures2)

# Optional: Visualization
cv2.imshow("Preprocessed Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
