import cv2
import numpy as np

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Canny filter
canny_image = cv2.Canny(grayscale_image, 80, 180)

# Create a blob detector
# detector = cv2.SimpleBlobDetector_create()

# Define the parameters
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 20
params.maxArea = 40000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.8

# Distance Between Blobs
params.minDistBetweenBlobs = 20

# Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect objects
blobs = detector.detect(canny_image)

# Print how many objects are in the image
print(len(blobs))

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)