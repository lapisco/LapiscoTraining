import cv2
import numpy as np
import random

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Canny filter
canny_image = cv2.Canny(grayscale_image, 80, 180)

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

rows, cols = image.shape[:2]

# Draw blobs
for k in blobs:

    # Get the coordinates of up_left and bottom_right
    x_up_left = int(k.pt[0] - k.size)
    y_up_left = int(k.pt[1] - k.size)

    x_bottom_right = int(k.pt[0] + k.size)
    y_bottom_right = int(k.pt[1] + k.size)

    # Verify if this coordinates are inside the limits of the image, correct if it is not inside the limits
    if x_up_left < 0:
        x_up_left = 0
    if y_up_left < 0:
        y_up_left = 0

    if x_bottom_right > cols:
        x_bottom_right = cols
    if y_bottom_right > rows:
        y_bottom_right = rows

    # Draw the rectangle
    cv2.rectangle(image, (x_up_left + 15, y_up_left + 15), (x_bottom_right - 15, y_bottom_right - 15), (255, 0, 0), 2)

# Show the result
cv2.imshow('Result', image)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)
