import cv2
import numpy as np

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Laplacian Filter
laplace = cv2.Laplacian(grayscale_image, ddepth=cv2.CV_16S, ksize=3)

# Convert to uint8
laplace = cv2.convertScaleAbs(laplace)

# Equalize the image of the laplacian filter
equalized_laplacian = cv2.equalizeHist(laplace)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Show the result of the laplacian filter
cv2.imshow('Laplacian filter result', laplace)

# Show the result of the equalized image of the laplacian filter
cv2.imshow('Equalized Laplacian', equalized_laplacian)

cv2.waitKey(0)
