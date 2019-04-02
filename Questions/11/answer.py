import cv2
import numpy as np

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Get the rows and columns of the image
rows, cols = grayscale_image.shape[:2]

# Create a matrix of zeros
new_image = np.zeros((rows, cols), dtype=np.uint8)

# Copy the grayscale image to the new_image
for row in range(rows):
    for col in range(cols):
        new_image[row, col] = grayscale_image[row, col]


# Initialize the centroid coordinates
xc = 0
yc = 0
count = 0

# Loop through the image and find the points of the square
for row in range(rows):
    for col in range(cols):
        if new_image[row, col] == 0:
            xc += row
            yc += col
            count += 1

# Calculates the mean point
xc = int(xc/count)
yc = int(yc/count)

# Draw a circle in the centroid of the square
cv2.circle(new_image, (xc, yc), 5, (255, 255, 255), -1)

# Show the centroid
cv2.imshow('Centroid', new_image)
cv2.waitKey(0)

# Save the results
cv2.imwrite('centroid.jpg', new_image)
