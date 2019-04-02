import cv2
import numpy as np

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)

# Get the rows and columns of the image
rows, cols = grayscale_image.shape[:2]

# Create a matrix of zeros
threshold_matrix = np.zeros((rows, cols), dtype=np.uint8)

for row in range(rows):
    for col in range(cols):
        threshold_matrix[row, col] = grayscale_image[row, col]

# Save all pixels in a txt file
with open('result.txt', 'w') as outfile:
    for row in range(rows):
        for col in range(cols):
            # Define the limits of the threshold
            if threshold_matrix[row, col] < 127:
                threshold_matrix[row, col] = 0
            else:
                threshold_matrix[row, col] = 255

            outfile.write(str(threshold_matrix[row, col]) + ' ')
        outfile.write('\n')
