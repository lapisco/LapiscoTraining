import cv2
import numpy as np

# Read a rgb image
image = cv2.imread('image.png')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Canny filter
canny_image = cv2.Canny(grayscale_image, 80, 180)

# Find how many contours are in the image
contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(hierarchy)

# Create a copy of the image to draw the contours
contour_img = np.copy(image)

# Draw all the contours found
cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 3)

# Print the area of each contour, notice that there are areas close to other areas. Therefore, can be considered
# areas of the same object
for i, contour in enumerate(contours):
    print('Area ' + str(i + 1) + ': ' + str(cv2.contourArea(contour)))

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Show the contours found
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
