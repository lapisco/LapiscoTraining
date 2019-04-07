import cv2
import numpy as np

# Read a rgb image
image = cv2.imread('image.png')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply hough transform
circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 30,
                           param1=150, param2=25, minRadius=0, maxRadius=0)

try:
    circles = np.uint16(np.around(circles))
except AttributeError:
    print('None circles found! Try change the parameters.')
    exit()


# Create a copy of the original image to draw the circles
circles_img = np.copy(image)

# Draw all the circles found
for xc, yc, radius in circles[0, :]:
    cv2.circle(circles_img, (xc, yc), radius, (0, 0, 255), 2)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Show the circles found
cv2.imshow('Threshold result', circles_img)
cv2.waitKey(0)
