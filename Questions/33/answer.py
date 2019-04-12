import cv2
import numpy as np

# Read a rgb image
image = cv2.imread('image.png')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Canny filter
canny_image = cv2.Canny(grayscale_image, 80, 180)

# Find how many contours are in the image. Note the new parameter cv2.RETR_EXTERNAL. Now the function gives only the
# external rects
contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find a polygon approximation for each contour and then find the bounding rect for every polygon
contours_poly = [None] * len(contours)
bound_rect = [None] * len(contours)

for i, contour in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(contour, 2, True)
    bound_rect[i] = cv2.boundingRect(contours_poly[i])


# Create a copy of the image to draw the contours
contour_img = np.copy(image)

# Draw the rectangles for every object
for i, contour in enumerate(contours_poly):
    cv2.rectangle(contour_img, (int(bound_rect[i][0]), int(bound_rect[i][1])),
                  (int(bound_rect[i][0]) + int(bound_rect[i][2]), int(bound_rect[i][1]) + bound_rect[i][3]),
                  (255, 0, 0), 2)

    # Crop each object
    crop = contour_img[int(bound_rect[i][1]):int(bound_rect[i][1]) + bound_rect[i][3],
                       int(bound_rect[i][0]):int(bound_rect[i][0]) + int(bound_rect[i][2])]

    cv2.imshow('Object ' + str(i + 1), crop)
    cv2.waitKey(10)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Show the contours found
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
