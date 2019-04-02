import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the threshold
ret, threshold_image = cv2.threshold(grayscale_image, 70, 255, cv2.THRESH_BINARY)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Show the result of the threshold
cv2.imshow('Canny filter result', threshold_image)

cv2.waitKey(0)

# Save the results
cv2.imwrite('canny_filter_result.jpg', threshold_image)
