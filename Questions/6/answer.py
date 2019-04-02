import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny filter (change the inferior and superior limit and see the difference)
canny_image = cv2.Canny(grayscale_image, 80, 180)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Show the result of the canny filter
cv2.imshow('Canny filter result', canny_image)

cv2.waitKey(0)

# Save the results
cv2.imwrite('canny_filter_result.jpg', canny_image)
