import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the median and blur filters
median_image = cv2.medianBlur(grayscale_image, ksize=5)
blur_image = cv2.blur(grayscale_image, ksize=(5, 5))

# Show the HSV image and its channels
cv2.imshow('Median filter result', median_image)
cv2.imshow('Blur filter result', blur_image)

cv2.waitKey(0)

# Save the results
cv2.imwrite('median_filter_result.jpg', median_image)
cv2.imwrite('blur_filter_result.jpg', blur_image)
