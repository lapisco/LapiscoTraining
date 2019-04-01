import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Show a rgb image
cv2.imshow('Image', image)
cv2.waitKey(0)

# Save the loaded image
cv2.imwrite('saved_image', image)
