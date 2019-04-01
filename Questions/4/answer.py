import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_image)

# Show the HSV image and its channels
cv2.imshow('HSV Image', hsv_image)

cv2.imshow('H Channel', h)
cv2.imshow('S Channel', s)
cv2.imshow('V Channel', v)

cv2.waitKey(0)

# Save the results
cv2.imwrite('hsv_image.jpg', hsv_image)
cv2.imwrite('h_channel.jpg', h)
cv2.imwrite('s_channel.jpg', s)
cv2.imwrite('v_channel.jpg', v)