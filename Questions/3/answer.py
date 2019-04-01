import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Separate the rgb channels
blue_channel, green_channel, red_channel = cv2.split(image)

# Show the channels individually
cv2.imshow('Blue Channel', blue_channel)
cv2.imshow('Green Channel', green_channel)
cv2.imshow('Red Channel', red_channel)

cv2.waitKey(0)

# Save the results
cv2.imwrite('blue_channel.jpg', blue_channel)
cv2.imwrite('green_channel.jpg', green_channel)
cv2.imwrite('red_channel.jpg', red_channel)
