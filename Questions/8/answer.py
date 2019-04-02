import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Get the rows and columns of the image
rows, cols = grayscale_image.shape[:2]

# Resize the image to be 2 times of its original size
double_sized_image = cv2.resize(grayscale_image, (2 * rows, 2 * cols))

# Resize the image to be half of its original size
half_sized_image = cv2.resize(grayscale_image, (int(rows/2), int(cols/2)))

# Show the result of the threshold
cv2.imshow('Double sized image', double_sized_image)
cv2.imshow('Half sized image', half_sized_image)

cv2.waitKey(0)

# Save the results
cv2.imwrite('double_sized_image.jpg', double_sized_image)
cv2.imwrite('half_sized_image.jpg', half_sized_image)
