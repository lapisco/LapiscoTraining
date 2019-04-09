import cv2
import matplotlib.pyplot as plt

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)

# Apply the threshold
ret, threshold_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Show the result of the threshold
cv2.imshow('Threshold image', threshold_image)

# Create the structuring element
structuring_elements = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]


# Apply erosion and show the results for each structuring element
plt.figure(1)
titles = ['Rectangle element', 'Elliptic element', 'Cross element']
for i, element in enumerate(structuring_elements):
    kernel = cv2.getStructuringElement(element, (5, 5))
    erosion = cv2.erode(threshold_image, kernel, iterations=7)

    fig = 130 + (i+1)
    plt.subplot(fig)
    plt.title(titles[i])
    plt.imshow(erosion, cmap='gray')

plt.show()
cv2.waitKey(0)
