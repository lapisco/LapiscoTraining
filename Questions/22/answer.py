import cv2
import numpy as np
from numba import njit

seed = (0, 0)

@njit
def region_growing(image, seed=None):

    # Get the rows and columns of the image
    rows, cols = image.shape[:2]

    # Get the seed point
    xc, yc = seed

    # Create a matrix that will contain the segmented region
    segmented = np.zeros_like(image)

    # Mark the seed point in the image
    segmented[xc, yc] = 255

    # Loop through the image until the region stop growing
    current_found = 0
    previous_points = 1

    while previous_points != current_found:

        previous_points = current_found
        current_found = 0
        for row in range(rows):
            for col in range(cols):
                # Verify if we reach the ROI and search through the neighborhood to see if the pixel is of the same
                # object, then if the pixel is part of the object put them in the segmented image
                if segmented[row, col] == 255:
                    if image[row - 1, col - 1] < 127:
                        segmented[row - 1, col - 1] = 255
                        current_found += 1
                    if image[row - 1, col] < 127:
                        segmented[row - 1, col] = 255
                        current_found += 1
                    if image[row - 1, col + 1] < 127:
                        segmented[row - 1, col + 1] = 255
                        current_found += 1
                    if image[row, col - 1] < 127:
                        segmented[row, col - 1] = 255
                        current_found += 1
                    if image[row, col + 1] < 127:
                        segmented[row, col + 1] = 255
                        current_found += 1
                    if image[row + 1, col - 1] < 127:
                        segmented[row + 1, col - 1] = 255
                        current_found += 1
                    if image[row + 1, col] < 127:
                        segmented[row + 1, col] = 255
                        current_found += 1
                    if image[row + 1, col + 1] < 127:
                        segmented[row + 1, col + 1] = 255
                        current_found += 1

    return segmented


def mouse_event(event, x, y, flags, param):
    # Verify if the left button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed

        # Update the seed point
        seed = (y, x)


def get_centroid(image):

    # Initialize the centroid
    xc, yc = 0, 0

    rows, cols = image.shape[:2]
    count = 0
    # Loop through the image and find the points of the square
    for row in range(rows):
        for col in range(cols):
            if image[row, col] == 255:
                xc += row
                yc += col
                count += 1

    # Calculates the mean point
    xc = int(xc / count)
    yc = int(yc / count)

    return xc, yc


if __name__ == '__main__':
    # Read a rgb image
    image = cv2.imread('image.jpg')

    # Transform to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a window, show the original image and wait for the click
    cv2.namedWindow('Original Image', 1)
    cv2.imshow('Original Image', grayscale_image)
    cv2.setMouseCallback('Original Image', mouse_event)
    cv2.waitKey(0)

    # Apply the region growing algorithm
    segmented_image = region_growing(grayscale_image, seed)

    # With the segmented image we can calculate the centroid
    xc, yc = get_centroid(segmented_image)

    # Create a rgb image
    rows, cols = segmented_image.shape[:2]
    new_img = np.zeros([rows, cols, 3], np.uint8)

    # Paint the segmented region as blue
    new_img[np.where(segmented_image == 255)] = [255, 0, 0]

    # Draw a circle in the centroid of the segmented region
    cv2.circle(new_img, (yc, xc), 5, (0, 255, 0), -1)

    # Show the result
    cv2.imshow('Segmented image', new_img)
    cv2.waitKey(0)
