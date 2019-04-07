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
                    if 130 < image[row - 1, col - 1] < 230:
                        segmented[row - 1, col - 1] = 255
                        current_found += 1
                    if 130 < image[row - 1, col] < 230:
                        segmented[row - 1, col] = 255
                        current_found += 1
                    if 130 < image[row - 1, col + 1] < 230:
                        segmented[row - 1, col + 1] = 255
                        current_found += 1
                    if 130 < image[row, col - 1] < 230:
                        segmented[row, col - 1] = 255
                        current_found += 1
                    if 130 < image[row, col + 1] < 230:
                        segmented[row, col + 1] = 255
                        current_found += 1
                    if 130 < image[row + 1, col - 1] < 230:
                        segmented[row + 1, col - 1] = 255
                        current_found += 1
                    if 130 < image[row + 1, col] < 230:
                        segmented[row + 1, col] = 255
                        current_found += 1
                    if 130 < image[row + 1, col + 1] < 230:
                        segmented[row + 1, col + 1] = 255
                        current_found += 1

    return segmented


def mouse_event(event, x, y, flags, param):
    # Verify if the left button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed

        # Update the seed point
        seed = (y, x)


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

    # Show the result
    cv2.imshow('Segmented image', segmented_image)
    cv2.waitKey(0)

