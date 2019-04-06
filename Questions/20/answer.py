import cv2
import numpy as np
from numba import njit


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


if __name__ == '__main__':
    # Read a rgb image
    image = cv2.imread('image.jpg')

    # Transform to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    segmented_image = region_growing(grayscale_image,
                                     seed=(int(grayscale_image.shape[0]/2), int(grayscale_image.shape[1]/2)))

    cv2.imshow('Segmented image', segmented_image)
    cv2.waitKey(0)

