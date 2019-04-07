import cv2
import numpy as np
from numba import njit

seed = (0, 0)


# @njit
def region_growing(image, seed=None):

    # Get the rows and columns of the image
    rows, cols = image.shape[:2]

    # Get the seed point
    xc, yc = seed

    # Get the reference color
    ref_color = image[xc, yc]

    # Create a matrix that will contain the segmented region
    segmented = np.zeros_like(image)

    # Mark the seed point in the image
    segmented[xc, yc] = ref_color

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
                if np.array_equal(segmented[row, col], ref_color):
                    if np.array_equal(image[row - 1, col - 1], ref_color):
                        segmented[row - 1, col - 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row - 1, col], ref_color):
                        segmented[row - 1, col] = ref_color
                        current_found += 1
                    if np.array_equal(image[row - 1, col + 1], ref_color):
                        segmented[row - 1, col + 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row, col - 1], ref_color):
                        segmented[row, col - 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row, col + 1], ref_color):
                        segmented[row, col + 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row + 1, col - 1], ref_color):
                        segmented[row + 1, col - 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row + 1, col], ref_color):
                        segmented[row + 1, col] = ref_color
                        current_found += 1
                    if np.array_equal(image[row + 1, col + 1], ref_color):
                        segmented[row + 1, col + 1] = ref_color
                        current_found += 1

        cv2.imshow('Segmentation', segmented)
        cv2.waitKey(1)

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

    # This algorithm is not the most fast implementation, but is one of the most easy to understand. So, we will resize
    # the image to execute the code faster. Note: it is not possible to use the numba package, since it not interpret
    # the python form of rgb image required in the question
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    # Create a window, show the original image and wait for the click
    cv2.namedWindow('Original Image', 1)
    cv2.imshow('Original Image', image)
    cv2.setMouseCallback('Original Image', mouse_event)
    cv2.waitKey(0)

    # Apply the region growing algorithm
    segmented_image = region_growing(image, seed)

    # Show the result
    cv2.imshow('Segmented image', segmented_image)
    cv2.waitKey(0)

