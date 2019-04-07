import cv2
import numpy as np
import random
from numba import njit


@njit
def region_growing(image):

    # Get the rows and columns of the image
    rows, cols = image.shape[:2]

    # Create a matrix that will contain the segmented region
    segmented = np.zeros_like(image)

    # Create a variable to store the number of objects found
    num_objects = 0

    for ext_row in range(rows):
        for ext_col in range(cols):
            if segmented[ext_row, ext_col] == 0 and image[ext_row, ext_col] < 230:
                num_objects += 1
                segmented[ext_row, ext_col] = num_objects

                # Loop through the image until the region stop growing
                current_found = 0
                previous_points = 1

                while previous_points != current_found:

                    previous_points = current_found
                    current_found = 0

                    for row in range(rows):
                        for col in range(cols):
                            # Verify if we reach the ROI and search through the neighborhood to see if the pixel is of
                            # the same object, then if the pixel is part of the object put them in the segmented image
                            if segmented[row, col] == num_objects:
                                if image[row - 1, col - 1] < 230:
                                    segmented[row - 1, col - 1] = num_objects
                                    current_found += 1
                                if image[row - 1, col] < 230:
                                    segmented[row - 1, col] = num_objects
                                    current_found += 1
                                if image[row - 1, col + 1] < 230:
                                    segmented[row - 1, col + 1] = num_objects
                                    current_found += 1
                                if image[row, col - 1] < 230:
                                    segmented[row, col - 1] = num_objects
                                    current_found += 1
                                if image[row, col + 1] < 230:
                                    segmented[row, col + 1] = num_objects
                                    current_found += 1
                                if image[row + 1, col - 1] < 230:
                                    segmented[row + 1, col - 1] = num_objects
                                    current_found += 1
                                if image[row + 1, col] < 230:
                                    segmented[row + 1, col] = num_objects
                                    current_found += 1
                                if image[row + 1, col + 1] < 230:
                                    segmented[row + 1, col + 1] = num_objects
                                    current_found += 1

    return segmented, num_objects


if __name__ == '__main__':
    # Read a rgb image
    image = cv2.imread('image.jpg')

    # Transform to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original', grayscale_image)

    # Apply the region growing algorithm
    segmented_image, n_objects = region_growing(grayscale_image)

    # Create a rgb image
    rows, cols = segmented_image.shape[:2]
    new_img = np.zeros([rows, cols, 3], np.uint8)

    # Paint the segmented region
    for n in range(n_objects):
        # create a lambda function to generate new colors for each object
        color = lambda: random.randint(0, 255)

        # Find the extreme coordinates of each object
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0

        for row in range(rows):
            for col in range(cols):
                if segmented_image[row, col] == n + 1:
                    if x_max < col:
                        x_max = col
                    if x_min > col:
                        x_min = col
                    if y_max < row:
                        y_max = row
                    if y_min > row:
                        y_min = row

                    # If is the first pixel found, then the min coordinates must be equal to the max coordinates
                    if x_min == 0:
                        x_min = x_max
                    if y_min == 0:
                        y_min = y_max

        # Paint the object with a random color
        new_img[np.where(segmented_image == n + 1)] = [color(), color(), color()]

        # Crop the object in the coordinates found
        crop_image = new_img[y_min:y_max, x_min:x_max]

        # Show each object individually
        cv2.imshow('Object ' + str(n), crop_image)
        cv2.waitKey(10)

    cv2.waitKey(0)

