import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create one vector to contain the original histogram and other to contain the equalized histogram
original_hist = np.zeros([256], np.uint8)
equalized_hist = np.zeros([256], np.uint8)

# Calculating the initial histogram
# rows, cols = grayscale_image.shape[:2]

img_flat = grayscale_image.flatten()

for pixel in img_flat:
    original_hist[pixel] += 1


# Calculates the cumulative distribution function of the histogram
cdf = [sum(original_hist[:i + 1]) for i in range(len(original_hist))]
cdf = np.array(cdf)

# Normalize the cdf to be between 0-255
normal_cdf = ((cdf - cdf.min())*255)/(cdf.max() - cdf.min())
normal_cdf = normal_cdf.astype('uint8')


equalized_image = normal_cdf[img_flat]

equalized_image = np.reshape(equalized_image, grayscale_image.shape)

plt.figure(1)
plt.subplot(221)
plt.imshow(grayscale_image, cmap='gray')
plt.subplot(222)
plt.hist(grayscale_image.ravel(), 256, [0, 256])
plt.subplot(223)
plt.imshow(equalized_image, cmap='gray')
plt.subplot(224)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.show()

# TODO - Try to get a better result


# # Normalize the histogram
# original_hist = np.array(original_hist)/(rows*cols)
#
# # Calculates the cumulative distribution function of the histogram
# cdf = [sum(original_hist[:i + 1]) for i in range(len(original_hist))]
#
# # Finding the coefficient values to apply a linear transformation
# coef_linear_transform = [round(x*255) for x in cdf]
#
# # pr = np.zeros([256], float)
# # for i in range(256):
# #     cdf[i] =
#
#
# print(coef_linear_transform)
#
# # Create a new matrix to the equalized image
# equalized_image = np.zeros_like(grayscale_image)
#
# # Apply the linear transform with the coefficient values founded
# for row in range(rows):
#     for col in range(cols):
#         equalized_image[row, col] = coef_linear_transform[grayscale_image[row, col]]
#         # print(equalized_image[row, col])
#         # print(grayscale_image[row, col])
#
# # Calculate the histogram of the equalized image
#
# for i in range(256):
#     for j in range(256):
#         if coef_linear_transform[i] == j:
#             equalized_hist[j] += original_hist[i]
#
#
#
# # for row in range(rows):
# #     for col in range(cols):
# #         equalized_hist[equalized_image[row, col]] += 1
#
# # Normalize the histogram
# equalized_hist = np.array(equalized_hist)/(rows*cols)






