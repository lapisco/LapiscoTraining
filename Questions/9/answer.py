import cv2

# Read a rgb image
image = cv2.imread('image.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the input image
cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)

# Get the rows and columns of the image
rows, cols = grayscale_image.shape[:2]

# Save all pixels in a txt file
with open('result.txt', 'w') as outfile:
    for row in range(rows):
        for col in range(cols):
            outfile.write(str(grayscale_image[row, col]) + ' ')
        outfile.write('\n')



