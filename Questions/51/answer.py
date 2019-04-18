import cv2

# Read the rgb images
image = cv2.imread('image.jpg')
logo = cv2.imread('logo.jpg')

# Convert the images to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

# Create a SIFT detector (Obs.: in order to use sift descriptor you will have to install opencv-contrib
# from scratch. An alternative to that is to use opencv implementation called orb, it has almost the
# same result. To use orb, comment the lines with sift and uncomment the lines with orb.)
sift = cv2.xfeatures2d.SIFT_create()
# orb = cv2.ORB_create()

# Find the keypoints and descriptors with SIFT
keypoints_1, descriptors_1 = sift.detectAndCompute(image, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(logo, None)

# keypoints_1, descriptors_1 = orb.detectAndCompute(image, None)
# keypoints_2, descriptors_2 = orb.detectAndCompute(logo, None)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors_1, descriptors_2)

# Draw only the first match
result = cv2.drawMatches(image, keypoints_1, logo, keypoints_2, matches[:2], None, flags=2)

# Show the result
cv2.imshow('Result', result)
cv2.waitKey(0)

# Save the result
cv2.imwrite('sift_result.jpg', result)
