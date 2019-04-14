import cv2
import os
import glob
import csv
import numpy as np
from skimage import feature


def extract_lbp(images, number_points, radius, eps=1e-7):
    print('[INFO] Extracting LBP.')
    lbp_features = []

    for i, image in enumerate(images):

        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(images)))

        # Load the rgb image
        file = cv2.imread(image)

        # Convert to grayscale
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        # Extract lbp
        lbp = feature.local_binary_pattern(file, number_points, radius, method='uniform')

        # Calculates the histogram of the lbp image
        hist, ret = np.histogram(lbp.ravel(), bins=np.arange(0, number_points + 3), range=(0, number_points + 2))

        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        # Create the feature vector extracted by lbp
        image_lbp = [item for item in list(hist)]

        lbp_features.append(image_lbp)

    print('\n')

    return lbp_features


def save_results(extractor_name, features):

    # Show the extracted features in command prompt
    for vector in features:
        print(vector)

    # Save all features in a csv file
    with open(extractor_name + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(features)


if __name__ == '__main__':

    # Inform the path to the rgb images
    dataset = 'dataset/'

    # Grab all the paths to the images with extension .jpg
    image_paths = glob.glob(os.path.join(dataset, '*.jpg'))

    # Extract LBP
    features = extract_lbp(image_paths, number_points=24, radius=8)

    # Save the results in a csv file.
    save_results('LBP', features)
