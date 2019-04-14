import cv2
import os
import glob
import csv


def extract_central_moments(images):
    print('[INFO] Extracting central moments.')
    central_moments = []

    for i, image in enumerate(images):

        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(images)))

        # Load the rgb image
        file = cv2.imread(image)

        # Convert to grayscale
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        # Extract the moments
        moments = cv2.moments(file)

        # Create a list with the features extracted
        central_moments.append([moments['mu20'], moments['mu11'], moments['mu02'], moments['mu30'],
                                moments['mu21'], moments['mu12'], moments['mu03']])

    print('\n')

    return central_moments


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

    # Extract central Moments
    features = extract_central_moments(image_paths)

    # Save the results in a csv file.
    save_results('CentralMoments', features)
