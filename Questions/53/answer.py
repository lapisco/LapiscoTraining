import numpy as np
import math
import pandas as pd
import operator


def hold_out(df, train_size, shuffle=True):

    # Shuffle the dataframe if the shuffle is set to true
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Convert the rows of the dataframe into a list of lists
    data = []
    for row in df.iterrows():
        index, values = row
        data.append(values.tolist())

    # Split the data into train and test
    X_train = data[:int(train_size*len(data))]
    X_test = data[int(train_size*len(data)):]

    # Get the correspondent labels to each feature vector
    y_train = [int(x[-1]) for x in X_train]
    y_test = [int(x[-1]) for x in X_test]

    # Remove the labels from the train and test vectors
    X_train = [x[:-1] for x in X_train]
    X_test = [x[:-1] for x in X_test]

    return X_train, X_test, y_train, y_test


def leave_one_out(df, shuffle=True):

    # Shuffle the dataframe if the shuffle is set to true
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Convert the rows of the dataframe into a list of lists
    data = []
    for row in df.iterrows():
        index, values = row
        data.append(values.tolist())

    # Create a list of lists, in which each iteration of leave one out will be stored
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(len(data)):
        train = data.copy()
        train.remove(data[i])

        test = data[i]

        # Get the correspondent labels to each feature vector
        y_train.append([int(x[-1]) for x in train])
        y_test.append(int(test[-1]))

        # Remove the labels from the train and test vectors
        X_train.append([x[:-1] for x in train])
        X_test.append(test[:-1])

    return X_train, X_test, y_train, y_test


def read_data(file):
    # Load the features of a file in a dataframe
    return pd.read_csv(file, sep=',', header=None)


def euclidean_dist(x1, x2):
    dist = 0.0
    for x, y in zip(x1, x2):
        dist += pow(float(x) - float(y), 2)
    dist = math.sqrt(dist)

    return dist


def knn_clf(X_train, X_test, y_train, k_neighbors=3):
    assert (k_neighbors % 2), 'Number of neighbors must be odd!'

    predict = []
    for x1 in X_test:
        class_prediction = np.zeros(max(y_train) + 1)
        euclidean_distance = []

        for x2, label2 in zip(X_train, y_train):
            eu_dist = euclidean_dist(x1, x2)
            euclidean_distance.append((label2, eu_dist))
            euclidean_distance.sort(key=operator.itemgetter(1))
            smaller_k_distances = euclidean_distance[:k_neighbors]

            for label, dist in smaller_k_distances:
                class_prediction[int(label)] += 1

        predict.append(max(range(len(class_prediction)), key=class_prediction.__getitem__))

    return predict


if __name__ == '__main__':

    # Read the file with the features to classify
    filename = 'features.txt'
    features = read_data(filename)

    # Split the database using hold out
    X_train, X_test, y_train, y_test = hold_out(features, train_size=0.8)

    # Apply knn (you can change the number of neighbors)
    predictions = knn_clf(X_train, X_test, y_train, k_neighbors=7)

    # Calculates the accuracy using hold out
    count = 0
    for x, y in zip(y_test, predictions):
        if x == y:
            count += 1

    accuracy = count/len(y_test)
    print('Accuracy using hold out: {:.4f}'.format(accuracy))

    # Split the database using leave one out
    X_train, X_test, y_train, y_test = leave_one_out(features)

    # Apply knn (you can change the number of neighbors)
    predictions = np.zeros(int(max(y_train[0])) + 1)
    count = 0
    for train_set, test_set, label_train, label_test in zip(X_train, X_test, y_train, y_test):

        test_list = []
        test_list.append(test_set)
        predict = knn_clf(train_set, test_list, label_train, k_neighbors=7)
        predictions[predict] += 1

        if predict[0] == label_test:
            count += 1

    accuracy = count / len(y_test)

    print('Accuracy using leave one out: {:.4f}'.format(accuracy))
