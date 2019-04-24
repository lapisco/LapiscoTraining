import pandas as pd
from sklearn.naive_bayes import GaussianNB


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


if __name__ == '__main__':

    # Read the file with the features to classify
    filename = 'features.txt'
    features = read_data(filename)

    # Split the database using hold out
    X_train, X_test, y_train, y_test = hold_out(features, train_size=0.9)

    # Create a naive-bayes object
    bayes = GaussianNB()

    # Train the model
    bayes.fit(X_train, y_train)

    # Evaluate in the test data
    predictions = bayes.predict(X_test)

    # Convert the results to a list
    predictions = list(predictions)

    # Calculates the accuracy using hold out
    count = 0
    for x, y in zip(y_test, predictions):
        if x == y:
            count += 1

    accuracy = count/len(y_test)
    print('Accuracy using hold out: {:.4f}'.format(accuracy))

    # Split the database using leave one out
    X_train, X_test, y_train, y_test = leave_one_out(features)

    # Apply SVM
    print('[INFO] Starting leave one out training...')
    count = 0
    sample_count = 0
    for train_set, test_set, label_train, label_test in zip(X_train, X_test, y_train, y_test):
        print('[INFO] Training sample {}/{}'.format(sample_count+1, len(y_train)))
        sample_count += 1

        # Create a naive-bayes object
        bayes = GaussianNB()

        # Train the model
        bayes.fit(train_set, label_train)

        # Evaluate in the test data
        new_list = []
        new_list.append(test_set)
        prediction = bayes.predict(new_list)

        if prediction == label_test:
            count += 1

    accuracy = count / len(y_test)

    print('Accuracy using leave one out: {:.4f}'.format(accuracy))
