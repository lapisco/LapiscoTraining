import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Get the names of the files in a list
files = ['true_and_predict_' + str(num) + '.csv' for num in range(53, 59)]

# Loop through files
for file in files:
    print(file, '\n\n')
    with open(file, 'r') as infile:
        # Read the csv file
        read = csv.reader(infile, delimiter=',')

        # Get the data from the read files
        data = []
        for row in read:
            data.append(row)

        # Create a list for true and predict labels
        true = [int(x) for x in data[0]]
        pred = [int(x) for x in data[1]]

        # Get the confusion matrix
        cm = confusion_matrix(true, pred)

        # Calculate the accuracy
        print('Accuracy: {}'.format(accuracy_score(true, pred)))

        # Show the classification report (precision, recall, f1-score,)
        print('Classification report: \n', classification_report(true, pred))

        print('\n\n###############################################')
