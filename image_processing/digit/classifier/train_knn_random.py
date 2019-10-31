# Trains the k-NN classifiers based on the features extracted from the randomly generated digit templates
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import neighbors, model_selection
from image_processing.constants import ROOT_DIR


def train_classifier(data, weights, k):
    """Trains a k-NN classifier using the feature data extracted from the randomly generated digit templates"""

    data = data[data['class'].notnull()]  # loads in the dataset
    X = np.array(data.drop(['class'], 1).iloc[1:])  # features
    y = np.array(data['class'].iloc[1:])  # class labels
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)  # split the data

    if weights == 1:  # create the classfier
        clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="kd_tree")

    else:
        clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights="uniform", algorithm="kd_tree")

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    result = np.ndarray.tolist(clf.predict(X_test))

    n = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ]

    misclassification = {}

    for index, el in enumerate(result):
        if result[index] == y_test[index]:
            n[int(result[index])].append(True)
        else:
            if el in misclassification:
                misclassification[el] += 1
            else:
                misclassification[el] = 1
            n[int(y_test[index])].append(False)

    for i, el in enumerate(n):
        if len(el) > 0:
            print("Accuracy for number %d is %.2f with length %d " % (i, (sum(el) / len(el)), len(el)))

    return clf, y_train


if __name__ == '__main__':

    if not os.path.exists(ROOT_DIR + "\\data\\classifiers"):
        os.mkdir(ROOT_DIR + "\\data\\classifiers")

    if not os.path.exists(ROOT_DIR + "\\data\\result_labels"):
        os.mkdir(ROOT_DIR + "\\data\\result_labels")

    # Run this script to create the classifiers

    fd = [10, 20, 30, 40, 50]  # Fourier descriptors
    n = [5, 7, 9, 11]  # neighbors
    method = [0, 1]  # 0 is uniform weight distribution, 1 is distance weighted k-NN classification

    for q in method:
        for i in fd:
            for j in n:

                data_file = "\\data\\knn_random_templates\\random_FD_%d.csv" % i
                data = pd.read_csv(ROOT_DIR + data_file)

                if q is 0:
                    clf, y_train = train_classifier(data, 0, j)
                    filename = ROOT_DIR + "\\data\\classifiers\\digits_classifier_random_uniform_%dn_%d.joblib.pkl" % (j, i)
                    joblib.dump(clf, filename, compress=9)  # saves the classifier object
                    pd.DataFrame(data=y_train, columns=['training']).to_csv(
                        ROOT_DIR + "\\data\\result_labels\\ytrain_random_uniform_%dN_%d.csv" % (j, i),
                        encoding='utf-8')  # stores the class labels
                else:
                    clf, y_train = train_classifier(data, 1, j)
                    filename = ROOT_DIR + "\\data\\classifiers\\digits_classifier_random_weighted_%dn_%d.joblib.pkl" % (
                    j, i)
                    joblib.dump(clf, filename, compress=9)
                    pd.DataFrame(data=y_train, columns=['training']).to_csv(
                        ROOT_DIR + "\\data\\result_labels\\ytrain_random_weighted_%dN_%d.csv" % (j, i),
                        encoding='utf-8')

