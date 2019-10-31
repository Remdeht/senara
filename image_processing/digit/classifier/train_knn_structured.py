import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import neighbors
from image_processing.constants import ROOT_DIR
from image_processing.digit.digit_templates.structured_digit_templates import split_templates
import dataset

DATA_DIR_CLASSES = ROOT_DIR + "\\data\\structured_templates"


def classifier(X_train, y_train, X_test, y_test, n, meta, dir, method):
    """Creates and tests the accuracy of the classifier based on the subclasses"""
    if method is 0:
        clf = neighbors.KNeighborsClassifier(n_neighbors=n, weights="uniform", algorithm="kd_tree")
    else:
        clf = neighbors.KNeighborsClassifier(n_neighbors=n, weights="distance", algorithm="kd_tree")

    clf.fit(X_train, y_train)
    result = np.ndarray.tolist(clf.predict(X_test))
    meta_vals = meta[:, 1]

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
        result[index] = meta_vals[el]
        if result[index] == y_test[index]:
            n[result[index]].append(True)
        else:
            if el in misclassification:
                misclassification[el] += 1
            else:
                misclassification[el] = 1
            n[y_test[index]].append(False)

    for i, el in enumerate(n):
        if len(el) > 0:
            print("Accuracy for number %d is %.2f with length %d " % (i, (sum(el) / len(el)), len(el)))

    result_bin = result == y_test
    result_bin_higher = result == (y_test - 1)

    acc = sum(result_bin) / len(result_bin)

    print(acc)
    return acc


def classifier_final(X_train, y_train, X_test, y_test, n, method):
    """Creates and tests the accuracy of the classifier"""
    if method is 0:
        clf = neighbors.KNeighborsClassifier(n_neighbors=n, weights="uniform", algorithm="kd_tree")
    else:
        clf = neighbors.KNeighborsClassifier(n_neighbors=n, weights="distance", algorithm="kd_tree")

    clf.fit(X_train, y_train)
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
            n[result[index]].append(True)
        else:
            if el in misclassification:
                misclassification[el] += 1
            else:
                misclassification[el] = 1
            n[y_test[index]].append(False)

    for i, el in enumerate(n):
        if len(el) > 0:
            print("Accuracy for number %d is %.2f with length %d " % (i, (sum(el) / len(el)), len(el)))

    print(clf.score(X_test, y_test))
    return (clf.score(X_test, y_test)), clf, y_train


def trim_dataset(df, df_ud, n, meta, dir, method):
    X_train = np.array(df.drop(['class', 'digit class'], 1))
    y_train = np.array(df['class'])
    y_final = np.array(df['digit class'])
    X_ud = np.array(df_ud.drop(['class'], 1).iloc[1:])
    y_ud = np.array(df_ud['class'].iloc[1:])

    acc = classifier(X_train, y_train, X_ud, y_ud, n, meta, dir, method)
    acc_final, clf, y_train = classifier_final(X_train, y_final, X_ud, y_ud, n, method)

    return acc, acc_final, clf, y_train


def train_classifier(df, df_ud, meta, fd, n, method):
    """trains the classifer"""
    acc, acc_final, clf, y_train = trim_dataset(df, df_ud, n, meta, DATA_DIR_CLASSES, method)
    if acc < .85:
        return dataset.create_knn_dataset_structured_templates(fd), clf, y_train, acc
    else:
        print("Final Result: Accuracy %.2f for FD: %d and KNN: %d" % (acc_final, fd, n))
        if method is 0:
            filename = ROOT_DIR + "\\data\\classifiers\\digits_classifier_uniform_%dn_%d.joblib.pkl" % (n, fd)
        else:
            filename = ROOT_DIR + "\\data\\classifiers\\digits_classifier_weighted_%dn_%d.joblib.pkl" % (n, fd)

        pd.DataFrame(data=y_train,  columns=['training']).to_csv(
            ROOT_DIR + "\\data\\result_labels\\ytrain_%dn_%d.csv" % (n, fd), encoding='utf-8')
        joblib.dump(clf, filename, compress=9)
        return True


if __name__ == '__main__':
    # run this to train classfiers using the digit templates

    fd = [10, 20, 30, 40, 50]
    n = [5, 7, 9, 11]
    method = [0,1]
    split_templates()
    for q in method:
        for i in fd:
            for j in n:
                print("Fourier Descriptors: %d & K-NearestNeighbors:  %d" % (i, j))
                _ = dataset.create_knn_dataset_structured_templates(i)
                df_filename = "\\data\\knn_structured_templates\\structured_FD_%d.csv" % i
                df_ud_filename = "\\data\\knn_random_templates\\random_FD_%d.csv" % i

                # Fourier Descriptors based on the fft only
                df = pd.read_csv(ROOT_DIR + df_filename)
                df_ud = pd.read_csv(ROOT_DIR + df_ud_filename)
                meta = np.array(pd.read_csv(ROOT_DIR + "\\data\\knn_structured_templates\\fourier_classifiers_meta.csv"))

                z = 0
                b, clf, y_train, acc = train_classifier(df, df_ud, meta, i, j, q)

                best_clf = [clf, y_train, acc]

                while b is False and z < 0:
                    z += 1
                    df = pd.read_csv(ROOT_DIR + df_filename)
                    meta = np.array(pd.read_csv(
                        ROOT_DIR + "\\data\\knn_structured_templates\\fourier_classifiers_meta.csv"))
                    b, clf, y_train, acc = train_classifier(df, df_ud, meta, i, j, q)

                    if acc > best_clf[2]:
                        best_clf = (clf, y_train, acc)

                if z is 0:
                    if q is 0:
                        filename = ROOT_DIR + "\\data\\classifiers\\digits_classifier_uniform_%dn_%d.joblib.pkl" % (j, i)
                    else:
                        filename = ROOT_DIR + "\\data\\classifiers\\digits_classifier_weighted_%dn_%d.joblib.pkl" % (j, i)
                    pd.DataFrame(data=best_clf[1], columns=['training']).to_csv(
                        ROOT_DIR + "\\data\\result_labels\\ytrain_%dn_%d.csv" % (j, i), encoding='utf-8')
                    joblib.dump(best_clf[0], filename, compress=9)
                    print("Final Result: Accuracy %.2f for FD: %d and KNN: %d" % (best_clf[2], i, j))




