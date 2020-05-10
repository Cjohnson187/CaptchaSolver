"""
Chris Johnson
Midterm HW
CS 3120-001 Machine learning
3/29/2020

Program to run 3 different classifiers for the mnst data set.
I added a prompt so i could run each without editing and re running
the program.

Classifiers-
model1   svm.SVC(kernel='poly', C=C,gamma=Gamma)
model2  LogisticRegression()
model3  DecisionTreeClassifier()

Sources -
data sets- train.csv
used 7,000 / 42,000
70 train  /   30 test split  no validation


resources-
class resource- 8_SVM1_midterm.pdf
class demo- 5_SVM_basic(1).py
class demo- 5_SVM_Kernels_iris(1).py
class demo- 5_SVM_MNISTcsv_demo(1).py
"""

import numpy as np
import pandas as pd
import sys

from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

import sklearn.svm as svm                               #model resource
from sklearn.tree import DecisionTreeClassifier         #model resource
from sklearn.linear_model import LogisticRegression     #model resource


# ******************************************************************
def classify(user_input):

    print("Importing and splitting data...\n")
    # import
    ordered_data = np.array(pd.read_csv('train.csv'))
    X = ordered_data[0:7000, 1:]
    y = ordered_data[0:7000, 0]

    # split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, train_size=0.70,
                                                                        random_state=0)

    # ******************************************************************
    # support vector machine
    if user_input == "1":
        print("Starting SVM...\n\n")

        gamma = 0.001
        c = 1  # 0.001
        model = svm.SVC(kernel='poly', C=c, gamma=gamma)

        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(classification_report(y_test, y_predict))

    # ******************************************************************
    # Logistic regression
    elif user_input == "2":
        print("Starting Logistic Regression...\n\n")

        # max iterations reached maybe try pre-processing data
        X_scaled = preprocessing.scale(X_train)
        # raised max iterations from 100 to 400
        log_reg = LogisticRegression(max_iter=400)
        log_reg.fit(X_scaled, y_train)
        y_predict = log_reg.predict(X_test)

        print(metrics.classification_report(y_test, y_predict))

    # ******************************************************************
    # Decision tree classifier
    elif user_input == "3":
        print("Starting Decision Tree classifier...\n\n")

        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

        print(metrics.classification_report(y_test, y_predict))

    # ******************************************************************
    # try again or quit
    else:
        user_input_end = input("input not recognized \n\n "
                               "Would you like to run another model?\n\n"
                               "Type \"y\" to continue or \"n\" to quit.\n\n")
        if user_input_end == "y":
            prompt()
        elif user_input_end == "n":
            sys.exit(0)
        else:
            print("Input not recognized, exiting program.")

    # Try another?
    new_input = input("Would you like to try a different model?\n"
                      "press \"y\" to try another or press any key to quit.\n\n")
    if new_input == "y":
        prompt()
    else:
        sys.exit(0)


# ******************************************************************
def prompt():
    print("\nStarting Classification program\n")

    user_input = input("Press \"1\" for  SVM.\n"
          "Press \"2\" for Logistic Regression.\n"
          "Press \"3\" for Decision tree classifier.\n\n"
          "or press \"q\" to quit\n\n")

    if user_input == "q":
        print("Bye.")

    elif user_input == "1":
        classify(user_input)

    elif user_input == "2":
        classify(user_input)

    elif user_input == "3":
        classify(user_input)

    else:
        print("input not recognized, try again")
        prompt()


# ******************************************************************
# Run program
if __name__ == "__main__":
    prompt()
