"""
Chris Johnson
Marc Inouye
5/9/2020
CS 3120-001
Machine Learning
Final Project
Solving captcha with machine learning.
- use computer vision to analyze letters
in an image and return the correct string
- simple convolutional network
- make pictures black and white
- detect (blobs) of text whose width is less than double the height

resources-
https://www.tensorflow.org/tutorials/images/cnn
https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
"""

import imutils
import pytesseract
import os
import cv2
import keras as kr
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imutils import paths

# from assignment #3
def load(imagePath_list, verbose=-1):

    data = []
    labels = []

    for (i, imagePath) in enumerate(imagePath_list):
        image = cv2.imread(imagePath)

        label = imagePath.split(os.path.sep)[-2]
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        data.append(image)
        labels.append(label)

        # show and update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
                                                  len(imagePath_list)))

    return (np.array(data), np.array(labels))


imagePath_list = list(paths.list_images("C:/Users/chris/IdeaProjects/CaptchaSolver/classes/letters"))
(data, labels) = load(imagePath_list, verbose=500)


data = data.reshape((data.shape[0], 3072))


#  split data 70% training, 10% validation, 20% testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
(trainX, validateX, trainY, validateY) = train_test_split(trainX, trainY, test_size=0.1, random_state=42)

# encode elabels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# 4) Train the classifier

neighbors = 7
# p(1)= Manhattan distance     P(2)= Euclidean distance(default)
model = KNeighborsClassifier(n_neighbors=neighbors, p=1)
model.fit(trainX, trainY)

model.fit(validateX, validateY)

print(classification_report(testY, model.predict(testX), target_names=le.classes_))
