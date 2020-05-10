"""
Chris Johnson
Assigment 3
CS 3120-001
3/10/2020

Given:
  = [3, 5, 7]
Distance metrics: L1 and L2
Dataset: animals.zip

Source -
Slide- 6_KNN_ImageClassification.pdf

"""

#   *******************************************************
# imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# from pyimagesearch.preprocessing import SimplePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader

from imutils import paths
import numpy as np
#from scipy.misc import imread, imresize
import cv2
import os

#   *******************************************************
# 1)  Gather Datasets
my_list = os.listdir('../animals/')

def load(imagePath_list, verbose=-1):
    data = []
    labels = []

    for (i, imagePath) in enumerate(imagePath_list):
        image = cv2.imread(imagePath)

        # imread(imagePath)

        label = imagePath.split(os.path.sep)[-2]
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        data.append(image)
        labels.append(label)

        # show and update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
                                                  len(imagePath_list)))

    return (np.array(data), np.array(labels))


imagePath_list = list(paths.list_images("C:/Users/chris/spring2020/cs_3120_hw/hw/HW3/animals"))
(data, labels) = load(imagePath_list, verbose=500)

print("data size    ", len(data), "       label size   ", len(labels) )


data = data.reshape((data.shape[0], 3072))

#   *******************************************************
# 2)  split data 70% training, 10% validation, 20% testing

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.2, random_state=42)

(trainX, validateX, trainY, validateY) = train_test_split(trainX, trainY,
                                                          test_size=0.1, random_state=42)

#   *******************************************************
# 3) Encode the labels as integers

le = LabelEncoder()
labels = le.fit_transform(labels)

#   *******************************************************
# 4) Train the classifier

model = KNeighborsClassifier(n_neighbors=7)
model.fit(trainX, trainY)

#   *******************************************************
# 5) Evaluate

print(classification_report(testY, model.predict(testX), target_names=le.classes_))
