#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USAGE
# python knn.py --dataset ../datasets/animals

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from pyimagesearch.preprocessing import SimplePreprocessor
#from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import numpy as np
from scipy.misc import imread, imresize
import cv2
import os

def load(imagePath_list, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePath_list):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
           
            # imread(imagePath)
            
            #imread(imagePath)
            
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			image = cv2.resize(image, (32, 32),interpolation=cv2.INTER_CUBIC)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagePath_list)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))

# grab the list of images that we'll be describing
print("[INFO] loading images...")

imagePath_list = list(paths.list_images("../Data/KNN/animals"))
# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
(data, labels) = load(imagePath_list, verbose=500)

print("data size    ", len(data), "       label size   ", len(labels) )


data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state= 42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")

model = KNeighborsClassifier(n_neighbors=7)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))