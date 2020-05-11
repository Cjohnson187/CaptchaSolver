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

from imutils import paths



#imagePath_list = list(paths.list_images("C:\Users\chris\IdeaProjects\CaptchaSolver\data2\samples"))
#(data, labels) = load(imagePath_list, verbose=500)

# get contours detect blobs(each character)
letters = cv2.findContours(black.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
letter_group = []




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