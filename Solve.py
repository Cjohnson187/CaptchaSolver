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


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import os
import cv2


# directory for captcha samples
captcha_samples = "/data/captcha samples"
# output for captcha samples that are split into separate letters
split = "/data/split images"

# creating list of captcha file names/identities
image_list = os.listdir("data/captcha samples")

# testing to see if the list import worked
for item in image_list:

    # strip extension
    captcha_letters = os.path.splitext(item)[0]
    print (captcha_letters)


    # convert to gray scale
    current_image = cv2.imread("/data/captcha samples/3FNF.png", 0)
    print(type(current_image))
    cv2.imshow("pic", current_image)
    cv2.waitKey()
    break

    gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    # adding border for padding
    gray_image = cv2.copyMakeBorder(gray_image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    # convert to black and white
    black = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # get contours detect blobs(each character)
    letters = cv2.findContours(black.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_group = []

    # get letter from each pixel group
    for letter in letters:
        (x, y, width, height) = cv2.boundingRect(letter)
        # some letters are written over each other so we need to check if the
        # pixel group as a single letter
        if width / height > 1.25:
            # can split the letters into two images
            half = int (width / 2)
            letter_group.append((x, y, half, height))
            letter_group.append((x + half, y, half, height))
        else:
            letter_group.append((x, y, width, height))

    # this means the captcha has more than 4 letters or something went wrong
    # so we skip it for now
    if len(letter_group) != 4:
        continue
