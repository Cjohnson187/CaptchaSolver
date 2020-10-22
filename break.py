"""
Chris Johnson
Marc Inouye
5/9/2020
CS 3120-001
Machine Learning
Final Project
Solving captcha with machine learning.
- train knn with sample letters
- detect (blobs) of text whose width is less than double the height
- predict each character and build string

resources-
https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
"""

import os
import cv2
import glob
import imutils
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


imagePath_list = list(paths.list_images("captcha_classes_for_captcha_solver-master/"))  # training data
MODEL_LABELS_FILENAME = "model_labels.dat"  # model labels for training
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"  # image to predict with trained model
OUTPUT_FOLDER = "extracted_letter_images"  # captcha images split by each glob(potential letter)


neighbors = 5
knn = KNeighborsClassifier(n_neighbors=neighbors, p=1)


def load(image_path_list, verbose=-1):
    data = []
    labels = []

    for (i, imagePath) in enumerate(image_path_list):
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]

        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)

        data.append(image)
        labels.append(label)

        # show and update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
                                                  len(image_path_list)))

    return np.array(data), np.array(labels)


def train(model):
    # load training images
    (data, labels) = load(imagePath_list, verbose=500)
    data = data.reshape((data.shape[0], 3072))

    # split data 70% training, 10% validation, 20% testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    (trainX, validateX, trainY, validateY) = train_test_split(trainX, trainY, test_size=0.1, random_state=42)

    # train classifier
    model.fit(trainX, trainY)
    # validate after training
    model.fit(validateX, validateY)

    return model

"""
***************************************************************************************/
*    Title: How to break a CAPTCHA system in 15 minutes with Machine Learning 
*    Author: Geitgey, Adam
*    Date: Dec 13, 2017
*    Availability: https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
*    We used the articles method of extracting image blobs for each character in the captcha to test each character separately
*    Methods: resize_to_fit from helpers and split_captcha
***************************************************************************************/
"""


def resize_to_fit(image, width, height):
    # grab the dimensions of the image, then initialize the padding values
    (height, height) = image.shape[:2]

    # if the width is greater than the height then resize along the width
    if height > height:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to obtain the target dimensions
    pad_w = int((width - image.shape[1]) / 2.0)
    pad_h = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image


def split_captcha(captcha_image_files):
    counts = {}
    split_images = []

    # loop over the image paths
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
        # grab the base filename as the text
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]

        # Load the image and convert it to grayscale
        image = cv2.imread(captcha_image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add some extra padding around the image
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image (convert it to pure black and white)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find the contours (continuous blobs of pixels) the image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hack for compatibility with different OpenCV versions
        contours = contours[1] if imutils.is_cv3() else contours[0]

        letter_image_regions = []
        # Now we can loop through each of the four contours and extract the letter inside of each one
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that are conjoined into one chunk
            if w / h > 1.25:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                # This is a normal letter by itself
                letter_image_regions.append((x, y, w, h))

        # If we found more or less than 4 letters in the captcha, our letter extraction
        # didn't work correctly. Skip the image instead of saving bad training data!
        if len(letter_image_regions) != 4:
            continue

        # Sort the detected letter images based on the x coordinate to make sure
        # we are processing them from left-to-right so we match the right image
        # with the right letter
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # Save out each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):

            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

            split_images.append(letter_image)

            # write the letter image to a file
            count = counts.get(letter_text, 1)

            # increment the count for the current key
            counts[letter_text] = count + 1

    return split_images


if __name__ == "__main__":

    # KNN classifier
    model_knn = train(knn)

    # directory that contains extracted captcha to analyze
    captcha_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

    # calling method to split up each character in the target image
    split_images = split_captcha(captcha_files)

    # string to print with each prediction
    prediction = ""

    for items in split_images:

        # adding border for split up characters to match training sets
        current_image = cv2.copyMakeBorder(items, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
        current_image = np.array(cv2.resize(current_image, (32, 32), interpolation=cv2.INTER_CUBIC))

        # convert back to 3 dimensional array
        current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)

        # reshape to test data size
        current_image_shaped = current_image.reshape((1, 3072))

        # make prediction
        guess = model_knn.predict(current_image_shaped)
        print("guess - ", guess, " , ",type(guess))

        # add each predicted character to the string being built
        prediction += str(guess)



    print("Charcters found in captcha - ", prediction)

    # TODO I wanted to print the prediction with imshow
    cv2.imshow("Target image- ", current_image)
    cv2.waitKey()







