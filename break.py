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

import os
import cv2
import glob
import imutils
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imutils import paths

imagePath_list = list(paths.list_images("C:/Users/chris/IdeaProjects/CaptchaSolver/classes/letters")) # training data
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"  # image to predict with trained model
OUTPUT_FOLDER = "extracted_letter_images" # captcha images split by each glob(potential letter)


neighbors = 2
model = KNeighborsClassifier(n_neighbors=neighbors, p=1)


def load(imagePath_list, verbose=-1):

    data = []
    labels = []

    for (i, imagePath) in enumerate(imagePath_list):
        image = cv2.imread(imagePath)

        label = imagePath.split(os.path.sep)[-2]
        print("1 train data shape / type- ", image.shape, " / ", type(image))

        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        print("train data shape / type- ", image.shape, " / ", type(image), "______ training images after resize cv2")

        data.append(image)
        labels.append(label)

        # show and update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
                                                  len(imagePath_list)))
   #print(image)

    return (np.array(data), np.array(labels))


def train(model):
    (data, labels) = load(imagePath_list, verbose=500)
    data = data.reshape((data.shape[0], 3072))

    #  split data 70% training, 10% validation, 20% testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    (trainX, validateX, trainY, validateY) = train_test_split(trainX, trainY, test_size=0.1, random_state=42)

    # encode labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # 4) Train the classifier

    #neighbors = 7
    # p(1)= Manhattan distance     P(2)= Euclidean distance(default)
    #model = KNeighborsClassifier(n_neighbors=neighbors, p=1)
    print("train data shape / type- ", trainX.shape, " / ", type(trainX), " _______ after formatiing for training")
    model.fit(trainX, trainY)
    model.fit(validateX, validateY)
    print(classification_report(testY, model.predict(testX), target_names=le.classes_))

    return model


"""
***************************************************************************************/
*    Title: How to break a CAPTCHA system in 15 minutes with Machine Learning 
*    Author: Geitgey, Adam
*    Date: Dec 13, 2017
*    Availability: https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
*    We used the articles method of extracting image blobs for each character in the captcha to test each character separately
*    Methods: resize_to_fit from helpers.py and solve_captchas_with_model.py
***************************************************************************************/
"""

def resize_to_fit(image, width, height):
    # grab the dimensions of the image, then initialize
    # the padding values
    (height, height) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if height > height:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

def split_captcha(captcha_image_files):
    split_images = []
    counts = {}

    # loop over the image paths
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

        # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
        # grab the base filename as the text
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]

        # Load the image and convert it to grayscale
        image = cv2.imread(captcha_image_file)
        print("train data shape / type- ", image.shape, " / ", type(image), "____ loaded sample to predict ")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add some extra padding around the image
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image (convert it to pure black and white)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find the contours (continuous blobs of pixels) the image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hack for compatibility with different OpenCV versions
        contours = contours[0] if imutils.is_cv2() else contours[1]

        letter_image_regions = []

        # Now we can loop through each of the four contours and extract the letter
        # inside of each one
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that
            # are conjoined into one chunk
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

            # Get the folder to save the image in
            #save_path = os.path.join(OUTPUT_FOLDER, letter_text)

            # if the output directory does not exist, create it
            #if not os.path.exists(save_path):
                #os.makedirs(save_path)

            # write the letter image to a file
            count = counts.get(letter_text, 1)
            #p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            #cv2.imwrite(p, letter_image)

            # increment the count for the current key
            counts[letter_text] = count + 1

    return split_images



if __name__ == "__main__":
    model = train(model)
    # directory that contains extracted captcha to analyze
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
    split_images = split_captcha(captcha_image_files)
    model = train(model)
    print(len(split_images))
    prediction = ""
    word = ""
    print("train data shape [1]/ type- ", split_images[1].shape, " / ", type(split_images[1]), " _______ letter bounding box")



    for items in split_images:
        image = cv2.resize(items, (32, 32), interpolation=cv2.INTER_CUBIC)
        print("train data shape / type- ", image.shape, " / ", type(image), " ______ after cv2 resize to use the model")


    #image = image.reshape((image.shape[0], 3072))

    #item = items.reshape((data.shape[0], 3072))

    #item = resize_to_fit(items, 20, 20)
        print("predicted- ", model.predict(image))
        #prediction += model.predict(image)
        #cv2.imshow("Guess- ", output)
        #cv2.waitKey()
        # Convert the one-hot-encoded prediction back to a normal letter


    print(prediction)






