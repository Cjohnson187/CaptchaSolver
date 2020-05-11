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
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


from imutils import paths




import os
import cv2
import keras as kr
import numpy as np
import pickle


# training directories
TRAINING_SAMPLES = "data/captcha samples"
MODEL_LABELS_FILENAME = "data/model_labels.dat"
# captchas split into each letter
CAPTCHA_IMAGE_FOLDER = "data/split images"

# NN model
MODEL_FILENAME = "model.hdf5"

def test_captcha():
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        label = pickle.load(f)
    # TODO need to find samples to test instead of training set
    model = kr.load_model(TRAINING_SAMPLES)
    captcha_image_files = list(imutils.paths.list_images(CAPTCHA_IMAGE_FOLDER))
    captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)


    #captcha_samples = "/data/captcha samples"
    # output for captcha samples that are split into separate letters
    #split = "/data/split images"

    # creating list of captcha file names/identities
    #image_list = os.listdir("data/captcha samples")

    # testing to see if the list import worked
    for item in image_list:

        # strip extension
        captcha_letters = os.path.splitext(item)[0]
        print(captcha_letters)


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

        # sort characters so we are reading from left to right using x coordinate
        letter_group = sorted(letter_group, key=lambda x: x[0])


        #make an output image
        output = cv2.merge([image] * 3)
        predictions = []

        # loop over letters to check each

        for piece in letter_group:
            # coordinates of letter
            x, y, width, height = piece
            # extract letter
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
            # resize to fit training data 20x20
            letter_image = resize_to_fit(letter_image, 20, 20)
            # set image to 4d list to fit keras model
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # get a prediction
            prediction = model.predict(letter_image)
            # convert to normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            # draw the prediction on the output image
            cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # print the predicted text
        captcha_text = "".join(predictions)
        print("CAPTCHA text is: {}".format(captcha_text))


        # Show the annotated image
        cv2.imshow("Output", output)
        cv2.waitKey()


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
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


def train_model():
    data = []
    labels = []

    # loop over the input images
    for image_file in paths.list_images(TRAINING_SAMPLES):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the letter so it fits in a 20x20 pixel box
        image = resize_to_fit(image, 20, 20)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)


    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the training data into separate train and test sets
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

    # TODO use train validation and test
    # # 2)  split data 70% training, 10% validation, 20% testing
    # (trainX, testX, trainY, testY) = train_test_split(data, labels,
    #                                                   test_size=0.2, random_state=42)
    # (trainX, validateX, trainY, validateY) = train_test_split(trainX, trainY,

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    # Build the neural network!
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # Output layer with 32 nodes (one for each possible letter/number we predict)
    model.add(Dense(32, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the neural network
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

    # Save the trained model to disk
    model.save(MODEL_FILENAME)


if __name__ == "__main__":
    # train model
    train_model()
    print("training finished")

    #test_captchas