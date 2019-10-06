"""
description:    ---
author:         Johann Schmidt
date:           October 2019
refs.:          https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math


class NumberClassifier:
    """ A classifier for number images.
    """

    def __init__(self, load_existing_model=False):
        """ Initialization method.
        :param load_existing_model:
        """
        if load_existing_model:
            self.model = self.load()
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_dataset()
            self.normalize()
            self.model = self.build_ff_model()
            self.add_flatten_layer()
            self.add_dense_layer(input_shape=self.x_train.shape[1:])
            self.add_dense_layer()
            self.add_dense_layer(units=10, activation=tf.nn.softmax)
            self.configure()
            self.train()
            self.evaluate()
            self.save()

    def load_dataset(self):
        """ Loads the default dataset.
        :return dataset
        """
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def normalize(self):
        """ Normalize each image pixel color value from [0,256] to [0,1].
        """
        self.x_train = tf.keras.utils.normalize(self.x_train, axis=1)
        self.x_test = tf.keras.utils.normalize(self.x_test, axis=1)

    def build_ff_model(self):
        """ Builds a feed forward model.
        :return: model
        """
        return tf.keras.models.Sequential()

    def add_flatten_layer(self):
        """ Adds a flatten input layer.
        (28x28 image -> 1x784 vector)
        """
        if self.model is not None:
            self.model.add(tf.keras.layers.Flatten())

    def add_dense_layer(self, units=128, activation=tf.nn.relu, input_shape=None):
        """ Adds a hidden layer (dense = fully connected).
        :param units number of units
        :param activation activation function
        :param input_shape shape of the input
        """
        if self.model is not None:
            if input_shape is None:
                self.model.add(tf.keras.layers.Dense(
                    units=units, activation=activation,
                    kernel_constraint=None, bias_constraint=None))
            else:
                self.model.add(tf.keras.layers.Dense(
                    units=units, activation=activation, input_shape=input_shape,
                    kernel_constraint=None, bias_constraint=None))

    def configure(self, optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']):
        """
        Configures the model for training.
        :param optimizer:
        :param loss:
        :param metrics:
        """
        if self.model is not None:
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    def train(self, epochs=3):
        """ Start training phase.
        :param epochs: number of epochs
        """
        if self.model is not None:
            self.model.fit(self.x_train, self.y_train, epochs=epochs)

    def evaluate(self, output=True):
        """ Evaluates the model.
        :param output: Output the result in the console.
        :return: results
        """
        if self.model is None:
            return None
        val_loss, val_acc = self.model.evaluate(self.x_test, self.y_test)
        if output:
            print("Evaluation loss: {}".format(val_loss))
            print("Evaluation accuracy: {}".format(val_acc))
        return val_loss, val_acc

    def save(self, name='num_reader.model'):
        """ Saves the model.
        :param name: name of the model
        """
        if self.model is not None:
            self.model.save(name)

    def load(self, filename='num_reader.model'):
        """ Loads a model.
        :param filename: the name of the model
        :return model
        """
        return tf.keras.models.load_model(filename)

    def shape(self):
        """ Returns the model input data shape.
        :return: shape
        """
        if self.model is None:
            return None
        return self.model._build_input_shape[1], self.model._build_input_shape[2]

    def predict(self, img):
        """ Predicts the content of an image.
        :param img:
        :return: predicted label
        """
        if self.model is None:
            return None
        if not isinstance(img, np.ndarray):
            print("ERROR: Unable to predict type {}".format(type(img)))
            return None
        if img.shape != self.shape():
            print("ERROR: Wrong image shape {}".format(img.shape))
        img = tf.keras.utils.normalize(img, axis=1)
        prediction = self.model.predict(np.array([img, ]))
        return np.argmax(prediction[0])

    def display_img(self, img):
        """ Displays a image.
        :param img:
        """
        if img is not None:
            plt.imshow(img, cmap=plt.cm.binary)
            plt.show()
