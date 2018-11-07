from __future__ import print_function
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *

import numpy as np

from copy import copy, deepcopy
from itertools import chain
import os

class AI:
    def __init__(self):
        self.x = None
        self.y = None
        self.model = None



    def load_data(self, x, y):
        self.x = x
        self.y = y

    def create_model(self):
        if self.x is None or self.y is None:
            raise Exception("NEED TO LOAD DATA FIRST")

        # no idea what this means

        ai = Sequential()
        ai.add(Dense(20, input_shape=(184,)))
        ai.add(Activation('relu'))
        ai.add(Dense(2))
        ai.add(Activation('sigmoid'))
        ai.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

        self.model = ai


    def initialize_network_layers(self):
        #no idea what the fuck this is doing
        main_dense_layers = [Dense(n, activation='relu') for n in self.layer_size]

        main_dense_input = keras.layers.concatenate(self.x_inputs)

        next_layer = main_dense_input
        for layer in main_dense_layers:
            next_layer = layer(next_layer)

        return next_layer

    def make_prediction(self, x):
        predictions = self.model.predict(x)
        rounded = [x[1] for x in predictions]
        return rounded

    def save_model(self, main_name,  index=0, verbose=False):
        filename = '{main_name}_{index}.h5'.format(
            main_name=main_name,
            index=index
        )
        if verbose:
            print('saving {filename}'.format(filename=filename))
        self.model.save(filename)

    def load_model(self, main_name, index=0, verbose=False):
        filename = '{main_name}_{index}.h5'.format(
            main_name=main_name,
            index=index
        )
        if verbose:
            print('loading {filename}'.format(filename=filename))
        self.model = load_model(filename)

    def train_model(self, n_epochs=10, batch_size=100, verbose=0):
        categorical_y = keras.utils.to_categorical(self.y, 2)
        x = self.x
        y = categorical_y
        self.model.fit(x, categorical_y, epochs=n_epochs, batch_size=batch_size, verbose=verbose)
        scores = self.model.evaluate(x, y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))