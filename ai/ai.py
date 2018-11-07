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

lr= 0.001
dropout= 0.3
epochs= 10
batch_size= 64
num_channels= 512

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
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=64, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


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
        rounded = [round(x[0]) for x in predictions]
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

    def train_model(self, n_epochs=50, batch_size=200, verbose=0):
        self.model.fit(self.x, self.y, epochs=n_epochs, batch_size=batch_size, verbose=verbose)
        scores = self.model.evaluate(self.x, self.y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))