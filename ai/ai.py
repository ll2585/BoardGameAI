from __future__ import print_function
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
# from sklearn.model_selection import train_test_split

import numpy as np

from copy import copy, deepcopy
from itertools import chain
import os

class AI:
    def __init__(self):
        self.x = None
        self.y = None
        self.model = None
        self.n_players = 2



    def load_data(self, x, y):
        self.x = x
        self.y = y

    def create_model(self):
        if self.x is None or self.y is None:
            raise Exception("NEED TO LOAD DATA FIRST")

        # no idea what this means
        board_input = Input(shape=(60,), dtype='int32', name='board_input')
        x = Embedding(output_dim =4, input_dim=4, input_length=60)(board_input)
        x = Flatten()(x)
        x = Dense(200)(x)

        player_inputs = Input(shape=(124,))
        player_dense = Dense(200)(player_inputs)

        inputs = [board_input, player_inputs]
        ai = keras.layers.concatenate([x, player_dense])
        ai = Dense(100)(ai)
        ai = Activation('relu')(ai)
        ai = Dense(50)(ai)
        ai = Activation('relu')(ai)
        ai = Dense(25)(ai)
        ai = Activation('relu')(ai)
        ai = Dense(2)(ai)
        ai = Activation('sigmoid')(ai)
        model = Model(inputs = inputs, outputs = ai)
        model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

        self.model = model


    def initialize_network_layers(self):
        #no idea what the fuck this is doing
        main_dense_layers = [Dense(n, activation='relu') for n in self.layer_size]

        main_dense_input = keras.layers.concatenate(self.x_inputs)

        next_layer = main_dense_input
        for layer in main_dense_layers:
            next_layer = layer(next_layer)

        return next_layer

    def make_prediction(self, x):
        board_x = x[:, :60]
        player_x = x[:, 60:]
        predictions = self.model.predict([np.array(board_x), np.array(player_x)])
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

    def train_model(self, n_epochs=100, batch_size=1000, verbose=0):
        categorical_y = keras.utils.to_categorical(self.y, 2)
        board_x = self.x[:,:60]
        player_x = self.x[:,60:]

        self.model.fit([board_x, player_x], categorical_y, epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                       validation_split=.33, shuffle=True)
        scores = self.model.evaluate([board_x, player_x], categorical_y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def evaluate_data(self, x, y):
        categorical_y = keras.utils.to_categorical(y, 2)
        board_x = x[:, :60]
        player_x = x[:, 60:]
        scores = self.model.evaluate([board_x, player_x], categorical_y)
        return scores[1]