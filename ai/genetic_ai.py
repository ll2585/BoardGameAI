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

class GeneticAI:
    def __init__(self, board_layers, board_neurons, player_layers, player_neurons, hidden_layers, hidden_neurons):
        self.x = None
        self.y = None
        self.model = None
        assert board_layers == len(board_neurons)
        assert player_layers == len(player_neurons)
        assert hidden_layers == len(hidden_neurons)
        self.board_layers = board_layers
        self.board_neurons = board_neurons
        self.player_layers = player_layers
        self.player_neurons = player_neurons
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.hist = None

    def load_data(self, x, y):
        self.x = x
        self.y = y

    def create_model(self):
        # no idea what this means
        board_input = Input(shape=(60,), dtype='int32', name='board_input')
        x = Embedding(output_dim =4, input_dim=4, input_length=60)(board_input)
        x = Flatten()(x)
        for i in range(self.board_layers):
            x = Dense(self.board_neurons[i], activation = 'relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(.5)(x)

        player_inputs = Input(shape=(124,))
        player_dense = player_inputs
        for i in range(self.player_layers):
            player_dense = Dense(self.player_neurons[i], activation='relu')(player_dense)
            player_dense = BatchNormalization()(player_dense)
            player_dense = Dropout(.5)(player_dense)

        inputs = [board_input, player_inputs]
        ai = keras.layers.concatenate([x, player_dense])
        ai = Dropout(.5)(ai)
        for i in range(self.hidden_layers):
            ai = Dense(self.hidden_neurons[i],activation='relu')(ai)
        ai = BatchNormalization()(ai)
        ai = Dense(3, activation='softmax')(ai)#3 categories for draw
        model = Model(inputs = inputs, outputs = ai)
        model.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

        self.model = model

    def make_prediction(self, x):
        board_x = x[:, :60]
        player_x = x[:, 60:]
        predictions = self.model.predict([np.array(board_x), np.array(player_x)])
        print(predictions)
        rounded = [x[0] for x in predictions]
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

    def train_model(self, x = None, y = None, n_epochs=30, batch_size=1000, verbose=False):
        if y is None:
            categorical_y = keras.utils.to_categorical(self.y, 3)
        else:
            categorical_y = keras.utils.to_categorical(y, 3)
        if x is None:
            board_x = self.x[:,:60]
            player_x = self.x[:,60:]
        else:
            board_x = x[:, :60]
            player_x = x[:, 60:]
        self.hist = self.model.fit([board_x, player_x], categorical_y, epochs=n_epochs, batch_size=batch_size, verbose=False,
                       validation_split=.33, shuffle=True)
        # print('loss: {0}'.format(hist.history['loss'][-1]))
        # print('val loss: {0}'.format(hist.history['val_loss'][-1]))
        # self.scores = self.model.evaluate([board_x, player_x], categorical_y, verbose=False)
        # print("\n%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1] * 100))

    def evaluate_data(self, x, y):
        categorical_y = keras.utils.to_categorical(y, 3)
        board_x = x[:, :60]
        player_x = x[:, 60:]
        scores = self.model.evaluate([board_x, player_x], categorical_y)
        return scores[1]

    def filter_by_tiles_collected(self, total_tiles_collected):
        assert self.x is not None
        assert self.y is not None
        np.set_printoptions(threshold=np.nan)
        player_tiles_collected = 181
        other_player_tiles_collected_index = 183
        subset = self.x
        x = subset[(subset[:, player_tiles_collected] + subset[:, other_player_tiles_collected_index]) >= total_tiles_collected]
        y = self.y[(subset[:, player_tiles_collected] + subset[:, other_player_tiles_collected_index]) >= total_tiles_collected]
        return x, y
