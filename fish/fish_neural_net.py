import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
import numpy as np

class FishNeuralNet:
    def __init__(self, id, game):
        self.id = id
        self.board = Input(shape=(60,))
        self.players = Input(shape=(6,2))
        self.model_inputs = (
            self.board +
            self.players
        )
        self.model = None

    def map_game_input_to_network_inputs(self, input):
        """
        will align the numpy/dict input returned by Player.full_serialization()
        to the inputs used by the neural network
        """

        # player inputs, game input, game objectives, player reserved, game cards
        # print(input)
        self_raw_input = [np.concatenate([input['self'][k] for k in ['gems','discount','points','order']])]

        self_reserved_input = input['self']['reserved_cards']

        player_raw_inputs = [

            np.concatenate(
                [
                    serializations[k] for k in ['gems','discount','points','order']
                ]
            )
            for serializations in input['other_players']

        ]


        player_reserved_inputs = lchain([x['reserved_cards'] for x in input['other_players']])

        game_raw_input = [np.concatenate([
            input['game'][k] for k in ['gems','turn']
        ])
        ] # note that the turn number + last turn is in the 'turn' key


        game_objective_input = input['game']['objectives']

        game_card_input = lchain(input['game']['available_cards'])

        return (
            self_raw_input + player_raw_inputs +
            game_raw_input +
            game_objective_input +
            self_reserved_input + player_reserved_inputs +
            game_card_input
        )

    def load_extended_history_from_player(self, player):
        self.extended_serialized_history = player.extended_serialized_action_history
        self.lagged_q_state_history = player.extended_lagged_q_state_history


    def prepare_data(self):
        x_unstacked = [self.map_game_input_to_network_inputs(row) for row in
                       self.extended_serialized_history]  # np.vstack(self.extended_serialized_history)
        x = [np.vstack(input_array) for input_array in zip(*x_unstacked)]
        y = np.asarray([row[model_name] for row in self.lagged_q_state_history])
        return x, y

    def train_model(self, n_epochs=10, batch_size=1000,verbose=0):
        x, y = self.prepare_data()
        self.model.fit(x, y, epochs=n_epochs, batch_size=batch_size, verbose=verbose)