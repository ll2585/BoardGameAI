import numpy as np
import random
#from .fish_neural_net import FishNeuralNet
"""
Random and Human-ineracting players for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloPlayers by Surag Nair.

"""
class Player:
    def __init__(self, player_id, name, game):
        self.player_id = player_id
        self.name = name
        self.game = game
        self.penguins = []
        self.score = 0
        self.tiles_collected = 0

    def reset(self):
        self.penguins = []
        self.score = 0
        self.tiles_collected = 0

    def get_score(self):
        return self.score

    def get_tiles(self):
        return self.tiles_collected

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def set_player_id(self, player_id):
        self.player_id = player_id

    def get_player_id(self):
        return self.player_id

    def setup(self):
        self.penguins = []
        self.score = 0
        self.tiles_collected = 0

    def get_state(self):
        return [self.penguins, self.score, self.tiles_collected]

    def get_penguins(self):
        return self.penguins

    def get_score(self):
        return self.score

    def get_num_tiles(self):
        return self.tiles_collected

    def penguin_moved(self, start, end):
        for i, penguin in enumerate(self.penguins):
            if penguin == start:
                if start == end:
                    #penguin died
                    self.penguins[i] = -1
                else:
                    self.penguins[i] = end
                break

    def has_penguin_at_index(self, index):
        return index in self.penguins

    def get_point_card_serialization(self):
        return [self.score, self.tiles_collected]

    def new_penguin_at(self, location):
        if len(self.penguins) == 4:
            raise Exception("TOO MANY PENGUINS")
        self.penguins.append(location)

    def move(self):
        pass



class RandomPlayer(Player):
    def __init__(self, player_id, name, game):
        Player.__init__(self, player_id, name, game)
        #self.ai = FishNeuralNet(id=id, game=game)
        self.extended_serialized_action_history = []

    def move(self):
        possible_moves = self.game.get_possible_moves(self)
        move = random.choice(possible_moves)
        return move




class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True:
            # Python 3.x
            a = input()
            # Python 2.x
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
