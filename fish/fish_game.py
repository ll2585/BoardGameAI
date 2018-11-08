from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .fish_board import FishBoard
from .fish_state import FishState
from .fish_move import  FishMove
from .fish_players import Player
from copy import deepcopy
import numpy as np

import time

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""
class FishGame:
    def __init__(self):
        self.players = None
        self.board = None
        self.current_player = None
        self.player_who_moved = None
        self.game_history = None

    def add_player(self, player):
        self.players.append(player)

    def set_players(self, players):
        self.players = players

    def set_up(self):
        self.players = []
        self.board = FishBoard()
        self.game_history = []

    def get_players(self):
        return self.players

    def get_player_ids(self):
        return [p.get_player_id() for p in self.players]

    def start(self):
        self.current_player = self.players[0]
        self.player_who_moved = -1

    def get_next_player(self, player_id):
        next_player_index = 0
        for i, player in enumerate(self.players):
            if player.get_player_id() == player_id:
                next_player_index = i + 1
        if next_player_index == len(self.players):
            next_player_index = 0
        return self.players[next_player_index]

    def do_move(self, action, save_history = True):
        player_id = action.player.get_player_id()
        cur_player = self.get_current_player()
        if player_id != cur_player.get_player_id():
            raise Exception("Current Player not playing!")
        self.player_who_moved = cur_player.get_player_id()
        if action.type == "move":

            hex_from = self.board.pieces[action.start]
            hex_from.move_penguin_away()

            cur_player.tiles_collected += 1
            cur_player.score += hex_from.value

            hex_from.empty()

            if action.start != action.end:
                hex_to = self.board.pieces[action.end]
                hex_to.move_penguin_here()

            cur_player.penguin_moved(start=action.start, end=action.end)

            next_player = self.get_next_player(player_id)
            if next_player.get_penguins() == [-1, -1, -1, -1]:
                next_player = cur_player
            while self.no_penguin_moves(next_player):
                self.remove_penguins(next_player)
                next_player = cur_player
                if self.is_over():
                    break
        elif action.type == "place":
            self.player_who_moved = cur_player.get_player_id()
            hex = self.board.pieces[action.start]
            hex.move_penguin_here()
            cur_player.new_penguin_at(action.start)
            next_player = self.get_next_player(player_id)
        else:
            raise Exception("WRONG ACTION TYPE")
        self.current_player = next_player
        if save_history:
            self.game_history.append(self.get_state().serialize())

    def no_penguin_moves(self, player):
        next_player_penguin_moves = []
        for penguin in player.penguins:
            next_player_penguin_moves += self.board.get_legal_moves(penguin, player)

        return len(next_player_penguin_moves) == 0

    def remove_penguins(self, player):
        for p in player.penguins:
            penguin_at_hex = self.board.pieces[p]
            penguin_at_hex.move_penguin_away()
            player.tiles_collected += 1
            player.score += penguin_at_hex.value
            penguin_at_hex.empty()
            player.penguins = [-1, -1, -1, -1]

    def get_full_game_history_for_neural_net(self):
        #0 is player 1, 1 is player 2
        winner = self.get_winner_id()
        x = []
        y = []
        for serialization in self.game_history:
            player_who_moved = serialization['player_who_moved']
            state = serialization['serialization']
            x.append(state)
            if player_who_moved == winner:
                y.append(1)
            else:
                y.append(0)

        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def get_current_player_id(self):
        return self.current_player.get_player_id()

    def get_possible_moves(self, player=None):
        legal_moves = []
        if self.is_over():
            return legal_moves
        if player:
            if player.get_player_id() != self.get_current_player_id():
                return []
        cur_player = self.get_current_player()
        penguins = cur_player.get_penguins()
        if len(penguins) < 4:
            for i, hex in enumerate(self.board.pieces):
                if not hex.has_penguin_here():
                    legal_moves.append(FishMove(i, player=player, type="place"))
        else:
            for penguin in penguins:
                legal_moves += self.board.get_legal_moves(penguin, player)
        return legal_moves

    def get_current_player(self):
        return self.current_player

    def get_board(self):
        return self.board

    def get_state(self):
        board = self.board.pieces
        players = [p.get_state() for p in self.players]
        current_player_id = self.get_current_player_id()
        player_who_moved = self.player_who_moved
        player_ids = [p.get_player_id() for p in self.players]
        return FishState(deepcopy(board), deepcopy(players), deepcopy(current_player_id), deepcopy(player_who_moved), deepcopy(player_ids))

    def is_over(self):
        player_1 = self.players[0]
        player_2 = self.players[1]

        return player_1.get_penguins() == [-1, -1, -1, -1] and player_2.get_penguins() == [-1, -1, -1, -1]

    def get_winner(self):
        if self.get_winner_id() == self.players[0].get_player_id():
            return "Player 0"
        elif self.get_winner_id() == self.players[1].get_player_id():
            return "Player 1"
        else:
            return "None! It's a draw!"

    def get_winner_name(self):
        if self.get_winner_id() == self.players[0].get_player_id():
            return self.players[0].get_name()
        elif self.get_winner_id() == self.players[1].get_player_id():
            return self.players[1].get_name()
        else:
            return "None! It's a draw!"

    def get_winner_id(self):
        player_1 = self.players[0]
        player_2 = self.players[1]

        player_1_score = player_1.get_score()
        player_2_score = player_2.get_score()
        if player_1_score > player_2_score:
            return player_1.get_player_id()
        elif player_2_score > player_1_score:
            return player_2.get_player_id()
        elif player_1_score == player_2_score:
            player_1_tiles = player_1.get_num_tiles()
            player_2_tiles = player_2.get_num_tiles()
            if player_1_tiles > player_2_tiles:
                return player_1.get_player_id()
            elif player_2_tiles > player_1_tiles:
                return player_2.get_player_id()
            else:
                # draw has a very little value
                return -1

#   00  01  02  03  04  05  06
# 07  08  09  10  11  12  13  14
#   15  16  17  18  19  20  21
# 22  23  24  25  26  27  28  29
#   30  31  32  33  34  35  36
# 37  38  39  40  41  42  43  44
#   45  46  47  48  49  50  51
# 52  53  54  55  56  57  58  59

def display(game):
    board = game.get_board()
    players = game.get_players()
    p1_penguins = players[0].get_penguins()
    p2_penguins = players[1].get_penguins()
    rows = [' ', '',
            ' ', '',
            ' ', '',
            ' ', '']
    ends = [7, 15, 22, 30, 37, 45, 52, 60]
    for i in range(len(board.pieces)):
        for j, end in enumerate(ends):
            if i < end:
                value = board.pieces[i].value
                if value == -1:
                    value = 0
                if i in p1_penguins:
                    value = "P"
                elif i in p2_penguins:
                    value = "Q"
                rows[j] += str(value) + " "
                break
    print ('\n'.join(rows))
