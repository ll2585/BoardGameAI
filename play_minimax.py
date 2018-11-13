from pprint import pprint
import numpy as np
from fish.fish_game import FishGame, display
from fish.fish_players import RandomPlayer
import tables
from pathlib import Path
# Setup
from tqdm import tqdm
from fish.fish_state import display_state
from minimax import *
import random

num_games = 250
x = None
y = None
write_data = True
min_rows_desired = 30000
total_rows = 0
BREAK_MOVES = 50
total_games_played = 0

def get_total_moves(state):
    total_moves = 0
    current_player_id = state.current_player_id
    state_moves = state.get_possible_moves(current_player_id)
    if len(state_moves) == 0:
        return 1
    else:
        for move in state_moves:
            total_moves += get_total_moves(state.get_state_from_move(move))
    return total_moves

#for i in tqdm(range(num_games)):
while total_rows < min_rows_desired:
    total_games_played += 1
    print("GAME #{0}".format(total_games_played))
    game = FishGame()
    player_1 = RandomPlayer(0,"BOB",game)
    player_2 = RandomPlayer(1,"CHARLA",game)
    game.set_up()
    players = [player_1, player_2]
    for player in players:
        player.reset()
        game.add_player(player)
    game.start()
    moves = 0

    while not game.is_over():
        moves += 1
        if moves == BREAK_MOVES:
            print("MOVE {0} GETTING TOTAL MOVES LEFT".format(BREAK_MOVES))
            #display(game)
            game_tree = GameTree(game.get_current_player_id())
            game_tree.build_tree(game.get_state())
            print("Game tree size: {0}".format(game_tree.size))
            if game_tree.abort:
                break
            minimax = MiniMax(game_tree)
            best_state = minimax.minimax()
            if x is None:
                x = minimax.get_minimax_history_for_neural_net()[0]
                y = minimax.get_minimax_history_for_neural_net()[1]
            else:
                x = np.concatenate((x, minimax.get_minimax_history_for_neural_net()[0]), axis=0)
                y = np.concatenate((y, minimax.get_minimax_history_for_neural_net()[1]), axis=0)
            total_rows = x.shape[0]
            print('Total rows: {0}'.format(total_rows))
            break
        cur_player = game.get_current_player()
        #seed=10124386
        move = cur_player.move()
        game.do_move(move)

if write_data:

    hdf5_file = tables.open_file('./data/minimax_50.hdf5', 'w')
    filters = tables.Filters(complevel=5, complib='blosc')
    x_storage = hdf5_file.create_earray(hdf5_file.root, 'x',
                                          tables.Atom.from_dtype(x.dtype),
                                          shape=(0, x.shape[-1]),
                                          filters=filters,
                                          expectedrows=len(x))
    y_storage = hdf5_file.create_earray(hdf5_file.root, 'y',
                                              tables.Atom.from_dtype(y.dtype),
                                              shape=(0,),
                                              filters=filters,
                                              expectedrows=len(y))
    for n, (d, c) in enumerate(zip(x, y)):
        x_storage.append(x[n][None])
        y_storage.append(y[n][None])
    hdf5_file.close()
