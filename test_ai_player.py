from pprint import pprint
import numpy as np
from fish.fish_game import FishGame, display
from fish.fish_players import RandomPlayer
from fish.fish_naive_ai_player import NaiveAIPlayer
import tables
from pathlib import Path
# Setup
from tqdm import tqdm
import keras
import constants

constants.use_gpu()

num_games = 1
#if random wins 9 or more games, keep random
#otherwise keep the other model
x = None
y = None
wins = {

}

write_data = False
write_new_file = False
data_filename = 'data_new'
write_score_threshold = .7

times_to_write = 20
times_written = 0

#while times_written < times_to_write:
show_game = True
game = FishGame()
player_1 = RandomPlayer(0,"BOB",game)
player_2 = NaiveAIPlayer(1,"CHARLA",game,main_name='minimax_50', index=0, filtered_moves=50)
for i in tqdm(range(num_games)):
    game.set_up()
    players = [player_1, player_2]
    for player in players:
        player.reset()
        game.add_player(player)
    game.start()
    turns = 0
    while not game.is_over():
        turns += 1
        if show_game:
            print('---------------------------------------------------------------------')
            print('turn: {0}'.format(turns))
            display(game)
        cur_player = game.get_current_player()
        if cur_player.get_player_id() == 1:
            move = cur_player.get_move_by_total_moves(turns)
        else:
            move = cur_player.move()
        game.do_move(move)
        if show_game:
            print(game.get_player_scores())
    if show_game:
        print('---------------------------------------------------------------------')
        display(game)

    #0 is the Player[0], 1 is Player[1]
    winner = game.get_winner()
    if winner not in wins:
        wins[winner] = 0
    wins[winner] += 1

    this_x = game.get_full_game_history_for_neural_net()[0]
    this_y = game.get_full_game_history_for_neural_net()[1]

    player_2.ai.load_data(this_x, this_y)
    print("winner: {0}".format(winner))
    print(wins)


print(wins)


if write_new_file:

    hdf5_file = tables.open_file('./data/{0}.hdf5'.format(data_filename), 'w')
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

if write_data and x is not None:
    print("writing data")
    hdf5_path = "./data/{0}.hdf5".format(data_filename)
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='a')
    extendable_hdf5_x = extendable_hdf5_file.root.x
    extendable_hdf5_y = extendable_hdf5_file.root.y
    for n, (d, c) in enumerate(zip(x, y)):
        extendable_hdf5_x.append(x[n][None])
        extendable_hdf5_y.append(y[n][None])
    extendable_hdf5_file.close()