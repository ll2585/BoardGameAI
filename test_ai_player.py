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
player_2 = NaiveAIPlayer(1,"CHARLA",game,main_name='seeded_last_25', index=0, filtered_moves=20)
for i in tqdm(range(num_games)):
    game.set_up()
    players = [player_1, player_2]
    for player in players:
        player.reset()
        game.add_player(player)
    game.start()

    while not game.is_over():
        if show_game:
            print('---------------------------------------------------------------------')
            display(game)
        cur_player = game.get_current_player()
        if cur_player.get_player_id() == 0:
            move = cur_player.move(seed=212)
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


    board_x = this_x[:, :60]
    player_x = this_x[:, 60:]
    categorical_y = keras.utils.to_categorical(this_y, 3)

    scores = player_2.ai.model.evaluate([board_x, player_x], categorical_y, verbose = False)
    print("\n%s: %.2f%%" % (player_2.ai.model.metrics_names[1], scores[1] * 100))
    print("winner: {0}".format(winner))
    print(wins)

    if (write_score_threshold is not None and scores[1] < write_score_threshold) or write_score_threshold is None:
        times_written += 1
        if x is None:
            x = this_x
            y = this_y
        else:
            x = np.concatenate((x, this_x), axis=0)
            y = np.concatenate((y, this_y), axis=0)


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