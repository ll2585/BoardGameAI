from pprint import pprint
import numpy as np
from fish.fish_game import FishGame, display
from fish.fish_players import RandomPlayer
from fish.fish_naive_ai_player import NaiveAIPlayer
import tables
from pathlib import Path
# Setup
from tqdm import tqdm
from ai.ai import AI
import keras
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

max_iters = 3
num_games = 25
win_threshold = .6
last_model = 0
best_model = None
random_player_1 = False


iters = 0
times_to_write = -1
write_new_file = True
write_data = True
max_games = 80

while iters < max_iters:
    data_filename = 'data_self_learn_{0}'.format(iters)
    times_written = 0
    num_games = num_games
    #if random wins 9 or more games, keep random
    #otherwise keep the other model
    x = None
    y = None
    wins = {

    }
    player_1 = None
    player_2 = None
    game = None
    new_challenger_name = "Model {0}".format(last_model+iters)
    games_played = 0
    while (games_played < num_games) or (times_written < times_to_write and games_played < max_games): #quit if 20 shit and 25 games played, or 80 games played and < 20 shit (so its not that bad)
        game = FishGame()
        if iters == 0:
            player_1 = NaiveAIPlayer(-1, "Empty Model", game, initialize=True)
            player_2 = NaiveAIPlayer(-1, "Empty Model", game, initialize=True)
        else:
            player_1 = NaiveAIPlayer(best_model+1, "Model {0}".format(best_model), game, main_name='model', index=best_model)
            player_2 = NaiveAIPlayer(iters+last_model+1, new_challenger_name,game,main_name='model', index=last_model+iters)
        game.set_up()
        players = [player_1, player_2]
        for player in players:
            player.reset()
            game.add_player(player)
        game.start()

        while not game.is_over():
            cur_player = game.get_current_player()
            move = cur_player.move()
            game.do_move(move)


        #0 is the Player[0], 1 is Player[1]
        winner = game.get_winner_name()
        if winner not in wins:
            wins[winner] = 0
        wins[winner] += 1
        games_played += 1

        this_x = game.get_full_game_history_for_neural_net()[0]
        this_y = game.get_full_game_history_for_neural_net()[1]

        board_x = this_x[:, :60]
        player_x = this_x[:, 60:]
        categorical_y = keras.utils.to_categorical(this_y, 2)
        scores = player_2.ai.model.evaluate([board_x, player_x], categorical_y)
        print("\n%s: %.2f%%" % (player_2.ai.model.metrics_names[1], scores[1] * 100))
        print("winner: {0}".format(winner))
        print(wins)

        if write_score_threshold is not None and scores[1] < write_score_threshold:
            times_written += 1
            print('shitty accuracy #{0}'.format(times_written))
        if x is None:
            x = this_x
            y = this_y
        else:
            x = np.concatenate((x, this_x), axis=0)
            y = np.concatenate((y, this_y), axis=0)



    print(wins)
    if write_data and x is not None:
        print("writing data")
        hdf5_path = "./data/data.hdf5".format(data_filename)
        extendable_hdf5_file = tables.open_file(hdf5_path, mode='a')
        extendable_hdf5_x = extendable_hdf5_file.root.x
        extendable_hdf5_y = extendable_hdf5_file.root.y
        for n, (d, c) in enumerate(zip(x, y)):
            extendable_hdf5_x.append(x[n][None])
            extendable_hdf5_y.append(y[n][None])
        extendable_hdf5_file.close()

    if write_new_file and x is not None:

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

    if wins[new_challenger_name] > (win_threshold*games_played):
        if random_player_1:
            random_player_1 = False
        print("NEW BEST PLAYER")
        best_model = last_model+iters
    else:
        best_model = best_model


    print('TRAINING NEW MODEL')
    hdf5_path = "./data/data.hdf5"
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    x = extendable_hdf5_file.root.x[:]
    y = extendable_hdf5_file.root.y[:]
    extendable_hdf5_file.close()

    ai = AI()
    ai.load_data(x, y)
    ai.create_model()
    ai.train_model()
    ai.save_model('model', index=iters+last_model+1)
    print("MODEL TRAINED LETS GO")
    iters += 1