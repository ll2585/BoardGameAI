# initialize model -> no training
# play yourself 25 times, log data into memory (and write i guess)
# train model (v1)
# play v0 vs v1 25 times, log data into memory and write
# train model (v2)
# best vs v2 etc. do this 10 times total
# then delete all the memory, play v11 vs best of first ten
# repeat i dunno until 50 times (5 generations)

import numpy as np
from fish.fish_game import FishGame, display
from fish.fish_players import RandomPlayer
from fish.fish_naive_ai_player import NaiveAIPlayer
import tables
from ai.ai import AI
import keras
import tensorflow as tf

x1 = [[1,2,3],[7,4,6]]
x2 = [[54,72,32],[37,4534,636]]
xs = [np.asarray(x1), np.asarray(x2)]

total_generations = 5
models_per_generation = 10
num_games = 25
best_model = None
win_threshold = .75

write_new_file = True

cur_generation = 0
trained_models = 0

all_games_x = []
all_games_y = []
while cur_generation < total_generations:
    cur_iter = 0
    games_to_train_x = None
    games_to_train_y = None
    while cur_iter < models_per_generation:
        data_filename = 'data_gen_{0}_iter_{0}'.format(cur_generation, cur_iter)
        x = None
        y = None
        wins = {

        }
        player_1 = None
        player_2 = None
        game = None
        new_challenger_name = "Model {0}".format(trained_models)
        games_played = 0
        while games_played < num_games:
            game = FishGame()
            if best_model is None: #start with 2 idiots
                player_1 = NaiveAIPlayer(-1, "Model 0", game, initialize=True)
                player_2 = NaiveAIPlayer(0, "Model 0", game, initialize=True)
            else:
                player_1 = NaiveAIPlayer(best_model, "Model {0}".format(best_model), game, main_name='model', index=best_model, initialize =(best_model == 0))
                player_2 = NaiveAIPlayer(trained_models, new_challenger_name, game, main_name='model', index=trained_models)
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
            categorical_y = keras.utils.to_categorical(this_y, 3)
            scores = player_2.ai.model.evaluate([board_x, player_x], categorical_y, verbose=False)
            print("\n%s: %.2f%%" % (player_2.ai.model.metrics_names[1], scores[1] * 100))
            print("winner: {0}".format(winner))
            print(wins)

            all_games_x.append(this_x)
            all_games_y.append(this_y)

            if x is None:
                x = this_x
                y = this_y
            else:
                x = np.concatenate((x, this_x), axis=0)
                y = np.concatenate((y, this_y), axis=0)

        print(wins)
        if (cur_generation != total_generations - 1) or (cur_iter != models_per_generation - 1):

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

            if wins[new_challenger_name] > (win_threshold*games_played) or (trained_models == 0):
                print("NEW BEST PLAYER")
                best_model = trained_models
            else:
                best_model = best_model


            print('TRAINING NEW MODEL')
            total_games_played = len(all_games_x)
            if total_games_played > models_per_generation:
                x = np.concatenate(all_games_x[total_games_played-models_per_generation:])
                y = np.concatenate(all_games_y[total_games_played-models_per_generation:])
            else:
                x = np.concatenate(all_games_x)
                y = np.concatenate(all_games_y)


            ai = AI()
            ai.load_data(x, y)
            ai.create_model()
            ai.train_model()
            ai.save_model('model', index=trained_models+1)
            print("MODEL TRAINED LETS GO")
            trained_models += 1
        cur_iter += 1
    cur_generation += 1