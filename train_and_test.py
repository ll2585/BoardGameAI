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
import constants

constants.use_gpu()


total_generations = 25
models_per_generation = 10 # data capped here
num_games = 50
run = 1

best_model = 48 #has to be 1 less than total models...
trained_models = -1 if best_model == 0 else best_model+1 #-1 if best model is 0

challenger_name = "Model {0}".format(trained_models)

game = FishGame()
best_player = NaiveAIPlayer(best_model, "Model {0}".format(best_model), game, main_name='model', index=best_model, initialize =(best_model == 0))
challenger = NaiveAIPlayer(trained_models, challenger_name, game, main_name='model', index=trained_models, initialize =(best_model == 0))

win_threshold = .55

write_new_file = True

cur_generation = 0


all_games_x = []
all_games_y = []
while cur_generation < total_generations:
    cur_iter = 0
    games_to_train_x = None
    games_to_train_y = None
    while cur_iter < models_per_generation:
        data_filename = 'data_run_{0}_gen_{1}_iter_{2}'.format(run, cur_generation, cur_iter)
        print(data_filename)
        x = None
        y = None
        wins = {

        }
        player_1 = None
        player_2 = None
        challenger_name = "Model {0}".format(trained_models)
        challenger_id = trained_models
        challenger.set_player_id(challenger_id)
        challenger.set_name(challenger_name)
        games_played = 0

        print("Setting up players")
        if best_model is None: #start with 2 idiots
            player_1 = NaiveAIPlayer(-1, "Model 0", game, initialize=True)
            player_2 = NaiveAIPlayer(0, "Model 0", game, initialize=True)
        else:
            player_1 = best_player
            player_2 = challenger
        players = [player_1, player_2]
        print("players set up.")
        while games_played < num_games:
            game.set_up()
            for player in players:
                player.reset()
                game.add_player(player)
            game.start()

            while not game.is_over():
                cur_player = game.get_current_player()
                move = cur_player.move()
                game.do_move(move)
            print("Game over")
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

                filename = './data/{0}.hdf5'.format(data_filename)
                constants.write_to_new_file(filename, x, y)

            if wins[challenger_name] > (win_threshold*games_played) or (trained_models == 0):
                print("NEW BEST PLAYER")
                best_model = trained_models
                new_challenger = best_player
                best_player = challenger
                challenger = new_challenger
            else:
                best_model = best_model
                best_player = best_player
                challenger = challenger

            print('TRAINING NEW MODEL')
            total_games_played = len(all_games_x)

            if total_games_played > models_per_generation*num_games:
                print("breaking it out")
                x = np.concatenate(all_games_x[total_games_played-models_per_generation*num_games:])
                y = np.concatenate(all_games_y[total_games_played-models_per_generation*num_games:])
            else:
                print("all x")
                x = np.concatenate(all_games_x)
                y = np.concatenate(all_games_y)
            if trained_models == -1:
                trained_models = 0
            challenger.train_and_save_new_model(x,y,trained_models+1)
            print("MODEL TRAINED LETS GO")
            trained_models += 1
        cur_iter += 1
    cur_generation += 1

filename = './data/all_games_run_{0}.hdf5'.format(run)
constants.write_to_new_file(filename, np.concatenate(all_games_x), np.concatenate(all_games_y))
