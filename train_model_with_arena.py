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

max_iters = 10
num_games = 15
win_threshold = 7
last_model = 0
best_model = None
random_player_1 = True


for iters in tqdm(range(max_iters)):
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
    for i in tqdm(range(num_games)):
        game = FishGame()
        if random_player_1:
            player_1 = RandomPlayer(0,"Random Bob",game)
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
        print("winner: {0}".format(winner))
        if x is None:
            x = game.get_full_game_history_for_neural_net()[0]
            y = game.get_full_game_history_for_neural_net()[1]
        else:
            x = np.concatenate((x, game.get_full_game_history_for_neural_net()[0]), axis=0)
            y = np.concatenate((y, game.get_full_game_history_for_neural_net()[1]), axis=0)

    print(wins)
    hdf5_path = "./data/data.hdf5"
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='a')
    extendable_hdf5_x = extendable_hdf5_file.root.x
    extendable_hdf5_y = extendable_hdf5_file.root.y
    for n, (d, c) in enumerate(zip(x, y)):
        extendable_hdf5_x.append(x[n][None])
        extendable_hdf5_y.append(y[n][None])
    extendable_hdf5_file.close()

    if wins[new_challenger_name] < win_threshold:
        #check player 0 first because that's the random guy and we want to keep him if we can
        best_model = best_model
    else:
        if random_player_1:
            random_player_1 = False
        print("NEW BEST PLAYER")
        best_model = last_model+iters

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