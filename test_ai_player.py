from pprint import pprint
import numpy as np
from fish.fish_game import FishGame, display
from fish.fish_players import RandomPlayer
from fish.fish_naive_ai_player import NaiveAIPlayer
import tables
from pathlib import Path
# Setup
from tqdm import tqdm

num_games = 20
#if random wins 9 or more games, keep random
#otherwise keep the other model
x = None
y = None
wins = {

}

write_data = True

for i in tqdm(range(num_games)):
    game = FishGame()
    player_1 = RandomPlayer(0,"CHARLA",game)
    player_2 = NaiveAIPlayer(1,"CHARLA",game,main_name='model', index=2)
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
    winner = game.get_winner()
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

if write_data:
    hdf5_path = "./data/data.hdf5"
    extendable_hdf5_file = tables.openFile(hdf5_path, mode='a')
    extendable_hdf5_x = extendable_hdf5_file.root.x
    extendable_hdf5_y = extendable_hdf5_file.root.y
    for n, (d, c) in enumerate(zip(x, y)):
        extendable_hdf5_x.append(x[n][None])
        extendable_hdf5_y.append(y[n][None])
    extendable_hdf5_file.close()