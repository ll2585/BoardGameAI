from pprint import pprint
import numpy as np
from fish.fish_game import FishGame
from fish.fish_players import RandomPlayer
import tables
from pathlib import Path
# Setup
from tqdm import tqdm

num_games = 100
x = None
y = None
wins = {

}
for i in tqdm(range(num_games)):
    game = FishGame()
    player_1 = RandomPlayer(0,"BOB",game)
    player_2 = RandomPlayer(1,"CHARLA",game)
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

    winner = game.get_winner()
    if winner not in wins:
        wins[winner] = 0
    wins[winner] += 1
    #print("winner: {0}".format(winner))
    if x is None:
        x = game.get_full_game_history_for_neural_net()[0]
        y = game.get_full_game_history_for_neural_net()[1]
    else:
        x = np.concatenate((x, game.get_full_game_history_for_neural_net()[0]), axis=0)
        y = np.concatenate((y, game.get_full_game_history_for_neural_net()[1]), axis=0)
print(wins)

hdf5_file = tables.openFile('../data/data.hdf5', 'w')
filters = tables.Filters(complevel=5, complib='blosc')
x_storage = hdf5_file.createEArray(hdf5_file.root, 'x',
                                      tables.Atom.from_dtype(x.dtype),
                                      shape=(0, x.shape[-1]),
                                      filters=filters,
                                      expectedrows=len(x))
y_storage = hdf5_file.createEArray(hdf5_file.root, 'y',
                                          tables.Atom.from_dtype(y.dtype),
                                          shape=(0,),
                                          filters=filters,
                                          expectedrows=len(y))
for n, (d, c) in enumerate(zip(x, y)):
    x_storage.append(x[n][None])
    y_storage.append(y[n][None])
hdf5_file.close()

#np.save('../data/x.npy', x)
#np.save('../data/y.npy', y)


