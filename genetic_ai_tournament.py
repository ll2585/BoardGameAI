from ai.genetic_ai import GeneticAI
import random
import tables
import numpy as np
from fish.fish_game import FishGame
from fish.fish_players import RandomPlayer
from tqdm import tqdm
from fish.fish_naive_ai_player import NaiveAIPlayer
import keras

cur_gen = 0
max_gens = 1
ais_per_gen = 2
keep_mother = True
keep_father = True
mutation_factor = {
    'board_layers' : .05,
    'board_neurons': .05,
    'player_layers': .05,
    'player_neurons': .05,
    'hidden_layers': .05,
    'hidden_neurons': .05
}
initial_mins = {
    'board_layers' : 0,
    'board_neurons': 0,
    'player_layers': 0,
    'player_neurons': 0,
    'hidden_layers': 0,
    'hidden_neurons': 0
}
initial_max = {
    'board_layers' : 5,
    'board_neurons': 500,
    'player_layers': 5,
    'player_neurons': 500,
    'hidden_layers': 5,
    'hidden_neurons': 500
}
print("Loading data")
hdf5_path = "./data/100_games.hdf5"
extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
x = extendable_hdf5_file.root.x[:]
y = extendable_hdf5_file.root.y[:]
extendable_hdf5_file.close()
print("Data loaded")
test_with_games = False

while cur_gen < max_gens:
    competitors = []
    if cur_gen == 0:
        mother = None
        father = None
        for i in range(ais_per_gen):
            board_layers = random.randrange(initial_mins['board_layers'],initial_max['board_layers'])
            board_neurons = [random.randrange(initial_mins['board_neurons'],initial_max['board_neurons']) for _ in range(board_layers)]
            player_layers = random.randrange(initial_mins['player_layers'],initial_max['player_layers'])
            player_neurons =[random.randrange(initial_mins['player_neurons'],initial_max['player_neurons']) for _ in range(player_layers)]
            hidden_layers = random.randrange(initial_mins['player_layers'],initial_max['board_layers'])
            hidden_neurons = [random.randrange(initial_mins['hidden_neurons'],initial_max['hidden_neurons']) for _ in range(hidden_layers)]
            competitors.append(GeneticAI(board_layers=board_layers,
                                         board_neurons=board_neurons,
                                         player_layers=player_layers,
                                         player_neurons = player_neurons,
                                         hidden_layers=hidden_layers,
                                         hidden_neurons = hidden_neurons))
    latest_version = None
    for i, ai in enumerate(competitors):
        ai_name = 'gen_{0}_ai_{1}'.format(cur_gen, i)
        ai.load_data(x, y)
        print("AI Loaded data")
        if latest_version is None:
            ai.create_model()
            latest_version = -1
        else:
            ai.load_model(ai_name, index=latest_version)
            print("AI Model Loaded")
        ai.train_model(x=x, y=y)
        ai.save_model(ai_name, index=latest_version + 1)
        print(ai.hist.history['val_loss'][-1])
        raise Exception


        if test_with_games:
            num_games = 5

            x = None
            y = None
            wins = {

            }

            game = FishGame()
            player_1 = RandomPlayer(0, "BOB", game)
            player_2 = NaiveAIPlayer(1, "CHARLA", game, main_name=ai_name, index=latest_version + 1)
            for i in tqdm(range(num_games)):
                game.set_up()
                players = [player_1, player_2]
                for player in players:
                    player.reset()
                    game.add_player(player)
                game.start()

                while not game.is_over():
                    cur_player = game.get_current_player()
                    if cur_player.get_player_id() == 0:
                        move = cur_player.move(seed=212)
                    else:
                        move = cur_player.move()
                    game.do_move(move)

                # 0 is the Player[0], 1 is Player[1]
                winner = game.get_winner()
                if winner not in wins:
                    wins[winner] = 0
                wins[winner] += 1

                this_x = game.get_full_game_history_for_neural_net()[0]
                this_y = game.get_full_game_history_for_neural_net()[1]

                board_x = this_x[:, :60]
                player_x = this_x[:, 60:]
                categorical_y = keras.utils.to_categorical(this_y, 3)

                scores = player_2.ai.model.evaluate([board_x, player_x], categorical_y, verbose=False)
                print("\n%s: %.2f%%" % (player_2.ai.model.metrics_names[1], scores[1] * 100))
                print("winner: {0}".format(winner))
                print(wins)

                if x is None:
                    x = this_x
                    y = this_y
                else:
                    x = np.concatenate((x, this_x), axis=0)
                    y = np.concatenate((y, this_y), axis=0)

            print(wins)
    else:
        print()
    cur_gen += 1