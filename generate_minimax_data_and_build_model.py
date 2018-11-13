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
from ai.ai import AI


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

def generate_data(BREAK_MOVES, LAST_MODEL, MODEL_AGO, num_games=100):
    num_games = num_games
    x = None
    y = None
    write_data = True

    BREAK_MOVES = BREAK_MOVES
    LAST_MODEL = LAST_MODEL
    MODEL_AGO = MODEL_AGO
    data_file = 'minimax_{0}_to_{1}'.format(BREAK_MOVES, LAST_MODEL)
    model_name = 'minimax_{0}_to_{1}'.format(LAST_MODEL, MODEL_AGO)

    ai = AI()
    ai.load_model(model_name, 0)
    model = ai.model
    for i in tqdm(range(num_games)):

        game = FishGame()
        player_1 = RandomPlayer(0, "BOB", game)
        player_2 = RandomPlayer(1, "CHARLA", game)
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
                # display(game)
                game_tree = GameTree(game.get_current_player_id(), cur_moves=moves, max_moves=LAST_MODEL, model=model)
                game_tree.build_tree(game.get_state())
                minimax = MiniMax(game_tree)
                minimax.alpha_beta_search()
                if x is None:
                    x = minimax.get_minimax_history_for_neural_net()[0]
                    y = minimax.get_minimax_history_for_neural_net()[1]
                else:
                    x = np.concatenate((x, minimax.get_minimax_history_for_neural_net()[0]), axis=0)
                    y = np.concatenate((y, minimax.get_minimax_history_for_neural_net()[1]), axis=0)
                break
            cur_player = game.get_current_player()
            move = cur_player.move()
            game.do_move(move)

    if write_data:

        hdf5_file = tables.open_file('./data/{0}.hdf5'.format(data_file), 'w')
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

def train_model(BREAK_MOVES, LAST_MODEL):
    print("Loading data")
    BREAK_MOVES = BREAK_MOVES
    LAST_MODEL = LAST_MODEL
    data_file = 'minimax_{0}_to_{1}'.format(BREAK_MOVES, LAST_MODEL)
    hdf5_path = "./data/{0}.hdf5".format(data_file)
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    x = extendable_hdf5_file.root.x[:]
    y = extendable_hdf5_file.root.y[:]
    extendable_hdf5_file.close()
    print("Data loaded")

    latest_version = None

    ai = AI()
    ai_name = data_file
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

for i in range(42, 0, -2):
    BREAK_MOVES = i-2
    LAST_MODEL = i
    MODEL_AGO = i+2
    generate_data(BREAK_MOVES=BREAK_MOVES, LAST_MODEL=LAST_MODEL, MODEL_AGO=MODEL_AGO)
    train_model(BREAK_MOVES=BREAK_MOVES, LAST_MODEL=LAST_MODEL)