from fish.fish_players import Player
from ai.ai import AI
from copy import deepcopy
import numpy as np
import random
import time
from fish.fish_state import FishState
import random, math

def choose_from_probs(probs):
    # will almost always make optimal decision;
    if len(probs) > 4:
        sorted_probs = np.sort(probs)[::-1] #descending
        #only look at top 25%
        last_index = .25*len(probs)
        probs = sorted_probs[:math.floor(last_index)]
    weights = probs / np.sum(probs)
    choice = np.random.choice(range(len(probs)), size=1, p=weights)
    return choice[0]

class NaiveAIPlayer(Player):
    def __init__(self, player_id, name, game, main_name=None, index=None, initialize=False, filtered_moves = None):
        Player.__init__(self, player_id, name, game)
        self.ai = AI()
        if not initialize:
            assert main_name is not None
            assert index is not None
            self.ai.load_model(main_name, index=index)
        else:
            self.ai.create_model()
        self.last_model_loaded = None
        self.filtered_moves = filtered_moves

    def get_model_move(self, possible_moves):
        new_states = None
        for move in possible_moves:
            '''
            start = time.time()
            cloned_game = deepcopy(self.game)
            print(time.time() - start)

            start = time.time()
            cloned_game.do_move(move, save_history=False)
            print(time.time() - start)
            start = time.time()
            new_state = cloned_game.get_state().convert_to_xs_for_neural_net()
            print(time.time() - start)
            '''
            new_state = self.state_from_move(move).convert_to_xs_for_neural_net()
            if new_states is None:
                new_states = new_state
            else:
                new_states = np.concatenate((new_states, new_state), axis=0)
        predictions = self.ai.make_prediction(new_states)
        print(predictions)
        choice_index = choose_from_probs(predictions)
        return possible_moves[choice_index]

    def get_random_move(self, possible_moves):
        return random.choice(possible_moves)

    def get_move_by_total_moves(self, total_moves):
        possible_moves = self.game.get_possible_moves(self.player_id)
        if total_moves < 50:
            move = self.get_random_move(possible_moves)
        else:
            if total_moves < 40:
                model_name = 'minimax_38_to_40'
            elif total_moves < 42:
                model_name = 'minimax_40_to_42'
            elif total_moves < 44:
                model_name = 'minimax_42_to_44'
            elif total_moves < 47:
                model_name = 'minimax_44_to_47'
            elif total_moves < 50:
                model_name = 'minimax_47_to_50'
            else:
                model_name = 'minimax_50'
            if self.last_model_loaded != model_name:
                self.ai.load_model(model_name, index=0)
                self.last_model_loaded = model_name
            move = self.get_model_move(possible_moves)
        return move

    def move(self):
        possible_moves = self.game.get_possible_moves(self.player_id)
        if self.filtered_moves is not None:
            print(self.game.get_moves_played())
            if (self.game.get_moves_played()-1) >= self.filtered_moves: # -1 because the next state is equal to the
                # filtered moves...also should be tiles collected but whatevr
                move = self.get_model_move(possible_moves)
            else:
                move = self.get_random_move(possible_moves)
        else:
            move = self.get_random_move(possible_moves)
        return move



    def state_from_move(self, action):
        player_id = action.player_id
        player_who_moved = player_id
        new_board = deepcopy(self.game.board)
        new_player_index = None
        new_players = deepcopy([p.get_state() for p in self.game.players])  # [penguins, score, tiles_collected]
        for i, player in enumerate(self.game.players):
            if player_id == player.get_player_id():
                new_player_index = i
            else:
                next_player_id = player.get_player_id()
        if new_player_index is None:
            raise Exception("This player not playing")
        cur_player = new_players[new_player_index]
        next_player_index = 1 if new_player_index == 0 else 0
        next_player = new_players[next_player_index]
        player_ids = deepcopy(self.game.get_player_ids())

        if action.type == "move":
            hex_from = new_board.pieces[action.start]
            hex_from.move_penguin_away()

            cur_player[2] += 1
            cur_player[1] += hex_from.value

            hex_from.empty()

            if action.start != action.end:
                hex_to = new_board.pieces[action.end]
                hex_to.move_penguin_here()

            start = action.start
            end = action.end
            for i, penguin in enumerate(cur_player[0]):
                if penguin == start:
                    if start == end:
                        # penguin died
                        cur_player[0][i] = end = -1
                    else:
                        cur_player[0][i] = end
                    break

            if next_player[0] == [-1, -1, -1, -1]:
                next_player_id = player_id
        elif action.type == "place":
            player_who_moved = player_id
            hex = new_board.pieces[action.start]
            hex.move_penguin_here()
            cur_player[0].append(action.start)
        else:
            raise Exception("WRONG ACTION TYPE")
        return FishState(deepcopy(new_board.pieces), deepcopy(new_players), deepcopy(next_player_id), deepcopy(player_who_moved), deepcopy(player_ids))

    def train_and_save_new_model(self,x,y, new_index):
        self.ai.load_data(x, y)
        self.ai.create_model()
        self.ai.train_model()
        self.ai.save_model('model', index=new_index)