import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from fish.fish_game import FishGame
from fish.fish_players import RandomPlayer
from fish.fish_board import get_all_actions
from fish.fish_state import get_state_from_serialization, display_state

class FishEnv(gym.Env):
    #p2 is random player
    def __init__(self):
        self.game = None
        self.state = None
        #need this i guess
        self.player_id = 0
        self.move_map = get_all_actions(self.player_id)
        self.hashed_move_map = [move.get_hash() for move in self.move_map]
        self.action_space = spaces.Discrete(len(self.move_map))
        self.observation_space = None

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        this_action = self.move_map[action]
        #return np array
        #penalty for invalid action
        actual_action = self.move_map[action]
        move_index = self.hashed_move_map.index(this_action.get_hash())
        move = self.move_map[move_index]
        self.game.do_move(move)
        while not self.game.is_over() and self.game.get_current_player_id() != self.player_id:
            cur_player = self.game.get_current_player()
            seed=10124386
            move = cur_player.move(seed)
            self.game.do_move(move)
        done = self.game.is_over()
        if done:
            reward = 1 if self.game.get_winner_id() == self.player_id else -1
        else:
            reward = 0
        self.state = self.game.get_state()
        if reward != 0:
            print("REWARD IS ", reward)
        xs = self.state.convert_to_xs_for_neural_net()[0]
        return xs, reward, done, {}

    def reset(self):
        #return np array - of the state?
        #you are player 1
        self.game = FishGame()
        player_1 = RandomPlayer(self.player_id, "BOB", self.game)
        player_2 = RandomPlayer(1, "CHARLA", self.game)
        self.game.set_up()
        players = [player_1, player_2]
        for player in players:
            player.reset()
            self.game.add_player(player)
        self.game.start()
        self.state = self.game.get_state()
        xs = self.state.convert_to_xs_for_neural_net()[0]
        self.observation_space = spaces.Box(low=0,high=3*10+2*20+1*30, dtype=np.uint8, shape=(185,)) # num of columns... or it is just shape, will have to test
        return xs

    def render(self, mode='human'):
        display_state(self.game.get_state())

    def close(self):
        pass