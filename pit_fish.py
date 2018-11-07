import Arena
#from MCTS import MCTS
from fish.fish_game import FishGame, display
from fish.fish_players import *
from fish.fish_naive_ai_player import NaiveAIPlayer
#from tictactoe.keras.NNet import NNetWrapper as NNet

import numpy as np
#from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = FishGame()

# all players
player_1 = RandomPlayer(0,"BOB",g)
player_2 = RandomPlayer(1,"CHARLA",g)
naive_1_iter = NaiveAIPlayer(2,"Anna",g,main_name='model', index=0)


arena = Arena.Arena(g, player_1, naive_1_iter, display=display)
print(arena.playGames(20, verbose=False))
