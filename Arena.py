import numpy as np
from tqdm import tqdm
import time
import math
import time
class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, game, player_1, player_2, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player_1 = player_1
        self.player_2 = player_2
        self.game = game
        self.display = display

    def playGame(self, players, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        game = self.game
        game.set_up()
        for player in players:
            player.reset()
            game.add_player(player)
        game.start()
        turn = 0
        while not game.is_over():
            cur_player = game.get_current_player()
            turn += 1
            if verbose:
                assert self.display
                print("Turn {turn}, Player {player}".format(turn = turn, player = cur_player.get_name()))
                self.display(game)
            move = cur_player.move()
            game.do_move(move)

        winner = game.get_winner_id()
        if verbose:
            assert (self.display)
            print("Game over: Turn {turn}, Result {winner}".format(turn=turn, winner=winner))
            self.display(game)
        return winner

    def playGames(self, num_games, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        player_1_wins = 0
        player_2_wins = 0
        player_1_id = self.player_1.get_player_id()
        player_2_id = self.player_2.get_player_id()
        draws = 0
        player_1_starts = math.floor(num_games/2)
        for _ in tqdm(range(player_1_starts)):
            winner = self.playGame([self.player_1, self.player_2], verbose=verbose)
            print('result', winner)
            if winner == player_1_id:
                player_1_wins += 1
            elif winner == player_2_id:
                player_2_wins += 1
            else:
                draws += 1

        player_2_starts = num_games - player_1_starts
        print(player_2_starts)
        for _ in tqdm(range(player_2_starts)):
            winner = self.playGame([self.player_2, self.player_1], verbose=verbose)
            print('result', winner)
            if winner == player_1_id:
                player_1_wins += 1
            elif winner == player_2_id:
                player_2_wins += 1
            else:
                draws += 1

        return player_1_wins, player_2_wins, draws
