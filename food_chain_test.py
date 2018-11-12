from food_chain.food_chain_game import FoodChainGame, display
from food_chain.food_chain_players import RandomPlayer

wins = {}
game = FoodChainGame()
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