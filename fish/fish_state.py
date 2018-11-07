import numpy as np

class FishState:
    def __init__(self, board, players, current_player_id, player_who_moved):
        self.board = board
        self.players = players
        self.current_player_id = current_player_id
        self.player_who_moved = player_who_moved

    def serialize(self):
        adjusted_board = []
        player_1_penguin = 0
        player_1 = self.players[0]
        player_1_penguins = player_1[0]
        player_2_penguin = 1
        player_2 = self.players[1]
        player_2_penguins = player_2[0]

        player_who_moved = self.player_who_moved

        empty_cell = -1
        one_point = 1
        two_points = 2
        three_points = 3
        adjusted_one_point = 11
        adjusted_two_points = 12
        adjusted_three_points = 13
        point_mapping = {
            empty_cell: empty_cell,
            one_point: adjusted_one_point,
            two_points: adjusted_two_points,
            three_points: adjusted_three_points
        }
        for i, cell in enumerate(self.board):
            if cell.has_penguin_here():
                if i in player_1_penguins:
                    adjusted_board.append(player_1_penguin)
                else:
                    assert i in player_2_penguins
                    adjusted_board.append(player_2_penguin)
            else:
                adjusted_board.append(point_mapping[cell.value])

        player_1_serialization = player_1[1:]
        player_2_serialization = player_2[1:]

        adjusted_board.extend(player_1_serialization)
        adjusted_board.extend(player_2_serialization)

        serialization = np.asarray(adjusted_board)

        return {
            'serialization': serialization,
            'player_who_moved': player_who_moved
        }

    def convert_to_xs_for_neural_net(self):
        serialization = self.serialize()['serialization']
        return np.asarray([serialization])