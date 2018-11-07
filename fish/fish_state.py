import numpy as np

class FishState:
    def __init__(self, board, players, current_player_id, player_who_moved, player_ids):
        self.board = board
        self.players = players
        self.current_player_id = current_player_id
        self.player_who_moved = player_who_moved
        self.player_ids = player_ids

    def serialize(self):
        serialization = []
        board_points = []
        board_penguins = []
        player_who_moved = self.player_who_moved
        player_ids = self.player_ids

        if player_who_moved == player_ids[0]:
            player_1 = self.players[0]
            player_2 = self.players[1]
        else:
            assert player_who_moved == player_ids[1]
            player_1 = self.players[1]
            player_2 = self.players[0]

        player_1_penguins = player_1[0]
        player_2_penguins = player_2[0]



        #from player who moved's perspective

        for i, cell in enumerate(self.board):
            if i in player_1_penguins:
                board_penguins.append(1)
            else:
                board_penguins.append(0)
            if i in player_2_penguins:
                board_penguins.append(1)
            else:
                board_penguins.append(0)
            board_points.append(cell.value)

        player_1_serialization = player_1[1:]
        player_2_serialization = player_2[1:]

        serialization.extend(board_points)
        serialization.extend(board_penguins)
        serialization.extend(player_1_serialization)
        serialization.extend(player_2_serialization)

        serialization = np.asarray(serialization)

        return {
            'serialization': serialization,
            'player_who_moved': player_who_moved
        }

    def convert_to_xs_for_neural_net(self):
        serialization = self.serialize()['serialization']
        return np.asarray([serialization])