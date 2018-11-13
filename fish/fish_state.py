import numpy as np

class FishState:
    def __init__(self, board, players, current_player_id, player_who_moved, player_ids):
        self.board = board #already pieces
        self.players = players #[self.penguins, self.score, self.tiles_collected]
        self.current_player_id = current_player_id
        self.player_who_moved = player_who_moved
        self.player_ids = player_ids #[p.get_player_id() for p in self.players]

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
            if cell.value == -1:
                board_points.append(0)
            else:
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

    def get_state_from_move(self, action):
        #print("ACTION", action)
        def no_penguin_moves(board, penguins, player_id):
            #print("CHECK P", penguins, "AND PID", player_id)
            from .fish_board import get_board_from_pieces
            next_player_penguin_moves = []
            board_clone = get_board_from_pieces(board)
            for peng in penguins:
                next_player_penguin_moves += board_clone.get_legal_moves(peng, player_id=player_id)
            #print("DAMOVES", next_player_penguin_moves)
            return len(next_player_penguin_moves) == 0
        from copy import deepcopy
        player_id = action.player_id
        player_who_moved = player_id
        new_player_index = None
        next_player_id = None
        new_board = deepcopy(self.board)
        new_players = deepcopy(self.players)  # [penguins, score, tiles_collected]
        player_ids = self.player_ids
        for i, p_id in enumerate(player_ids):
            if player_id == p_id:
                new_player_index = i
            else:
                next_player_id = p_id
        if new_player_index is None:
            raise Exception("This player not playing")
        if next_player_id is None:
            raise Exception("No next player")
        cur_player = new_players[new_player_index]
        next_player_index = 1 if new_player_index == 0 else 0
        next_player = new_players[next_player_index]
        if action.type == "move":
            hex_from = new_board[action.start]
            hex_from.move_penguin_away()

            cur_player[2] += 1
            cur_player[1] += hex_from.value

            hex_from.empty()

            if action.start != action.end:
                hex_to = new_board[action.end]
                hex_to.move_penguin_here()

            start = action.start
            end = action.end
            for i, penguin in enumerate(cur_player[0]):
                if penguin == start:
                    if start == end:
                        # penguin died
                        cur_player[0][i] = -1
                    else:
                        cur_player[0][i] = end
                    break

            if next_player[0] == [-1, -1, -1, -1]:
                next_player_id = player_id

            while no_penguin_moves(new_board, new_players[next_player_id][0], player_ids[next_player_id]):
                #print("RUN TWICE")
                penguins = new_players[next_player_id][0]
                for p in penguins:
                    penguin_at_hex = new_board[p]
                    penguin_at_hex.move_penguin_away()
                    new_players[next_player_id][2] += 1
                    new_players[next_player_id][1] += penguin_at_hex.value
                    penguin_at_hex.empty()
                    new_players[next_player_id][0] = [-1, -1, -1, -1]
                next_player_id = player_id
                if new_players[0][0] == [-1, -1, -1, -1] and new_players[1][0] == [-1, -1, -1, -1]:
                    break
        elif action.type == "place":
            player_who_moved = player_id
            hex = new_board[action.start]
            hex.move_penguin_here()
            cur_player[0].append(action.start)
        else:
            raise Exception("WRONG ACTION TYPE")
        return FishState(new_board, new_players, next_player_id, player_who_moved, player_ids)

    def get_possible_moves(self, player_id=None):
        from .fish_move import FishMove
        from .fish_board import get_board_from_pieces

        def is_over():
            player_1 = self.players[0]
            player_2 = self.players[1]
            return player_1[0] == [-1, -1, -1, -1] and player_2[0] == [-1, -1, -1, -1]
        legal_moves = []
        if is_over():
            return legal_moves
        if player_id is not None:
            if player_id != self.current_player_id:
                return []
        cur_player = self.current_player_id
        cur_player_index = 0 if cur_player == self.player_ids[0] else 1
        penguins = self.players[cur_player_index][0]
        if len(penguins) < 4:
            for i, hex in enumerate(self.board):
                if not hex.has_penguin_here():
                    legal_moves.append(FishMove(i, player_id=player_id, type="place"))
        else:
            board_clone = get_board_from_pieces(self.board)
            for penguin in penguins:
                legal_moves += board_clone.get_legal_moves(penguin, player_id=player_id)
            #print("LEGA",legal_moves)
        return legal_moves

    def convert_to_xs_for_neural_net(self):
        serialization = self.serialize()['serialization']
        return np.asarray([serialization])

    def get_value_of_state(self, player_id):
        def is_over():
            player_1 = self.players[0]
            player_2 = self.players[1]
            return player_1[0] == [-1, -1, -1, -1] and player_2[0] == [-1, -1, -1, -1]
        if not is_over():
            return 0

        player_1 = self.players[0]
        player_2 = self.players[1]

        player_1_score = player_1[1]
        player_2_score = player_2[1]
        if player_1_score > player_2_score:
            winner = self.player_ids[0]
        elif player_2_score > player_1_score:
            winner = self.player_ids[1]
        elif player_1_score == player_2_score:
            player_1_tiles = player_1[2]
            player_2_tiles = player_2[2]
            if player_1_tiles > player_2_tiles:
                winner = self.player_ids[0]
            elif player_2_tiles > player_1_tiles:
                winner = self.player_ids[1]
            else:
                # draw has a very little value
                return .00000001
        if winner == player_id:
            return 1
        else:
            return -1

    def get_hash(self):
        serialization = self.serialize()['serialization']
        return '_'.join([str(i) for i in serialization.tolist()])

def display_state(state):
    board_pieces = state.board
    players = state.players
    p1_penguins = players[0][0]
    p2_penguins = players[1][0]
    rows = ['  ', '',
            '  ', '',
            '  ', '',
            '  ', '']
    ends = [7, 15, 22, 30, 37, 45, 52, 60]
    for i in range(len(board_pieces)):
        for j, end in enumerate(ends):
            if i < end:
                value = board_pieces[i].value
                if value == -1:
                    value = '_'
                if i in p1_penguins:
                    penguin = "P"
                elif i in p2_penguins:
                    penguin = "Q"
                else:
                    penguin = "_"
                value = '{0}{1}'.format(str(value), penguin)
                rows[j] += str(value) + "  "
                break
    rows.extend(["Player 0: points/tiles: {0}/{1}".format(state.players[0][1], state.players[0][2])])
    rows.extend(["Player 1: points/tiles: {0}/{1}".format(state.players[1][1], state.players[1][2])])
    print ('\n'.join(rows))