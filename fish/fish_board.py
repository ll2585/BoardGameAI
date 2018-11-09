import random
from .fish_move import FishMove

class FishBoard():
    def __init__(self):
        "Set up initial board configuration."

        # Create the empty board array.
        self.pieces = []
        self.location_dict = {}
        # figure out some pattern but otherwise do this
        # xs
        #  0 1 2 3 4 5 6
        # 1 0 1 2 3 4 5 6
        #  1 0 1 2 3 4 5
        # 2 1 0 1 2 3 4 5
        #  2 1 0 1 2 3 4
        # 3 2 1 0 1 2 3 4
        #  3 2 1 0 1 2 3
        # 4 3 2 1 0 1 2 3
        #
        # ys
        #  0 1 2 3 4 5 6
        # 0 1 2 3 4 5 6 7
        #  1 2 3 4 5 6 7
        # 1 2 3 4 5 6 7 8
        #  2 3 4 5 6 7 8
        # 2 3 4 5 6 7 8 9
        #  3 4 5 6 7 8 9
        # 3 4 5 6 7 8 9 10
        #
        # zs
        #  0 0 0 0 0 0 0
        # 1 1 1 1 1 1 1 1
        #  2 2 2 2 2 2 2
        # 3 3 3 3 3 3 3 3
        #  4 4 4 4 4 4 4
        # 5 5 5 5 5 5 5 5
        #  6 6 6 6 6 6 6
        # 7 7 7 7 7 7 7 7

        for i in range(7):
            self.pieces.append(Hexagon(i, i, 0))
        for i in range(8):
            self.pieces.append(Hexagon(i - 1, i, 1))
        for i in range(7):
            self.pieces.append(Hexagon(i - 1, i + 1, 2))
        for i in range(8):
            self.pieces.append(Hexagon(i - 2, i + 1, 3))
        for i in range(7):
            self.pieces.append(Hexagon(i - 2, i + 2, 4))
        for i in range(8):
            self.pieces.append(Hexagon(i - 3, i + 2, 5))
        for i in range(7):
            self.pieces.append(Hexagon(i - 3, i + 3, 6))
        for i in range(8):
            self.pieces.append(Hexagon(i - 4, i + 3, 7))
        values = [1] * 30 + [2] * 20 + [3] * 10
        random.shuffle(values)
        for i, piece in enumerate(self.pieces):
            self.location_dict['{x}-{y}-{z}'.format(x=piece.x, y=piece.y, z=piece.z)] = i
            piece.set_value(values[i])

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, fish_index, player):
        """Returns all the legal moves for the given player.
        """
        moves = []
        if fish_index == -1:
            return moves
        for direction in ["left", "right", "top_left", "bottom_right", "top_right", "bottom_left"]:
            moves += self.get_moves_direction(fish_index, direction, player)
        if len(moves) == 0:
            # penguin cant move - can kill itself - JK
            pass
            #moves.append(FishMove(fish_index, fish_index, player=player))
        return moves

    def get_number_of_possible_moves(self):
        moves = 0
        for i in range(len(self.pieces)):
            for direction in ["left", "right", "top_left", "bottom_right", "top_right", "bottom_left"]:
                moves += self.get_all_moves(i, direction)
        return moves


    def get_all_moves(self, start, direction):
        my_hex = self.pieces[start]
        my_x = my_hex.x
        my_y = my_hex.y
        my_z = my_hex.z
        x = my_x
        y = my_y
        z = my_z
        total_moves = 0

        while True:
            if direction == 'left':
                x -= 1
                y -= 1
            elif direction == 'right':
                x += 1
                y += 1
            elif direction == 'top_left':
                y -= 1
                z -= 1
            elif direction == 'bottom_right':
                y += 1
                z += 1
            elif direction == 'top_right':
                z -= 1
                x += 1
            elif direction == 'bottom_left':
                z += 1
                x -= 1
            else:
                raise Exception("WRONG DIRECTION")
            index = '{x}-{y}-{z}'.format(x=x, y=y, z=z)
            if index not in self.location_dict:
                break
            total_moves += 1

        return total_moves

    def get_moves_direction(self, start, direction, player):
        my_hex = self.pieces[start]
        my_x = my_hex.x
        my_y = my_hex.y
        my_z = my_hex.z
        x = my_x
        y = my_y
        z = my_z
        moves = []

        while True:
            if direction == 'left':
                x -= 1
                y -= 1
            elif direction == 'right':
                x += 1
                y += 1
            elif direction == 'top_left':
                y -= 1
                z -= 1
            elif direction == 'bottom_right':
                y += 1
                z += 1
            elif direction == 'top_right':
                z -= 1
                x += 1
            elif direction == 'bottom_left':
                z += 1
                x -= 1
            else:
                raise Exception("WRONG DIRECTION")
            index = '{x}-{y}-{z}'.format(x=x, y=y, z=z)
            if index not in self.location_dict:
                break
            hex = self.pieces[self.location_dict[index]]

            if hex.has_penguin_here() or hex.is_empty():
                break
            moves.append(FishMove(start, self.location_dict[index], player=player))

        return moves

    def has_legal_moves(self):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction;
        @param color (1=white,-1=black)
        """
        win = self.n
        # check y-strips
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
            if count == win:
                return True
        # check x-strips
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self[x][y] == color:
                    count += 1
            if count == win:
                return True
        # check two diagonal strips
        count = 0
        for d in range(self.n):
            if self[d][d] == color:
                count += 1
        if count == win:
            return True
        count = 0
        for d in range(self.n):
            if self[d][self.n - d - 1] == color:
                count += 1
        if count == win:
            return True

        return False

    def execute_move(self, move, color):
        """Perform the given move on the board;
        color gives the color pf the piece to play (1=white,-1=black)
        """

        (x, y) = move

        # Add the piece to the empty square.
        assert self[x][y] == 0
        self[x][y] = color


class Hexagon:
    def __init__(self, x, y, z, value=-1):
        self.x = x
        self.y = y
        self.z = z
        self.value = value
        self.has_penguin = False

    def set_value(self, value):
        self.value = value

    def move_penguin_away(self):
        self.has_penguin = False

    def move_penguin_here(self):
        self.has_penguin = True

    def has_penguin_here(self):
        return self.has_penguin

    def is_empty(self):
        return self.value == -1

    def empty(self):
        self.value = -1

    def __repr__(self):
        return str({
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'value': self.value
        })