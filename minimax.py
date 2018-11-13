from fish.fish_state import display_state
import numpy as np

class MiniMax:
    def __init__(self, game_tree):
        self.game_tree = game_tree
        self.root = game_tree.root
        self.current_node = None
        self.children = []
        self.xs = None
        self.ys = None
        self.value_dict = {}

    def alpha_beta_search(self, node=None):
        if node is None:
            node = self.root
        self.xs = []
        self.ys = []
        infinity = float('inf')
        best_val = -infinity
        beta = infinity

        children = self.get_children(node)
        best_state = None
        for state in children:
            value = self.value(state, best_val, beta, max_or_min='min')
            self.value_dict[state] = value
            if value > best_val:
                best_val = value
                best_state = state
        return best_state

    def minimax(self, node=None):
        if node is None:
            node = self.root
        self.xs = []
        self.ys = []
        # first, find the max value
        best_val = self.value(node, max_or_min='max')  # should be root node of tree

        successors = self.get_children(node)
        # find the node with our best move
        best_move = None
        for elem in successors:  # ---> Need to propagate values up tree for this to work
            if elem.value == best_val:
                best_move = elem
                break

        # return that best value that we've found
        return best_move

    def append_to_x_y(self, node, value):
        state = node.state
        #print('########################')
        #display_state(state)
        #print(state.get_hash())
        #print('value: {0}'.format(value))
        #print('starting player: {0}'.format(node.player_id))
        #print('last person to move: {0}'.format(state.player_who_moved))
        #print('########################')
        initial_player_id = node.player_id
        serialization = state.serialize()
        x = serialization['serialization']
        self.xs.append(x)
        player_who_moved = serialization['player_who_moved']

        if value == 1:
            if player_who_moved == initial_player_id:
                self.ys.append(0)
            else:
                # draw
                self.ys.append(1)
        elif value == -1:
            if player_who_moved == initial_player_id:
                # lose
                self.ys.append(1)
            else:
                # win
                self.ys.append(0)
        else:
            self.ys.append(2) #draw

    def value(self, node, max_or_min):
        if max_or_min != 'max' and max_or_min != 'min':
            raise Exception("MAX OR MIN NEEDS TO BE max OR min")
        if self.is_terminal(node):
            value = self.get_value(node)
            self.append_to_x_y(node, value)
            return value

        infinity = float('inf')
        if max_or_min == 'max':
            value = -infinity
        else:
            value = infinity

        children = self.get_children(node)
        for child in children:
            if max_or_min == 'max':
                value = max(value, self.value(child, max_or_min='min'))
            else:
                value = min(value, self.value(child, max_or_min='max'))
        self.append_to_x_y(node, value)
        return value

    def value_alpha_beta(self, node, alpha, beta, max_or_min):
        if max_or_min != 'max' and max_or_min != 'min':
            raise Exception("MAX OR MIN NEEDS TO BE max OR min")
        if self.is_terminal(node):
            value = self.get_value(node)
            self.append_to_x_y(node, value)
            return value

        infinity = float('inf')
        if max_or_min == 'max':
            value = -infinity
        else:
            value = infinity

        children = self.get_children(node)
        for child in children:
            if max_or_min == 'max':
                value = max(value, self.value_alpha_beta(child, alpha, beta, max_or_min='min'))
                if value >= beta:
                    self.append_to_x_y(child, value)
                    return value
                alpha = max(alpha, value)
            else:
                value = min(value, self.value_alpha_beta(child, alpha, beta, max_or_min='max'))
                if value <= alpha:
                    self.append_to_x_y(child, value)
                    return value
                beta = min(beta, value)
        self.append_to_x_y(node, value)
        return value

    def get_children(self, node):
        assert node is not None
        return node.children

    def is_terminal(self, node):
        assert node is not None
        return len(node.children) == 0

    def get_value(self, node):
        assert node is not None
        return node.value

    def get_minimax_history_for_neural_net(self):
        return np.asarray(self.xs, dtype=np.float32), np.asarray(self.ys, dtype=np.float32)

class GameNode:
    def __init__(self, state, initial_player_id, value=0, parent=None):
        self.state = state
        self.player_id = initial_player_id
        self.value = value
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

class GameTree:
    def __init__(self, player_id, cur_moves = None, max_moves = None, model = None):
        self.root = None
        self.player_id = player_id
        self.cur_moves = cur_moves
        if cur_moves is not None:
            assert max_moves is not None
            assert model is not None
            self.max_moves = max_moves
            self.model = model
        self.size = 1
        self.abort = False
        self.ABORT_LIMIT = 3000

    def build_tree(self, state):
        self.root = GameNode(state, self.player_id)
        current_player_id = state.current_player_id
        state_moves = state.get_possible_moves(current_player_id)
        for possible_move in state_moves:
            next_state = state.get_state_from_move(possible_move)
            self.add_child(next_state, self.root, cur_moves=self.cur_moves)

    def add_child(self, state, parent, cur_moves=None):
        self.size += 1
        if self.size > self.ABORT_LIMIT:
            self.abort = True
            print("ABORTTTTT")
            return
        current_player_id = state.current_player_id
        state_moves = state.get_possible_moves(current_player_id)
        # print('-----------------------')
        # display_state(state)
        # print(state.get_hash())
        # print('-----------------------')
        if len(state_moves) == 0:
            leaf_node = GameNode(state, self.player_id)
            leaf_node.parent = parent
            parent.add_child(leaf_node)
            value_of_state = state.get_value_of_state(self.player_id)
            assert value_of_state != 0
            leaf_node.value = value_of_state
            return
        elif cur_moves is not None:
            if cur_moves >= self.max_moves:
                leaf_node = GameNode(state, self.player_id)
                leaf_node.parent = parent
                parent.add_child(leaf_node)
                x = state.convert_to_xs_for_neural_net()
                board_x = x[:, :60]
                player_x = x[:, 60:]
                predictions = self.model.predict([np.array(board_x), np.array(player_x)])
                if state.player_who_moved == self.player_id:
                    value_of_state = predictions[0][0]
                else:
                    value_of_state = predictions[0][1]
                    #no draws
                leaf_node.value = value_of_state
                return
        # recursive case
        tree_node = GameNode(state, self.player_id)
        # make connections
        tree_node.parent = parent
        parent.add_child(tree_node)
        for possible_move in state_moves:
            next_state = state.get_state_from_move(possible_move)
            if self.size > self.ABORT_LIMIT:
                self.abort = True
                print("ABORTTTTT")
                return
            #print("PM",possible_move)
            self.add_child(next_state, tree_node, cur_moves = None if cur_moves is None else cur_moves + 1)
