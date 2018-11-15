from rl.agents.dqn import DQNAgent
from fish.fish_state import get_state_from_serialization
from fish.fish_game import get_moves_from_state
from fish.fish_board import get_all_actions

class FishAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        #mask out invalid moves
        real_state = get_state_from_serialization(state[0])
        real_moves = get_moves_from_state(real_state)
        player_id = real_state.current_player_id
        move_map = get_all_actions(player_id)
        hashed_move_map = [move.get_hash() for move in move_map]
        hashed_real_moves = [move.get_hash() for move in real_moves]
        for i, move in enumerate(hashed_move_map):
            if move not in hashed_real_moves:
                q_values[i] = -5000000000000000000000
        return q_values
