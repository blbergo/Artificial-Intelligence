import gymnasium as gym
import chess
import numpy as np

class ChessEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.board = chess.Board()
        self.action_space = gym.spaces.Discrete(4672)  # 4672 possible moves in chess - AlphaZero
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,8), dtype=np.float32)  # Start with 12 channels for pieces for now
        self._piece_map_to_index = {
            chess.Piece.from_symbol('P'): 1,
            chess.Piece.from_symbol('N'): 2,
            chess.Piece.from_symbol('B'): 3,
            chess.Piece.from_symbol('R'): 4,
            chess.Piece.from_symbol('Q'): 5,
            chess.Piece.from_symbol('K'): 6,
            chess.Piece.from_symbol('p'): -1,
            chess.Piece.from_symbol('n'): -2,
            chess.Piece.from_symbol('b'): -3,
            chess.Piece.from_symbol('r'): -4,
            chess.Piece.from_symbol('q'): -5,
            chess.Piece.from_symbol('k'): -6,
        }
        
        self.render_mode = render_mode
            
        
    def _get_obs(self):
        piece_map = self.board.piece_map()
        obs = np.zeros((8, 8), dtype=np.int8)
        for square, piece in piece_map.items():
            obs[chess.square_rank(square), chess.square_file(square)] = self._piece_map_to_index[piece]
        return obs.astype(np.float32)
       
    
    def _get_info(self):
        return {
            'legal_moves': [self._piece_map_to_index[piece] for square, piece in self.board.piece_map().items()],
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_insufficient_material': self.board.is_insufficient_material(),
            'is_seventyfive_moves': self.board.is_seventyfive_moves(),
            'is_fivefold_repetition': self.board.is_fivefold_repetition(),
        }
    
    def _get_action_dict(self):
        action_to_move_dict = {}
        for i, move in enumerate(self.board.legal_moves):
            action_to_move_dict[i] = move
        return action_to_move_dict
    
    def step(self, action):
        candidate_moves = self._get_action_dict()
        
        legal_moves_mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        for idx in candidate_moves.keys():
            legal_moves_mask[idx] = 1
            
        action = action[0] # Unwrap the action from the tuple
        # if the chosen action is illegal, select a random legal move
        if action not in candidate_moves:
            action = np.random.choice(list(candidate_moves.keys()))
            
        move = candidate_moves[action]
        self.board.push(move)
        
        if self.render_mode == 'human':
            self.render()
            
        return self._get_obs(), 0, self.board.is_game_over(), False, {**self._get_info(), 'legal_moves_mask': legal_moves_mask}
    
    def reset(self):
        self.board.reset()
        return self._get_obs(), self._get_info()
    
    def render(self):
        print("".join(["-" for _ in range (15)]))
        print(self.board)
        