import torch
import numpy as np
from gym import Env
from gym import spaces

from board import Board  # Assuming Board is your game implementation

class ChessEnv(Env):
    def __init__(self):
        super().__init__()
        
        self.board = Board()
        # The observation space is an 8x8 board with values from -6 to 6
        self.observation_space = spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.int8)
        # The action space is a discrete space of size 64x64 (from one square to another)
        self.action_space = spaces.Discrete(64 * 64)
        self._action_to_pos = lambda x: ((x // 64) // 8, (x // 64) % 8, (x % 64) // 8, (x % 64) % 8)
        
    def _log_action(self, action):
        """Logs the action taken."""
        from_row, from_col, to_row, to_col = self._action_to_pos(action)
        piece = self.board.board[from_row, from_col]
        print(f"Action taken: Move {piece} from ({from_row}, {from_col}) to ({to_row}, {to_col})")
        
    def _get_obs(self):
        """Returns the current board state."""
        return self.board.get_state()
    
    def _get_info(self):
        """Returns additional information about the current state."""
        return {
            "current_player": self.board.current_player,
            "move_count": self.board.move_count,
        }
    
    def step(self, action):
        """Executes the action and returns the new state, reward, done, and info."""
        from_row, from_col, to_row, to_col = self._action_to_pos(action)
        valid_move = self.board.make_move((from_row, from_col), (to_row, to_col))
        
        if valid_move:
            reward = 1
        else:
            reward = -1
        terminated = self.board.is_game_over()
        truncated = False
        self._log_action(action)
        return self._get_obs(), reward, terminated, truncated, self._get_info()
        
    def reset(self):
        """Resets the environment to the initial state."""
        self.board = Board()
        return self._get_obs(), self._get_info()
        