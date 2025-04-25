import numpy as np
import gymnasium as gym
import csv

from board import Board  # Assuming Board is your game implementation

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.board = Board()
        # The observation space is an 8x8 board with values from -6 to 6
        self.observation_space = gym.spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.float32)
        # The action space is a discrete space of size 64x64 (from one square to another)
        self.action_space = gym.spaces.Discrete(64 * 64)
        self._action_to_pos = lambda x: ((x // 64) // 8, (x // 64) % 8, (x % 64) // 8, (x % 64) % 8)
        self.epoch = 0
        self.step_count = 0
        with open('actions.csv', 'w', newline='') as file:
            self.writer = csv.writer(file)
            self.writer.writerow(["step", "piece", "from", "to", "reward", "epoch"])
        
    def _log_action(self, action, reward):
        """Logs the action taken to a CSV file."""
        from_row, from_col, to_row, to_col = self._action_to_pos(action)
        piece = self.board.board[from_row, from_col]
        piece_name = self.board.int_to_piece[piece]
        with open('actions.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.step_count, self.board.move_count, piece_name, (from_row, from_col), (to_row, to_col), reward, self.epoch])
        
    def _get_obs(self):
        """Returns the current board state."""
        return np.array(self.board.get_state(), dtype=np.float32)
    
    def _get_info(self):
        """Returns additional information about the current state."""
        return {
            "move_count": self.board.move_count,
        }
    
    def step(self, action):
        """Executes the action and returns the new state, reward, done, and info."""
        from_row, from_col, to_row, to_col = self._action_to_pos(action)
        valid_move = self.board.make_move((from_row, from_col), (to_row, to_col))
        
        if valid_move:
            reward = 0.5
        else:
            reward = -1
            
        terminated = self.board.is_game_over()
        truncated = False
        self._log_action(action, reward)
        self.step_count += 1
        return self._get_obs(), reward, terminated, truncated, {}
        
    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        self.board.reset()
        self.epoch += 1
        self.step_count = 0
        return self._get_obs(), {}
        