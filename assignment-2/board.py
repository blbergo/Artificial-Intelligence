import numpy as np
import torch

class Board:
    """
    Represents the chess board and game logic.
    """
    def __init__(self):
        """Initializes the board to the standard starting position."""
        # 0: empty square
        # Positive integers: white pieces (1: Pawn, 2: Knight, 3: Bishop, 4: Rook, 5: Queen, 6: King)
        # Negative integers: black pieces (-1: Pawn, -2: Knight, -3: Bishop, -4: Rook, -5: Queen, -6: King)
        self.board = np.array([
            [-4, -2, -3, -5, -6, -3, -2, -4], # Row 8 (Black back rank)
            [-1, -1, -1, -1, -1, -1, -1, -1], # Row 7 (Black pawns)
            [ 0,  0,  0,  0,  0,  0,  0,  0], # Row 6
            [ 0,  0,  0,  0,  0,  0,  0,  0], # Row 5
            [ 0,  0,  0,  0,  0,  0,  0,  0], # Row 4
            [ 0,  0,  0,  0,  0,  0,  0,  0], # Row 3
            [ 1,  1,  1,  1,  1,  1,  1,  1], # Row 2 (White pawns)
            [ 4,  2,  3,  5,  6,  3,  2,  4]  # Row 1 (White back rank)
        ], dtype=np.int8)
        self.current_player = 1 # 1 for white, -1 for black
        self.move_count = 0
        # TODO: Add state for castling rights, en passant target square, halfmove clock, fullmove number for FEN/PGN compatibility if needed

    def get_state(self):
        """Returns the current board state as a NumPy array or PyTorch tensor."""
        # TorchRL often works best with tensors
        return torch.tensor(self.board, dtype=torch.float32) # Use float for NN input

    
    def make_move(self, from_pos, to_pos):
        """
        Applies a move to the board if it's valid.
        Args:
            from_pos: Tuple (row, col) of the starting square.
            to_pos: Tuple (row, col) of the ending square.
        Returns:
            True if the move was valid and made, False otherwise.
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        if(from_row == to_row and from_col == to_col):
            # No move made
            return False

        # Basic check: is there a piece at from_pos belonging to the current player?
        piece = self.board[from_row, from_col]
        if piece == 0 or (piece > 0 and self.current_player == -1) or (piece < 0 and self.current_player == 1):
             # Trying to move an empty square or opponent's piece
             # print(f"Warning: Attempted invalid move logic: from {from_pos} to {to_pos}, piece {piece}, player {self.current_player}")
             return False # Treat as invalid for reward purposes

        # Make the move
        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = 0

        # Switch player
        self.current_player *= -1
        self.move_count += 1

        # TODO: Handle captures, promotions, castling, en passant updates
        return True

    def is_game_over(self):
        """
        Checks if the game has ended (checkmate, stalemate, draw conditions).
        Returns:
            A tuple (is_over, winner), where winner is 1 for white, -1 for black, 0 for draw.
        """
        # Simple check: has a king been captured? (Shouldn't happen with valid moves)
        kings = [np.any(self.board == 6), np.any(self.board == -6)]
        if not kings[0]: return True, -1 # White king missing, black wins
        if not kings[1]: return True, 1  # Black king missing, white wins

        # Simple check: Max moves (to prevent infinite games in testing)
        if self.move_count >= 200: # Arbitrary limit
            # print("Game Over: Max moves reached.")
            return True # Draw

        return False # Game not over

    def reset(self):
        """Resets the board to the starting position."""
        self.__init__() # Re-initialize the board