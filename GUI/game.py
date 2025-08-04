import numpy as np
import random

class Game2048:
    """A class representing the 2048 game with all game logic."""
    
    def __init__(self):
        """Initialize a new game with a 4x4 board."""
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.moves = 0  # Track the number of valid moves
        self.highest_tile = 0  # Track the highest tile achieved
        self.done = False
        # Add initial two tiles
        self.add_random_tile()
        self.add_random_tile()
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.highest_tile = 0
        self.done = False
        self.add_random_tile()
        self.add_random_tile()
        return self.board
    
    def add_random_tile(self):
        """Add a 2 (90%) or 4 (10%) tile to a random empty cell."""
        empty_cells = np.where(self.board == 0)
        empty_cells = list(zip(empty_cells[0], empty_cells[1]))
        
        if not empty_cells:
            return self.board
        
        i, j = random.choice(empty_cells)
        self.board[i, j] = 4 if random.random() >= 0.9 else 2
        # Update highest tile
        self.highest_tile = max(self.highest_tile, self.board[i, j])
        return self.board
    
    def get_state(self):
        """Convert the game board to a state representation for the DQN."""
        # One-hot encode the board
        power_mat = np.zeros((1, 4, 4, 16), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if self.board[i, j] == 0:
                    power_mat[0, i, j, 0] = 1.0
                else:
                    power = int(np.log2(self.board[i, j]))
                    power_mat[0, i, j, power] = 1.0
        return power_mat
    
    def step(self, action):
        """Take a step in the game with the given action.
        
        Args:
            action: Integer in [0, 1, 2, 3] representing [up, left, right, down]
            
        Returns:
            board: The new game board
            reward: The reward for this step
            done: Whether the game is over
            info: Dictionary containing score, moves, and highest tile
        """
        prev_board = np.copy(self.board)  # Use NumPy copy for efficiency
        prev_max = np.max(self.board)
        prev_empty = np.sum(self.board == 0)
        
        # Apply the move
        self.board, move_made, move_score = self._apply_move(action)
        
        # If the move didn't change the board, return negative reward
        if np.array_equal(prev_board, self.board):
            return self.board, -1, self.done, {"score": self.score, "moves": self.moves, "highest_tile": self.highest_tile}
        
        # Increment moves counter for valid moves
        self.moves += 1
        
        # Add a new tile if the move was valid
        self.add_random_tile()
        
        # Update score
        self.score += move_score
        
        # Check if game is over
        self.done = self._check_game_over()
        
        # Calculate reward
        current_max = np.max(self.board)
        current_empty = np.sum(self.board == 0)
        
        # Reward calculation
        # Base reward: score from merges
        reward = move_score / 10.0  # Normalize the score contribution
        # Bonus for increasing the max tile
        if current_max > prev_max:
            reward += np.log2(current_max) * 2  # Larger bonus for higher tiles
        # Penalty for reducing empty cells
        empty_diff = current_empty - prev_empty
        reward += empty_diff * 0.5  # Small penalty/reward for empty cell changes
        
        # Log game state (optional, can be removed in production)
        if self.done:
            print(f"Game Over - Score: {self.score}, Moves: {self.moves}, Highest Tile: {self.highest_tile}")
        
        return self.board, reward, self.done, {"score": self.score, "moves": self.moves, "highest_tile": self.highest_tile}
    
    def _check_game_over(self):
        """Check if the game is over (no moves possible)."""
        # If there are empty cells, game is not over
        if np.any(self.board == 0):
            return False
        
        # Check for adjacent equal tiles (horizontally or vertically)
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1]:  # Horizontal
                    return False
            for j in range(4):
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:  # Vertical
                    return False
        
        return True
    
    def _apply_move(self, action):
        """Apply the move action to the board.
        
        Args:
            action: Integer in [0, 1, 2, 3] representing [up, left, right, down]
            
        Returns:
            board: The new game board
            move_made: Whether the move changed the board
            score: The score gained from this move
        """
        if action == 0:  # Up
            return self._up()
        elif action == 1:  # Left
            return self._left()
        elif action == 2:  # Right
            return self._right()
        elif action == 3:  # Down
            return self._down()
        else:
            raise ValueError(f"Invalid action: {action}. Must be in [0, 1, 2, 3]")
    
    def _cover_up(self, board):
        """Shift non-zero tiles to the left, filling with zeros."""
        new = np.zeros((4, 4), dtype=np.int32)
        done = False
        
        for i in range(4):
            count = 0
            for j in range(4):
                if board[i, j] != 0:
                    new[i, count] = board[i, j]
                    if j != count:
                        done = True
                    count += 1
        
        return new, done
    
    def _merge(self, board):
        """Merge equal adjacent tiles."""
        done = False
        score = 0
        
        for i in range(4):
            for j in range(3):
                if board[i, j] == board[i, j + 1] and board[i, j] != 0:
                    board[i, j] *= 2
                    score += board[i, j]
                    board[i, j + 1] = 0
                    done = True
        
        return board, done, score
    
    def _up(self):
        """Move tiles up."""
        board = np.transpose(self.board)
        board, done = self._cover_up(board)
        board, done_merge, score = self._merge(board)
        board = self._cover_up(board)[0]
        board = np.transpose(board)
        return board, done or done_merge, score
    
    def _down(self):
        """Move tiles down."""
        board = np.flip(np.transpose(self.board), axis=1)
        board, done = self._cover_up(board)
        board, done_merge, score = self._merge(board)
        board = self._cover_up(board)[0]
        board = np.transpose(np.flip(board, axis=1))
        return board, done or done_merge, score
    
    def _left(self):
        """Move tiles left."""
        board = self.board.copy()
        board, done = self._cover_up(board)
        board, done_merge, score = self._merge(board)
        board = self._cover_up(board)[0]
        return board, done or done_merge, score
    
    def _right(self):
        """Move tiles right."""
        board = np.flip(self.board, axis=1)
        board, done = self._cover_up(board)
        board, done_merge, score = self._merge(board)
        board = self._cover_up(board)[0]
        board = np.flip(board, axis=1)
        return board, done or done_merge, score