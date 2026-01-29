"""
Matrix trading environment module.

Based on the base environment, but state is a 2D matrix for CNN models.
"""

import numpy as np
import gym
from utils.environment import TradingEnvironment


class MatrixTradingEnvironment(TradingEnvironment):
    """Matrix-based trading environment (2D state)."""
    
    def __init__(self, config, data_loader, lim_calculator):
        """Initialize matrix-based trading environment."""
        # Initialize base environment
        super(MatrixTradingEnvironment, self).__init__(config, data_loader, lim_calculator)
        
        # Set observation space to image-like space
        # Compute matrix dimension
        state_dim = self.observation_space.shape[0]  # Original state dimension
        self.matrix_dim = int(np.ceil(np.sqrt(state_dim)))  # Matrix edge length
        
        # Redefine observation space to 2D matrix
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.matrix_dim, self.matrix_dim),
            dtype=np.float32
        )
        
        print(f"Matrix trading environment created, state shape: {self.observation_space.shape}")
    
    def _get_state(self):
        """Build state matrix and return 2D state."""
        # Get original 1D state vector
        original_state = super()._get_state()
        
        # Reshape to matrix
        matrix_state = self._reshape_to_matrix(original_state)
        
        return matrix_state
    
    def _reshape_to_matrix(self, state_vector):
        """Reshape 1D state vector to 2D matrix."""
        # Ensure state_vector is 1D
        state_vector = np.array(state_vector, dtype=np.float32).flatten()
        
        # Create matrix and pad with zeros
        matrix = np.zeros((self.matrix_dim, self.matrix_dim), dtype=np.float32)
        
        # Fill matrix
        for i in range(min(len(state_vector), self.matrix_dim * self.matrix_dim)):
            row = i // self.matrix_dim
            col = i % self.matrix_dim
            # Ensure assigned value is scalar
            matrix[row, col] = float(state_vector[i])
        
        return matrix
        
    def reset(self, start_date=None, end_date=None):
        """Reset environment and return matrix state."""
        # Call base reset to get original state
        original_state = super().reset(start_date, end_date)
        
        # Convert to matrix state
        try:
            matrix_state = self._reshape_to_matrix(original_state)
        except Exception as e:
            print(f"Error converting state to matrix: {e}")
            print(f"State type: {type(original_state)}, shape: {np.array(original_state).shape}")
            # On failure, return zeros
            matrix_state = np.zeros((self.matrix_dim, self.matrix_dim), dtype=np.float32)
        
        return matrix_state
    
    def step(self, action):
        """Execute one step and return matrix state."""
        # Call base step
        next_state, reward, done, info = super().step(action)
        
        # Convert state to matrix
        try:
            matrix_next_state = self._reshape_to_matrix(next_state)
        except Exception as e:
            print(f"Error converting next state to matrix: {e}")
            print(f"Next state type: {type(next_state)}, shape: {np.array(next_state).shape}")
            # On failure, return zeros
            matrix_next_state = np.zeros((self.matrix_dim, self.matrix_dim), dtype=np.float32)
        
        return matrix_next_state, reward, done, info 

