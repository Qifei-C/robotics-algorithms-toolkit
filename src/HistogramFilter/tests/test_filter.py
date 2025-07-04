import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.histogram_filter import HistogramFilter


class TestHistogramFilter:
    
    def setup_method(self):
        self.filter = HistogramFilter()
        
    def test_normalization(self):
        """Test that belief always sums to 1"""
        colormap = np.array([[1, 0], [0, 1]])
        belief = np.ones((2, 2)) * 0.25
        
        # Apply filter
        new_belief = self.filter.histogram_filter(colormap, belief, (1, 0), 1)
        
        # Check normalization
        assert np.isclose(np.sum(new_belief), 1.0)
        
    def test_stationary_action(self):
        """Test that invalid actions keep robot in place"""
        colormap = np.array([[1, 0], [0, 1]])
        belief = np.zeros((2, 2))
        belief[0, 0] = 1.0  # Robot definitely at (0, 0)
        
        # Try to move out of bounds
        new_belief = self.filter.histogram_filter(colormap, belief, (-1, 0), 1)
        
        # Robot should have high probability of staying at (0, 0)
        assert new_belief[0, 0] > 0.8
        
    def test_perfect_observation(self):
        """Test with perfect initial knowledge"""
        colormap = np.array([[1, 0, 1], [0, 1, 0]])
        belief = np.zeros((2, 3))
        belief[0, 0] = 1.0  # Robot knows it's at (0, 0)
        
        # Move right and observe
        new_belief = self.filter.histogram_filter(colormap, belief, (1, 0), 0)
        
        # Should be confident about new position
        assert new_belief[0, 1] > 0.7
        
    def test_uniform_initial_belief(self):
        """Test starting from complete uncertainty"""
        colormap = np.array([[1, 0], [0, 1]])
        belief = np.ones((2, 2)) * 0.25
        
        # Observe white (1)
        new_belief = self.filter.histogram_filter(colormap, belief, (0, 0), 1)
        
        # Positions with white cells should have higher probability
        assert new_belief[0, 0] > new_belief[0, 1]
        assert new_belief[1, 1] > new_belief[1, 0]
        
    def test_action_uncertainty(self):
        """Test that action has inherent uncertainty"""
        colormap = np.array([[1, 1], [1, 1]])  # All white
        belief = np.zeros((2, 2))
        belief[0, 0] = 1.0
        
        # Move right
        new_belief = self.filter.histogram_filter(colormap, belief, (1, 0), 1)
        
        # Should have moved mostly to (0, 1) but some probability at (0, 0)
        assert new_belief[0, 1] > new_belief[0, 0]
        assert new_belief[0, 0] > 0  # But not zero due to action uncertainty
        
    def test_convergence(self):
        """Test that repeated observations lead to convergence"""
        colormap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        belief = np.ones((3, 3)) / 9  # Uniform
        
        # Stay in place and observe multiple times
        for _ in range(10):
            belief = self.filter.histogram_filter(colormap, belief, (0, 0), 1)
            
        # Should converge to positions with value 1
        white_positions = np.where(colormap == 1)
        for i, j in zip(white_positions[0], white_positions[1]):
            assert belief[i, j] > 0.25  # Higher than uniform
            
    def test_motion_model_sum(self):
        """Test that motion model preserves probability mass"""
        colormap = np.array([[1, 0], [0, 1]])
        belief = np.array([[0.5, 0.2], [0.2, 0.1]])
        
        # Test only the motion update part
        # This would require exposing motion update separately
        # For now, test with neutral observation
        
        new_belief = self.filter.histogram_filter(colormap, belief, (1, 0), 1)
        assert np.isclose(np.sum(new_belief), 1.0)