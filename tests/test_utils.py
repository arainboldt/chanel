"""
Tests for utility functions.
"""

import pytest
import numpy as np


class TestLinearRegression:
    """Test linear regression utilities."""
    
    def test_simple_line(self):
        """Test fitting a perfect line."""
        from chanel.core.utils import linear_regression, calculate_r_squared
        
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float64)  # y = 2x
        
        slope, intercept = linear_regression(x, y)
        
        assert abs(slope - 2.0) < 0.001
        assert abs(intercept - 0.0) < 0.001
        
        r_squared = calculate_r_squared(x, y, slope, intercept)
        assert abs(r_squared - 1.0) < 0.001  # Perfect fit
    
    def test_horizontal_line(self):
        """Test fitting a horizontal line."""
        from chanel.core.utils import linear_regression
        
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        y = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        
        slope, intercept = linear_regression(x, y)
        
        assert abs(slope) < 0.001
        assert abs(intercept - 5.0) < 0.001


class TestClustering:
    """Test clustering utilities."""
    
    def test_cluster_values(self):
        """Test value clustering."""
        from chanel.core.utils import cluster_values, cluster_representative
        
        values = np.array([100.0, 100.1, 100.2, 105.0, 105.1, 110.0], dtype=np.float64)
        tolerance = 0.005  # 0.5% tolerance
        
        labels = cluster_values(values, tolerance)
        
        # Should have 3 clusters
        assert len(np.unique(labels)) == 3
        
        # Get representative for first cluster
        rep = cluster_representative(values, labels, 0)
        assert 99.5 < rep < 100.5


class TestRecency:
    """Test recency weighting."""
    
    def test_recency_weight(self):
        """Test recency weight calculation."""
        from chanel.core.utils import recency_weight
        
        current_time = 1000000
        recent_time = 990000
        old_time = 500000
        
        recent_weight = recency_weight(current_time, recent_time)
        old_weight = recency_weight(current_time, old_time)
        
        assert recent_weight > old_weight
        assert 0.0 <= recent_weight <= 1.0
        assert 0.0 <= old_weight <= 1.0


class TestPriceTolerance:
    """Test price tolerance checking."""
    
    def test_within_tolerance(self):
        """Test price tolerance check."""
        from chanel.core.utils import are_prices_within_tolerance
        
        price1 = 100.0
        price2 = 100.15
        
        # Within 0.2% tolerance
        assert are_prices_within_tolerance(price1, price2, 0.002) == 1
        
        # Not within 0.1% tolerance
        assert are_prices_within_tolerance(price1, price2, 0.001) == 0

