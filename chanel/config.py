"""
Configuration system for Chanel library.

Defines default parameters and timeframe hierarchy for hierarchical
support/resistance detection.
"""

from typing import List, Dict, Optional
from pathlib import Path
import json


class AnalysisConfig:
    """
    Configuration for hierarchical support/resistance analysis.
    
    Attributes:
        timeframe_hierarchy: List of timeframes in order from highest to lowest
        swing_window: Window size for swing point detection
        tolerance: Price tolerance for level touches as fraction
        epsilon: Slope difference threshold for boundary line grouping
        max_swing_points: Maximum number of swing points to analyze (default: 50)
        min_touches: Minimum number of touches required for a valid level
        min_r_squared: Minimum R² for diagonal trendlines
        detect_diagonal: Whether to detect diagonal trendlines
    """
    
    def __init__(
        self,
        timeframe_hierarchy: Optional[List[str]] = None,
        swing_window: int = 10,
        tolerance: float = 0.002,
        epsilon: float = 0.0001,
        max_swing_points: int = 50,
        min_touches: int = 3,
        min_r_squared: float = 0.8,
        detect_diagonal: bool = True
    ):
        """
        Initialize configuration.
        
        Args:
            timeframe_hierarchy: List of timeframes (e.g., ['1hr', '15min', '5min', '1min'])
            swing_window: Window size for swing detection
            tolerance: Price tolerance as fraction (0.002 = 0.2%)
            epsilon: Slope difference threshold for grouping (0.0001 = 0.01%)
            max_swing_points: Maximum swing points to analyze
            min_touches: Minimum touches for valid level
            min_r_squared: Minimum R² for trendlines
            detect_diagonal: Enable diagonal detection
        """
        if timeframe_hierarchy is None:
            timeframe_hierarchy = ['1hr', '15min', '5min', '1min']
        
        self.timeframe_hierarchy = timeframe_hierarchy
        self.swing_window = swing_window
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.max_swing_points = max_swing_points
        self.min_touches = min_touches
        self.min_r_squared = min_r_squared
        self.detect_diagonal = detect_diagonal
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'timeframe_hierarchy': self.timeframe_hierarchy,
            'swing_window': self.swing_window,
            'tolerance': self.tolerance,
            'epsilon': self.epsilon,
            'max_swing_points': self.max_swing_points,
            'min_touches': self.min_touches,
            'min_r_squared': self.min_r_squared,
            'detect_diagonal': self.detect_diagonal
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AnalysisConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'AnalysisConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_timeframe_index(self, timeframe: str) -> int:
        """
        Get the index of a timeframe in the hierarchy.
        
        Args:
            timeframe: Timeframe string (e.g., '1hr', '15min')
        
        Returns:
            Index in hierarchy, or -1 if not found
        """
        try:
            return self.timeframe_hierarchy.index(timeframe)
        except ValueError:
            return -1
    
    def get_next_timeframe(self, timeframe: str) -> Optional[str]:
        """
        Get the next lower timeframe in the hierarchy.
        
        Args:
            timeframe: Current timeframe
        
        Returns:
            Next timeframe, or None if at the bottom
        """
        idx = self.get_timeframe_index(timeframe)
        if idx < 0 or idx >= len(self.timeframe_hierarchy) - 1:
            return None
        return self.timeframe_hierarchy[idx + 1]
    
    def get_previous_timeframe(self, timeframe: str) -> Optional[str]:
        """
        Get the previous higher timeframe in the hierarchy.
        
        Args:
            timeframe: Current timeframe
        
        Returns:
            Previous timeframe, or None if at the top
        """
        idx = self.get_timeframe_index(timeframe)
        if idx <= 0:
            return None
        return self.timeframe_hierarchy[idx - 1]


# Default configuration instance
DEFAULT_CONFIG = AnalysisConfig()


def get_default_config() -> AnalysisConfig:
    """Get the default configuration."""
    return DEFAULT_CONFIG


def load_config(path: Optional[Path] = None) -> AnalysisConfig:
    """
    Load configuration from file or return default.
    
    Args:
        path: Path to config file (optional)
    
    Returns:
        AnalysisConfig instance
    """
    if path is None:
        # Try to find config.toml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'chanel_config.json'
        if config_path.exists():
            return AnalysisConfig.load(config_path)
        return get_default_config()
    
    if path.exists():
        return AnalysisConfig.load(path)
    
    return get_default_config()

