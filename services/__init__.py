"""
Services Module

This module contains all the core tracking and detection services for football match analysis.

Modules:
- ball_detector: Detects ball position using color-based detection
- ball_tracker: Tracks ball position and velocity across frames
- possession_tracker: Detects and tracks ball possession by players
- event_detector: Detects passes and other football events
"""

from services.ball_detector import ColorBallDetector
from services.ball_tracker import BallTracker
from services.possession_tracker import PossessionTracker
from services.event_detector import EventDetector

__all__ = [
    'ColorBallDetector',
    'BallTracker',
    'PossessionTracker',
    'EventDetector'
]
