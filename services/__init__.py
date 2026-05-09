"""
Services Module

This module contains all the core tracking and detection services for football match analysis.

Modules:
- ball_detector: Detects ball position using color-based detection
- ball_tracker: Tracks ball position and velocity across frames
- possession_tracker: Detects and tracks ball possession by players
- event_detector: Detects passes and other football events
"""

# Lazy imports to avoid loading heavy dependencies when not needed
__all__ = [
    'ColorBallDetector',
    'BallTracker',
    'PossessionTracker',
    'EventDetector',
    'BatchProcessor',
    'RosterSyncService',
]

def __getattr__(name):
    if name == 'ColorBallDetector':
        from services.ball_detector import ColorBallDetector
        return ColorBallDetector
    elif name == 'BallTracker':
        from services.ball_tracker import BallTracker
        return BallTracker
    elif name == 'PossessionTracker':
        from services.possession_tracker import PossessionTracker
        return PossessionTracker
    elif name == 'EventDetector':
        from services.event_detector import EventDetector
        return EventDetector
    elif name == 'BatchProcessor':
        from services.batch_processor import BatchProcessor
        return BatchProcessor
    elif name == 'RosterSyncService':
        from services.roster_sync_service import RosterSyncService
        return RosterSyncService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
