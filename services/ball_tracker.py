"""
Ball Tracker Module - Enhanced with Trajectory Prediction

Tracks ball position and calculates velocity across frames.
NOW WITH: Prediction for missed detections to improve tracking continuity.

WHY PREDICTION MATTERS:
- Ball can be temporarily occluded by players
- Detection can fail in fast motion (blur)
- Prediction fills gaps using physics (velocity + direction)
- Dramatically improves tracking continuity

ENHANCEMENTS:
1. Trajectory prediction when detection fails
2. Automatic gap filling (up to 5 frames)
3. Bounds checking for predictions
"""

from typing import Tuple, Optional, List, Dict, Any
from collections import deque
import numpy as np


class BallTracker:
    """
    Tracks ball position and calculates velocity across frames.
    
    ENHANCED with trajectory prediction for missed detections.
    """

    def __init__(self, max_history: int = 30, velocity_window: int = 3):
        """
        Initialize the ball tracker.

        Args:
            max_history: Maximum number of positions to store (default: 30 frames = 1 second at 30fps)
            velocity_window: Number of recent positions for velocity calculation (default: 3)
        """
        self.ball_history = deque(maxlen=max_history)
        self.current_velocity = 0.0
        self.current_direction = np.array([0.0, 0.0])
        self.max_history = max_history
        self.velocity_window = velocity_window
        
        # NEW: Prediction tracking
        self._missing_frames = 0
        self._last_detected_frame = None

    def update(self, ball_bbox: Tuple[float, float, float, float],
               frame_idx: int, timestamp: float) -> None:
        """
        Updates ball position and recalculates velocity.

        Args:
            ball_bbox: Bounding box (x1, y1, x2, y2) in pixels
            frame_idx: Current frame number
            timestamp: Current timestamp in seconds
        """
        # Extract center point from bounding box
        x1, y1, x2, y2 = ball_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Store position with metadata
        self.ball_history.append({
            'x': cx,
            'y': cy,
            'frame': frame_idx,
            'timestamp': timestamp,
            'bbox': ball_bbox
        })

        # Recalculate velocity and direction
        self._calculate_velocity()
        self._calculate_direction()

    def _calculate_velocity(self) -> None:
        """
        Calculates current ball velocity from recent positions.

        Algorithm:
        1. Take last N positions (where N = velocity_window)
        2. Calculate displacement between consecutive positions
        3. Calculate time delta between consecutive positions
        4. Velocity = total_displacement / total_time
        """
        if len(self.ball_history) < 2:
            self.current_velocity = 0.0
            return

        recent_positions = list(self.ball_history)[-self.velocity_window:]

        if len(recent_positions) < 2:
            self.current_velocity = 0.0
            return

        total_distance = 0.0
        total_time = 0.0

        for i in range(len(recent_positions) - 1):
            pos1 = recent_positions[i]
            pos2 = recent_positions[i + 1]

            dx = pos2['x'] - pos1['x']
            dy = pos2['y'] - pos1['y']
            distance = np.sqrt(dx**2 + dy**2)

            dt = pos2['timestamp'] - pos1['timestamp']

            total_distance += distance
            total_time += dt

        if total_time > 0:
            self.current_velocity = total_distance / total_time
        else:
            self.current_velocity = 0.0

    def _calculate_direction(self) -> None:
        """
        Calculates normalized direction vector from recent movement.
        """
        if len(self.ball_history) < 2:
            self.current_direction = np.array([0.0, 0.0])
            return

        recent = list(self.ball_history)[-self.velocity_window:]

        if len(recent) < 2:
            self.current_direction = np.array([0.0, 0.0])
            return

        first = recent[0]
        last = recent[-1]

        dx = last['x'] - first['x']
        dy = last['y'] - first['y']

        magnitude = np.sqrt(dx**2 + dy**2)

        if magnitude > 0:
            self.current_direction = np.array([dx / magnitude, dy / magnitude])
        else:
            self.current_direction = np.array([0.0, 0.0])

    def get_velocity(self) -> float:
        """Returns current ball speed in pixels/second."""
        return self.current_velocity

    def get_velocity_mps(self, pixels_per_meter: float) -> float:
        """Returns current ball speed in meters/second."""
        return self.current_velocity / pixels_per_meter

    def get_position(self) -> Optional[Tuple[float, float]]:
        """Returns current ball position (x, y) or None if no tracking data."""
        if not self.ball_history:
            return None
        last = self.ball_history[-1]
        return (last['x'], last['y'])

    def get_direction(self) -> np.ndarray:
        """Returns normalized direction vector (dx, dy)."""
        return self.current_direction

    def get_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns latest bounding box (x1, y1, x2, y2) or None."""
        if not self.ball_history:
            return None
        return self.ball_history[-1]['bbox']

    def is_moving(self, threshold: float = 50.0) -> bool:
        """Check if ball is moving faster than threshold."""
        return self.current_velocity > threshold

    def get_history(self, n: int = None) -> List[dict]:
        """Get recent position history."""
        history = list(self.ball_history)
        if n is not None:
            return history[-n:]
        return history

    def reset(self) -> None:
        """Reset all tracking data."""
        self.ball_history.clear()
        self.current_velocity = 0.0
        self.current_direction = np.array([0.0, 0.0])
        self._missing_frames = 0
        self._last_detected_frame = None

    # ============================================================
    # NEW: TRAJECTORY PREDICTION METHODS
    # ============================================================

    def predict_next_position(self, time_delta: float = 0.04) -> Optional[Tuple[float, float]]:
        """
        Predict ball position when detection fails.
        
        Uses velocity and direction to estimate where ball should be.
        Useful for filling gaps when ball is temporarily occluded.
        
        WHY THIS WORKS:
        - Ball follows physics: position += velocity * direction * time
        - Short-term predictions (1-5 frames) are quite accurate
        - Helps maintain tracking continuity during occlusion
        
        Args:
            time_delta: Time into future to predict (default: 0.04s = 1 frame at 25fps)
            
        Returns:
            Predicted (x, y) position or None if insufficient data
        """
        if len(self.ball_history) < 2:
            return None
        
        current_pos = self.get_position()
        if current_pos is None:
            return None
        
        direction = self.get_direction()
        velocity = self.get_velocity()
        
        # Calculate displacement: velocity * time
        displacement = velocity * time_delta
        
        # Apply direction to get new position
        pred_x = current_pos[0] + direction[0] * displacement
        pred_y = current_pos[1] + direction[1] * displacement
        
        return (pred_x, pred_y)
    
    def update_with_prediction(
        self, 
        detected_bbox: Optional[Tuple[float, float, float, float]], 
        frame_idx: int, 
        timestamp: float, 
        frame_width: int, 
        frame_height: int,
        max_missing_frames: int = 5
    ) -> Tuple[bool, str]:
        """
        Update ball tracker with detection OR prediction if detection failed.
        
        This method handles temporary detection failures by predicting ball position
        based on recent trajectory. Improves tracking continuity dramatically.
        
        WHEN TO USE PREDICTION:
        - Ball occluded by players (1-3 frames)
        - Motion blur during fast movement (1-2 frames)
        - Detection confidence just below threshold (1-2 frames)
        
        WHEN TO STOP PREDICTING:
        - Too many consecutive misses (>5 frames)
        - Prediction goes out of bounds
        - No velocity data available
        
        Args:
            detected_bbox: Detected bounding box or None if no detection
            frame_idx: Current frame index
            timestamp: Current timestamp
            frame_width: Frame width for bounds checking
            frame_height: Frame height for bounds checking
            max_missing_frames: Max frames to predict before giving up (default: 5)
            
        Returns:
            (tracked: bool, method: str) - Whether ball was tracked and how
        """
        if detected_bbox is not None:
            # Normal detection - update tracker
            self.update(detected_bbox, frame_idx, timestamp)
            self._missing_frames = 0
            self._last_detected_frame = frame_idx
            return (True, "detected")
        
        # No detection - try prediction
        self._missing_frames += 1
        
        if self._missing_frames > max_missing_frames:
            # Too many missing frames - stop predicting
            return (False, "lost")
        
        # Predict position
        predicted_pos = self.predict_next_position()
        if predicted_pos is None:
            return (False, "no_data")
        
        pred_x, pred_y = predicted_pos
        
        # Check if prediction is within frame bounds (with small margin)
        margin = 20  # pixels
        if not (-margin <= pred_x <= frame_width + margin and 
                -margin <= pred_y <= frame_height + margin):
            return (False, "out_of_bounds")
        
        # Clamp to frame bounds
        pred_x = max(0, min(pred_x, frame_width - 1))
        pred_y = max(0, min(pred_y, frame_height - 1))
        
        # Create predicted bounding box (assume small ball size)
        ball_size = 10  # pixels
        predicted_bbox = (
            pred_x - ball_size/2,
            pred_y - ball_size/2,
            pred_x + ball_size/2,
            pred_y + ball_size/2
        )
        
        # Update with prediction
        self.update(predicted_bbox, frame_idx, timestamp)
        return (True, "predicted")

    def get_prediction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about predictions vs detections.
        
        Returns:
            Dict with prediction statistics
        """
        if not self.ball_history:
            return {
                'total_frames': 0,
                'detected_frames': 0,
                'predicted_frames': 0,
                'current_missing_frames': self._missing_frames
            }
        
        # Note: This is a simplified version - in production,
        # you'd track this more precisely
        return {
            'total_frames': len(self.ball_history),
            'current_missing_frames': self._missing_frames,
            'last_detected_frame': self._last_detected_frame
        }
