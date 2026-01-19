"""
Ball Tracking Module

This module implements ball position tracking and velocity calculation for football analysis.
The ball is the most important object in football - all events (passes, shots, possession)
depend on accurate ball tracking.

Key Features:
- Frame-by-frame position tracking
- Velocity calculation using smoothing window
- Direction vector computation
- Noise filtering through multi-frame averaging
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, List


class BallTracker:
    """
    Tracks ball position and calculates velocity across frames.

    Why use a history window?
    - Smooths out detection noise and jitter
    - More stable velocity than single-frame differences
    - Filters out false detections (ball appears/disappears briefly)
    - Represents actual ball movement better

    Attributes:
        ball_history: Deque storing recent ball positions (x, y, frame_idx, timestamp)
        current_velocity: Current ball speed in pixels/second
        max_history: Number of positions to keep for calculations
        velocity_window: Number of recent positions used for velocity calculation
    """

    def __init__(self, max_history: int = 30, velocity_window: int = 3):
        """
        Initialize the ball tracker.

        Args:
            max_history: Maximum number of positions to store (default: 30 frames = 1 second at 30fps)
            velocity_window: Number of recent positions for velocity calculation (default: 3)
                           - Too small (1-2): noisy, unstable
                           - Too large (>5): laggy, misses quick changes
                           - 3 is the sweet spot for real-time accuracy
        """
        self.ball_history = deque(maxlen=max_history)
        self.current_velocity = 0.0
        self.current_direction = np.array([0.0, 0.0])  # Direction vector (dx, dy)
        self.max_history = max_history
        self.velocity_window = velocity_window

    def update(self, ball_bbox: Tuple[float, float, float, float],
               frame_idx: int, timestamp: float) -> None:
        """
        Updates ball position and recalculates velocity.

        Why calculate velocity from last 3 positions?
        - Single frame difference: velocity = position_change / time_delta
          Problem: Very noisy due to detection errors
        - Multi-frame average: velocity = avg(last_N_position_changes) / avg(last_N_time_deltas)
          Benefit: Smooth, stable, filters noise

        Args:
            ball_bbox: Bounding box (x1, y1, x2, y2) in pixels
            frame_idx: Current frame number
            timestamp: Current timestamp in seconds
        """
        # Extract center point from bounding box
        x1, y1, x2, y2 = ball_bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Add to history with all tracking info
        self.ball_history.append({
            'x': center_x,
            'y': center_y,
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'bbox': ball_bbox
        })

        # Recalculate velocity and direction from recent positions
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

        Why this approach?
        - Averages out detection noise
        - Handles variable frame rates
        - More robust than single-frame calculations

        Example:
        Frame 1: ball at (100, 200), t=0.000s
        Frame 2: ball at (110, 205), t=0.033s  -> moved 11.2 pixels in 0.033s
        Frame 3: ball at (120, 210), t=0.066s  -> moved 11.2 pixels in 0.033s
        Average velocity = 22.4 pixels / 0.066s = 339 pixels/second
        """
        if len(self.ball_history) < 2:
            self.current_velocity = 0.0
            return

        # Get recent positions for velocity calculation
        recent_positions = list(self.ball_history)[-self.velocity_window:]

        if len(recent_positions) < 2:
            self.current_velocity = 0.0
            return

        # Calculate total displacement and time
        total_displacement = 0.0
        total_time = 0.0

        for i in range(1, len(recent_positions)):
            prev = recent_positions[i - 1]
            curr = recent_positions[i]

            # Euclidean distance between consecutive positions
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            displacement = np.sqrt(dx**2 + dy**2)

            # Time difference
            time_delta = curr['timestamp'] - prev['timestamp']

            # Avoid division by zero
            if time_delta > 0:
                total_displacement += displacement
                total_time += time_delta

        # Calculate average velocity
        if total_time > 0:
            self.current_velocity = total_displacement / total_time
        else:
            self.current_velocity = 0.0

    def _calculate_direction(self) -> None:
        """
        Calculates current ball direction as a unit vector.

        Direction is useful for:
        - Determining pass direction (towards which player?)
        - Shot detection (moving towards goal?)
        - Possession change (ball direction reversed?)

        Returns normalized vector (dx, dy) where:
        - (1, 0) = moving right
        - (0, 1) = moving down
        - (-1, 0) = moving left
        - (0, -1) = moving up
        """
        if len(self.ball_history) < 2:
            self.current_direction = np.array([0.0, 0.0])
            return

        # Use first and last position in window for direction
        recent_positions = list(self.ball_history)[-self.velocity_window:]
        first = recent_positions[0]
        last = recent_positions[-1]

        # Direction vector
        dx = last['x'] - first['x']
        dy = last['y'] - first['y']

        # Normalize to unit vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            self.current_direction = np.array([dx / magnitude, dy / magnitude])
        else:
            self.current_direction = np.array([0.0, 0.0])

    def get_velocity(self) -> float:
        """
        Returns current ball speed in pixels/second.

        Typical values:
        - 0-100 px/s: Ball stationary or slow roll
        - 100-300 px/s: Dribbling
        - 300-800 px/s: Pass
        - 800+ px/s: Shot or long pass

        Note: Convert to m/s using field calibration for real-world speeds
        """
        return self.current_velocity

    def get_velocity_mps(self, pixels_per_meter: float) -> float:
        """
        Returns current ball speed in meters/second.

        Args:
            pixels_per_meter: Calibration factor (pixels per meter in the scene)
                            Calculate from known field dimensions

        Returns:
            Ball velocity in m/s

        Typical football velocities:
        - Pass: 2-10 m/s
        - Shot: 10-30 m/s
        - Professional shot: 20-35 m/s
        """
        return self.current_velocity / pixels_per_meter

    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Returns current ball position (x, y) or None if no tracking data.

        Returns:
            Tuple of (center_x, center_y) in pixels, or None
        """
        if len(self.ball_history) == 0:
            return None

        latest = self.ball_history[-1]
        return (latest['x'], latest['y'])

    def get_direction(self) -> np.ndarray:
        """
        Returns current ball direction as normalized vector (dx, dy).

        Returns:
            Numpy array [dx, dy] representing direction
        """
        return self.current_direction

    def get_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns current ball bounding box or None if no tracking data.

        Returns:
            Tuple of (x1, y1, x2, y2) in pixels, or None
        """
        if len(self.ball_history) == 0:
            return None

        latest = self.ball_history[-1]
        return latest['bbox']

    def is_moving(self, threshold: float = 50.0) -> bool:
        """
        Checks if ball is moving above a velocity threshold.

        Args:
            threshold: Minimum velocity in pixels/second to consider "moving"
                      Default: 50 px/s (slow roll threshold)

        Returns:
            True if ball velocity > threshold

        Useful for:
        - Detecting active play vs. dead ball
        - Filtering out noise when ball is stationary
        - Event detection (pass started, shot taken)
        """
        return self.current_velocity > threshold

    def get_history(self, n: int = None) -> List[dict]:
        """
        Returns recent ball position history.

        Args:
            n: Number of recent positions to return (None = all)

        Returns:
            List of position dictionaries with keys:
            - 'x', 'y': center position
            - 'frame_idx': frame number
            - 'timestamp': time in seconds
            - 'bbox': bounding box (x1, y1, x2, y2)
        """
        if n is None:
            return list(self.ball_history)
        else:
            return list(self.ball_history)[-n:]

    def reset(self) -> None:
        """
        Resets the tracker, clearing all history.

        Use when:
        - Starting a new video
        - After scene change
        - After tracking loss (ball out of frame for extended period)
        """
        self.ball_history.clear()
        self.current_velocity = 0.0
        self.current_direction = np.array([0.0, 0.0])

    def predict_position(self, time_delta: float) -> Optional[Tuple[float, float]]:
        """
        Predicts future ball position based on current velocity and direction.

        Args:
            time_delta: Time in seconds to predict forward

        Returns:
            Predicted (x, y) position or None if insufficient data

        Useful for:
        - Tracking through occlusion (player blocks ball)
        - Pass destination prediction
        - Shot trajectory analysis

        Note: Simple linear prediction. For curved trajectories, use physics model.
        """
        current_pos = self.get_position()
        if current_pos is None or self.current_velocity == 0:
            return None

        # Predict using: future_pos = current_pos + velocity * direction * time
        displacement = self.current_velocity * time_delta
        predicted_x = current_pos[0] + self.current_direction[0] * displacement
        predicted_y = current_pos[1] + self.current_direction[1] * displacement

        return (predicted_x, predicted_y)


# Example usage and testing helper
if __name__ == "__main__":
    """
    Example usage demonstrating ball tracking on simulated data.
    """
    print("Ball Tracker Module - Example Usage\n")

    # Initialize tracker
    tracker = BallTracker(max_history=30, velocity_window=3)

    # Simulate ball moving diagonally across frame
    print("Simulating ball movement...")
    for frame in range(10):
        # Simulate ball moving from (100, 100) to (200, 150)
        x = 100 + frame * 10
        y = 100 + frame * 5
        bbox = (x - 5, y - 5, x + 5, y + 5)  # 10x10 ball
        timestamp = frame / 30.0  # 30 fps

        tracker.update(bbox, frame, timestamp)

        if frame >= 2:  # Need at least 3 positions for velocity
            pos = tracker.get_position()
            vel = tracker.get_velocity()
            direction = tracker.get_direction()
            print(f"Frame {frame}: pos={pos}, velocity={vel:.1f} px/s, direction={direction}")

    print(f"\nFinal velocity: {tracker.get_velocity():.1f} px/s")
    print(f"Is moving? {tracker.is_moving()}")

    # Predict future position
    predicted = tracker.predict_position(1.0)  # 1 second ahead
    print(f"Predicted position in 1s: {predicted}")
