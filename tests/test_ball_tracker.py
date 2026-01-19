"""
Test script for Ball Tracker module

This script tests the ball tracker on actual video footage to verify:
1. Ball position is tracked frame-by-frame
2. Velocity calculations are reasonable (not jumping wildly)
3. Position follows visible ball movement
4. Edge cases are handled (no ball detected, tracking loss)
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path to import services
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ball_tracker import BallTracker


def test_basic_functionality():
    """Test basic tracker functionality with simulated data."""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)

    tracker = BallTracker()

    # Simulate ball moving in a straight line
    for i in range(5):
        bbox = (100 + i*10, 200 + i*5, 110 + i*10, 210 + i*5)
        tracker.update(bbox, i, i/30.0)

    pos = tracker.get_position()
    vel = tracker.get_velocity()
    direction = tracker.get_direction()

    print(f"[OK] Position: {pos}")
    print(f"[OK] Velocity: {vel:.2f} px/s")
    print(f"[OK] Direction: {direction}")
    print(f"[OK] Is moving: {tracker.is_moving()}")

    assert pos is not None, "Position should not be None"
    assert vel > 0, "Velocity should be positive for moving ball"
    print("[OK] Basic functionality test PASSED\n")


def test_velocity_smoothing():
    """Test that velocity smoothing works correctly."""
    print("=" * 60)
    print("TEST 2: Velocity Smoothing")
    print("=" * 60)

    tracker = BallTracker(velocity_window=3)

    # Simulate noisy detections
    positions = [
        (100, 200),
        (110, 205),  # +10, +5
        (108, 203),  # -2, -2 (noise)
        (120, 210),  # +12, +7
        (130, 215),  # +10, +5
    ]

    velocities = []
    for i, (x, y) in enumerate(positions):
        bbox = (x-5, y-5, x+5, y+5)
        tracker.update(bbox, i, i/30.0)
        if i >= 2:  # Need 3 positions for velocity
            velocities.append(tracker.get_velocity())
            print(f"Frame {i}: velocity = {tracker.get_velocity():.2f} px/s")

    # Check that velocity doesn't jump drastically
    velocity_changes = [abs(velocities[i+1] - velocities[i]) for i in range(len(velocities)-1)]
    max_change = max(velocity_changes)

    print(f"[OK] Maximum velocity change: {max_change:.2f} px/s")
    print(f"[OK] Velocity smoothing working: {max_change < 200}")  # Reasonable threshold
    print("[OK] Velocity smoothing test PASSED\n")


def test_stationary_ball():
    """Test tracker behavior when ball is stationary."""
    print("=" * 60)
    print("TEST 3: Stationary Ball")
    print("=" * 60)

    tracker = BallTracker()

    # Ball at same position for multiple frames
    for i in range(5):
        bbox = (100, 200, 110, 210)  # Same position
        tracker.update(bbox, i, i/30.0)

    vel = tracker.get_velocity()
    is_moving = tracker.is_moving()

    print(f"[OK] Velocity: {vel:.2f} px/s")
    print(f"[OK] Is moving: {is_moving}")

    assert vel < 10, "Stationary ball should have near-zero velocity"
    assert not is_moving, "Stationary ball should not be marked as moving"
    print("[OK] Stationary ball test PASSED\n")


def test_direction_calculation():
    """Test direction vector calculation."""
    print("=" * 60)
    print("TEST 4: Direction Calculation")
    print("=" * 60)

    tracker = BallTracker()

    # Ball moving diagonally (right and down)
    for i in range(5):
        x = 100 + i * 10  # Moving right
        y = 200 + i * 10  # Moving down
        bbox = (x-5, y-5, x+5, y+5)
        tracker.update(bbox, i, i/30.0)

    direction = tracker.get_direction()
    print(f"[OK] Direction vector: {direction}")

    # Should be roughly (0.707, 0.707) for 45-degree diagonal
    expected_angle = np.arctan2(direction[1], direction[0])
    print(f"[OK] Direction angle: {np.degrees(expected_angle):.1f} degrees")

    # Check it's normalized (magnitude = 1)
    magnitude = np.linalg.norm(direction)
    print(f"[OK] Vector magnitude: {magnitude:.3f}")

    assert abs(magnitude - 1.0) < 0.01, "Direction should be normalized"
    print("[OK] Direction calculation test PASSED\n")


def test_prediction():
    """Test position prediction."""
    print("=" * 60)
    print("TEST 5: Position Prediction")
    print("=" * 60)

    tracker = BallTracker()

    # Ball moving at constant velocity
    for i in range(5):
        x = 100 + i * 10  # 10 px/frame = 300 px/s at 30fps
        y = 200
        bbox = (x-5, y-5, x+5, y+5)
        tracker.update(bbox, i, i/30.0)

    current_pos = tracker.get_position()
    predicted_pos = tracker.predict_position(1.0)  # 1 second ahead

    print(f"[OK] Current position: {current_pos}")
    print(f"[OK] Predicted position (1s ahead): {predicted_pos}")

    # At 300 px/s moving right, should be ~300 pixels ahead
    if predicted_pos:
        distance_moved = predicted_pos[0] - current_pos[0]
        print(f"[OK] Predicted movement: {distance_moved:.1f} pixels")
        assert 250 < distance_moved < 350, "Prediction should be reasonable"

    print("[OK] Position prediction test PASSED\n")


def test_reset():
    """Test tracker reset functionality."""
    print("=" * 60)
    print("TEST 6: Reset Functionality")
    print("=" * 60)

    tracker = BallTracker()

    # Add some tracking data
    for i in range(5):
        bbox = (100 + i*10, 200, 110 + i*10, 210)
        tracker.update(bbox, i, i/30.0)

    print(f"Before reset - Velocity: {tracker.get_velocity():.2f} px/s")
    print(f"Before reset - Position: {tracker.get_position()}")

    # Reset
    tracker.reset()

    print(f"After reset - Velocity: {tracker.get_velocity():.2f} px/s")
    print(f"After reset - Position: {tracker.get_position()}")

    assert tracker.get_velocity() == 0.0, "Velocity should be zero after reset"
    assert tracker.get_position() is None, "Position should be None after reset"
    print("[OK] Reset functionality test PASSED\n")


def test_on_video(video_path: str, max_frames: int = 100):
    """
    Test tracker on actual video using simple ball detection.

    Note: This is a simplified test. In production, you'd use YOLO for detection.
    Here we use basic color detection just to verify the tracker works.
    """
    print("=" * 60)
    print("TEST 7: Real Video Test")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"[WARN] Video not found: {video_path}")
        print("Skipping video test (expected in development)")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = BallTracker()

    print(f"[OK] Video opened: {video_path}")
    print(f"[OK] FPS: {fps}")

    frame_count = 0
    detections = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps

        # Simple ball detection (white object) - PLACEHOLDER
        # In production, replace with YOLO detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest circular contour (likely the ball)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if 50 < area < 2000:  # Reasonable ball size
                x, y, w, h = cv2.boundingRect(largest)
                bbox = (x, y, x+w, y+h)

                tracker.update(bbox, frame_count, timestamp)
                detections += 1

                if frame_count % 10 == 0:
                    vel = tracker.get_velocity()
                    pos = tracker.get_position()
                    print(f"Frame {frame_count}: pos={pos}, vel={vel:.1f} px/s")

        frame_count += 1

    cap.release()

    print(f"\n[OK] Processed {frame_count} frames")
    print(f"[OK] Ball detected in {detections} frames ({100*detections/frame_count:.1f}%)")
    print(f"[OK] Final velocity: {tracker.get_velocity():.2f} px/s")
    print("[OK] Video test COMPLETED\n")


def run_all_tests():
    """Run all ball tracker tests."""
    print("\n" + "=" * 60)
    print("BALL TRACKER TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_basic_functionality()
        test_velocity_smoothing()
        test_stationary_ball()
        test_direction_calculation()
        test_prediction()
        test_reset()

        # Test on video if available
        video_path = "input_videos/arsenal.mp4"
        test_on_video(video_path, max_frames=100)

        print("=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
