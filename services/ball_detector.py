"""
Color-Based Ball Detection Module
Fallback detector for when YOLO fails to detect footballs
Detects white/light colored balls on green field
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class ColorBallDetector:
    """
    Detects balls using color segmentation and contour analysis.
    More reliable than YOLO for small, fast-moving footballs.
    """
    
    def __init__(self, 
                 min_radius: int = 5,
                 max_radius: int = 50,
                 color_mode: str = 'white'):
        """
        Initialize ball detector.
        
        Args:
            min_radius: Minimum ball radius in pixels
            max_radius: Maximum ball radius in pixels
            color_mode: 'white' for white balls, 'orange' for orange balls
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.color_mode = color_mode
        
        # Define color ranges in HSV
        if color_mode == 'white':
            # White ball: high value, low saturation
            self.lower_color = np.array([0, 0, 200])
            self.upper_color = np.array([180, 30, 255])
        elif color_mode == 'orange':
            # Orange ball
            self.lower_color = np.array([5, 100, 100])
            self.upper_color = np.array([15, 255, 255])
        else:
            raise ValueError(f"Unknown color_mode: {color_mode}")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Detect balls in a frame using color segmentation.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for ball color
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Calculate equivalent radius
            if area < np.pi * self.min_radius**2:
                continue
            if area > np.pi * self.max_radius**2:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Ball should be reasonably circular (> 0.6)
            if circularity < 0.6:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add to detections
            detections.append((float(x), float(y), float(x + w), float(y + h)))
        
        return detections
    
    def detect_best(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect the most likely ball (largest, most circular).
        
        Returns:
            Single bounding box (x1, y1, x2, y2) or None
        """
        detections = self.detect(frame)
        
        if not detections:
            return None
        
        # Return the detection with largest area (most likely the ball)
        best = max(detections, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        return best


# Testing function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ball_detector.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print("Testing Color-Based Ball Detection...")
    detector = ColorBallDetector(min_radius=5, max_radius=50, color_mode='white')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        sys.exit(1)
    
    detections_count = 0
    frames_processed = 0
    
    while frames_processed < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        ball_box = detector.detect_best(frame)
        if ball_box is not None:
            detections_count += 1
            # Draw detection
            x1, y1, x2, y2 = map(int, ball_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "BALL", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 255), 2)
        
        frames_processed += 1
        
        cv2.imshow("Ball Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nResults: {detections_count}/{frames_processed} frames ({detections_count/frames_processed*100:.1f}%)")
