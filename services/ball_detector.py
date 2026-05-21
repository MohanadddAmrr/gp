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
    
    def __init__(
        self,
        min_radius: int = 3,      # Reduced from 5 for small balls
        max_radius: int = 60,     # Increased from 50 for close-up shots
        min_circularity: float = 0.5,  # Reduced from 0.6 for partially occluded balls
        hsv_lower: Tuple[int, int, int] = (0, 0, 180),   # White ball default
        hsv_upper: Tuple[int, int, int] = (180, 50, 255),
        enable_multi_color: bool = True  # New: detect multiple ball colors
    ):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.enable_multi_color = enable_multi_color
        
        # Additional color ranges for different ball types
        self.ball_color_ranges = [
            # White ball (default)
            {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 50, 255])},
            # Yellow ball
            {'lower': np.array([20, 100, 150]), 'upper': np.array([35, 255, 255])},
            # Orange ball
            {'lower': np.array([10, 150, 150]), 'upper': np.array([25, 255, 255])},
            # Classic white with some color
            {'lower': np.array([0, 0, 150]), 'upper': np.array([180, 80, 255])},
        ]

    
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
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
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
    
    def detect_best(self, frame: np.ndarray, 
                    last_known_pos: Optional[Tuple[float, float]] = None,
                    max_jump_px: float = 200.0) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect the most likely ball (largest, most circular).
        
        If last_known_pos is provided, reject detections that are too far
        from the last known ball position (likely false positives).
        
        Args:
            frame: BGR image from OpenCV
            last_known_pos: (x, y) of last known ball position, or None
            max_jump_px: Maximum allowed distance from last known position
            
        Returns:
            Single bounding box (x1, y1, x2, y2) or None
        """
        detections = self.detect(frame)
        
        if not detections:
            return None
        
        # If we have a reference position, filter by distance
        if last_known_pos is not None:
            ref_x, ref_y = last_known_pos
            valid = []
            for box in detections:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                dist = np.hypot(cx - ref_x, cy - ref_y)
                if dist <= max_jump_px:
                    valid.append(box)
            
            if valid:
                return max(valid, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            # All detections too far from last known position - reject all
            return None
        
        # No reference position - return largest detection
        best = max(detections, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        return best
    
    def detect_multi_color(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect balls using multiple color ranges.
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if not self.enable_multi_color:
            return self.detect(frame)
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        all_detections = []
        
        for color_range in self.ball_color_ranges:
            lower = color_range['lower']
            upper = color_range['upper']
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < np.pi * self.min_radius ** 2:
                    continue
                if area > np.pi * self.max_radius ** 2:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < self.min_circularity:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Aspect ratio check (ball should be roughly square)
                aspect = w / h if h > 0 else 0
                if aspect < 0.6 or aspect > 1.7:
                    continue
                    
                all_detections.append((x, y, w, h))
        
        # Remove duplicate detections (overlapping boxes)
        return self._remove_duplicates(all_detections)
    
    def _remove_duplicates(self, detections: List[Tuple[int, int, int, int]], 
                           iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping detections."""
        if not detections:
            return []
            
        # Sort by area (larger first)
        detections = sorted(detections, key=lambda d: d[2] * d[3], reverse=True)
        
        kept = []
        for det in detections:
            x1, y1, w1, h1 = det
            is_duplicate = False
            
            for kept_det in kept:
                x2, y2, w2, h2 = kept_det
                
                # Calculate IoU
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                
                if xi2 > xi1 and yi2 > yi1:
                    inter_area = (xi2 - xi1) * (yi2 - yi1)
                    union_area = w1 * h1 + w2 * h2 - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                kept.append(det)
        
        return kept




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
