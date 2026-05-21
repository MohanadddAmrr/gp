"""
Accuracy Measurement Framework for TactiVision Pro

This module provides tools to measure the actual accuracy of tracking and detection systems
by comparing system outputs against ground truth annotations.

Usage:
    1. First, create ground truth annotations for sample frames
    2. Run the accuracy measurement
    3. View detailed accuracy reports

Example:
    python tests/test_accuracy_measurement.py --create-ground-truth --video input_videos/match.mp4 --frames 100
    python tests/test_accuracy_measurement.py --measure --video input_videos/match.mp4
    python tests/test_accuracy_measurement.py --report
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.ball_tracker import BallTracker
from services.possession_tracker import PossessionTracker
from services.event_detector import EventDetector
from services.sprint_detector import SprintDetector
from ultralytics import YOLO


@dataclass
class GroundTruthAnnotation:
    """Single frame ground truth annotation."""
    frame_idx: int
    timestamp: float
    player_positions: Dict[int, Tuple[float, float, str]]  # player_id -> (x, y, team)
    ball_position: Optional[Tuple[float, float]] = None
    events: List[Dict] = field(default_factory=list)  # passes, shots, etc.
    sprints: List[int] = field(default_factory=list)  # list of player_ids sprinting


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a specific component."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_ground_truth: int = 0
    total_predictions: int = 0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def accuracy(self) -> float:
        """Accuracy = TP / Total_GT (for detection tasks)"""
        if self.total_ground_truth == 0:
            return 0.0
        return self.true_positives / self.total_ground_truth


@dataclass
class TrackingAccuracyMetrics:
    """Accuracy metrics for tracking (includes position error)."""
    detection_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    position_errors: List[float] = field(default_factory=list)  # Euclidean distances
    id_switches: int = 0
    total_frames: int = 0
    
    @property
    def mae(self) -> float:
        """Mean Absolute Error (pixels)."""
        if not self.position_errors:
            return 0.0
        return np.mean(self.position_errors)
    
    @property
    def rmse(self) -> float:
        """Root Mean Square Error (pixels)."""
        if not self.position_errors:
            return 0.0
        return np.sqrt(np.mean([e**2 for e in self.position_errors]))
    
    @property
    def id_switch_rate(self) -> float:
        """ID switches per frame."""
        if self.total_frames == 0:
            return 0.0
        return self.id_switches / self.total_frames


class GroundTruthCreator:
    """Interactive tool to create ground truth annotations."""
    
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.annotations: Dict[int, GroundTruthAnnotation] = {}
        self.current_frame = 0
        
        # Annotation state
        self.selected_player = None
        self.player_id_counter = 1
        self.current_team = "A"
        
    def create_annotations(self, frame_indices: List[int]):
        """Create ground truth annotations for specified frames."""
        print(f"\n{'='*70}")
        print("GROUND TRUTH ANNOTATION TOOL")
        print(f"{'='*70}")
        print(f"Video: {self.video_path.name}")
        print(f"Total frames: {self.total_frames}")
        print(f"Annotating {len(frame_indices)} frames")
        print(f"\nControls:")
        print("  Click: Add/select player")
        print("  Right-click: Remove player")
        print("  'b': Switch to Team B")
        print("  'a': Switch to Team A")
        print("  'n': Next frame")
        print("  'p': Previous frame")
        print("  's': Save annotations")
        print("  'q': Quit")
        print(f"{'='*70}\n")
        
        for frame_idx in frame_indices:
            self.current_frame = frame_idx
            self._annotate_frame()
            
        self._save_annotations()
        
    def _annotate_frame(self):
        """Annotate a single frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if not ret:
            return
            
        timestamp = self.current_frame / self.fps
        annotation = GroundTruthAnnotation(
            frame_idx=self.current_frame,
            timestamp=timestamp,
            player_positions={}
        )
        
        # Load existing annotation if available
        if self.current_frame in self.annotations:
            annotation = self.annotations[self.current_frame]
        
        # Create display
        display = frame.copy()
        self._draw_annotations(display, annotation)
        
        # Mouse callback
        cv2.namedWindow("Ground Truth Annotation")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add or select player
                player_id = self._find_nearest_player(annotation, x, y)
                if player_id is None:
                    player_id = self.player_id_counter
                    self.player_id_counter += 1
                annotation.player_positions[player_id] = (x, y, self.current_team)
                self._draw_annotations(display, annotation)
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Remove player
                player_id = self._find_nearest_player(annotation, x, y)
                if player_id is not None:
                    del annotation.player_positions[player_id]
                    self._draw_annotations(display, annotation)
        
        cv2.setMouseCallback("Ground Truth Annotation", mouse_callback)
        
        while True:
            cv2.imshow("Ground Truth Annotation", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.current_team = "A"
                print("Switched to Team A")
            elif key == ord('b'):
                self.current_team = "B"
                print("Switched to Team B")
            elif key == ord('s'):
                self.annotations[self.current_frame] = annotation
                self._save_annotations()
                print(f"Saved annotations for frame {self.current_frame}")
            elif key == ord('n'):
                self.annotations[self.current_frame] = annotation
                break
            elif key == ord('p'):
                self.annotations[self.current_frame] = annotation
                self.current_frame = max(0, self.current_frame - 1)
                break
                
        cv2.destroyAllWindows()
        
    def _find_nearest_player(self, annotation: GroundTruthAnnotation, x: int, y: int, threshold: int = 30) -> Optional[int]:
        """Find nearest player to click position."""
        min_dist = float('inf')
        nearest = None
        
        for player_id, (px, py, _) in annotation.player_positions.items():
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest = player_id
                
        return nearest
    
    def _draw_annotations(self, display: np.ndarray, annotation: GroundTruthAnnotation):
        """Draw annotations on display."""
        # Clear display
        display[:] = display.copy()
        
        # Draw players
        for player_id, (x, y, team) in annotation.player_positions.items():
            color = (0, 0, 255) if team == "A" else (255, 0, 0)
            cv2.circle(display, (int(x), int(y)), 10, color, 2)
            cv2.putText(display, str(player_id), (int(x)-5, int(y)-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw info
        cv2.putText(display, f"Frame: {self.current_frame}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Team: {self.current_team}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Players: {len(annotation.player_positions)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _save_annotations(self):
        """Save annotations to file."""
        output_file = self.output_dir / f"{self.video_path.stem}_ground_truth.json"
        
        data = []
        for frame_idx, annotation in sorted(self.annotations.items()):
            data.append({
                'frame_idx': annotation.frame_idx,
                'timestamp': annotation.timestamp,
                'player_positions': {
                    str(k): {'x': v[0], 'y': v[1], 'team': v[2]}
                    for k, v in annotation.player_positions.items()
                },
                'ball_position': annotation.ball_position,
                'events': annotation.events,
                'sprints': annotation.sprints
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\nSaved {len(data)} annotations to: {output_file}")


class AccuracyMeasurer:
    """Measure accuracy by comparing system output to ground truth."""
    
    def __init__(self, video_path: Path, ground_truth_path: Path):
        self.video_path = video_path
        self.ground_truth_path = ground_truth_path
        
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            data = json.load(f)
            
        self.ground_truth: Dict[int, GroundTruthAnnotation] = {}
        for item in data:
            annotation = GroundTruthAnnotation(
                frame_idx=item['frame_idx'],
                timestamp=item['timestamp'],
                player_positions={
                    int(k): (v['x'], v['y'], v['team'])
                    for k, v in item['player_positions'].items()
                },
                ball_position=item.get('ball_position'),
                events=item.get('events', []),
                sprints=item.get('sprints', [])
            )
            self.ground_truth[annotation.frame_idx] = annotation
            
        # Initialize system components
        self.model = YOLO('yolov8n.pt')
        self.ball_tracker = BallTracker()
        self.possession_tracker = PossessionTracker()
        self.event_detector = EventDetector()
        self.sprint_detector = SprintDetector()
        
        # Metrics
        self.player_tracking_metrics = TrackingAccuracyMetrics()
        self.ball_tracking_metrics = TrackingAccuracyMetrics()
        self.pass_detection_metrics = AccuracyMetrics()
        self.shot_detection_metrics = AccuracyMetrics()
        self.sprint_detection_metrics = AccuracyMetrics()
        
    def measure_accuracy(self) -> Dict[str, Any]:
        """Run accuracy measurement."""
        print(f"\n{'='*70}")
        print("ACCURACY MEASUREMENT")
        print(f"{'='*70}")
        print(f"Video: {self.video_path.name}")
        print(f"Ground truth frames: {len(self.ground_truth)}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Process each annotated frame
        for frame_idx in sorted(self.ground_truth.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            gt = self.ground_truth[frame_idx]
            timestamp = gt.timestamp
            
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Extract detections
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > 0.5:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        detections.append((cx, cy, conf))
            
            # Measure player tracking accuracy
            self._measure_player_tracking(gt, detections, frame_idx)
            
            # Measure ball tracking
            self._measure_ball_tracking(gt, frame, timestamp)
            
            # Update other trackers
            self._update_trackers(detections, timestamp, frame_idx)
            
            if frame_idx % 10 == 0:
                print(f"  Processed frame {frame_idx}/{max(self.ground_truth.keys())}")
        
        cap.release()
        
        # Compile results
        results = self._compile_results()
        self._print_results(results)
        self._save_results(results)
        
        return results
    
    def _measure_player_tracking(self, gt: GroundTruthAnnotation, 
                                  detections: List[Tuple[float, float, float]],
                                  frame_idx: int):
        """Measure player tracking accuracy."""
        gt_players = gt.player_positions
        
        self.player_tracking_metrics.total_frames += 1
        self.player_tracking_metrics.detection_metrics.total_ground_truth += len(gt_players)
        self.player_tracking_metrics.detection_metrics.total_predictions += len(detections)
        
        # Match detections to ground truth
        matched_gt = set()
        matched_det = set()
        
        for det_idx, (dx, dy, conf) in enumerate(detections):
            best_dist = float('inf')
            best_gt = None
            
            for gt_id, (gx, gy, _) in gt_players.items():
                if gt_id in matched_gt:
                    continue
                    
                dist = np.sqrt((dx - gx)**2 + (dy - gy)**2)
                if dist < best_dist and dist < 50:  # 50 pixel threshold
                    best_dist = dist
                    best_gt = gt_id
            
            if best_gt is not None:
                matched_gt.add(best_gt)
                matched_det.add(det_idx)
                self.player_tracking_metrics.detection_metrics.true_positives += 1
                self.player_tracking_metrics.position_errors.append(best_dist)
            else:
                self.player_tracking_metrics.detection_metrics.false_positives += 1
        
        # Count false negatives
        for gt_id in gt_players:
            if gt_id not in matched_gt:
                self.player_tracking_metrics.detection_metrics.false_negatives += 1
    
    def _measure_ball_tracking(self, gt: GroundTruthAnnotation, frame: np.ndarray, timestamp: float):
        """Measure ball tracking accuracy."""
        # Ball tracking is complex - simplified version
        # In practice, you'd need ball detections from your ball detector
        pass
    
    def _update_trackers(self, detections: List[Tuple[float, float, float]], timestamp: float, frame_idx: int):
        """Update all trackers with current detections."""
        # Update possession tracker
        if detections:
            ball_pos = detections[0][:2] if detections else (0, 0)
            player_positions = {i: (x, y, 'A') for i, (x, y, _) in enumerate(detections)}
            self.possession_tracker.detect_possession(ball_pos, player_positions, frame_idx, timestamp)
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile all metrics into results dictionary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'video': str(self.video_path),
            'ground_truth_file': str(self.ground_truth_path),
            'total_frames_evaluated': len(self.ground_truth),
            'player_tracking': {
                'precision': round(self.player_tracking_metrics.detection_metrics.precision, 3),
                'recall': round(self.player_tracking_metrics.detection_metrics.recall, 3),
                'f1_score': round(self.player_tracking_metrics.detection_metrics.f1_score, 3),
                'mae_pixels': round(self.player_tracking_metrics.mae, 2),
                'rmse_pixels': round(self.player_tracking_metrics.rmse, 2),
                'id_switches': self.player_tracking_metrics.id_switches,
                'true_positives': self.player_tracking_metrics.detection_metrics.true_positives,
                'false_positives': self.player_tracking_metrics.detection_metrics.false_positives,
                'false_negatives': self.player_tracking_metrics.detection_metrics.false_negatives,
            },
            'ball_tracking': {
                'precision': round(self.ball_tracking_metrics.detection_metrics.precision, 3),
                'recall': round(self.ball_tracking_metrics.detection_metrics.recall, 3),
                'f1_score': round(self.ball_tracking_metrics.detection_metrics.f1_score, 3),
                'mae_pixels': round(self.ball_tracking_metrics.mae, 2),
                'rmse_pixels': round(self.ball_tracking_metrics.rmse, 2),
            },
            'pass_detection': {
                'precision': round(self.pass_detection_metrics.precision, 3),
                'recall': round(self.pass_detection_metrics.recall, 3),
                'f1_score': round(self.pass_detection_metrics.f1_score, 3),
            },
            'shot_detection': {
                'precision': round(self.shot_detection_metrics.precision, 3),
                'recall': round(self.shot_detection_metrics.recall, 3),
                'f1_score': round(self.shot_detection_metrics.f1_score, 3),
            },
            'sprint_detection': {
                'precision': round(self.sprint_detection_metrics.precision, 3),
                'recall': round(self.sprint_detection_metrics.recall, 3),
                'f1_score': round(self.sprint_detection_metrics.f1_score, 3),
            }
        }
    
    def _print_results(self, results: Dict[str, Any]):
        """Print accuracy results."""
        print(f"\n{'='*70}")
        print("ACCURACY RESULTS")
        print(f"{'='*70}")
        
        print("\n📊 PLAYER TRACKING:")
        pt = results['player_tracking']
        print(f"  Precision: {pt['precision']:.3f} ({pt['precision']*100:.1f}%)")
        print(f"  Recall:    {pt['recall']:.3f} ({pt['recall']*100:.1f}%)")
        print(f"  F1 Score:  {pt['f1_score']:.3f}")
        print(f"  MAE:       {pt['mae_pixels']:.1f} pixels")
        print(f"  RMSE:      {pt['rmse_pixels']:.1f} pixels")
        print(f"  TP: {pt['true_positives']}, FP: {pt['false_positives']}, FN: {pt['false_negatives']}")
        
        print("\n⚽ BALL TRACKING:")
        bt = results['ball_tracking']
        print(f"  Precision: {bt['precision']:.3f} ({bt['precision']*100:.1f}%)")
        print(f"  Recall:    {bt['recall']:.3f} ({bt['recall']*100:.1f}%)")
        print(f"  F1 Score:  {bt['f1_score']:.3f}")
        
        print("\n🔄 PASS DETECTION:")
        pd = results['pass_detection']
        print(f"  Precision: {pd['precision']:.3f} ({pd['precision']*100:.1f}%)")
        print(f"  Recall:    {pd['recall']:.3f} ({pd['recall']*100:.1f}%)")
        print(f"  F1 Score:  {pd['f1_score']:.3f}")
        
        print("\n🎯 SHOT DETECTION:")
        sd = results['shot_detection']
        print(f"  Precision: {sd['precision']:.3f} ({sd['precision']*100:.1f}%)")
        print(f"  Recall:    {sd['recall']:.3f} ({sd['recall']*100:.1f}%)")
        print(f"  F1 Score:  {sd['f1_score']:.3f}")
        
        print("\n💨 SPRINT DETECTION:")
        sp = results['sprint_detection']
        print(f"  Precision: {sp['precision']:.3f} ({sp['precision']*100:.1f}%)")
        print(f"  Recall:    {sp['recall']:.3f} ({sp['recall']*100:.1f}%)")
        print(f"  F1 Score:  {sp['f1_score']:.3f}")
        
        print(f"\n{'='*70}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        output_file = Path("tests/accuracy_reports") / f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_native = convert_to_native(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_native, f, indent=2)
            
        print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='TactiVision Pro Accuracy Measurement')
    parser.add_argument('--create-ground-truth', action='store_true',
                       help='Create ground truth annotations')
    parser.add_argument('--measure', action='store_true',
                       help='Measure accuracy against ground truth')
    parser.add_argument('--report', action='store_true',
                       help='Show latest accuracy report')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--frames', type=int, default=50,
                       help='Number of frames to annotate (for --create-ground-truth)')
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth file (for --measure)')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    if args.create_ground_truth:
        # Create ground truth annotations
        output_dir = Path("tests/ground_truth")
        creator = GroundTruthCreator(video_path, output_dir)
        
        # Select evenly spaced frames
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_indices = np.linspace(0, total_frames-1, args.frames, dtype=int)
        creator.create_annotations(frame_indices.tolist())
        
    elif args.measure:
        # Measure accuracy
        if not args.ground_truth:
            # Auto-find ground truth file
            gt_file = Path(f"tests/ground_truth/{video_path.stem}_ground_truth.json")
            if not gt_file.exists():
                print(f"❌ Ground truth file not found: {gt_file}")
                print("   Run with --create-ground-truth first")
                return
            args.ground_truth = gt_file
            
        measurer = AccuracyMeasurer(video_path, Path(args.ground_truth))
        measurer.measure_accuracy()
        
    elif args.report:
        # Show latest report
        reports_dir = Path("tests/accuracy_reports")
        if not reports_dir.exists():
            print("❌ No accuracy reports found")
            return
            
        reports = sorted(reports_dir.glob("accuracy_report_*.json"))
        if not reports:
            print("❌ No accuracy reports found")
            return
            
        latest = reports[-1]
        with open(latest, 'r') as f:
            results = json.load(f)
            
        print(f"\n📊 Latest Accuracy Report: {latest.name}")
        print(f"   Generated: {results['timestamp']}")
        print(f"   Video: {results['video']}")
        print(f"   Frames: {results['total_frames_evaluated']}")
        
        # Print summary
        print("\n🎯 ACCURACY SUMMARY:")
        pt = results['player_tracking']
        print(f"  Player Tracking: P={pt['precision']:.3f}, R={pt['recall']:.3f}, F1={pt['f1_score']:.3f}")
        bt = results['ball_tracking']
        print(f"  Ball Tracking:   P={bt['precision']:.3f}, R={bt['recall']:.3f}, F1={bt['f1_score']:.3f}")
        pd = results['pass_detection']
        print(f"  Pass Detection:  P={pd['precision']:.3f}, R={pd['recall']:.3f}, F1={pd['f1_score']:.3f}")
        sd = results['shot_detection']
        print(f"  Shot Detection:  P={sd['precision']:.3f}, R={sd['recall']:.3f}, F1={sd['f1_score']:.3f}")
        sp = results['sprint_detection']
        print(f"  Sprint Detection: P={sp['precision']:.3f}, R={sp['recall']:.3f}, F1={sp['f1_score']:.3f}")


if __name__ == "__main__":
    main()
