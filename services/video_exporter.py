"""
Video Export with Overlays Module

Handles exporting processed video with overlays, tactical drawings,
highlight moments, and custom branding.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum

__all__ = ['VideoExporter', 'ExportConfig', 'TacticalDrawing', 'HighlightMoment',
           'WatermarkConfig', 'TacticalDrawer', 'ZoomEffect', 'WatermarkRenderer',
           'MultiAngleSync', 'create_video_exporter']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Video export formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"


@dataclass
class ExportConfig:
    """Configuration for video export."""
    output_path: str
    format: ExportFormat = ExportFormat.MP4
    resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0
    bitrate: str = "10M"
    codec: str = "h264"
    quality: int = 95  # 0-100
    include_audio: bool = True


@dataclass
class TacticalDrawing:
    """Tactical drawing element."""
    drawing_type: str  # 'arrow', 'circle', 'line', 'rectangle', 'freehand'
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 255)
    thickness: int = 3
    start_frame: int = 0
    end_frame: int = 0
    label: str = ""


@dataclass
class HighlightMoment:
    """Highlight moment configuration."""
    start_frame: int
    end_frame: int
    zoom_factor: float = 1.5
    zoom_center: Optional[Tuple[int, int]] = None
    slow_motion: bool = False
    slow_factor: float = 0.5
    title: str = ""


@dataclass
class WatermarkConfig:
    """Watermark and branding configuration."""
    text: str = ""
    logo_path: Optional[str] = None
    position: Tuple[int, int] = (20, 20)
    opacity: float = 0.7
    size: Tuple[int, int] = (150, 50)


class TacticalDrawer:
    """Draw tactical elements on video frames."""
    
    def __init__(self):
        self.drawings: List[TacticalDrawing] = []
        
    def add_arrow(self, start: Tuple[int, int], end: Tuple[int, int],
                  color: Tuple[int, int, int] = (0, 255, 255),
                  thickness: int = 3, frames: Optional[Tuple[int, int]] = None,
                  label: str = ""):
        """Add an arrow drawing."""
        drawing = TacticalDrawing(
            drawing_type='arrow',
            points=[start, end],
            color=color,
            thickness=thickness,
            start_frame=frames[0] if frames else 0,
            end_frame=frames[1] if frames else 999999,
            label=label
        )
        self.drawings.append(drawing)
        
    def add_circle(self, center: Tuple[int, int], radius: int,
                   color: Tuple[int, int, int] = (0, 255, 255),
                   thickness: int = 3, frames: Optional[Tuple[int, int]] = None,
                   label: str = ""):
        """Add a circle drawing."""
        drawing = TacticalDrawing(
            drawing_type='circle',
            points=[center],
            color=color,
            thickness=thickness,
            start_frame=frames[0] if frames else 0,
            end_frame=frames[1] if frames else 999999,
            label=label
        )
        drawing.radius = radius
        self.drawings.append(drawing)
        
    def add_line(self, points: List[Tuple[int, int]],
                 color: Tuple[int, int, int] = (0, 255, 255),
                 thickness: int = 3, frames: Optional[Tuple[int, int]] = None,
                 label: str = ""):
        """Add a line drawing."""
        drawing = TacticalDrawing(
            drawing_type='line',
            points=points,
            color=color,
            thickness=thickness,
            start_frame=frames[0] if frames else 0,
            end_frame=frames[1] if frames else 999999,
            label=label
        )
        self.drawings.append(drawing)
        
    def add_rectangle(self, top_left: Tuple[int, int], 
                     bottom_right: Tuple[int, int],
                     color: Tuple[int, int, int] = (0, 255, 255),
                     thickness: int = 3, frames: Optional[Tuple[int, int]] = None,
                     label: str = ""):
        """Add a rectangle drawing."""
        drawing = TacticalDrawing(
            drawing_type='rectangle',
            points=[top_left, bottom_right],
            color=color,
            thickness=thickness,
            start_frame=frames[0] if frames else 0,
            end_frame=frames[1] if frames else 999999,
            label=label
        )
        self.drawings.append(drawing)
        
    def clear_drawings(self):
        """Clear all drawings."""
        self.drawings.clear()
        
    def render(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render all active drawings on frame."""
        result = frame.copy()
        
        for drawing in self.drawings:
            if drawing.start_frame <= frame_number <= drawing.end_frame:
                result = self._draw_element(result, drawing)
                
        return result
        
    def _draw_element(self, frame: np.ndarray, drawing: TacticalDrawing) -> np.ndarray:
        """Draw a single tactical element."""
        result = frame.copy()
        
        if drawing.drawing_type == 'arrow':
            if len(drawing.points) >= 2:
                start, end = drawing.points[0], drawing.points[1]
                result = cv2.arrowedLine(
                    result, start, end, drawing.color, 
                    drawing.thickness, tipLength=0.3
                )
                
        elif drawing.drawing_type == 'circle':
            if len(drawing.points) >= 1:
                center = drawing.points[0]
                radius = getattr(drawing, 'radius', 30)
                result = cv2.circle(
                    result, center, radius, drawing.color, drawing.thickness
                )
                
        elif drawing.drawing_type == 'line':
            if len(drawing.points) >= 2:
                points = np.array(drawing.points, np.int32)
                result = cv2.polylines(
                    result, [points], False, drawing.color, drawing.thickness
                )
                
        elif drawing.drawing_type == 'rectangle':
            if len(drawing.points) >= 2:
                top_left, bottom_right = drawing.points[0], drawing.points[1]
                result = cv2.rectangle(
                    result, top_left, bottom_right, drawing.color, drawing.thickness
                )
                
        # Draw label if present
        if drawing.label:
            label_pos = (drawing.points[0][0], drawing.points[0][1] - 10)
            cv2.putText(result, drawing.label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, drawing.color, 2)
                       
        return result


class ZoomEffect:
    """Handle zoom effects for highlight moments."""
    
    def __init__(self):
        self.highlights: List[HighlightMoment] = []
        
    def add_highlight(self, start_time: float, end_time: float,
                     zoom_factor: float = 1.5,
                     zoom_center: Optional[Tuple[int, int]] = None,
                     slow_motion: bool = False,
                     slow_factor: float = 0.5,
                     title: str = ""):
        """Add a highlight moment."""
        # Convert time to frames (will be set properly when fps is known)
        highlight = HighlightMoment(
            start_frame=0,
            end_frame=0,
            zoom_factor=zoom_factor,
            zoom_center=zoom_center,
            slow_motion=slow_motion,
            slow_factor=slow_factor,
            title=title
        )
        highlight.start_time = start_time
        highlight.end_time = end_time
        self.highlights.append(highlight)
        
    def set_fps(self, fps: float):
        """Set FPS and convert time-based highlights to frames."""
        for highlight in self.highlights:
            if hasattr(highlight, 'start_time'):
                highlight.start_frame = int(highlight.start_time * fps)
                highlight.end_frame = int(highlight.end_time * fps)
                
    def apply_zoom(self, frame: np.ndarray, highlight: HighlightMoment) -> np.ndarray:
        """Apply zoom effect to frame."""
        h, w = frame.shape[:2]
        
        # Calculate zoom center
        if highlight.zoom_center:
            cx, cy = highlight.zoom_center
        else:
            cx, cy = w // 2, h // 2
            
        # Calculate crop region
        new_w = int(w / highlight.zoom_factor)
        new_h = int(h / highlight.zoom_factor)
        
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        
        # Adjust if out of bounds
        if x2 - x1 < new_w:
            x1 = max(0, x2 - new_w)
        if y2 - y1 < new_h:
            y1 = max(0, y2 - new_h)
            
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return zoomed
        
    def is_in_highlight(self, frame_number: int) -> Optional[HighlightMoment]:
        """Check if frame is within a highlight moment."""
        for highlight in self.highlights:
            if highlight.start_frame <= frame_number <= highlight.end_frame:
                return highlight
        return None


class WatermarkRenderer:
    """Render watermarks and branding on video."""
    
    def __init__(self):
        self.watermarks: List[WatermarkConfig] = []
        
    def add_text_watermark(self, text: str, position: Tuple[int, int] = (20, 20),
                          opacity: float = 0.7, size: int = 24):
        """Add text watermark."""
        config = WatermarkConfig(
            text=text,
            position=position,
            opacity=opacity,
            size=(size * len(text), size + 10)
        )
        self.watermarks.append(config)
        
    def add_logo_watermark(self, logo_path: str, 
                          position: Tuple[int, int] = (20, 20),
                          opacity: float = 0.7,
                          size: Tuple[int, int] = (150, 50)):
        """Add logo watermark."""
        config = WatermarkConfig(
            logo_path=logo_path,
            position=position,
            opacity=opacity,
            size=size
        )
        self.watermarks.append(config)
        
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render all watermarks on frame."""
        result = frame.copy()
        
        for watermark in self.watermarks:
            if watermark.logo_path and Path(watermark.logo_path).exists():
                result = self._render_logo(result, watermark)
            elif watermark.text:
                result = self._render_text(result, watermark)
                
        return result
        
    def _render_text(self, frame: np.ndarray, config: WatermarkConfig) -> np.ndarray:
        """Render text watermark."""
        overlay = frame.copy()
        
        # Draw semi-transparent background
        x, y = config.position
        w, h = config.size
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, config.text, (x + 10, y + h - 15), 
                   font, 0.7, (255, 255, 255), 2)
        
        # Blend with opacity
        result = cv2.addWeighted(overlay, config.opacity, frame, 1 - config.opacity, 0)
        return result
        
    def _render_logo(self, frame: np.ndarray, config: WatermarkConfig) -> np.ndarray:
        """Render logo watermark."""
        try:
            logo = cv2.imread(config.logo_path, cv2.IMREAD_UNCHANGED)
            if logo is None:
                return frame
                
            # Resize logo
            logo = cv2.resize(logo, config.size)
            
            x, y = config.position
            h, w = logo.shape[:2]
            
            # Handle alpha channel
            if logo.shape[2] == 4:
                alpha = logo[:, :, 3] / 255.0 * config.opacity
                for c in range(3):
                    frame[y:y+h, x:x+w, c] = (
                        alpha * logo[:, :, c] + 
                        (1 - alpha) * frame[y:y+h, x:x+w, c]
                    )
            else:
                overlay = frame.copy()
                overlay[y:y+h, x:x+w] = logo
                frame = cv2.addWeighted(overlay, config.opacity, frame, 1 - config.opacity, 0)
                
        except Exception as e:
            logger.warning(f"Error rendering logo: {e}")
            
        return frame


class MultiAngleSync:
    """Synchronize and export multi-angle video."""
    
    def __init__(self):
        self.video_sources: Dict[str, Dict] = {}
        self.sync_offsets: Dict[str, float] = {}
        
    def add_video_source(self, name: str, video_path: str, 
                        sync_offset: float = 0.0):
        """Add a video source with sync offset."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
            
        self.video_sources[name] = {
            'capture': cap,
            'path': video_path,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        self.sync_offsets[name] = sync_offset
        return True
        
    def get_synced_frames(self, timestamp: float) -> Dict[str, np.ndarray]:
        """Get synchronized frames from all sources at given timestamp."""
        frames = {}
        
        for name, source in self.video_sources.items():
            offset = self.sync_offsets[name]
            adjusted_time = timestamp + offset
            frame_number = int(adjusted_time * source['fps'])
            
            # Clamp to valid range
            frame_number = max(0, min(frame_number, source['frames'] - 1))
            
            source['capture'].set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = source['capture'].read()
            
            if ret:
                frames[name] = frame
                
        return frames
        
    def create_split_screen(self, frames: Dict[str, np.ndarray],
                           layout: str = "2x2") -> np.ndarray:
        """Create split screen view from multiple angles."""
        if not frames:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
            
        frame_list = list(frames.values())
        
        if layout == "2x2" and len(frame_list) >= 4:
            # Resize all to same size
            h, w = 540, 960
            resized = [cv2.resize(f, (w, h)) for f in frame_list[:4]]
            
            # Create 2x2 grid
            top = np.hstack([resized[0], resized[1]])
            bottom = np.hstack([resized[2], resized[3]])
            return np.vstack([top, bottom])
            
        elif layout == "side_by_side" and len(frame_list) >= 2:
            h, w = 1080, 960
            resized = [cv2.resize(f, (w, h)) for f in frame_list[:2]]
            return np.hstack(resized)
            
        elif layout == "picture_in_picture" and len(frame_list) >= 2:
            main = cv2.resize(frame_list[0], (1920, 1080))
            pip = cv2.resize(frame_list[1], (480, 270))
            
            # Place PiP in bottom right
            h, w = pip.shape[:2]
            y1, y2 = 1080 - h - 20, 1080 - 20
            x1, x2 = 1920 - w - 20, 1920 - 20
            
            # Blend with border
            main[y1:y2, x1:x2] = pip
            cv2.rectangle(main, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 2)
            
            return main
            
        # Default: return first frame resized
        return cv2.resize(frame_list[0], (1920, 1080))


class VideoExporter:
    """Main video export handler."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.tactical_drawer = TacticalDrawer()
        self.zoom_effect = ZoomEffect()
        self.watermark_renderer = WatermarkRenderer()
        self.multi_angle = MultiAngleSync()
        
        self.frame_processors: List[Callable[[np.ndarray, int], np.ndarray]] = []
        self.progress_callback: Optional[Callable[[float], None]] = None
        
    def add_frame_processor(self, processor: Callable[[np.ndarray, int], np.ndarray]):
        """Add a custom frame processor."""
        self.frame_processors.append(processor)
        
    def set_progress_callback(self, callback: Callable[[float], None]):
        """Set callback for export progress updates."""
        self.progress_callback = callback
        
    def export_video(self, input_path: str, 
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None) -> bool:
        """
        Export video with all overlays and effects.
        
        Args:
            input_path: Path to input video
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end)
            
        Returns:
            True if export successful
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open input video: {input_path}")
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set zoom effect FPS
        self.zoom_effect.set_fps(fps)
        
        # Calculate frame range
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.config.codec == "h264":
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        elif self.config.codec == "mpeg4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        writer = cv2.VideoWriter(
            self.config.output_path,
            fourcc,
            self.config.fps,
            self.config.resolution
        )
        
        if not writer.isOpened():
            logger.error(f"Could not create output video: {self.config.output_path}")
            cap.release()
            return False
            
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        processed_frames = 0
        
        logger.info(f"Starting export: {self.config.output_path}")
        
        try:
            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed = self._process_frame(frame, frame_count)
                
                # Resize to output resolution
                processed = cv2.resize(processed, self.config.resolution)
                
                # Write frame
                writer.write(processed)
                
                frame_count += 1
                processed_frames += 1
                
                # Update progress
                if self.progress_callback:
                    progress = (frame_count - start_frame) / (end_frame - start_frame) * 100
                    self.progress_callback(progress)
                    
                # Log progress every 100 frames
                if processed_frames % 100 == 0:
                    logger.info(f"Processed {processed_frames} frames...")
                    
        except Exception as e:
            logger.error(f"Error during export: {e}")
            return False
            
        finally:
            cap.release()
            writer.release()
            
        logger.info(f"Export complete: {self.config.output_path}")
        return True
        
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Apply all processing to a frame."""
        result = frame.copy()
        
        # Apply tactical drawings
        result = self.tactical_drawer.render(result, frame_number)
        
        # Apply zoom effects
        highlight = self.zoom_effect.is_in_highlight(frame_number)
        if highlight:
            result = self.zoom_effect.apply_zoom(result, highlight)
            
        # Apply watermarks
        result = self.watermark_renderer.render(result)
        
        # Apply custom processors
        for processor in self.frame_processors:
            result = processor(result, frame_number)
            
        return result
        
    def export_highlights(self, input_path: str, 
                         highlights: List[Tuple[float, float]],
                         transition_duration: float = 1.0) -> bool:
        """
        Export highlight reel with transitions.
        
        Args:
            input_path: Path to input video
            highlights: List of (start_time, end_time) tuples in seconds
            transition_duration: Transition duration in seconds
            
        Returns:
            True if export successful
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set up writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            self.config.output_path,
            fourcc,
            self.config.fps,
            self.config.resolution
        )
        
        for i, (start, end) in enumerate(highlights):
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Add transition effect at start (except for first clip)
            if i > 0:
                self._add_transition(writer, transition_duration, fps, 'fade_in')
                
            # Write highlight frames
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed = self._process_frame(frame, frame_num)
                processed = cv2.resize(processed, self.config.resolution)
                writer.write(processed)
                
            # Add transition at end (except for last clip)
            if i < len(highlights) - 1:
                self._add_transition(writer, transition_duration, fps, 'fade_out')
                
        cap.release()
        writer.release()
        return True
        
    def _add_transition(self, writer: cv2.VideoWriter, 
                       duration: float, fps: float, 
                       transition_type: str):
        """Add transition effect."""
        num_frames = int(duration * fps)
        
        if transition_type == 'fade_in':
            for i in range(num_frames):
                alpha = i / num_frames
                frame = np.zeros((self.config.resolution[1], self.config.resolution[0], 3), dtype=np.uint8)
                frame = cv2.addWeighted(frame, 1 - alpha, frame, alpha, 0)
                writer.write(frame)
                
        elif transition_type == 'fade_out':
            for i in range(num_frames):
                alpha = 1 - (i / num_frames)
                frame = np.zeros((self.config.resolution[1], self.config.resolution[0], 3), dtype=np.uint8)
                frame = cv2.addWeighted(frame, 1 - alpha, frame, alpha, 0)
                writer.write(frame)
                
    def export_multi_angle(self, output_layout: str = "2x2",
                          duration: Optional[float] = None) -> bool:
        """
        Export synchronized multi-angle video.
        
        Args:
            output_layout: Layout type ("2x2", "side_by_side", "picture_in_picture")
            duration: Export duration in seconds (None for full duration)
            
        Returns:
            True if export successful
        """
        if not self.multi_angle.video_sources:
            logger.error("No video sources added")
            return False
            
        # Get minimum duration
        min_duration = min(
            s['frames'] / s['fps'] 
            for s in self.multi_angle.video_sources.values()
        )
        
        if duration:
            min_duration = min(min_duration, duration)
            
        # Set up writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            self.config.output_path,
            fourcc,
            self.config.fps,
            self.config.resolution
        )
        
        fps = self.config.fps
        total_frames = int(min_duration * fps)
        
        for frame_num in range(total_frames):
            timestamp = frame_num / fps
            
            # Get synced frames
            frames = self.multi_angle.get_synced_frames(timestamp)
            
            # Create layout
            combined = self.multi_angle.create_split_screen(frames, output_layout)
            
            # Apply processing
            processed = self._process_frame(combined, frame_num)
            processed = cv2.resize(processed, self.config.resolution)
            
            writer.write(processed)
            
        writer.release()
        return True


# Convenience functions
def create_video_exporter(output_path: str, 
                         resolution: Tuple[int, int] = (1920, 1080),
                         fps: float = 30.0) -> VideoExporter:
    """Create and return a configured video exporter."""
    config = ExportConfig(
        output_path=output_path,
        resolution=resolution,
        fps=fps
    )
    return VideoExporter(config)
