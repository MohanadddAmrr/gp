"""
Real-time Streaming Support Module

Handles RTMP stream input, real-time processing pipeline, low-latency
overlay injection, and WebRTC for browser streaming.
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import json
import asyncio
import time
from enum import Enum

# Optional import for websockets
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    
__all__ = ['StreamingManager', 'RTMPStreamHandler', 'RealTimeProcessor', 
           'LowLatencyOverlay', 'WebRTCServer', 'StreamOutput', 'StreamConfig',
           'StreamStats', 'StreamProtocol', 'create_streaming_manager',
           'start_rtmp_stream']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamProtocol(Enum):
    """Supported streaming protocols."""
    RTMP = "rtmp"
    RTSP = "rtsp"
    HLS = "hls"
    WEBRTC = "webrtc"
    SRT = "srt"


@dataclass
class StreamConfig:
    """Configuration for stream handling."""
    input_url: str
    protocol: StreamProtocol = StreamProtocol.RTMP
    buffer_size: int = 1024 * 1024  # 1MB buffer
    latency_target: int = 500  # Target latency in ms
    output_resolution: Tuple[int, int] = (1920, 1080)
    output_fps: int = 30
    enable_overlays: bool = True


@dataclass
class StreamStats:
    """Stream statistics."""
    fps: float = 0.0
    bitrate: float = 0.0
    latency: float = 0.0
    dropped_frames: int = 0
    total_frames: int = 0
    start_time: Optional[datetime] = None


class RTMPStreamHandler:
    """Handle RTMP stream input and processing."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self.stats = StreamStats()
        self.frame_callbacks: List[Callable[[np.ndarray], np.ndarray]] = []
        
    def add_frame_callback(self, callback: Callable[[np.ndarray], np.ndarray]):
        """Add a frame processing callback."""
        self.frame_callbacks.append(callback)
        
    def start(self) -> bool:
        """Start RTMP stream capture."""
        try:
            self.capture = cv2.VideoCapture(self.config.input_url)
            
            if not self.capture.isOpened():
                logger.error(f"Failed to open RTMP stream: {self.config.input_url}")
                return False
                
            # Set buffer size
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            self.is_running = True
            self.stats.start_time = datetime.now()
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info(f"Started RTMP stream: {self.config.input_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting RTMP stream: {e}")
            return False
            
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        frame_time = 1.0 / self.config.output_fps
        last_frame_time = datetime.now()
        
        while self.is_running:
            ret, frame = self.capture.read()
            
            if not ret:
                self.stats.dropped_frames += 1
                continue
                
            # Update stats
            self.stats.total_frames += 1
            current_time = datetime.now()
            elapsed = (current_time - last_frame_time).total_seconds()
            
            if elapsed > 0:
                self.stats.fps = 1.0 / elapsed
                
            last_frame_time = current_time
            
            # Process frame through callbacks
            processed = frame
            for callback in self.frame_callbacks:
                try:
                    processed = callback(processed)
                except Exception as e:
                    logger.warning(f"Frame callback error: {e}")
                    
            # Resize to output resolution
            processed = cv2.resize(processed, self.config.output_resolution)
            
            # Add to queue (drop old frames if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put(processed)
            
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get latest processed frame."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop stream capture."""
        self.is_running = False
        
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        if self.capture:
            self.capture.release()
            
        logger.info("Stopped RTMP stream")
        
    def get_stats(self) -> StreamStats:
        """Get current stream statistics."""
        return self.stats


class RealTimeProcessor:
    """Real-time video processing pipeline."""
    
    def __init__(self, target_latency: float = 0.5):
        self.target_latency = target_latency
        self.processors: List[Dict] = []
        self.processing_times: List[float] = []
        
    def add_processor(self, name: str, 
                     processor: Callable[[np.ndarray], np.ndarray],
                     priority: int = 0):
        """
        Add a processing step.
        
        Args:
            name: Processor identifier
            processor: Processing function
            priority: Higher priority = processed first
        """
        self.processors.append({
            'name': name,
            'func': processor,
            'priority': priority,
            'enabled': True
        })
        
        # Sort by priority
        self.processors.sort(key=lambda x: x['priority'], reverse=True)
        
    def remove_processor(self, name: str):
        """Remove a processor by name."""
        self.processors = [p for p in self.processors if p['name'] != name]
        
    def enable_processor(self, name: str, enabled: bool = True):
        """Enable/disable a processor."""
        for p in self.processors:
            if p['name'] == name:
                p['enabled'] = enabled
                break
                
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process frame through pipeline."""
        import time
        start_time = time.time()
        
        result = frame
        
        for processor in self.processors:
            if processor['enabled']:
                try:
                    result = processor['func'](result)
                except Exception as e:
                    logger.warning(f"Processor {processor['name']} error: {e}")
                    
        # Track processing time
        elapsed = time.time() - start_time
        self.processing_times.append(elapsed)
        
        # Keep only last 100 measurements
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            
        return result
        
    def get_average_latency(self) -> float:
        """Get average processing latency."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
        
    def adjust_quality(self):
        """Dynamically adjust processing quality based on latency."""
        avg_latency = self.get_average_latency()
        
        if avg_latency > self.target_latency:
            # Disable lower priority processors
            for p in sorted(self.processors, key=lambda x: x['priority']):
                if p['enabled'] and p['priority'] < 5:
                    p['enabled'] = False
                    logger.info(f"Disabled {p['name']} to reduce latency")
                    break


class LowLatencyOverlay:
    """Low-latency overlay injection system."""
    
    def __init__(self):
        self.overlays: Dict[str, Dict] = {}
        self.overlay_lock = threading.Lock()
        
    def add_text_overlay(self, overlay_id: str, text: str,
                        position: Tuple[int, int] = (10, 10),
                        font_scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        thickness: int = 2,
                        background: bool = True):
        """Add a text overlay."""
        with self.overlay_lock:
            self.overlays[overlay_id] = {
                'type': 'text',
                'text': text,
                'position': position,
                'font_scale': font_scale,
                'color': color,
                'thickness': thickness,
                'background': background,
                'enabled': True
            }
            
    def add_image_overlay(self, overlay_id: str, image_path: str,
                         position: Tuple[int, int] = (0, 0),
                         scale: float = 1.0,
                         alpha: float = 1.0):
        """Add an image overlay."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                with self.overlay_lock:
                    self.overlays[overlay_id] = {
                        'type': 'image',
                        'image': image,
                        'position': position,
                        'scale': scale,
                        'alpha': alpha,
                        'enabled': True
                    }
        except Exception as e:
            logger.error(f"Error loading overlay image: {e}")
            
    def update_text(self, overlay_id: str, text: str):
        """Update text overlay content."""
        with self.overlay_lock:
            if overlay_id in self.overlays and self.overlays[overlay_id]['type'] == 'text':
                self.overlays[overlay_id]['text'] = text
                
    def remove_overlay(self, overlay_id: str):
        """Remove an overlay."""
        with self.overlay_lock:
            if overlay_id in self.overlays:
                del self.overlays[overlay_id]
                
    def enable_overlay(self, overlay_id: str, enabled: bool = True):
        """Enable/disable an overlay."""
        with self.overlay_lock:
            if overlay_id in self.overlays:
                self.overlays[overlay_id]['enabled'] = enabled
                
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render all overlays on frame."""
        result = frame.copy()
        
        with self.overlay_lock:
            for overlay in self.overlays.values():
                if not overlay.get('enabled', True):
                    continue
                    
                if overlay['type'] == 'text':
                    result = self._render_text_overlay(result, overlay)
                elif overlay['type'] == 'image':
                    result = self._render_image_overlay(result, overlay)
                    
        return result
        
    def _render_text_overlay(self, frame: np.ndarray, overlay: Dict) -> np.ndarray:
        """Render text overlay."""
        x, y = overlay['position']
        text = overlay['text']
        
        # Draw background if enabled
        if overlay.get('background', True):
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 
                overlay['font_scale'], overlay['thickness']
            )
            cv2.rectangle(frame, (x - 5, y - text_h - 5), 
                         (x + text_w + 5, y + 5), (0, 0, 0), -1)
            
        # Draw text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   overlay['font_scale'], overlay['color'], overlay['thickness'])
                   
        return frame
        
    def _render_image_overlay(self, frame: np.ndarray, overlay: Dict) -> np.ndarray:
        """Render image overlay."""
        image = overlay['image']
        x, y = overlay['position']
        scale = overlay['scale']
        alpha = overlay['alpha']
        
        # Resize image
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Handle alpha channel
        if resized.shape[2] == 4:
            # Extract alpha channel
            alpha_channel = resized[:, :, 3] / 255.0 * alpha
            
            # Blend
            for c in range(3):
                frame[y:y+new_h, x:x+new_w, c] = (
                    alpha_channel * resized[:, :, c] +
                    (1 - alpha_channel) * frame[y:y+new_h, x:x+new_w, c]
                )
        else:
            # Simple overlay
            overlay_region = frame[y:y+new_h, x:x+new_w]
            blended = cv2.addWeighted(resized, alpha, overlay_region, 1 - alpha, 0)
            frame[y:y+new_h, x:x+new_w] = blended
            
        return frame


class WebRTCServer:
    """WebRTC server for browser-based streaming."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.clients: set = set()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.server = None
        
    async def handle_client(self, websocket, path):
        """Handle WebRTC client connection."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets module not available")
            return
            
        self.clients.add(websocket)
        logger.info(f"WebRTC client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                # Handle signaling messages
                data = json.loads(message)
                
                if data.get('type') == 'offer':
                    # Handle WebRTC offer
                    await self._handle_offer(websocket, data)
                elif data.get('type') == 'ice_candidate':
                    # Handle ICE candidate
                    pass
                    
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"WebRTC client disconnected")
            
    async def _handle_offer(self, websocket, data: Dict):
        """Handle WebRTC offer."""
        # Simplified - in production, use a proper WebRTC library
        response = {
            'type': 'answer',
            'sdp': data.get('sdp', '')
        }
        await websocket.send(json.dumps(response))
        
    async def broadcast_frame(self, frame: np.ndarray):
        """Broadcast frame to all connected clients."""
        if not self.clients:
            return
            
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_data = buffer.tobytes()
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(frame_data)
            except:
                disconnected.add(client)
                
        # Remove disconnected clients
        self.clients -= disconnected
        
    def start(self):
        """Start WebRTC server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets module not available, WebRTC server not started")
            return
            
        self.is_running = True
        
        # Start WebSocket server
        self.server = websockets.serve(
            self.handle_client, self.host, self.port
        )
        
        logger.info(f"WebRTC server started on {self.host}:{self.port}")
        
        # Run in separate thread
        def run_server():
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.get_event_loop().run_until_complete(self.server)
            asyncio.get_event_loop().run_forever()
            
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
    def stop(self):
        """Stop WebRTC server."""
        self.is_running = False
        
        if self.server:
            self.server.close()
            
        logger.info("WebRTC server stopped")
        
    async def send_frame(self, frame: np.ndarray):
        """Send frame to clients (to be called from main loop)."""
        await self.broadcast_frame(frame)


class StreamOutput:
    """Handle stream output to various destinations."""
    
    def __init__(self, output_url: str, protocol: StreamProtocol = StreamProtocol.RTMP):
        self.output_url = output_url
        self.protocol = protocol
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.is_streaming = False
        
    def start(self, width: int = 1920, height: int = 1080, fps: int = 30, 
             bitrate: str = "4M") -> bool:
        """Start streaming output using FFmpeg."""
        try:
            # FFmpeg command for streaming
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',  # Input from pipe
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'veryfast',
                '-b:v', bitrate,
                '-g', str(fps * 2),
                '-f', 'flv' if self.protocol == StreamProtocol.RTMP else 'mpegts',
                self.output_url
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.is_streaming = True
            logger.info(f"Started stream output to {self.output_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting stream output: {e}")
            return False
            
    def write_frame(self, frame: np.ndarray):
        """Write frame to stream."""
        if self.ffmpeg_process and self.is_streaming:
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
            except Exception as e:
                logger.error(f"Error writing frame: {e}")
                self.is_streaming = False
                
    def stop(self):
        """Stop streaming output."""
        self.is_streaming = False
        
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
            
        logger.info("Stopped stream output")


class StreamingManager:
    """Main manager for all streaming operations."""
    
    def __init__(self):
        self.input_handler: Optional[RTMPStreamHandler] = None
        self.processor = RealTimeProcessor()
        self.overlay = LowLatencyOverlay()
        self.webrtc = WebRTCServer()
        self.output: Optional[StreamOutput] = None
        self.is_running = False
        
    def setup_input(self, input_url: str, protocol: StreamProtocol = StreamProtocol.RTMP):
        """Setup input stream."""
        config = StreamConfig(
            input_url=input_url,
            protocol=protocol
        )
        self.input_handler = RTMPStreamHandler(config)
        
        # Add processing pipeline
        self.input_handler.add_frame_callback(self._process_frame)
        
    def setup_output(self, output_url: str, protocol: StreamProtocol = StreamProtocol.RTMP):
        """Setup output stream."""
        self.output = StreamOutput(output_url, protocol)
        
    def add_processor(self, name: str, processor: Callable[[np.ndarray], np.ndarray],
                     priority: int = 0):
        """Add a frame processor."""
        self.processor.add_processor(name, processor, priority)
        
    def add_overlay(self, overlay_id: str, text: str, **kwargs):
        """Add a text overlay."""
        self.overlay.add_text_overlay(overlay_id, text, **kwargs)
        
    def start(self) -> bool:
        """Start all streaming components."""
        if not self.input_handler:
            logger.error("No input configured")
            return False
            
        # Start input
        if not self.input_handler.start():
            return False
            
        # Start WebRTC
        self.webrtc.start()
        
        # Start output if configured
        if self.output:
            self.output.start()
            
        self.is_running = True
        
        # Start main processing loop
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
        
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame through pipeline."""
        # Apply processors
        result = self.processor.process(frame)
        
        # Apply overlays
        result = self.overlay.render(result)
        
        return result
        
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            frame = self.input_handler.get_frame(timeout=0.1)
            
            if frame is not None:
                # Send to output
                if self.output and self.output.is_streaming:
                    self.output.write_frame(frame)
                    
                # Send to WebRTC clients
                if self.webrtc.clients:
                    asyncio.run(self.webrtc.send_frame(frame))
                    
    def stop(self):
        """Stop all streaming."""
        self.is_running = False
        
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        if self.input_handler:
            self.input_handler.stop()
            
        if self.output:
            self.output.stop()
            
        self.webrtc.stop()
        
    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        stats = {
            'processing_latency': self.processor.get_average_latency(),
            'webrtc_clients': len(self.webrtc.clients),
            'is_streaming': self.is_running
        }
        
        if self.input_handler:
            input_stats = self.input_handler.get_stats()
            stats['input_fps'] = input_stats.fps
            stats['dropped_frames'] = input_stats.dropped_frames
            stats['total_frames'] = input_stats.total_frames
            
        return stats


# Convenience functions
def create_streaming_manager() -> StreamingManager:
    """Create and return a configured streaming manager."""
    return StreamingManager()


def start_rtmp_stream(input_url: str, output_url: str) -> StreamingManager:
    """Quick start RTMP stream with processing."""
    manager = StreamingManager()
    manager.setup_input(input_url, StreamProtocol.RTMP)
    manager.setup_output(output_url, StreamProtocol.RTMP)
    manager.start()
    return manager
