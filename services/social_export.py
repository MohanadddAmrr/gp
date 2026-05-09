"""
Social Media Export Module

Auto-generates clips for social media platforms including Twitter/X,
Instagram Reels, TikTok with captions, hashtags, and transitions.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import json
import textwrap
import random
from enum import Enum

__all__ = ['SocialMediaExporter', 'SocialPlatform', 'SocialClip', 'CaptionStyle',
           'PlatformSpecs', 'CaptionGenerator', 'HashtagGenerator', 
           'TransitionEffects', 'SocialMediaScheduler', 'create_social_exporter',
           'quick_export_for_platform']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from enum import Enum

class SocialPlatform(Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    INSTAGRAM_REELS = "instagram_reels"
    TIKTOK = "tiktok"
    YOUTUBE_SHORTS = "youtube_shorts"
    FACEBOOK = "facebook"


@dataclass
class PlatformSpecs:
    """Platform-specific video specifications."""
    resolution: Tuple[int, int]
    aspect_ratio: str
    max_duration: float  # seconds
    max_file_size: int  # MB
    recommended_fps: int
    format: str


# Platform specifications
PLATFORM_SPECS = {
    SocialPlatform.TWITTER: PlatformSpecs(
        resolution=(1280, 720),
        aspect_ratio="16:9",
        max_duration=140.0,
        max_file_size=512,
        recommended_fps=30,
        format="mp4"
    ),
    SocialPlatform.INSTAGRAM_REELS: PlatformSpecs(
        resolution=(1080, 1920),
        aspect_ratio="9:16",
        max_duration=90.0,
        max_file_size=250,
        recommended_fps=30,
        format="mp4"
    ),
    SocialPlatform.TIKTOK: PlatformSpecs(
        resolution=(1080, 1920),
        aspect_ratio="9:16",
        max_duration=180.0,
        max_file_size=287,
        recommended_fps=30,
        format="mp4"
    ),
    SocialPlatform.YOUTUBE_SHORTS: PlatformSpecs(
        resolution=(1080, 1920),
        aspect_ratio="9:16",
        max_duration=60.0,
        max_file_size=100,
        recommended_fps=30,
        format="mp4"
    ),
    SocialPlatform.FACEBOOK: PlatformSpecs(
        resolution=(1280, 720),
        aspect_ratio="16:9",
        max_duration=240.0,
        max_file_size=1000,
        recommended_fps=30,
        format="mp4"
    )
}


@dataclass
class CaptionStyle:
    """Caption styling options."""
    font_name: str = "Arial"
    font_size: int = 48
    font_color: Tuple[int, int, int] = (255, 255, 255)
    stroke_color: Tuple[int, int, int] = (0, 0, 0)
    stroke_width: int = 3
    position: str = "bottom"  # top, bottom, center
    max_width: int = 80  # characters


@dataclass
class SocialClip:
    """Social media clip configuration."""
    video_path: str
    start_time: float
    end_time: float
    platform: SocialPlatform
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    music_track: Optional[str] = None
    transition_type: str = "fade"  # fade, slide, zoom, none
    thumbnail_time: Optional[float] = None
    output_path: Optional[str] = None


class CaptionGenerator:
    """Generate captions for social media clips."""
    
    def __init__(self):
        self.style = CaptionStyle()
        
    def set_style(self, style: CaptionStyle):
        """Set caption style."""
        self.style = style
        
    def generate_caption_frame(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Render caption on frame."""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Load font
        try:
            font = ImageFont.truetype(self.style.font_name, self.style.font_size)
        except:
            font = ImageFont.load_default()
            
        # Wrap text
        wrapped_text = textwrap.fill(text, width=self.style.max_width)
        lines = wrapped_text.split('\n')
        
        # Calculate position
        img_w, img_h = pil_image.size
        line_height = self.style.font_size + 10
        
        if self.style.position == "bottom":
            start_y = img_h - (len(lines) * line_height) - 50
        elif self.style.position == "top":
            start_y = 50
        else:  # center
            start_y = (img_h - (len(lines) * line_height)) // 2
            
        # Draw each line
        for i, line in enumerate(lines):
            y = start_y + (i * line_height)
            
            # Calculate x position (centered)
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            x = (img_w - text_w) // 2
            
            # Draw stroke
            for dx in range(-self.style.stroke_width, self.style.stroke_width + 1):
                for dy in range(-self.style.stroke_width, self.style.stroke_width + 1):
                    draw.text((x + dx, y + dy), line, font=font, 
                             fill=self.style.stroke_color)
                    
            # Draw text
            draw.text((x, y), line, font=font, fill=self.style.font_color)
            
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    def auto_generate_caption(self, event_type: str, player_name: str,
                             team_name: str, minute: int) -> str:
        """Auto-generate caption based on event."""
        templates = {
            'goal': [
                f"⚽ GOAL! {player_name} scores for {team_name} in the {minute}th minute!",
                f"🎯 {player_name} finds the back of the net! {team_name} take the lead!",
                f"⚡ What a strike from {player_name}! {team_name} are on fire!"
            ],
            'assist': [
                f"🅰️ Brilliant assist from {player_name}!",
                f"👏 {player_name} with the perfect pass!",
                f"🎯 Vision from {player_name} to set up the goal!"
            ],
            'save': [
                f"🧤 Incredible save! {player_name} denies the goal!",
                f"🛑 {player_name} with a world-class stop!",
                f"💪 What a reaction from {player_name}!"
            ],
            'tackle': [
                f"💥 Crucial tackle from {player_name}!",
                f"🛡️ {player_name} with the defensive masterclass!",
                f"⚔️ {player_name} wins the ball back!"
            ],
            'skill': [
                f"✨ Magic from {player_name}!",
                f"🎩 {player_name} showing off the skills!",
                f"🔥 {player_name} with the silky dribble!"
            ]
        }
        
        import random
        options = templates.get(event_type, [f"Great moment from {player_name}!"])
        return random.choice(options)


class HashtagGenerator:
    """Generate relevant hashtags for clips."""
    
    def __init__(self):
        self.common_hashtags = [
            "#football", "#soccer", "#futbol", "#footballhighlights",
            "#goals", "#premierleague", "#championsleague", "#fifa",
            "#footballskills", "#soccerlife", "#footballgame", "#matchday"
        ]
        
        self.team_hashtags: Dict[str, List[str]] = {}
        self.player_hashtags: Dict[str, List[str]] = {}
        
    def add_team_hashtags(self, team_name: str, hashtags: List[str]):
        """Add team-specific hashtags."""
        self.team_hashtags[team_name.lower()] = hashtags
        
    def add_player_hashtags(self, player_name: str, hashtags: List[str]):
        """Add player-specific hashtags."""
        self.player_hashtags[player_name.lower()] = hashtags
        
    def generate_hashtags(self, team_name: Optional[str] = None,
                         player_name: Optional[str] = None,
                         event_type: Optional[str] = None,
                         count: int = 10) -> List[str]:
        """Generate relevant hashtags."""
        hashtags = []
        
        # Add common hashtags
        hashtags.extend(self.common_hashtags[:3])
        
        # Add team hashtags
        if team_name and team_name.lower() in self.team_hashtags:
            hashtags.extend(self.team_hashtags[team_name.lower()][:2])
            
        # Add player hashtags
        if player_name and player_name.lower() in self.player_hashtags:
            hashtags.extend(self.player_hashtags[player_name.lower()][:2])
            
        # Add event-specific hashtags
        event_tags = {
            'goal': ["#goal", "#golazo", "#scorer"],
            'assist': ["#assist", "#playmaker"],
            'save': ["#save", "#goalkeeper", "#cleansheet"],
            'tackle': ["#tackle", "#defense"],
            'skill': ["#skills", "#dribble", "#nutmeg"]
        }
        
        if event_type and event_type in event_tags:
            hashtags.extend(event_tags[event_type])
            
        # Remove duplicates and limit
        hashtags = list(dict.fromkeys(hashtags))
        return hashtags[:count]


class TransitionEffects:
    """Video transition effects."""
    
    @staticmethod
    def fade_transition(frame1: np.ndarray, frame2: np.ndarray, 
                       progress: float) -> np.ndarray:
        """Fade between two frames."""
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        
    @staticmethod
    def slide_transition(frame1: np.ndarray, frame2: np.ndarray,
                        progress: float, direction: str = "left") -> np.ndarray:
        """Slide transition."""
        h, w = frame1.shape[:2]
        offset = int(w * progress)
        
        result = np.zeros_like(frame1)
        
        if direction == "left":
            result[:, :w-offset] = frame1[:, offset:]
            result[:, w-offset:] = frame2[:, :offset]
        elif direction == "right":
            result[:, offset:] = frame1[:, :w-offset]
            result[:, :offset] = frame2[:, w-offset:]
        elif direction == "up":
            result[:h-offset, :] = frame1[offset:, :]
            result[h-offset:, :] = frame2[:offset, :]
        elif direction == "down":
            result[offset:, :] = frame1[:h-offset, :]
            result[:offset, :] = frame2[h-offset:, :]
            
        return result
        
    @staticmethod
    def zoom_transition(frame1: np.ndarray, frame2: np.ndarray,
                       progress: float) -> np.ndarray:
        """Zoom transition."""
        h, w = frame1.shape[:2]
        
        if progress < 0.5:
            # Zoom out frame1
            scale = 1 + (progress * 2)
            scaled = cv2.resize(frame1, None, fx=scale, fy=scale)
            sh, sw = scaled.shape[:2]
            y1, x1 = (sh - h) // 2, (sw - w) // 2
            return scaled[y1:y1+h, x1:x1+w]
        else:
            # Zoom in frame2
            scale = 2 - ((progress - 0.5) * 2)
            scaled = cv2.resize(frame2, None, fx=scale, fy=scale)
            sh, sw = scaled.shape[:2]
            y1, x1 = (sh - h) // 2, (sw - w) // 2
            return scaled[y1:y1+h, x1:x1+w]
            
    @staticmethod
    def apply_transition(frames: List[np.ndarray], transition_type: str,
                        duration_frames: int = 30) -> List[np.ndarray]:
        """Apply transition between clips."""
        if len(frames) < 2 or transition_type == "none":
            return frames
            
        result = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Add frame1
            result.append(frame1)
            
            # Add transition frames
            for j in range(duration_frames):
                progress = j / duration_frames
                
                if transition_type == "fade":
                    transition = TransitionEffects.fade_transition(
                        frame1, frame2, progress
                    )
                elif transition_type == "slide":
                    transition = TransitionEffects.slide_transition(
                        frame1, frame2, progress
                    )
                elif transition_type == "zoom":
                    transition = TransitionEffects.zoom_transition(
                        frame1, frame2, progress
                    )
                else:
                    continue
                    
                result.append(transition)
                
        # Add last frame
        result.append(frames[-1])
        
        return result


class SocialMediaExporter:
    """Export clips optimized for social media platforms."""
    
    def __init__(self):
        self.caption_generator = CaptionGenerator()
        self.hashtag_generator = HashtagGenerator()
        self.transition_effects = TransitionEffects()
        
    def export_clip(self, clip: SocialClip, 
                   progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Export a clip for social media.
        
        Args:
            clip: SocialClip configuration
            progress_callback: Optional progress callback
            
        Returns:
            True if export successful
        """
        specs = PLATFORM_SPECS[clip.platform]
        
        # Validate duration
        duration = clip.end_time - clip.start_time
        if duration > specs.max_duration:
            logger.warning(f"Clip duration {duration}s exceeds platform maximum {specs.max_duration}s")
            clip.end_time = clip.start_time + specs.max_duration
            
        # Open input video
        cap = cv2.VideoCapture(clip.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {clip.video_path}")
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(clip.start_time * fps)
        end_frame = int(clip.end_time * fps)
        
        # Set output path
        if not clip.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clip.output_path = f"social_export_{clip.platform.value}_{timestamp}.mp4"
            
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            clip.output_path,
            fourcc,
            specs.recommended_fps,
            specs.resolution
        )
        
        if not writer.isOpened():
            logger.error(f"Could not create output video: {clip.output_path}")
            cap.release()
            return False
            
        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_count = start_frame
        total_frames = end_frame - start_frame
        
        logger.info(f"Exporting {clip.platform.value} clip: {clip.output_path}")
        
        try:
            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize to platform resolution
                frame = self._resize_for_platform(frame, specs)
                
                # Add caption if provided
                if clip.caption:
                    frame = self.caption_generator.generate_caption_frame(
                        frame, clip.caption
                    )
                    
                writer.write(frame)
                
                frame_count += 1
                
                if progress_callback:
                    progress = (frame_count - start_frame) / total_frames * 100
                    progress_callback(progress)
                    
        except Exception as e:
            logger.error(f"Error exporting clip: {e}")
            return False
            
        finally:
            cap.release()
            writer.release()
            
        logger.info(f"Export complete: {clip.output_path}")
        return True
        
    def _resize_for_platform(self, frame: np.ndarray, 
                            specs: PlatformSpecs) -> np.ndarray:
        """Resize frame for platform specifications."""
        target_w, target_h = specs.resolution
        h, w = frame.shape[:2]
        
        # Calculate scaling
        scale_w = target_w / w
        scale_h = target_h / h
        
        if specs.aspect_ratio == "9:16":
            # Vertical video - fit height, crop width
            scale = scale_h
            new_w = int(w * scale)
            resized = cv2.resize(frame, (new_w, target_h))
            
            # Center crop
            if new_w > target_w:
                x_start = (new_w - target_w) // 2
                return resized[:, x_start:x_start+target_w]
            else:
                # Pad sides
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                return cv2.copyMakeBorder(
                    resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT
                )
        else:
            # Horizontal video - standard resize
            return cv2.resize(frame, specs.resolution)
            
    def create_compilation(self, clips: List[SocialClip],
                          output_path: str,
                          transition_type: str = "fade",
                          add_music: bool = False,
                          music_path: Optional[str] = None) -> bool:
        """
        Create a compilation of multiple clips.
        
        Args:
            clips: List of clips to compile
            output_path: Output file path
            transition_type: Type of transition between clips
            add_music: Whether to add background music
            music_path: Path to music file
            
        Returns:
            True if successful
        """
        if not clips:
            logger.error("No clips provided for compilation")
            return False
            
        # Use specs from first clip
        specs = PLATFORM_SPECS[clips[0].platform]
        
        # Collect all frames
        all_frames = []
        
        for clip in clips:
            cap = cv2.VideoCapture(clip.video_path)
            if not cap.isOpened():
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(clip.start_time * fps)
            end_frame = int(clip.end_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            clip_frames = []
            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = self._resize_for_platform(frame, specs)
                
                if clip.caption:
                    frame = self.caption_generator.generate_caption_frame(
                        frame, clip.caption
                    )
                    
                clip_frames.append(frame)
                
            all_frames.extend(clip_frames)
            cap.release()
            
        # Apply transitions
        if transition_type != "none":
            all_frames = self.transition_effects.apply_transition(
                all_frames, transition_type
            )
            
        # Write compilation
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, specs.recommended_fps, specs.resolution
        )
        
        for frame in all_frames:
            writer.write(frame)
            
        writer.release()
        
        logger.info(f"Compilation exported: {output_path}")
        return True
        
    def auto_clip_from_events(self, video_path: str,
                             events: List[Dict],
                             platform: SocialPlatform,
                             output_dir: str = "social_clips") -> List[str]:
        """
        Automatically generate clips from match events.
        
        Args:
            video_path: Source video path
            events: List of event dictionaries with 'time', 'type', 'player', etc.
            platform: Target social media platform
            output_dir: Output directory
            
        Returns:
            List of exported clip paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        exported = []
        
        for i, event in enumerate(events):
            event_time = event.get('time', 0)
            event_type = event.get('type', 'highlight')
            player_name = event.get('player', 'Player')
            team_name = event.get('team', 'Team')
            
            # Generate caption
            caption = self.caption_generator.auto_generate_caption(
                event_type, player_name, team_name, int(event_time / 60)
            )
            
            # Generate hashtags
            hashtags = self.hashtag_generator.generate_hashtags(
                team_name, player_name, event_type
            )
            
            # Create clip config
            clip = SocialClip(
                video_path=video_path,
                start_time=max(0, event_time - 5),  # 5 seconds before
                end_time=event_time + 10,  # 10 seconds after
                platform=platform,
                caption=caption,
                hashtags=hashtags,
                output_path=f"{output_dir}/clip_{i+1}_{event_type}.mp4"
            )
            
            if self.export_clip(clip):
                exported.append(clip.output_path)
                
        return exported
        
    def generate_thumbnail(self, video_path: str, 
                          output_path: str,
                          time_position: float = 0.0,
                          add_text: Optional[str] = None) -> bool:
        """
        Generate thumbnail image from video.
        
        Args:
            video_path: Source video
            output_path: Thumbnail output path
            time_position: Time to capture (seconds)
            add_text: Optional text overlay
            
        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(time_position * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        cap.release()
        
        if not ret:
            return False
            
        # Add text if provided
        if add_text:
            frame = self.caption_generator.generate_caption_frame(frame, add_text)
            
        # Save thumbnail
        cv2.imwrite(output_path, frame)
        logger.info(f"Thumbnail saved: {output_path}")
        return True


class SocialMediaScheduler:
    """Schedule and manage social media posts."""
    
    def __init__(self):
        self.scheduled_posts: List[Dict] = []
        
    def schedule_post(self, clip_path: str, platform: SocialPlatform,
                     caption: str, hashtags: List[str],
                     scheduled_time: datetime,
                     auto_post: bool = False):
        """Schedule a social media post."""
        post = {
            'clip_path': clip_path,
            'platform': platform,
            'caption': caption,
            'hashtags': hashtags,
            'scheduled_time': scheduled_time,
            'auto_post': auto_post,
            'posted': False
        }
        self.scheduled_posts.append(post)
        
    def get_pending_posts(self) -> List[Dict]:
        """Get posts that haven't been posted yet."""
        now = datetime.now()
        return [
            p for p in self.scheduled_posts 
            if not p['posted'] and p['scheduled_time'] <= now
        ]
        
    def export_schedule(self, output_path: str):
        """Export posting schedule to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.scheduled_posts, f, indent=2, default=str)


# Convenience functions
def create_social_exporter() -> SocialMediaExporter:
    """Create and return a configured social media exporter."""
    return SocialMediaExporter()


def quick_export_for_platform(video_path: str, platform: SocialPlatform,
                              start_time: float, end_time: float,
                              caption: str = "") -> str:
    """Quick export a clip for a specific platform."""
    exporter = SocialMediaExporter()
    
    clip = SocialClip(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        platform=platform,
        caption=caption
    )
    
    if exporter.export_clip(clip):
        return clip.output_path
    return ""
