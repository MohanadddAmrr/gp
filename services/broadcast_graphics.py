"""
Broadcast Graphics Generator Module

Generates professional overlay graphics for football match broadcasts including
scoreboards, player stats, event popups, and team statistics.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import csv

# Re-export EventType for backward compatibility
__all__ = ['EventType', 'TeamInfo', 'ScoreboardState', 'PlayerStatsOverlay', 
           'EventPopup', 'BroadcastGraphicsManager', 'create_team_info', 
           'create_broadcast_manager']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of match events for graphics."""
    GOAL = "goal"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    PENALTY = "penalty"
    VAR = "var"


@dataclass
class TeamInfo:
    """Team information for graphics."""
    name: str
    short_name: str
    color_primary: Tuple[int, int, int] = (255, 0, 0)
    color_secondary: Tuple[int, int, int] = (255, 255, 255)
    logo: Optional[np.ndarray] = None


@dataclass
class ScoreboardState:
    """Current scoreboard state."""
    home_team: TeamInfo
    away_team: TeamInfo
    home_score: int = 0
    away_score: int = 0
    match_time: int = 0  # Minutes
    match_period: str = "1H"  # 1H, 2H, ET, PEN
    added_time: int = 0
    show: bool = True


@dataclass
class PlayerStatsOverlay:
    """Player statistics overlay data."""
    player_name: str
    player_number: int
    team: str
    speed: float = 0.0
    distance: float = 0.0
    passes: int = 0
    shots: int = 0
    position: Tuple[int, int] = (50, 50)  # Screen position percentage
    show: bool = False


@dataclass
class EventPopup:
    """Event popup notification."""
    event_type: EventType
    player_name: str
    team: str
    minute: int
    additional_info: str = ""
    show_duration: float = 5.0  # Seconds
    start_time: Optional[datetime] = None


class GraphicsRenderer:
    """Base class for rendering graphics elements."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.font_large = self._load_font(48)
        self.font_medium = self._load_font(32)
        self.font_small = self._load_font(24)
        
    def _load_font(self, size: int):
        """Load font, fallback to default if not available."""
        try:
            # Try to load a common font
            return ImageFont.truetype("arial.ttf", size)
        except:
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except:
                return ImageFont.load_default()
                
    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class ScoreboardRenderer(GraphicsRenderer):
    """Render professional scoreboard overlay."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__(width, height)
        self.scoreboard_height = 80
        self.team_width = 300
        self.score_width = 120
        self.time_width = 100
        
    def render(self, frame: np.ndarray, state: ScoreboardState) -> np.ndarray:
        """Render scoreboard on frame."""
        if not state.show:
            return frame
            
        # Convert to PIL for easier text rendering
        pil_image = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate positions
        center_x = self.width // 2
        top_y = 20
        
        # Draw background bar
        bar_width = self.team_width * 2 + self.score_width + self.time_width + 40
        bar_left = center_x - bar_width // 2
        
        # Semi-transparent background
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [bar_left, top_y, bar_left + bar_width, top_y + self.scoreboard_height],
            fill=(0, 0, 0, 180)
        )
        
        # Blend overlay
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # Draw home team
        home_x = bar_left + 20
        self._draw_team_section(draw, state.home_team, home_x, top_y, True)
        
        # Draw score
        score_x = bar_left + self.team_width + 20
        self._draw_score(draw, state, score_x, top_y)
        
        # Draw away team
        away_x = bar_left + self.team_width + self.score_width + 40
        self._draw_team_section(draw, state.away_team, away_x, top_y, False)
        
        # Draw match time
        time_x = bar_left + bar_width - self.time_width - 10
        self._draw_time(draw, state, time_x, top_y)
        
        return self._pil_to_cv2(pil_image)
        
    def _draw_team_section(self, draw: ImageDraw.Draw, team: TeamInfo, 
                          x: int, y: int, is_home: bool):
        """Draw team name and color indicator."""
        # Draw color indicator
        indicator_width = 8
        indicator_x = x if is_home else x + self.team_width - indicator_width - 10
        draw.rectangle(
            [indicator_x, y + 10, indicator_x + indicator_width, y + self.scoreboard_height - 10],
            fill=team.color_primary
        )
        
        # Draw team name
        text_x = x + 20 if is_home else x + 10
        text_y = y + self.scoreboard_height // 2 - 16
        
        # Truncate if too long
        name = team.short_name if len(team.name) > 12 else team.name
        draw.text((text_x, text_y), name, font=self.font_medium, fill=(255, 255, 255))
        
    def _draw_score(self, draw: ImageDraw.Draw, state: ScoreboardState, x: int, y: int):
        """Draw score display."""
        score_text = f"{state.home_score} - {state.away_score}"
        text_y = y + self.scoreboard_height // 2 - 24
        
        # Draw score background
        draw.rectangle(
            [x, y + 5, x + self.score_width, y + self.scoreboard_height - 5],
            fill=(40, 40, 40)
        )
        
        draw.text((x + 20, text_y), score_text, font=self.font_large, fill=(255, 255, 255))
        
    def _draw_time(self, draw: ImageDraw.Draw, state: ScoreboardState, x: int, y: int):
        """Draw match time."""
        time_text = f"{state.match_time}'"
        if state.added_time > 0:
            time_text += f"+{state.added_time}"
            
        text_y = y + self.scoreboard_height // 2 - 16
        
        # Draw period indicator
        period_y = y + 10
        draw.text((x, period_y), state.match_period, font=self.font_small, fill=(200, 200, 200))
        
        # Draw time
        draw.text((x, text_y + 10), time_text, font=self.font_medium, fill=(255, 255, 255))


class PlayerStatsRenderer(GraphicsRenderer):
    """Render player statistics overlays."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__(width, height)
        self.card_width = 280
        self.card_height = 150
        
    def render(self, frame: np.ndarray, stats: PlayerStatsOverlay) -> np.ndarray:
        """Render player stats card on frame."""
        if not stats.show:
            return frame
            
        pil_image = self._cv2_to_pil(frame)
        
        # Calculate position
        x = int(self.width * stats.position[0] / 100) - self.card_width // 2
        y = int(self.height * stats.position[1] / 100)
        
        # Ensure within bounds
        x = max(10, min(x, self.width - self.card_width - 10))
        y = max(10, min(y, self.height - self.card_height - 10))
        
        # Create overlay
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Draw card background
        overlay_draw.rectangle(
            [x, y, x + self.card_width, y + self.card_height],
            fill=(0, 0, 0, 160),
            outline=(255, 255, 255, 200),
            width=2
        )
        
        # Blend overlay
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # Draw player info
        name_y = y + 10
        draw.text((x + 15, name_y), f"#{stats.player_number} {stats.player_name}", 
                 font=self.font_medium, fill=(255, 255, 255))
        
        # Draw team
        draw.text((x + 15, name_y + 35), stats.team, font=self.font_small, fill=(200, 200, 200))
        
        # Draw separator
        draw.line([(x + 15, y + 70), (x + self.card_width - 15, y + 70)], 
                 fill=(150, 150, 150), width=1)
        
        # Draw stats
        stats_y = y + 80
        
        # Speed
        draw.text((x + 15, stats_y), f"Speed: {stats.speed:.1f} km/h", 
                 font=self.font_small, fill=(255, 255, 255))
        
        # Distance
        draw.text((x + 15, stats_y + 25), f"Distance: {stats.distance:.1f} m", 
                 font=self.font_small, fill=(255, 255, 255))
        
        # Passes and shots on right side
        draw.text((x + 150, stats_y), f"Passes: {stats.passes}", 
                 font=self.font_small, fill=(255, 255, 255))
        draw.text((x + 150, stats_y + 25), f"Shots: {stats.shots}", 
                 font=self.font_small, fill=(255, 255, 255))
        
        return self._pil_to_cv2(pil_image)


class EventPopupRenderer(GraphicsRenderer):
    """Render event popups (goals, cards, etc.)."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__(width, height)
        self.popup_width = 400
        self.popup_height = 100
        self.event_colors = {
            EventType.GOAL: (0, 200, 0),
            EventType.YELLOW_CARD: (255, 220, 0),
            EventType.RED_CARD: (220, 0, 0),
            EventType.SUBSTITUTION: (0, 150, 255),
            EventType.PENALTY: (255, 150, 0),
            EventType.VAR: (150, 0, 200)
        }
        
    def render(self, frame: np.ndarray, popup: EventPopup) -> np.ndarray:
        """Render event popup on frame."""
        if not self._should_show(popup):
            return frame
            
        pil_image = self._cv2_to_pil(frame)
        
        # Position at bottom center
        x = (self.width - self.popup_width) // 2
        y = self.height - self.popup_height - 50
        
        # Create overlay
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        color = self.event_colors.get(popup.event_type, (100, 100, 100))
        
        # Draw popup background with gradient effect
        overlay_draw.rectangle(
            [x, y, x + self.popup_width, y + self.popup_height],
            fill=(0, 0, 0, 200)
        )
        
        # Draw color bar on left
        overlay_draw.rectangle(
            [x, y, x + 10, y + self.popup_height],
            fill=(*color, 255)
        )
        
        # Blend overlay
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # Draw event icon/text
        icon_x = x + 25
        icon_y = y + 20
        
        event_text = self._get_event_text(popup.event_type)
        draw.text((icon_x, icon_y), event_text, font=self.font_medium, fill=color)
        
        # Draw player name
        name_y = y + 55
        draw.text((icon_x, name_y), popup.player_name, font=self.font_medium, fill=(255, 255, 255))
        
        # Draw team and minute
        info_y = y + 20
        draw.text((x + self.popup_width - 120, info_y), 
                 f"{popup.team} | {popup.minute}'", 
                 font=self.font_small, fill=(200, 200, 200))
        
        # Draw additional info if present
        if popup.additional_info:
            draw.text((x + 200, name_y), popup.additional_info, 
                     font=self.font_small, fill=(200, 200, 200))
        
        return self._pil_to_cv2(pil_image)
        
    def _should_show(self, popup: EventPopup) -> bool:
        """Check if popup should still be displayed."""
        if popup.start_time is None:
            return False
            
        elapsed = (datetime.now() - popup.start_time).total_seconds()
        return elapsed < popup.show_duration
        
    def _get_event_text(self, event_type: EventType) -> str:
        """Get display text for event type."""
        texts = {
            EventType.GOAL: "⚽ GOAL!",
            EventType.YELLOW_CARD: "🟨 YELLOW CARD",
            EventType.RED_CARD: "🟥 RED CARD",
            EventType.SUBSTITUTION: "🔄 SUBSTITUTION",
            EventType.PENALTY: "⚽ PENALTY",
            EventType.VAR: "📺 VAR"
        }
        return texts.get(event_type, "EVENT")


class PossessionBarRenderer(GraphicsRenderer):
    """Render possession bar and team statistics."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__(width, height)
        self.bar_width = 600
        self.bar_height = 40
        
    def render(self, frame: np.ndarray, 
               home_possession: float,
               away_possession: float,
               home_stats: Dict[str, Any],
               away_stats: Dict[str, Any],
               home_team: TeamInfo,
               away_team: TeamInfo) -> np.ndarray:
        """Render possession bar and team stats."""
        pil_image = self._cv2_to_pil(frame)
        
        # Position at top, below scoreboard
        x = (self.width - self.bar_width) // 2
        y = 110
        
        # Create overlay
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Draw background
        overlay_draw.rectangle(
            [x, y, x + self.bar_width, y + self.bar_height],
            fill=(0, 0, 0, 150)
        )
        
        # Calculate split point
        split_x = x + int(self.bar_width * (home_possession / 100))
        
        # Draw home possession (left side)
        overlay_draw.rectangle(
            [x, y, split_x, y + self.bar_height],
            fill=(*home_team.color_primary, 200)
        )
        
        # Draw away possession (right side)
        overlay_draw.rectangle(
            [split_x, y, x + self.bar_width, y + self.bar_height],
            fill=(*away_team.color_primary, 200)
        )
        
        # Blend overlay
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # Draw possession percentages
        draw.text((x + 10, y + 5), f"{home_possession:.0f}%", 
                 font=self.font_small, fill=(255, 255, 255))
        draw.text((x + self.bar_width - 60, y + 5), f"{away_possession:.0f}%", 
                 font=self.font_small, fill=(255, 255, 255))
        
        # Draw "POSSESSION" label
        label_x = x + self.bar_width // 2 - 60
        draw.text((label_x, y + 5), "POSSESSION", font=self.font_small, fill=(255, 255, 255))
        
        # Draw team stats below possession bar
        stats_y = y + self.bar_height + 10
        self._draw_team_stats(draw, home_stats, away_stats, x, stats_y, home_team, away_team)
        
        return self._pil_to_cv2(pil_image)
        
    def _draw_team_stats(self, draw: ImageDraw.Draw,
                        home_stats: Dict, away_stats: Dict,
                        x: int, y: int,
                        home_team: TeamInfo, away_team: TeamInfo):
        """Draw team statistics comparison."""
        stats_to_show = [
            ('shots', 'Shots'),
            ('shots_on_target', 'On Target'),
            ('corners', 'Corners'),
            ('fouls', 'Fouls')
        ]
        
        stat_y = y
        for stat_key, stat_name in stats_to_show:
            home_val = home_stats.get(stat_key, 0)
            away_val = away_stats.get(stat_key, 0)
            
            # Draw stat name
            draw.text((x + self.bar_width // 2 - 50, stat_y), stat_name, 
                     font=self.font_small, fill=(200, 200, 200))
            
            # Draw home value
            draw.text((x + 10, stat_y), str(home_val), 
                     font=self.font_small, fill=home_team.color_primary)
            
            # Draw away value
            draw.text((x + self.bar_width - 40, stat_y), str(away_val), 
                     font=self.font_small, fill=away_team.color_primary)
            
            stat_y += 25


class BroadcastGraphicsManager:
    """Main manager for all broadcast graphics."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        
        # Initialize renderers
        self.scoreboard_renderer = ScoreboardRenderer(width, height)
        self.player_stats_renderer = PlayerStatsRenderer(width, height)
        self.event_renderer = EventPopupRenderer(width, height)
        self.possession_renderer = PossessionBarRenderer(width, height)
        
        # State
        self.scoreboard_state: Optional[ScoreboardState] = None
        self.active_player_stats: List[PlayerStatsOverlay] = []
        self.active_events: List[EventPopup] = []
        self.show_possession_bar = True
        self.home_possession = 50.0
        self.away_possession = 50.0
        self.home_stats: Dict[str, Any] = {}
        self.away_stats: Dict[str, Any] = {}
        
    def setup_scoreboard(self, home_team: TeamInfo, away_team: TeamInfo):
        """Initialize scoreboard with teams."""
        self.scoreboard_state = ScoreboardState(
            home_team=home_team,
            away_team=away_team
        )
        
    def update_score(self, home_score: int, away_score: int):
        """Update match score."""
        if self.scoreboard_state:
            self.scoreboard_state.home_score = home_score
            self.scoreboard_state.away_score = away_score
            
    def update_time(self, minutes: int, period: str = "1H", added_time: int = 0):
        """Update match time."""
        if self.scoreboard_state:
            self.scoreboard_state.match_time = minutes
            self.scoreboard_state.match_period = period
            self.scoreboard_state.added_time = added_time
            
    def show_player_stats(self, stats: PlayerStatsOverlay):
        """Display player statistics overlay."""
        # Remove existing stats for same player
        self.active_player_stats = [
            s for s in self.active_player_stats 
            if s.player_name != stats.player_name
        ]
        self.active_player_stats.append(stats)
        
    def hide_player_stats(self, player_name: str):
        """Hide player statistics overlay."""
        self.active_player_stats = [
            s for s in self.active_player_stats 
            if s.player_name != player_name
        ]
        
    def trigger_event(self, event_type: EventType, player_name: str, 
                     team: str, minute: int, additional_info: str = ""):
        """Trigger an event popup."""
        popup = EventPopup(
            event_type=event_type,
            player_name=player_name,
            team=team,
            minute=minute,
            additional_info=additional_info,
            start_time=datetime.now()
        )
        self.active_events.append(popup)
        
    def update_possession(self, home_possession: float, away_possession: float):
        """Update possession percentages."""
        self.home_possession = home_possession
        self.away_possession = away_possession
        
    def update_team_stats(self, home_stats: Dict[str, Any], away_stats: Dict[str, Any]):
        """Update team statistics."""
        self.home_stats = home_stats
        self.away_stats = away_stats
        
    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        """Render all active graphics on frame."""
        result = frame.copy()
        
        # Render scoreboard
        if self.scoreboard_state:
            result = self.scoreboard_renderer.render(result, self.scoreboard_state)
            
        # Render possession bar
        if self.show_possession_bar and self.scoreboard_state:
            result = self.possession_renderer.render(
                result,
                self.home_possession,
                self.away_possession,
                self.home_stats,
                self.away_stats,
                self.scoreboard_state.home_team,
                self.scoreboard_state.away_team
            )
            
        # Render player stats
        for stats in self.active_player_stats:
            if stats.show:
                result = self.player_stats_renderer.render(result, stats)
                
        # Render events and clean up expired ones
        active_events = []
        for event in self.active_events:
            result = self.event_renderer.render(result, event)
            if event.start_time and (datetime.now() - event.start_time).total_seconds() < event.show_duration:
                active_events.append(event)
        self.active_events = active_events
        
        return result
        
    def create_intro_graphic(self, home_team: TeamInfo, away_team: TeamInfo,
                            competition: str, match_date: str) -> np.ndarray:
        """Create match intro graphic."""
        # Create blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Background gradient (simulated with rectangles)
        for i in range(self.height):
            color_val = int(20 + (i / self.height) * 40)
            draw.line([(0, i), (self.width, i)], fill=(0, 0, color_val))
        
        # Draw competition name
        comp_font = self._load_font(36)
        draw.text((self.width // 2 - 200, 100), competition, 
                 font=comp_font, fill=(255, 255, 255))
        
        # Draw teams
        team_font = self._load_font(72)
        
        # Home team
        draw.text((200, self.height // 2 - 50), home_team.name, 
                 font=team_font, fill=home_team.color_primary)
        
        # VS
        draw.text((self.width // 2 - 50, self.height // 2 - 50), "VS", 
                 font=team_font, fill=(255, 255, 255))
        
        # Away team
        away_name_width = len(away_team.name) * 40
        draw.text((self.width - 200 - away_name_width, self.height // 2 - 50), 
                 away_team.name, font=team_font, fill=away_team.color_primary)
        
        # Draw date
        date_font = self._load_font(28)
        draw.text((self.width // 2 - 150, self.height - 200), match_date, 
                 font=date_font, fill=(200, 200, 200))
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    def _load_font(self, size: int):
        """Load font."""
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except:
                return ImageFont.load_default()


# Convenience functions
def create_broadcast_manager(width: int = 1920, height: int = 1080) -> BroadcastGraphicsManager:
    """Create and return a configured broadcast graphics manager."""
    return BroadcastGraphicsManager(width, height)


def create_team_info(name: str, short_name: str, 
                    primary_color: Tuple[int, int, int] = (255, 0, 0),
                    secondary_color: Tuple[int, int, int] = (255, 255, 255)) -> TeamInfo:
    """Create team info object."""
    return TeamInfo(
        name=name,
        short_name=short_name,
        color_primary=primary_color,
        color_secondary=secondary_color
    )
