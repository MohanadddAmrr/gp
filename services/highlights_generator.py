"""
Auto Highlights Generator Module

Automatically detects and generates highlights from match footage.
Identifies key moments like goals, big chances, tackles, saves, and other important events.
Scores events by importance and creates a highlights reel with timestamps.

Key Features:
- Automatic key moment detection
- Event importance scoring
- Highlight clip generation with timestamps
- Match summary creation
- Customizable highlight criteria
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np


class EventType(Enum):
    """Types of highlight events."""
    GOAL = "goal"
    BIG_CHANCE = "big_chance"
    SAVE = "save"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    DRIBBLE = "dribble"
    PASS_COMPLETION = "pass_completion"
    SET_PIECE = "set_piece"
    TRANSITION = "transition"
    CARD = "card"
    OFFSIDE = "offside"
    PRESSING = "pressing"


class ImportanceLevel(Enum):
    """Importance levels for events."""
    CRITICAL = 5  # Goals, red cards
    HIGH = 4      # Big chances, great saves
    MEDIUM = 3    # Good tackles, interceptions
    LOW = 2       # Pass completions, normal plays
    MINIMAL = 1   # Background events


@dataclass
class HighlightEvent:
    """Represents a highlight event."""
    event_type: EventType
    timestamp: float
    frame: int
    importance: ImportanceLevel
    
    # Participants
    primary_player_id: Optional[int] = None
    secondary_player_id: Optional[int] = None
    team: Optional[str] = None
    
    # Context
    description: str = ""
    score_at_time: Tuple[int, int] = (0, 0)
    
    # Metrics
    xg_value: float = 0.0
    velocity: float = 0.0
    distance: float = 0.0
    
    # Clip info
    clip_start: float = 0.0
    clip_end: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HighlightClip:
    """Represents a highlight clip."""
    start_time: float
    end_time: float
    events: List[HighlightEvent] = field(default_factory=list)
    total_importance: int = 0
    description: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class HighlightsGenerator:
    """
    Automatic Highlights Generator.
    
    Detects key moments from match data and generates highlight clips.
    Uses importance scoring to prioritize the best moments.
    """
    
    # Default clip duration settings (seconds)
    DEFAULT_CLIP_DURATION = 8.0
    MIN_CLIP_DURATION = 3.0
    MAX_CLIP_DURATION = 15.0
    
    # Importance thresholds
    HIGHLIGHT_THRESHOLD = ImportanceLevel.HIGH
    
    # Time windows for context (seconds)
    PRE_EVENT_CONTEXT = 3.0
    POST_EVENT_CONTEXT = 5.0
    
    def __init__(
        self,
        clip_duration: float = DEFAULT_CLIP_DURATION,
        min_importance: ImportanceLevel = ImportanceLevel.MEDIUM
    ):
        """
        Initialize the highlights generator.
        
        Args:
            clip_duration: Default duration for highlight clips
            min_importance: Minimum importance level to include
        """
        self.clip_duration = clip_duration
        self.min_importance = min_importance
        
        self.events: List[HighlightEvent] = []
        self.clips: List[HighlightClip] = []
        
        # Current match state
        self.current_score = {'A': 0, 'B': 0}
        self.match_duration = 0.0
        
        # Event weights for scoring
        self.event_weights = {
            EventType.GOAL: ImportanceLevel.CRITICAL,
            EventType.BIG_CHANCE: ImportanceLevel.HIGH,
            EventType.SAVE: ImportanceLevel.HIGH,
            EventType.TACKLE: ImportanceLevel.MEDIUM,
            EventType.INTERCEPTION: ImportanceLevel.MEDIUM,
            EventType.DRIBBLE: ImportanceLevel.LOW,
            EventType.PASS_COMPLETION: ImportanceLevel.MINIMAL,
            EventType.SET_PIECE: ImportanceLevel.MEDIUM,
            EventType.TRANSITION: ImportanceLevel.LOW,
            EventType.CARD: ImportanceLevel.HIGH,
            EventType.OFFSIDE: ImportanceLevel.LOW,
            EventType.PRESSING: ImportanceLevel.MINIMAL
        }
    
    def add_event(
        self,
        event_type: EventType,
        timestamp: float,
        frame: int,
        primary_player_id: int = None,
        secondary_player_id: int = None,
        team: str = None,
        description: str = "",
        xg_value: float = 0.0,
        velocity: float = 0.0,
        distance: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> HighlightEvent:
        """
        Add a highlight event.
        
        Args:
            event_type: Type of event
            timestamp: Time in seconds
            frame: Frame number
            primary_player_id: Main player involved
            secondary_player_id: Secondary player (if any)
            team: Team involved
            description: Event description
            xg_value: xG value (for shots)
            velocity: Ball/player velocity
            distance: Distance covered
            metadata: Additional data
            
        Returns:
            Created HighlightEvent
        """
        # Determine importance
        base_importance = self.event_weights.get(event_type, ImportanceLevel.LOW)
        
        # Adjust importance based on context
        importance = self._adjust_importance(
            event_type, base_importance, xg_value, velocity, metadata
        )
        
        event = HighlightEvent(
            event_type=event_type,
            timestamp=timestamp,
            frame=frame,
            importance=importance,
            primary_player_id=primary_player_id,
            secondary_player_id=secondary_player_id,
            team=team,
            description=description,
            score_at_time=(self.current_score['A'], self.current_score['B']),
            xg_value=xg_value,
            velocity=velocity,
            distance=distance,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Update score for goals
        if event_type == EventType.GOAL and team:
            self.current_score[team] += 1
        
        return event
    
    def _adjust_importance(
        self,
        event_type: EventType,
        base_importance: ImportanceLevel,
        xg_value: float,
        velocity: float,
        metadata: Dict[str, Any]
    ) -> ImportanceLevel:
        """Adjust importance based on context."""
        importance_value = base_importance.value
        
        # Big chances with high xG are more important
        if event_type == EventType.BIG_CHANCE and xg_value > 0.3:
            importance_value += 1
        
        # High velocity shots/saves are more spectacular
        if velocity > 30:  # m/s
            importance_value += 1
        
        # Late game events can be more important
        if metadata and metadata.get('match_time_percent', 0) > 0.8:
            importance_value += 1
        
        # Clutch moments (tied game, last minutes)
        if metadata and metadata.get('is_clutch', False):
            importance_value += 1
        
        # Cap at CRITICAL
        importance_value = min(importance_value, ImportanceLevel.CRITICAL.value)
        
        return ImportanceLevel(importance_value)
    
    def add_goal(
        self,
        timestamp: float,
        frame: int,
        scorer_id: int,
        team: str,
        assist_id: int = None,
        xg_value: float = 0.0,
        description: str = ""
    ) -> HighlightEvent:
        """Convenience method to add a goal event."""
        return self.add_event(
            event_type=EventType.GOAL,
            timestamp=timestamp,
            frame=frame,
            primary_player_id=scorer_id,
            secondary_player_id=assist_id,
            team=team,
            description=description or f"Goal by Player {scorer_id}",
            xg_value=xg_value
        )
    
    def add_big_chance(
        self,
        timestamp: float,
        frame: int,
        shooter_id: int,
        team: str,
        xg_value: float,
        saved: bool = False,
        description: str = ""
    ) -> HighlightEvent:
        """Convenience method to add a big chance event."""
        event_type = EventType.SAVE if saved else EventType.BIG_CHANCE
        return self.add_event(
            event_type=event_type,
            timestamp=timestamp,
            frame=frame,
            primary_player_id=shooter_id,
            team=team,
            description=description or f"Big chance ({xg_value:.2f} xG)",
            xg_value=xg_value
        )
    
    def add_tackle(
        self,
        timestamp: float,
        frame: int,
        tackler_id: int,
        team: str,
        successful: bool = True,
        description: str = ""
    ) -> HighlightEvent:
        """Convenience method to add a tackle event."""
        return self.add_event(
            event_type=EventType.TACKLE,
            timestamp=timestamp,
            frame=frame,
            primary_player_id=tackler_id,
            team=team,
            description=description or f"{'Successful' if successful else 'Failed'} tackle",
            metadata={'successful': successful}
        )
    
    def add_save(
        self,
        timestamp: float,
        frame: int,
        keeper_id: int,
        team: str,
        xg_faced: float,
        description: str = ""
    ) -> HighlightEvent:
        """Convenience method to add a save event."""
        return self.add_event(
            event_type=EventType.SAVE,
            timestamp=timestamp,
            frame=frame,
            primary_player_id=keeper_id,
            team=team,
            description=description or f"Save ({xg_faced:.2f} xG)",
            xg_value=xg_faced
        )
    
    def generate_clips(self, merge_window: float = 3.0) -> List[HighlightClip]:
        """
        Generate highlight clips from events.
        
        Args:
            merge_window: Time window to merge nearby events (seconds)
            
        Returns:
            List of HighlightClip objects
        """
        # Filter events by importance
        important_events = [
            e for e in self.events 
            if e.importance.value >= self.min_importance.value
        ]
        
        if not important_events:
            return []
        
        # Sort by timestamp
        important_events.sort(key=lambda e: e.timestamp)
        
        # Group events into clips
        clips = []
        current_clip_events = [important_events[0]]
        
        for i in range(1, len(important_events)):
            current_event = important_events[i]
            last_event = current_clip_events[-1]
            
            # Check if events are close enough to merge
            if current_event.timestamp - last_event.timestamp <= merge_window:
                current_clip_events.append(current_event)
            else:
                # Create clip from current group
                clip = self._create_clip(current_clip_events)
                clips.append(clip)
                current_clip_events = [current_event]
        
        # Don't forget the last group
        if current_clip_events:
            clip = self._create_clip(current_clip_events)
            clips.append(clip)
        
        self.clips = clips
        return clips
    
    def _create_clip(self, events: List[HighlightEvent]) -> HighlightClip:
        """Create a highlight clip from a group of events."""
        if not events:
            return None
        
        # Find time range
        first_event = min(events, key=lambda e: e.timestamp)
        last_event = max(events, key=lambda e: e.timestamp)
        
        # Add context buffers
        start_time = max(0, first_event.timestamp - self.PRE_EVENT_CONTEXT)
        end_time = last_event.timestamp + self.POST_EVENT_CONTEXT
        
        # Ensure minimum duration
        if end_time - start_time < self.MIN_CLIP_DURATION:
            end_time = start_time + self.MIN_CLIP_DURATION
        
        # Cap maximum duration
        end_time = min(end_time, start_time + self.MAX_CLIP_DURATION)
        
        # Calculate total importance
        total_importance = sum(e.importance.value for e in events)
        
        # Generate description
        descriptions = []
        for e in events:
            if e.importance.value >= ImportanceLevel.HIGH.value:
                descriptions.append(e.description)
        description = " | ".join(descriptions[:3])  # Top 3 events
        
        clip = HighlightClip(
            start_time=start_time,
            end_time=end_time,
            events=events,
            total_importance=total_importance,
            description=description
        )
        
        # Update event clip times
        for e in events:
            e.clip_start = start_time
            e.clip_end = end_time
        
        return clip
    
    def get_top_clips(self, n: int = 10) -> List[HighlightClip]:
        """Get top N clips by importance."""
        if not self.clips:
            self.generate_clips()
        
        sorted_clips = sorted(
            self.clips, 
            key=lambda c: c.total_importance, 
            reverse=True
        )
        return sorted_clips[:n]
    
    def generate_match_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive match summary.
        
        Returns:
            Dictionary with match summary data
        """
        if not self.clips:
            self.generate_clips()
        
        # Count events by type
        events_by_type = defaultdict(int)
        events_by_importance = defaultdict(int)
        
        for event in self.events:
            events_by_type[event.event_type.value] += 1
            events_by_importance[event.importance.name] += 1
        
        # Get key moments
        key_moments = [
            {
                'time': e.timestamp,
                'type': e.event_type.value,
                'importance': e.importance.name,
                'description': e.description,
                'team': e.team
            }
            for e in self.events
            if e.importance.value >= ImportanceLevel.HIGH.value
        ]
        
        # Sort by importance and time
        key_moments.sort(key=lambda m: (m['importance'], m['time']), reverse=True)
        
        return {
            'total_events': len(self.events),
            'total_clips': len(self.clips),
            'events_by_type': dict(events_by_type),
            'events_by_importance': dict(events_by_importance),
            'final_score': self.current_score.copy(),
            'key_moments': key_moments[:20],  # Top 20 moments
            'clip_timestamps': [
                {
                    'start': c.start_time,
                    'end': c.end_time,
                    'duration': c.duration,
                    'importance': c.total_importance,
                    'description': c.description
                }
                for c in self.get_top_clips(10)
            ]
        }
    
    def get_highlight_timeline(self) -> List[Dict[str, Any]]:
        """
        Get a timeline of all highlight events.
        
        Returns:
            List of event dictionaries sorted by time
        """
        timeline = []
        
        for event in sorted(self.events, key=lambda e: e.timestamp):
            timeline.append({
                'timestamp': event.timestamp,
                'frame': event.frame,
                'type': event.event_type.value,
                'importance': event.importance.value,
                'importance_name': event.importance.name,
                'team': event.team,
                'description': event.description,
                'xg': event.xg_value,
                'clip_start': event.clip_start,
                'clip_end': event.clip_end
            })
        
        return timeline
    
    def export_highlights_json(self) -> Dict[str, Any]:
        """Export highlights data as JSON-serializable dict."""
        return {
            'summary': self.generate_match_summary(),
            'timeline': self.get_highlight_timeline(),
            'clips': [
                {
                    'start_time': c.start_time,
                    'end_time': c.end_time,
                    'duration': c.duration,
                    'total_importance': c.total_importance,
                    'description': c.description,
                    'events': [
                        {
                            'type': e.event_type.value,
                            'timestamp': e.timestamp,
                            'importance': e.importance.name,
                            'description': e.description
                        }
                        for e in c.events
                    ]
                }
                for c in self.clips
            ]
        }
    
    def reset(self):
        """Reset all data."""
        self.events = []
        self.clips = []
        self.current_score = {'A': 0, 'B': 0}
        self.match_duration = 0.0
