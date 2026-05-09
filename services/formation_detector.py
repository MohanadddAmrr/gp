"""
Tactical Formation Detector Module

Detects team formations (4-3-3, 4-4-2, 3-5-2, etc.) from player positions.

FORMATION DETECTION LOGIC:
1. Calculate average positions over time for each player
2. Cluster players into defensive/midfield/attacking lines
3. Count players in each line to determine formation
4. Track formation changes during match

COMMON FORMATIONS:
- 4-3-3: 4 defenders, 3 midfielders, 3 attackers
- 4-4-2: 4 defenders, 4 midfielders, 2 attackers
- 3-5-2: 3 defenders, 5 midfielders, 2 attackers
- 5-3-2: 5 defenders, 3 midfielders, 2 attackers
- 4-2-3-1: 4 defenders, 2 holding, 3 attacking mid, 1 striker
- 4-5-1: 4 defenders, 5 midfielders, 1 attacker
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import numpy as np
from dataclasses import dataclass, field


@dataclass
class FormationLine:
    """Represents a horizontal line of players (defense, midfield, attack)."""
    line_type: str  # 'defense', 'midfield', 'attack', 'goalkeeper'
    avg_x: float
    players: List[int] = field(default_factory=list)
    y_spread: float = 0.0


@dataclass
class FormationSnapshot:
    """Formation at a specific point in time."""
    timestamp: float
    frame_idx: int
    formation_code: str
    lines: List[FormationLine]
    team: str
    confidence: float


class FormationDetector:
    """
    Detects team formations from player tracking data.
    
    Uses clustering to identify defensive, midfield, and attacking lines,
    then maps these to standard formation codes.
    """

    # Standard formations and their line distributions
    FORMATION_TEMPLATES = {
        # Standard formations
        '4-3-3': {'defense': 4, 'midfield': 3, 'attack': 3},
        '4-4-2': {'defense': 4, 'midfield': 4, 'attack': 2},
        '3-5-2': {'defense': 3, 'midfield': 5, 'attack': 2},
        '5-3-2': {'defense': 5, 'midfield': 3, 'attack': 2},
        '4-2-3-1': {'defense': 4, 'midfield': 5, 'attack': 1},
        '4-5-1': {'defense': 4, 'midfield': 5, 'attack': 1},
        '3-4-3': {'defense': 3, 'midfield': 4, 'attack': 3},
        '5-4-1': {'defense': 5, 'midfield': 4, 'attack': 1},
        '4-1-4-1': {'defense': 4, 'midfield': 5, 'attack': 1},
        '3-6-1': {'defense': 3, 'midfield': 6, 'attack': 1},
        # Additional formations
        '4-4-1-1': {'defense': 4, 'midfield': 5, 'attack': 1},
        '4-3-2-1': {'defense': 4, 'midfield': 5, 'attack': 1},
        '3-4-2-1': {'defense': 3, 'midfield': 6, 'attack': 1},
        '5-2-3': {'defense': 5, 'midfield': 2, 'attack': 3},
        '4-1-2-1-2': {'defense': 4, 'midfield': 4, 'attack': 2},
        '3-3-4': {'defense': 3, 'midfield': 3, 'attack': 4},
        '2-3-5': {'defense': 2, 'midfield': 3, 'attack': 5},  # Historical
    }
    
    # Fuzzy matching tolerance for player counts
    FORMATION_TOLERANCE = 1  # Allow ±1 player difference


    def __init__(
        self,
        position_window: int = 30,  # frames to average over (~1 second at 30fps)
        min_samples: int = 10,      # minimum samples for reliable average
        line_cluster_threshold: float = 30.0,  # pixels - max distance for same line
        pitch_length_px: float = 1920.0,  # default frame width
    ):
        """
        Initialize formation detector.
        
        Args:
            position_window: Number of frames to average positions over
            min_samples: Minimum samples needed for reliable position average
            line_cluster_threshold: Max pixel distance to consider players in same line
            pitch_length_px: Frame width (used for relative positioning)
        """
        self.position_window = position_window
        self.min_samples = min_samples
        self.line_cluster_threshold = line_cluster_threshold
        self.pitch_length_px = pitch_length_px

        # Player position history: {player_id: deque of (x, y, timestamp)}
        self._position_history: Dict[int, deque] = {}

        # Formation history
        self._formation_history: List[FormationSnapshot] = []

        # Current formation state per team
        self._current_formations: Dict[str, FormationSnapshot] = {}

        # Formation change tracking
        self._formation_changes: List[Dict[str, Any]] = []

        # Team assignments
        self._player_teams: Dict[int, str] = {}

    def update_player_position(
        self,
        player_id: int,
        position: Tuple[float, float],
        team: str,
        timestamp: float,
        frame_idx: int
    ) -> None:
        """
        Update position history for a player.
        
        Args:
            player_id: Player track ID
            position: (x, y) pixel coordinates
            team: Team label ('A' or 'B')
            timestamp: Current time in seconds
            frame_idx: Current frame index
        """
        self._player_teams[player_id] = team

        if player_id not in self._position_history:
            self._position_history[player_id] = deque(maxlen=self.position_window)

        self._position_history[player_id].append({
            'x': position[0],
            'y': position[1],
            'timestamp': timestamp,
            'frame_idx': frame_idx
        })

    def get_average_position(self, player_id: int) -> Optional[Tuple[float, float]]:
        """
        Get average position for a player over recent frames.
        
        Args:
            player_id: Player track ID
            
        Returns:
            Average (x, y) position or None if insufficient data
        """
        if player_id not in self._position_history:
            return None

        history = self._position_history[player_id]
        if len(history) < self.min_samples:
            return None

        avg_x = sum(p['x'] for p in history) / len(history)
        avg_y = sum(p['y'] for p in history) / len(history)
        return (avg_x, avg_y)

    def detect_formation(
        self,
        team: str,
        frame_idx: int,
        timestamp: float,
        exclude_goalkeeper: bool = True
    ) -> Optional[FormationSnapshot]:
        """
        Detect formation for a specific team.
        
        Args:
            team: Team label ('A' or 'B')
            frame_idx: Current frame index
            timestamp: Current timestamp
            exclude_goalkeeper: Whether to exclude the deepest player (GK)
            
        Returns:
            FormationSnapshot with detected formation or None
        """
        # Get all players for this team with valid average positions
        team_players = []
        for player_id, player_team in self._player_teams.items():
            if player_team != team:
                continue

            avg_pos = self.get_average_position(player_id)
            if avg_pos is not None:
                team_players.append((player_id, avg_pos))

        if len(team_players) < 7:  # Need at least 7 players for reliable formation
            return None

        # Sort by x position (depth on pitch)
        # Team A attacks right (higher x), Team B attacks left (lower x)
        if team == 'A':
            # For Team A, defense is at lower x values
            team_players.sort(key=lambda p: p[1][0])
        else:
            # For Team B, defense is at higher x values
            team_players.sort(key=lambda p: p[1][0], reverse=True)

        # Identify goalkeeper (deepest player)
        if exclude_goalkeeper and team_players:
            team_players = team_players[1:]  # Remove first (deepest) player

        if len(team_players) < 6:
            return None

        # Cluster players into lines based on x-position
        lines = self._cluster_into_lines(team_players, team)

        # Map lines to formation code
        formation_code, confidence = self._classify_formation(lines)

        snapshot = FormationSnapshot(
            timestamp=timestamp,
            frame_idx=frame_idx,
            formation_code=formation_code,
            lines=lines,
            team=team,
            confidence=confidence
        )

        # Track formation changes
        if team in self._current_formations:
            prev_formation = self._current_formations[team].formation_code
            if prev_formation != formation_code:
                self._formation_changes.append({
                    'timestamp': timestamp,
                    'frame_idx': frame_idx,
                    'team': team,
                    'from_formation': prev_formation,
                    'to_formation': formation_code,
                    'confidence': confidence
                })

        self._current_formations[team] = snapshot
        self._formation_history.append(snapshot)

        return snapshot

    def _cluster_into_lines(
        self,
        players: List[Tuple[int, Tuple[float, float]]],
        team: str
    ) -> List[FormationLine]:
        """
        Cluster players into defensive, midfield, and attacking lines.
        Uses adaptive clustering based on pitch position distribution.
        
        Args:
            players: List of (player_id, (x, y)) tuples, sorted by depth
            team: Team label
            
        Returns:
            List of FormationLine objects
        """
        if not players:
            return []

        if len(players) < 3:
            # Not enough players for meaningful formation
            return []

        # Extract x positions for clustering analysis
        x_positions = np.array([p[1][0] for p in players])
        
        # Use adaptive threshold based on pitch spread
        x_range = np.max(x_positions) - np.min(x_positions)
        adaptive_threshold = max(self.line_cluster_threshold, x_range / 6)
        
        # Group players into lines using clustering
        lines = []
        current_line = [players[0]]
        
        for player in players[1:]:
            player_x = player[1][0]
            current_line_x_positions = [p[1][0] for p in current_line]
            current_line_avg_x = np.mean(current_line_x_positions)
            x_distance = abs(player_x - current_line_avg_x)

            if x_distance <= adaptive_threshold:
                # Same line
                current_line.append(player)
            else:
                # New line - save current line
                avg_x = np.mean([p[1][0] for p in current_line])
                y_values = [p[1][1] for p in current_line]
                y_spread = max(y_values) - min(y_values) if len(y_values) > 1 else 0

                lines.append({
                    'avg_x': avg_x,
                    'players': current_line.copy(),
                    'y_spread': y_spread
                })

                # Start new line
                current_line = [player]

        # Don't forget the last line
        if current_line:
            avg_x = np.mean([p[1][0] for p in current_line])
            y_values = [p[1][1] for p in current_line]
            y_spread = max(y_values) - min(y_values) if len(y_values) > 1 else 0

            lines.append({
                'avg_x': avg_x,
                'players': current_line.copy(),
                'y_spread': y_spread
            })

        # Now classify lines into defense/midfield/attack based on position
        # Sort lines by x position (deepest = defense for the team)
        lines.sort(key=lambda l: l['avg_x'])
        
        # Determine line types based on number of lines and their distribution
        num_lines = len(lines)
        formation_lines = []
        
        if num_lines == 1:
            # All players in one line - likely all midfield
            formation_lines.append(FormationLine(
                line_type='midfield',
                avg_x=lines[0]['avg_x'],
                players=[p[0] for p in lines[0]['players']],
                y_spread=lines[0]['y_spread']
            ))
        elif num_lines == 2:
            # Two lines - defense and attack (compact formation)
            formation_lines.append(FormationLine(
                line_type='defense',
                avg_x=lines[0]['avg_x'],
                players=[p[0] for p in lines[0]['players']],
                y_spread=lines[0]['y_spread']
            ))
            formation_lines.append(FormationLine(
                line_type='attack',
                avg_x=lines[1]['avg_x'],
                players=[p[0] for p in lines[1]['players']],
                y_spread=lines[1]['y_spread']
            ))
        elif num_lines == 3:
            # Standard three lines - defense, midfield, attack
            formation_lines.append(FormationLine(
                line_type='defense',
                avg_x=lines[0]['avg_x'],
                players=[p[0] for p in lines[0]['players']],
                y_spread=lines[0]['y_spread']
            ))
            formation_lines.append(FormationLine(
                line_type='midfield',
                avg_x=lines[1]['avg_x'],
                players=[p[0] for p in lines[1]['players']],
                y_spread=lines[1]['y_spread']
            ))
            formation_lines.append(FormationLine(
                line_type='attack',
                avg_x=lines[2]['avg_x'],
                players=[p[0] for p in lines[2]['players']],
                y_spread=lines[2]['y_spread']
            ))
        else:
            # More than 3 lines - merge middle lines into midfield
            # First line is defense
            formation_lines.append(FormationLine(
                line_type='defense',
                avg_x=lines[0]['avg_x'],
                players=[p[0] for p in lines[0]['players']],
                y_spread=lines[0]['y_spread']
            ))
            
            # Last line is attack
            formation_lines.append(FormationLine(
                line_type='attack',
                avg_x=lines[-1]['avg_x'],
                players=[p[0] for p in lines[-1]['players']],
                y_spread=lines[-1]['y_spread']
            ))
            
            # Middle lines merged into midfield
            mid_players = []
            mid_x_values = []
            mid_y_spreads = []
            for line in lines[1:-1]:
                mid_players.extend([p[0] for p in line['players']])
                mid_x_values.append(line['avg_x'])
                mid_y_spreads.append(line['y_spread'])
            
            formation_lines.append(FormationLine(
                line_type='midfield',
                avg_x=np.mean(mid_x_values) if mid_x_values else 0,
                players=mid_players,
                y_spread=max(mid_y_spreads) if mid_y_spreads else 0
            ))
            
            # Sort by avg_x to maintain order
            formation_lines.sort(key=lambda l: l.avg_x)

        return formation_lines

    def _classify_line_type(
        self,
        existing_lines: List[FormationLine],
        total_outfield_players: int
    ) -> str:
        """
        Classify what type of line this is based on order and context.
        
        Args:
            existing_lines: Lines already identified
            total_outfield_players: Total number of outfield players
            
        Returns:
            Line type string
        """
        num_lines = len(existing_lines)

        if num_lines == 0:
            return 'defense'
        elif num_lines == 1:
            return 'midfield'
        elif num_lines == 2:
            return 'attack'
        else:
            return 'midfield'  # Additional lines classified as midfield

    
    # Valid formation ranges - used to filter out impossible formations
    VALID_DEFENDER_RANGE = (2, 6)  # Min 2, max 6 defenders
    VALID_MIDFIELD_RANGE = (2, 6)  # Min 2, max 6 midfielders
    VALID_ATTACKER_RANGE = (1, 5)  # Min 1, max 5 attackers
    VALID_TOTAL_OUTFIELD = (9, 11)  # Should have 9-11 outfield players

    def _is_valid_formation_counts(self, d: int, m: int, a: int) -> bool:
        """Check if formation counts are within valid ranges."""
        total = d + m + a
        return (
            self.VALID_DEFENDER_RANGE[0] <= d <= self.VALID_DEFENDER_RANGE[1] and
            self.VALID_MIDFIELD_RANGE[0] <= m <= self.VALID_MIDFIELD_RANGE[1] and
            self.VALID_ATTACKER_RANGE[0] <= a <= self.VALID_ATTACKER_RANGE[1] and
            self.VALID_TOTAL_OUTFIELD[0] <= total <= self.VALID_TOTAL_OUTFIELD[1]
        )

    def _normalize_to_valid_formation(self, d: int, m: int, a: int) -> Tuple[str, float]:
        """
        Normalize invalid formation counts to closest valid formation.
        Uses proportional distribution to ensure realistic formations.
        """
        total = d + m + a
        
        # If total is way off, we can't make a valid formation
        if total < 5 or total > 15:
            return 'unknown', 0.0
        
        # Proportionally distribute to valid ranges
        # Target: 10 outfield players (4-3-3, 4-4-2, etc.)
        target_total = 10
        
        # Calculate proportions
        if total > 0:
            d_ratio = d / total
            m_ratio = m / total
            a_ratio = a / total
        else:
            return 'unknown', 0.0
        
        # Apply proportions to target total, then clamp to valid ranges
        d_norm = max(self.VALID_DEFENDER_RANGE[0], min(self.VALID_DEFENDER_RANGE[1], round(d_ratio * target_total)))
        m_norm = max(self.VALID_MIDFIELD_RANGE[0], min(self.VALID_MIDFIELD_RANGE[1], round(m_ratio * target_total)))
        a_norm = max(self.VALID_ATTACKER_RANGE[0], min(self.VALID_ATTACKER_RANGE[1], round(a_ratio * target_total)))
        
        # Adjust to ensure total is exactly 10
        current_total = d_norm + m_norm + a_norm
        if current_total != target_total:
            # Adjust midfielders first (most flexible)
            diff = target_total - current_total
            m_norm = max(self.VALID_MIDFIELD_RANGE[0], min(self.VALID_MIDFIELD_RANGE[1], m_norm + diff))
            
            # Recalculate and adjust attackers if needed
            current_total = d_norm + m_norm + a_norm
            if current_total != target_total:
                diff = target_total - current_total
                a_norm = max(self.VALID_ATTACKER_RANGE[0], min(self.VALID_ATTACKER_RANGE[1], a_norm + diff))
        
        # Map to closest standard formation
        formation_code = f"{d_norm}-{m_norm}-{a_norm}"
        
        # Check if this matches a known formation
        if formation_code in self.FORMATION_TEMPLATES:
            return formation_code, 0.85
        
        # Try to find closest match
        return self._find_closest_standard_formation(d_norm, m_norm, a_norm)

    def _find_closest_standard_formation(self, d: int, m: int, a: int) -> Tuple[str, float]:
        """Find the closest standard formation to the given counts."""
        best_match = '4-3-3'  # Default fallback
        best_score = 0.0
        
        for formation, template in self.FORMATION_TEMPLATES.items():
            score = 0
            score += 3 - abs(d - template['defense'])
            score += 3 - abs(m - template['midfield'])
            score += 3 - abs(a - template['attack'])
            
            if score > best_score:
                best_score = score
                best_match = formation
        
        confidence = min(0.8, best_score / 9.0)  # Max confidence 0.8 for normalized formations
        return best_match, round(confidence, 2)

    def _classify_formation(self, lines: List[FormationLine]) -> Tuple[str, float]:
        """
        Classify formation with improved fuzzy matching and validation.
        
        Returns:
            Tuple of (formation_code, confidence)
        """
        if not lines:
            return 'unknown', 0.0

        # Count players in each line type
        line_counts = {'defense': 0, 'midfield': 0, 'attack': 0}
        for line in lines:
            if line.line_type in line_counts:
                line_counts[line.line_type] = len(line.players)

        total_players = sum(line_counts.values())
        
        # Need minimum players for detection
        if total_players < 7:
            return 'unknown', 0.0

        d = line_counts['defense']
        m = line_counts['midfield']
        a = line_counts['attack']
        
        # Check if counts form a valid formation
        if not self._is_valid_formation_counts(d, m, a):
            # Normalize to valid formation
            return self._normalize_to_valid_formation(d, m, a)

        # Find best matching formation with fuzzy matching
        best_match = 'unknown'
        best_score = 0.0
        
        for formation, template in self.FORMATION_TEMPLATES.items():
            score = 0
            matches = 0
            
            for line_type in ['defense', 'midfield', 'attack']:
                detected = line_counts.get(line_type, 0)
                expected = template.get(line_type, 0)
                diff = abs(detected - expected)
                
                # Perfect match
                if diff == 0:
                    score += 3
                    matches += 1
                # Within tolerance
                elif diff <= self.FORMATION_TOLERANCE:
                    score += 1
                    matches += 0.5
                # Poor match
                else:
                    score -= diff
            
            # Normalize score
            max_possible = 9  # 3 perfect matches
            confidence = max(0, (score + 3) / (max_possible + 3))  # Shift to 0-1 range
            
            if confidence > best_score:
                best_score = confidence
                best_match = formation

        # If confidence is too low, try to find closest valid formation
        if best_score < 0.3:
            return self._normalize_to_valid_formation(d, m, a)

        return best_match, round(best_score, 2)


    def get_current_formation(self, team: str) -> Optional[str]:
        """
        Get current formation code for a team.
        
        If current formation is unknown or None, returns the most common
        formation detected during the match for that team.
        """
        if team in self._current_formations:
            formation = self._current_formations[team].formation_code
            if formation and formation != 'unknown':
                return formation
        
        # Fallback to most common formation
        return self.get_most_common_formation(team)
    
    def get_most_common_formation(self, team: str) -> Optional[str]:
        """
        Get the most common formation for a team during the match.
        
        Args:
            team: Team label ('A' or 'B')
            
        Returns:
            Most common formation code or None if no data
        """
        if not self._formation_history:
            return None
        
        # Filter formations for this team
        team_formations = [
            snap for snap in self._formation_history 
            if snap.team == team and snap.formation_code != 'unknown'
        ]
        
        if not team_formations:
            return None
        
        # Count formations
        formation_counts = {}
        for snap in team_formations:
            code = snap.formation_code
            formation_counts[code] = formation_counts.get(code, 0) + 1
        
        if not formation_counts:
            return None
        
        # Return most common
        most_common = max(formation_counts.items(), key=lambda x: x[1])[0]
        return most_common

    def get_formation_history(self, team: str = None) -> List[FormationSnapshot]:
        """Get formation history, optionally filtered by team."""
        if team is None:
            return self._formation_history.copy()
        return [f for f in self._formation_history if f.team == team]

    def get_formation_changes(self) -> List[Dict[str, Any]]:
        """Get list of formation changes during match."""
        return self._formation_changes.copy()

    def get_formation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive formation statistics."""
        if not self._formation_history:
            return {
                'most_common': 'unknown',
                'formation_counts': {},
                'avg_confidence': 0.0,
                'formation_changes': 0
            }

        # Count formations
        formation_counts = {}
        for snapshot in self._formation_history:
            code = snapshot.formation_code
            formation_counts[code] = formation_counts.get(code, 0) + 1

        # Most common formation
        most_common = max(formation_counts.items(), key=lambda x: x[1])[0]

        # Average confidence
        avg_confidence = sum(f.confidence for f in self._formation_history) / len(self._formation_history)

        return {
            'most_common': most_common,
            'formation_counts': formation_counts,
            'avg_confidence': round(avg_confidence, 3),
            'formation_changes': len(self._formation_changes),
            'current_formations': {
                team: snap.formation_code
                for team, snap in self._current_formations.items()
            }
        }

    def get_team_shape_metrics(self, team: str) -> Dict[str, Any]:
        """
        Get team shape metrics (compactness, width, depth).
        
        Args:
            team: Team label
            
        Returns:
            Dictionary of shape metrics
        """
        if team not in self._current_formations:
            return {}

        snapshot = self._current_formations[team]
        lines = snapshot.lines

        if not lines:
            return {}

        # Calculate depth (distance between defense and attack)
        defense_line = next((l for l in lines if l.line_type == 'defense'), None)
        attack_line = next((l for l in lines if l.line_type == 'attack'), None)

        depth = 0.0
        if defense_line and attack_line:
            depth = abs(attack_line.avg_x - defense_line.avg_x)

        # Calculate width (average y-spread across all lines)
        total_width = sum(l.y_spread for l in lines)
        avg_width = total_width / len(lines) if lines else 0

        # Compactness (how close are the lines)
        if len(lines) >= 2:
            line_positions = [l.avg_x for l in lines]
            line_positions.sort()
            gaps = [line_positions[i+1] - line_positions[i] for i in range(len(line_positions)-1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            compactness = 1.0 / (1.0 + avg_gap / 100)  # Normalized compactness
        else:
            compactness = 1.0

        return {
            'depth_px': round(depth, 1),
            'avg_width_px': round(avg_width, 1),
            'compactness': round(compactness, 3),
            'num_lines': len(lines)
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        self._position_history.clear()
        self._formation_history.clear()
        self._current_formations.clear()
        self._formation_changes.clear()
        self._player_teams.clear()
