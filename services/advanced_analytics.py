"""
Advanced Analytics Module

Provides sophisticated football analytics including:
- Passing network analysis with centrality metrics
- Zone control analysis
- Pressure index calculation
- Ball progression analysis
- Defensive actions tracking

Uses network theory and spatial analysis for deep insights.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from enum import Enum


class Zone(Enum):
    """Pitch zones."""
    DEF_LEFT = "defensive_left"
    DEF_CENTER = "defensive_center"
    DEF_RIGHT = "defensive_right"
    MID_LEFT = "midfield_left"
    MID_CENTER = "midfield_center"
    MID_RIGHT = "midfield_right"
    ATT_LEFT = "attacking_left"
    ATT_CENTER = "attacking_center"
    ATT_RIGHT = "attacking_right"


@dataclass
class PassingNetwork:
    """Passing network for a team."""
    team: str
    nodes: Set[int] = field(default_factory=set)
    edges: Dict[Tuple[int, int], Dict] = field(default_factory=dict)
    
    # Centrality metrics
    degree_centrality: Dict[int, float] = field(default_factory=dict)
    betweenness_centrality: Dict[int, float] = field(default_factory=dict)
    closeness_centrality: Dict[int, float] = field(default_factory=dict)
    
    # Network properties
    density: float = 0.0
    clustering_coefficient: float = 0.0
    avg_path_length: float = 0.0


@dataclass
class ZoneControl:
    """Zone control metrics."""
    zone: Zone
    team_a_control: float = 0.0  # Percentage
    team_b_control: float = 0.0
    contested: float = 0.0  # Neither team in clear control
    
    # Activity metrics
    passes_in_zone: Dict[str, int] = field(default_factory=dict)
    touches_in_zone: Dict[str, int] = field(default_factory=dict)
    time_in_zone: Dict[str, float] = field(default_factory=dict)


@dataclass
class PressureEvent:
    """Represents a pressure event."""
    timestamp: float
    frame: int
    pressing_team: str
    target_team: str
    intensity: float  # 0-100
    duration: float
    players_involved: int
    recovered_possession: bool
    zone: Zone


@dataclass
class BallProgression:
    """Ball progression event."""
    timestamp: float
    frame: int
    team: str
    start_zone: Zone
    end_zone: Zone
    
    # Metrics
    distance: float  # meters
    time_taken: float  # seconds
    passes_involved: int
    players_involved: int
    successful: bool
    
    # Context
    under_pressure: bool
    method: str  # 'pass', 'dribble', 'long_ball'


@dataclass
class DefensiveAction:
    """Defensive action event."""
    timestamp: float
    frame: int
    player_id: int
    team: str
    action_type: str  # 'tackle', 'interception', 'block', 'clearance'
    
    # Context
    zone: Zone
    successful: bool
    led_to_possession: bool
    xg_prevented: float = 0.0


class AdvancedAnalytics:
    """
    Advanced Analytics Engine.
    
    Provides sophisticated analysis of football matches:
    - Network analysis of passing patterns
    - Spatial control of pitch zones
    - Pressure intensity tracking
    - Ball progression efficiency
    - Defensive effectiveness
    """
    
    # Pitch dimensions
    PITCH_LENGTH = 105.0  # meters
    PITCH_WIDTH = 68.0
    
    # Zone boundaries (normalized 0-1)
    ZONE_BOUNDS = {
        Zone.DEF_LEFT: (0.0, 0.0, 0.33, 0.33),
        Zone.DEF_CENTER: (0.0, 0.33, 0.33, 0.67),
        Zone.DEF_RIGHT: (0.0, 0.67, 0.33, 1.0),
        Zone.MID_LEFT: (0.33, 0.0, 0.67, 0.33),
        Zone.MID_CENTER: (0.33, 0.33, 0.67, 0.67),
        Zone.MID_RIGHT: (0.33, 0.67, 0.67, 1.0),
        Zone.ATT_LEFT: (0.67, 0.0, 1.0, 0.33),
        Zone.ATT_CENTER: (0.67, 0.33, 1.0, 0.67),
        Zone.ATT_RIGHT: (0.67, 0.67, 1.0, 1.0),
    }
    
    def __init__(self):
        """Initialize the advanced analytics engine."""
        # Passing networks
        self.passing_networks: Dict[str, PassingNetwork] = {}
        
        # Zone control
        self.zone_control: Dict[Zone, ZoneControl] = {
            zone: ZoneControl(zone=zone) for zone in Zone
        }
        
        # Pressure tracking
        self.pressure_events: List[PressureEvent] = []
        self.pressure_index: Dict[str, List[Tuple[float, float]]] = {
            'A': [], 'B': []  # (timestamp, intensity)
        }
        
        # Ball progression
        self.progression_events: List[BallProgression] = []
        
        # Defensive actions
        self.defensive_actions: List[DefensiveAction] = []
        
        # Tracking data for analysis
        self.position_history: List[Dict] = []
        self.possession_history: List[Dict] = []
    
    def get_zone(self, x: float, y: float) -> Zone:
        """Determine which zone a position falls into."""
        for zone, (x1, y1, x2, y2) in self.ZONE_BOUNDS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone
        return Zone.MID_CENTER  # Default
    
    # ============================================================
    # PASSING NETWORK ANALYSIS
    # ============================================================
    
    def build_passing_network(
        self,
        team: str,
        passes: List[Dict],
        min_passes: int = 2
    ) -> PassingNetwork:
        """
        Build a passing network from pass data.
        
        Args:
            team: Team identifier
            passes: List of pass events
            min_passes: Minimum passes for an edge
            
        Returns:
            PassingNetwork object
        """
        network = PassingNetwork(team=team)
        
        # Count passes between players
        pass_counts = defaultdict(lambda: defaultdict(int))
        pass_success = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0}))
        
        for pass_event in passes:
            if pass_event.get('team') != team:
                continue
                
            from_player = pass_event.get('from_player')
            to_player = pass_event.get('to_player')
            
            if from_player is None or to_player is None:
                continue
            
            network.nodes.add(from_player)
            network.nodes.add(to_player)
            
            pass_counts[from_player][to_player] += 1
            pass_success[from_player][to_player]['total'] += 1
            
            if pass_event.get('success', False):
                pass_success[from_player][to_player]['success'] += 1
        
        # Build edges
        for from_player, targets in pass_counts.items():
            for to_player, count in targets.items():
                if count >= min_passes:
                    edge_key = (from_player, to_player)
                    success_rate = (
                        pass_success[from_player][to_player]['success'] /
                        pass_success[from_player][to_player]['total']
                    )
                    
                    network.edges[edge_key] = {
                        'weight': count,
                        'success_rate': success_rate,
                        'from': from_player,
                        'to': to_player
                    }
        
        # Calculate centrality metrics
        network.degree_centrality = self._calculate_degree_centrality(network)
        network.betweenness_centrality = self._calculate_betweenness_centrality(network)
        network.closeness_centrality = self._calculate_closeness_centrality(network)
        
        # Calculate network properties
        network.density = self._calculate_network_density(network)
        
        self.passing_networks[team] = network
        return network
    
    def _calculate_degree_centrality(self, network: PassingNetwork) -> Dict[int, float]:
        """Calculate degree centrality for each node."""
        centrality = {}
        n = len(network.nodes)
        
        if n <= 1:
            return {node: 0.0 for node in network.nodes}
        
        for node in network.nodes:
            # Count connections (both in and out)
            degree = 0
            for (from_n, to_n), edge in network.edges.items():
                if from_n == node or to_n == node:
                    degree += edge['weight']
            
            # Normalize
            centrality[node] = degree / (n - 1)
        
        return centrality
    
    def _calculate_betweenness_centrality(self, network: PassingNetwork) -> Dict[int, float]:
        """Calculate betweenness centrality (simplified)."""
        centrality = {node: 0.0 for node in network.nodes}
        
        if len(network.nodes) <= 2:
            return centrality
        
        # For each pair of nodes, find shortest path
        nodes = list(network.nodes)
        for i, source in enumerate(nodes):
            for target in nodes[i+1:]:
                path = self._find_shortest_path(network, source, target)
                if path:
                    # Increment centrality for intermediate nodes
                    for node in path[1:-1]:
                        centrality[node] += 1
        
        # Normalize
        n = len(network.nodes)
        normalizer = (n - 1) * (n - 2) / 2
        if normalizer > 0:
            for node in centrality:
                centrality[node] /= normalizer
        
        return centrality
    
    def _calculate_closeness_centrality(self, network: PassingNetwork) -> Dict[int, float]:
        """Calculate closeness centrality."""
        centrality = {}
        
        for node in network.nodes:
            # Calculate average shortest path to all other nodes
            total_distance = 0
            reachable = 0
            
            for other in network.nodes:
                if other != node:
                    distance = self._shortest_path_length(network, node, other)
                    if distance is not None:
                        total_distance += distance
                        reachable += 1
            
            if reachable > 0 and total_distance > 0:
                centrality[node] = reachable / total_distance
            else:
                centrality[node] = 0.0
        
        return centrality
    
    def _find_shortest_path(
        self,
        network: PassingNetwork,
        source: int,
        target: int
    ) -> Optional[List[int]]:
        """Find shortest path using BFS."""
        if source == target:
            return [source]
        
        visited = {source}
        queue = [(source, [source])]
        
        while queue:
            current, path = queue.pop(0)
            
            # Find neighbors
            for (from_n, to_n) in network.edges.keys():
                if from_n == current and to_n not in visited:
                    if to_n == target:
                        return path + [to_n]
                    visited.add(to_n)
                    queue.append((to_n, path + [to_n]))
        
        return None
    
    def _shortest_path_length(
        self,
        network: PassingNetwork,
        source: int,
        target: int
    ) -> Optional[int]:
        """Get shortest path length."""
        path = self._find_shortest_path(network, source, target)
        return len(path) - 1 if path else None
    
    def _calculate_network_density(self, network: PassingNetwork) -> float:
        """Calculate network density."""
        n = len(network.nodes)
        if n <= 1:
            return 0.0
        
        max_edges = n * (n - 1)
        actual_edges = len(network.edges)
        
        return actual_edges / max_edges if max_edges > 0 else 0.0
    
    def get_key_players(self, team: str, top_n: int = 3) -> Dict[str, List[int]]:
        """
        Identify key players by different centrality metrics.
        
        Returns:
            Dictionary with key players by category
        """
        if team not in self.passing_networks:
            return {}
        
        network = self.passing_networks[team]
        
        # Sort by each centrality metric
        by_degree = sorted(
            network.degree_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        by_betweenness = sorted(
            network.betweenness_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        by_closeness = sorted(
            network.closeness_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return {
            'most_connected': [p[0] for p in by_degree],
            'playmakers': [p[0] for p in by_betweenness],
            'influential': [p[0] for p in by_closeness]
        }
    
    # ============================================================
    # ZONE CONTROL ANALYSIS
    # ============================================================
    
    def update_zone_control(
        self,
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_position: Tuple[float, float],
        possessing_team: Optional[str],
        timestamp: float
    ):
        """
        Update zone control based on player positions.
        
        Args:
            player_positions: {player_id: (x, y, team)}
            ball_position: (x, y) ball position
            possessing_team: Team in possession
            timestamp: Current timestamp
        """
        
        # Count players in each zone (exclude referees)
        zone_counts = defaultdict(lambda: {'A': 0, 'B': 0})

        for player_id, (x, y, team) in player_positions.items():
            if team not in ('A', 'B'):
                continue  # Skip referees and other non-team entities
            zone = self.get_zone(x, y)
            zone_counts[zone][team] += 1

        
        # Update control percentages
        for zone in Zone:
            counts = zone_counts[zone]
            total = counts['A'] + counts['B']
            
            if total > 0:
                control = self.zone_control[zone]
                control.team_a_control = (counts['A'] / total) * 100
                control.team_b_control = (counts['B'] / total) * 100
                
                # Contested if close to equal
                diff = abs(control.team_a_control - control.team_b_control)
                control.contested = 100 - diff
                
                # Update possession time
                if possessing_team:
                    if possessing_team not in control.time_in_zone:
                        control.time_in_zone[possessing_team] = 0
                    control.time_in_zone[possessing_team] += 0.1  # Assuming 10Hz
    
    def get_zone_control_summary(self) -> Dict[str, Any]:
        """Get summary of zone control."""
        summary = {
            'by_zone': {},
            'team_a_dominance': [],
            'team_b_dominance': [],
            'contested_zones': []
        }
        
        for zone, control in self.zone_control.items():
            summary['by_zone'][zone.value] = {
                'team_a': round(control.team_a_control, 1),
                'team_b': round(control.team_b_control, 1),
                'contested': round(control.contested, 1)
            }
            
            if control.team_a_control > 60:
                summary['team_a_dominance'].append(zone.value)
            elif control.team_b_control > 60:
                summary['team_b_dominance'].append(zone.value)
            elif control.contested > 40:
                summary['contested_zones'].append(zone.value)
        
        return summary
    
    # ============================================================
    # PRESSURE INDEX
    # ============================================================
    
    def calculate_pressure(
        self,
        timestamp: float,
        frame: int,
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_position: Tuple[float, float],
        possessing_team: str,
        possessing_player: int
    ) -> float:
        """
        Calculate pressure index on the ball carrier.
        
        Returns:
            Pressure index (0-100)
        """
        if possessing_player not in player_positions:
            return 0.0
        
        px, py, _ = player_positions[possessing_player]
        opposing_team = 'B' if possessing_team == 'A' else 'A'
        
        # Find opponents within pressing distance
        pressing_distance = 5.0  # meters (normalized ~0.05)
        pressing_players = 0
        total_pressure = 0.0
        
        for player_id, (x, y, team) in player_positions.items():
            if team != opposing_team:
                continue
            
            distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            
            if distance < pressing_distance:
                pressing_players += 1
                # Closer players = more pressure
                pressure_contribution = 1 - (distance / pressing_distance)
                total_pressure += pressure_contribution
        
        # Calculate pressure index
        # Base pressure from number of pressing players
        base_pressure = min(pressing_players * 25, 75)
        
        # Additional pressure from proximity
        proximity_pressure = min(total_pressure * 10, 25)
        
        pressure_index = base_pressure + proximity_pressure
        
        # Record event
        event = PressureEvent(
            timestamp=timestamp,
            frame=frame,
            pressing_team=opposing_team,
            target_team=possessing_team,
            intensity=pressure_index,
            duration=0.1,  # Will be updated
            players_involved=pressing_players,
            recovered_possession=False,  # Will be updated
            zone=self.get_zone(px, py)
        )
        
        self.pressure_events.append(event)
        self.pressure_index[opposing_team].append((timestamp, pressure_index))
        
        return pressure_index
    
    def get_pressure_statistics(self) -> Dict[str, Any]:
        """Get pressure statistics."""
        stats = {'A': {}, 'B': {}}
        
        for team in ['A', 'B']:
            pressures = [p for p in self.pressure_events if p.pressing_team == team]
            
            if pressures:
                intensities = [p.intensity for p in pressures]
                stats[team] = {
                    'total_pressures': len(pressures),
                    'avg_intensity': round(np.mean(intensities), 1),
                    'max_intensity': round(max(intensities), 1),
                    'high_intensity_pressures': len([i for i in intensities if i > 60]),
                    'possession_won': len([p for p in pressures if p.recovered_possession])
                }
            else:
                stats[team] = {
                    'total_pressures': 0,
                    'avg_intensity': 0,
                    'max_intensity': 0,
                    'high_intensity_pressures': 0,
                    'possession_won': 0
                }
        
        return stats
    
    # ============================================================
    # BALL PROGRESSION
    # ============================================================
    
    def track_progression(
        self,
        timestamp: float,
        frame: int,
        team: str,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        time_taken: float,
        passes: int,
        players: int,
        successful: bool,
        under_pressure: bool,
        method: str
    ) -> BallProgression:
        """Track a ball progression event."""
        
        start_zone = self.get_zone(start_pos[0], start_pos[1])
        end_zone = self.get_zone(end_pos[0], end_pos[1])
        
        # Calculate distance
        dx = (end_pos[0] - start_pos[0]) * self.PITCH_LENGTH
        dy = (end_pos[1] - start_pos[1]) * self.PITCH_WIDTH
        distance = np.sqrt(dx ** 2 + dy ** 2)
        
        progression = BallProgression(
            timestamp=timestamp,
            frame=frame,
            team=team,
            start_zone=start_zone,
            end_zone=end_zone,
            distance=distance,
            time_taken=time_taken,
            passes_involved=passes,
            players_involved=players,
            successful=successful,
            under_pressure=under_pressure,
            method=method
        )
        
        self.progression_events.append(progression)
        return progression
    
    def get_progression_statistics(self) -> Dict[str, Any]:
        """Get ball progression statistics."""
        stats = {'A': {}, 'B': {}}
        
        for team in ['A', 'B']:
            team_progressions = [p for p in self.progression_events if p.team == team]
            
            if team_progressions:
                successful = [p for p in team_progressions if p.successful]
                under_pressure = [p for p in team_progressions if p.under_pressure]
                
                stats[team] = {
                    'total_progressions': len(team_progressions),
                    'successful': len(successful),
                    'success_rate': round(len(successful) / len(team_progressions) * 100, 1),
                    'avg_distance': round(np.mean([p.distance for p in team_progressions]), 1),
                    'avg_time': round(np.mean([p.time_taken for p in team_progressions]), 2),
                    'avg_passes': round(np.mean([p.passes_involved for p in team_progressions]), 1),
                    'under_pressure': len(under_pressure),
                    'by_method': self._progressions_by_method(team_progressions)
                }
            else:
                stats[team] = {
                    'total_progressions': 0,
                    'successful': 0,
                    'success_rate': 0,
                    'avg_distance': 0,
                    'avg_time': 0,
                    'avg_passes': 0,
                    'under_pressure': 0,
                    'by_method': {}
                }
        
        return stats
    
    def _progressions_by_method(self, progressions: List[BallProgression]) -> Dict[str, int]:
        """Count progressions by method."""
        counts = defaultdict(int)
        for p in progressions:
            counts[p.method] += 1
        return dict(counts)
    
    # ============================================================
    # DEFENSIVE ACTIONS
    # ============================================================
    
    def record_defensive_action(
        self,
        timestamp: float,
        frame: int,
        player_id: int,
        team: str,
        action_type: str,
        position: Tuple[float, float],
        successful: bool,
        led_to_possession: bool,
        xg_prevented: float = 0.0
    ) -> DefensiveAction:
        """Record a defensive action."""
        
        zone = self.get_zone(position[0], position[1])
        
        action = DefensiveAction(
            timestamp=timestamp,
            frame=frame,
            player_id=player_id,
            team=team,
            action_type=action_type,
            zone=zone,
            successful=successful,
            led_to_possession=led_to_possession,
            xg_prevented=xg_prevented
        )
        
        self.defensive_actions.append(action)
        return action
    
    def get_defensive_statistics(self) -> Dict[str, Any]:
        """Get defensive action statistics."""
        stats = {'A': {}, 'B': {}}
        
        for team in ['A', 'B']:
            team_actions = [a for a in self.defensive_actions if a.team == team]
            
            if team_actions:
                successful = [a for a in team_actions if a.successful]
                led_to_poss = [a for a in team_actions if a.led_to_possession]
                
                # By type
                by_type = defaultdict(lambda: {'total': 0, 'successful': 0})
                for action in team_actions:
                    by_type[action.action_type]['total'] += 1
                    if action.successful:
                        by_type[action.action_type]['successful'] += 1
                
                stats[team] = {
                    'total_actions': len(team_actions),
                    'successful': len(successful),
                    'success_rate': round(len(successful) / len(team_actions) * 100, 1),
                    'led_to_possession': len(led_to_poss),
                    'xg_prevented': round(sum(a.xg_prevented for a in team_actions), 2),
                    'by_type': dict(by_type),
                    'by_zone': self._actions_by_zone(team_actions)
                }
            else:
                stats[team] = {
                    'total_actions': 0,
                    'successful': 0,
                    'success_rate': 0,
                    'led_to_possession': 0,
                    'xg_prevented': 0,
                    'by_type': {},
                    'by_zone': {}
                }
        
        return stats
    
    def _actions_by_zone(self, actions: List[DefensiveAction]) -> Dict[str, int]:
        """Count actions by zone."""
        counts = defaultdict(int)
        for a in actions:
            counts[a.zone.value] += 1
        return dict(counts)
    
    # ============================================================
    # COMPREHENSIVE REPORT
    # ============================================================
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive advanced analytics report."""
        return {
            'passing_networks': {
                team: {
                    'nodes': list(network.nodes),
                    'edges': len(network.edges),
                    'density': round(network.density, 3),
                    'key_players': self.get_key_players(team)
                }
                for team, network in self.passing_networks.items()
            },
            'zone_control': self.get_zone_control_summary(),
            'pressure': self.get_pressure_statistics(),
            'ball_progression': self.get_progression_statistics(),
            'defensive_actions': self.get_defensive_statistics()
        }
    
    def reset(self):
        """Reset all data."""
        self.passing_networks = {}
        self.zone_control = {zone: ZoneControl(zone=zone) for zone in Zone}
        self.pressure_events = []
        self.pressure_index = {'A': [], 'B': []}
        self.progression_events = []
        self.defensive_actions = []
        self.position_history = []
        self.possession_history = []
