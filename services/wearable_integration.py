"""
Wearable Data Integration Module

Handles integration with GPS vests, heart rate monitors, and accelerometer data
for comprehensive player performance tracking.
"""

import json
import csv
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

__all__ = ['WearableIntegrationManager', 'GPSDataPoint', 'PlayerLoadMetrics', 
           'WearableDataImporter', 'PlayerLoadCalculator', 'SprintAnalyzer',
           'create_wearable_integration']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPSDataPoint:
    """Single GPS data point from wearable device."""
    timestamp: float
    player_id: str
    x: float  # Pitch X coordinate (0-100)
    y: float  # Pitch Y coordinate (0-100)
    speed: float  # km/h
    distance: float  # meters from start
    acceleration: float  # m/s²
    heart_rate: Optional[int] = None
    
    
@dataclass
class PlayerLoadMetrics:
    """Calculated load and fatigue metrics for a player."""
    player_id: str
    total_distance: float
    sprint_distance: float
    high_intensity_runs: int
    avg_speed: float
    max_speed: float
    avg_heart_rate: float
    max_heart_rate: int
    metabolic_power: float
    player_load: float  # Arbitrary load unit
    fatigue_index: float  # 0-100 scale
    timestamp: datetime


class WearableDataImporter:
    """Import and process data from GPS vests and wearables."""
    
    def __init__(self):
        self.gps_data: Dict[str, List[GPSDataPoint]] = {}
        self.heart_rate_data: Dict[str, List[Tuple[float, int]]] = {}
        self.accelerometer_data: Dict[str, List[Tuple[float, float, float, float]]] = {}
        self.video_sync_offset: float = 0.0
        
    def import_catapult_data(self, file_path: str, player_mapping: Dict[str, str]) -> bool:
        """
        Import data from Catapult GPS vests (common format).
        
        Args:
            file_path: Path to CSV or JSON file
            player_mapping: Dict mapping device IDs to player IDs
            
        Returns:
            True if import successful
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            if path.suffix.lower() == '.csv':
                return self._import_catapult_csv(file_path, player_mapping)
            elif path.suffix.lower() == '.json':
                return self._import_catapult_json(file_path, player_mapping)
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return False
                
        except Exception as e:
            logger.error(f"Error importing Catapult data: {e}")
            return False
            
    def _import_catapult_csv(self, file_path: str, player_mapping: Dict[str, str]) -> bool:
        """Import Catapult CSV format."""
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                device_id = row.get('device_id', row.get('Device ID', ''))
                player_id = player_mapping.get(device_id, device_id)
                
                if player_id not in self.gps_data:
                    self.gps_data[player_id] = []
                    
                try:
                    point = GPSDataPoint(
                        timestamp=float(row.get('time', row.get('Time', 0))),
                        player_id=player_id,
                        x=float(row.get('x', row.get('X', 0))),
                        y=float(row.get('y', row.get('Y', 0))),
                        speed=float(row.get('speed', row.get('Speed', 0))),
                        distance=float(row.get('distance', row.get('Distance', 0))),
                        acceleration=float(row.get('accel', row.get('Acceleration', 0))),
                        heart_rate=int(row.get('hr', row.get('Heart Rate', 0))) if row.get('hr') else None
                    )
                    self.gps_data[player_id].append(point)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue
                    
        logger.info(f"Imported GPS data for {len(self.gps_data)} players")
        return True
        
    def _import_catapult_json(self, file_path: str, player_mapping: Dict[str, str]) -> bool:
        """Import Catapult JSON format."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for device_data in data.get('devices', []):
            device_id = device_data.get('device_id')
            player_id = player_mapping.get(device_id, device_id)
            
            if player_id not in self.gps_data:
                self.gps_data[player_id] = []
                
            for point in device_data.get('data', []):
                gps_point = GPSDataPoint(
                    timestamp=point.get('timestamp', 0),
                    player_id=player_id,
                    x=point.get('x', 0),
                    y=point.get('y', 0),
                    speed=point.get('speed', 0),
                    distance=point.get('distance', 0),
                    acceleration=point.get('acceleration', 0),
                    heart_rate=point.get('heart_rate')
                )
                self.gps_data[player_id].append(gps_point)
                
        logger.info(f"Imported GPS data for {len(self.gps_data)} players")
        return True
        
    def import_polar_hr(self, file_path: str, player_id: str) -> bool:
        """
        Import heart rate data from Polar H10 or similar monitors.
        
        Args:
            file_path: Path to heart rate data file
            player_id: Player identifier
            
        Returns:
            True if import successful
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
                
            hr_data = []
            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = float(row.get('time', row.get('timestamp', 0)))
                    hr = int(row.get('hr', row.get('heart_rate', 0)))
                    hr_data.append((timestamp, hr))
                    
            self.heart_rate_data[player_id] = hr_data
            logger.info(f"Imported {len(hr_data)} HR data points for {player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing HR data: {e}")
            return False
            
    def import_accelerometer_data(self, file_path: str, player_id: str) -> bool:
        """
        Import accelerometer data for sprint analysis.
        
        Args:
            file_path: Path to accelerometer data file
            player_id: Player identifier
            
        Returns:
            True if import successful
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
                
            accel_data = []
            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = float(row.get('time', 0))
                    x = float(row.get('x', row.get('accel_x', 0)))
                    y = float(row.get('y', row.get('accel_y', 0)))
                    z = float(row.get('z', row.get('accel_z', 0)))
                    magnitude = np.sqrt(x**2 + y**2 + z**2)
                    accel_data.append((timestamp, x, y, z, magnitude))
                    
            self.accelerometer_data[player_id] = accel_data
            logger.info(f"Imported {len(accel_data)} accelerometer points for {player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing accelerometer data: {e}")
            return False
            
    def sync_with_video(self, wearable_timestamp: float, video_timestamp: float):
        """
        Synchronize wearable data with video timestamps.
        
        Args:
            wearable_timestamp: Reference timestamp from wearable data
            video_timestamp: Corresponding timestamp in video
        """
        self.video_sync_offset = video_timestamp - wearable_timestamp
        logger.info(f"Video sync offset set to {self.video_sync_offset:.3f}s")
        
    def get_player_position_at_time(self, player_id: str, video_time: float) -> Optional[Tuple[float, float]]:
        """
        Get player position at a specific video timestamp.
        
        Args:
            player_id: Player identifier
            video_time: Timestamp in video
            
        Returns:
            (x, y) coordinates or None
        """
        if player_id not in self.gps_data:
            return None
            
        wearable_time = video_time - self.video_sync_offset
        points = self.gps_data[player_id]
        
        # Find closest data point
        closest = min(points, key=lambda p: abs(p.timestamp - wearable_time))
        
        if abs(closest.timestamp - wearable_time) < 1.0:  # Within 1 second
            return (closest.x, closest.y)
        return None
        
    def get_player_speed_at_time(self, player_id: str, video_time: float) -> Optional[float]:
        """Get player speed at specific video time."""
        if player_id not in self.gps_data:
            return None
            
        wearable_time = video_time - self.video_sync_offset
        points = self.gps_data[player_id]
        closest = min(points, key=lambda p: abs(p.timestamp - wearable_time))
        
        if abs(closest.timestamp - wearable_time) < 1.0:
            return closest.speed
        return None


class PlayerLoadCalculator:
    """Calculate player load and fatigue metrics from wearable data."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[PlayerLoadMetrics]] = {}
        
    def calculate_load_metrics(self, 
                              player_id: str, 
                              gps_data: List[GPSDataPoint],
                              time_window: Optional[Tuple[float, float]] = None) -> PlayerLoadMetrics:
        """
        Calculate comprehensive load metrics for a player.
        
        Args:
            player_id: Player identifier
            gps_data: List of GPS data points
            time_window: Optional (start, end) time tuple
            
        Returns:
            PlayerLoadMetrics object
        """
        if time_window:
            start, end = time_window
            gps_data = [p for p in gps_data if start <= p.timestamp <= end]
            
        if not gps_data:
            return PlayerLoadMetrics(
                player_id=player_id,
                total_distance=0,
                sprint_distance=0,
                high_intensity_runs=0,
                avg_speed=0,
                max_speed=0,
                avg_heart_rate=0,
                max_heart_rate=0,
                metabolic_power=0,
                player_load=0,
                fatigue_index=0,
                timestamp=datetime.now()
            )
            
        # Distance metrics
        total_distance = gps_data[-1].distance - gps_data[0].distance if len(gps_data) > 1 else 0
        
        # Speed metrics
        speeds = [p.speed for p in gps_data]
        avg_speed = np.mean(speeds)
        max_speed = max(speeds)
        
        # Sprint analysis (> 24 km/h)
        sprint_threshold = 24.0
        sprint_distance = sum(
            self._calculate_distance_between_points(gps_data[i], gps_data[i+1])
            for i in range(len(gps_data)-1)
            if gps_data[i].speed >= sprint_threshold
        )
        
        # High intensity runs
        high_intensity_runs = sum(
            1 for i in range(len(gps_data)-1)
            if gps_data[i].speed >= 21.0 and gps_data[i+1].speed < 21.0
        )
        
        # Heart rate metrics
        hr_values = [p.heart_rate for p in gps_data if p.heart_rate is not None]
        avg_hr = np.mean(hr_values) if hr_values else 0
        max_hr = max(hr_values) if hr_values else 0
        
        # Metabolic power calculation (simplified)
        metabolic_power = self._calculate_metabolic_power(gps_data)
        
        # Player load (weighted combination)
        player_load = (
            total_distance * 0.3 +
            sprint_distance * 0.4 +
            high_intensity_runs * 2.0 +
            metabolic_power * 0.3
        )
        
        # Fatigue index (based on HR variability and speed decline)
        fatigue_index = self._calculate_fatigue_index(gps_data)
        
        metrics = PlayerLoadMetrics(
            player_id=player_id,
            total_distance=total_distance,
            sprint_distance=sprint_distance,
            high_intensity_runs=high_intensity_runs,
            avg_speed=avg_speed,
            max_speed=max_speed,
            avg_heart_rate=avg_hr,
            max_heart_rate=max_hr,
            metabolic_power=metabolic_power,
            player_load=player_load,
            fatigue_index=fatigue_index,
            timestamp=datetime.now()
        )
        
        # Store in history
        if player_id not in self.metrics_history:
            self.metrics_history[player_id] = []
        self.metrics_history[player_id].append(metrics)
        
        return metrics
        
    def _calculate_distance_between_points(self, p1: GPSDataPoint, p2: GPSDataPoint) -> float:
        """Calculate distance between two GPS points in meters."""
        # Assuming pitch is 105m x 68m
        dx = (p2.x - p1.x) * 1.05  # Convert percentage to meters
        dy = (p2.y - p1.y) * 0.68
        return np.sqrt(dx**2 + dy**2)
        
    def _calculate_metabolic_power(self, gps_data: List[GPSDataPoint]) -> float:
        """Calculate metabolic power using di Prampero equation."""
        if len(gps_data) < 2:
            return 0.0
            
        total_power = 0.0
        for i in range(len(gps_data) - 1):
            speed = gps_data[i].speed / 3.6  # Convert to m/s
            accel = gps_data[i].acceleration
            
            # Simplified metabolic power calculation
            # P = kv³ + ka * a * v
            kv = 0.01  # Air resistance coefficient
            ka = 1.0   # Acceleration coefficient
            
            power = kv * speed**3 + ka * accel * speed
            total_power += power
            
        return total_power / len(gps_data)
        
    def _calculate_fatigue_index(self, gps_data: List[GPSDataPoint]) -> float:
        """Calculate fatigue index based on performance decline."""
        if len(gps_data) < 10:
            return 0.0
            
        # Split data into halves
        mid = len(gps_data) // 2
        first_half = gps_data[:mid]
        second_half = gps_data[mid:]
        
        # Compare average speeds
        first_speed = np.mean([p.speed for p in first_half])
        second_speed = np.mean([p.speed for p in second_half])
        
        if first_speed > 0:
            decline = (first_speed - second_speed) / first_speed
            return min(100, max(0, decline * 100))
        return 0.0
        
    def get_fatigue_trend(self, player_id: str) -> List[Tuple[datetime, float]]:
        """Get fatigue trend over time for a player."""
        if player_id not in self.metrics_history:
            return []
            
        return [(m.timestamp, m.fatigue_index) for m in self.metrics_history[player_id]]
        
    def detect_overload_risk(self, player_id: str, threshold: float = 80.0) -> bool:
        """Detect if player is at risk of overload."""
        if player_id not in self.metrics_history or not self.metrics_history[player_id]:
            return False
            
        recent_metrics = self.metrics_history[player_id][-5:]  # Last 5 measurements
        avg_load = np.mean([m.player_load for m in recent_metrics])
        avg_fatigue = np.mean([m.fatigue_index for m in recent_metrics])
        
        return avg_fatigue > threshold or avg_load > 1000


class SprintAnalyzer:
    """Analyze sprint performance from wearable data."""
    
    def __init__(self):
        self.sprint_threshold = 24.0  # km/h
        self.sprints: Dict[str, List[Dict]] = {}
        
    def detect_sprints(self, player_id: str, gps_data: List[GPSDataPoint]) -> List[Dict]:
        """
        Detect and analyze sprints from GPS data.
        
        Args:
            player_id: Player identifier
            gps_data: List of GPS data points
            
        Returns:
            List of sprint dictionaries
        """
        sprints = []
        in_sprint = False
        sprint_start = None
        sprint_points = []
        
        for point in gps_data:
            if point.speed >= self.sprint_threshold:
                if not in_sprint:
                    in_sprint = True
                    sprint_start = point
                sprint_points.append(point)
            else:
                if in_sprint and len(sprint_points) >= 3:  # Minimum 3 data points
                    sprint_info = self._analyze_sprint(sprint_start, sprint_points)
                    sprints.append(sprint_info)
                in_sprint = False
                sprint_points = []
                
        # Handle sprint at end of data
        if in_sprint and len(sprint_points) >= 3:
            sprint_info = self._analyze_sprint(sprint_start, sprint_points)
            sprints.append(sprint_info)
            
        self.sprints[player_id] = sprints
        return sprints
        
    def _analyze_sprint(self, start_point: GPSDataPoint, points: List[GPSDataPoint]) -> Dict:
        """Analyze a single sprint."""
        end_point = points[-1]
        
        duration = end_point.timestamp - start_point.timestamp
        distance = end_point.distance - start_point.distance
        
        speeds = [p.speed for p in points]
        max_speed = max(speeds)
        avg_speed = np.mean(speeds)
        
        # Calculate acceleration phase
        accel_points = [p for p in points if p.acceleration > 2.0]
        accel_duration = len(accel_points) * 0.1  # Assuming 10Hz data
        
        return {
            'start_time': start_point.timestamp,
            'end_time': end_point.timestamp,
            'duration': duration,
            'distance': distance,
            'max_speed': max_speed,
            'avg_speed': avg_speed,
            'acceleration_duration': accel_duration,
            'start_x': start_point.x,
            'start_y': start_point.y,
            'end_x': end_point.x,
            'end_y': end_point.y
        }
        
    def get_top_speeds(self, player_id: str, n: int = 5) -> List[float]:
        """Get top N speeds for a player."""
        if player_id not in self.sprints:
            return []
            
        speeds = [s['max_speed'] for s in self.sprints[player_id]]
        return sorted(speeds, reverse=True)[:n]
        
    def compare_sprints(self, player_id: str, sprint_indices: Tuple[int, int]) -> Dict:
        """Compare two sprints for a player."""
        if player_id not in self.sprints:
            return {}
            
        idx1, idx2 = sprint_indices
        if idx1 >= len(self.sprints[player_id]) or idx2 >= len(self.sprints[player_id]):
            return {}
            
        s1 = self.sprints[player_id][idx1]
        s2 = self.sprints[player_id][idx2]
        
        return {
            'duration_diff': s2['duration'] - s1['duration'],
            'distance_diff': s2['distance'] - s1['distance'],
            'max_speed_diff': s2['max_speed'] - s1['max_speed'],
            'avg_speed_diff': s2['avg_speed'] - s1['avg_speed'],
            'speed_decline': ((s1['max_speed'] - s2['max_speed']) / s1['max_speed'] * 100) if s1['max_speed'] > 0 else 0
        }


class WearableIntegrationManager:
    """Main manager class for wearable data integration."""
    
    def __init__(self):
        self.importer = WearableDataImporter()
        self.load_calculator = PlayerLoadCalculator()
        self.sprint_analyzer = SprintAnalyzer()
        self.active_players: set = set()
        
    def import_session_data(self, 
                           gps_file: str,
                           player_mapping: Dict[str, str],
                           hr_files: Optional[Dict[str, str]] = None,
                           accel_files: Optional[Dict[str, str]] = None) -> bool:
        """
        Import complete session data from multiple sources.
        
        Args:
            gps_file: Path to GPS data file
            player_mapping: Mapping of device IDs to player IDs
            hr_files: Optional dict of player_id to heart rate file
            accel_files: Optional dict of player_id to accelerometer file
            
        Returns:
            True if successful
        """
        # Import GPS data
        if not self.importer.import_catapult_data(gps_file, player_mapping):
            return False
            
        self.active_players = set(player_mapping.values())
        
        # Import heart rate data
        if hr_files:
            for player_id, file_path in hr_files.items():
                self.importer.import_polar_hr(file_path, player_id)
                
        # Import accelerometer data
        if accel_files:
            for player_id, file_path in accel_files.items():
                self.importer.import_accelerometer_data(file_path, player_id)
                
        # Calculate initial load metrics
        for player_id in self.active_players:
            if player_id in self.importer.gps_data:
                self.load_calculator.calculate_load_metrics(
                    player_id, 
                    self.importer.gps_data[player_id]
                )
                
        logger.info(f"Successfully imported session data for {len(self.active_players)} players")
        return True
        
    def get_player_stats(self, player_id: str) -> Dict:
        """Get comprehensive stats for a player."""
        stats = {
            'player_id': player_id,
            'has_gps_data': player_id in self.importer.gps_data,
            'has_hr_data': player_id in self.importer.heart_rate_data,
            'has_accel_data': player_id in self.importer.accelerometer_data,
            'load_metrics': None,
            'sprints': [],
            'top_speeds': []
        }
        
        if player_id in self.importer.gps_data:
            gps_data = self.importer.gps_data[player_id]
            
            # Get latest load metrics
            if player_id in self.load_calculator.metrics_history:
                latest = self.load_calculator.metrics_history[player_id][-1]
                stats['load_metrics'] = {
                    'total_distance': latest.total_distance,
                    'sprint_distance': latest.sprint_distance,
                    'high_intensity_runs': latest.high_intensity_runs,
                    'avg_speed': latest.avg_speed,
                    'max_speed': latest.max_speed,
                    'avg_heart_rate': latest.avg_heart_rate,
                    'max_heart_rate': latest.max_heart_rate,
                    'player_load': latest.player_load,
                    'fatigue_index': latest.fatigue_index
                }
                
            # Analyze sprints
            sprints = self.sprint_analyzer.detect_sprints(player_id, gps_data)
            stats['sprints'] = sprints
            stats['top_speeds'] = self.sprint_analyzer.get_top_speeds(player_id)
            
        return stats
        
    def get_team_summary(self) -> Dict:
        """Get summary statistics for entire team."""
        summary = {
            'total_players': len(self.active_players),
            'total_distance': 0.0,
            'total_sprint_distance': 0.0,
            'avg_team_speed': 0.0,
            'max_speed_recorded': 0.0,
            'players_at_risk': []
        }
        
        speeds = []
        
        for player_id in self.active_players:
            stats = self.get_player_stats(player_id)
            
            if stats['load_metrics']:
                summary['total_distance'] += stats['load_metrics']['total_distance']
                summary['total_sprint_distance'] += stats['load_metrics']['sprint_distance']
                speeds.append(stats['load_metrics']['avg_speed'])
                summary['max_speed_recorded'] = max(
                    summary['max_speed_recorded'],
                    stats['load_metrics']['max_speed']
                )
                
            if self.load_calculator.detect_overload_risk(player_id):
                summary['players_at_risk'].append(player_id)
                
        if speeds:
            summary['avg_team_speed'] = np.mean(speeds)
            
        return summary
        
    def export_to_json(self, output_path: str):
        """Export all wearable data to JSON."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'players': {},
            'team_summary': self.get_team_summary()
        }
        
        for player_id in self.active_players:
            export_data['players'][player_id] = self.get_player_stats(player_id)
            
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        logger.info(f"Exported wearable data to {output_path}")


# Convenience function for quick integration
def create_wearable_integration() -> WearableIntegrationManager:
    """Create and return a configured wearable integration manager."""
    return WearableIntegrationManager()
