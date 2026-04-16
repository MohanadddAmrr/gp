"""
Player Profile System for Football Analytics

Manages player identity across matches using:
- Face recognition with embeddings
- Jersey number tracking
- Name/team matching
- Performance statistics aggregation

This module provides a high-level interface for managing player identities
and tracking their performance across multiple matches.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from services.database_manager import DatabaseManager


@dataclass
class PlayerIdentity:
    """Player identity information."""
    profile_id: int = None
    name: str = "Unknown"
    team_name: str = None
    jersey_number: int = None
    face_encoding: np.ndarray = None
    position: str = None
    nationality: str = None
    height_cm: int = None
    weight_kg: int = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert numpy array to list for serialization
        if self.face_encoding is not None:
            data['face_encoding'] = self.face_encoding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlayerIdentity':
        """Create from dictionary."""
        if 'face_encoding' in data and data['face_encoding'] is not None:
            data['face_encoding'] = np.array(data['face_encoding'])
        return cls(**data)


@dataclass
class PlayerMatchPerformance:
    """Player performance in a single match."""
    match_id: int
    match_date: datetime
    opponent: str
    result: str
    
    # Physical metrics
    total_distance_m: float = 0
    sprint_distance_m: float = 0
    avg_speed_mps: float = 0
    max_speed_mps: float = 0
    sprints: int = 0
    minutes_played: float = 0
    workload_score: float = 0
    
    # Technical metrics
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_accuracy: float = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    assists: int = 0
    tackles: int = 0
    interceptions: int = 0
    
    # Possession metrics
    touches: int = 0
    possession_count: int = 0
    possession_duration_s: float = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        if isinstance(self.match_date, datetime):
            data['match_date'] = self.match_date.isoformat()
        return data


@dataclass
class PlayerFormMetrics:
    """Player form/trend metrics."""
    matches_analyzed: int = 0
    
    # Averages
    avg_distance_m: float = 0
    avg_speed_mps: float = 0
    avg_pass_accuracy: float = 0
    avg_touches: float = 0
    avg_workload: float = 0
    
    # Trends (percentage change)
    distance_trend: float = 0
    speed_trend: float = 0
    pass_accuracy_trend: float = 0
    workload_trend: float = 0
    
    # Form rating (0-100)
    form_rating: float = 50
    trend_direction: str = "stable"  # improving, declining, stable
    
    # Consistency (coefficient of variation, lower is more consistent)
    consistency_score: float = 0


class PlayerProfileSystem:
    """
    High-level system for managing player profiles and tracking performance.
    
    Features:
    - Face-based player recognition
    - Cross-match player tracking
    - Performance statistics aggregation
    - Form/trend analysis
    - Career statistics
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the player profile system.
        
        Args:
            db_manager: DatabaseManager instance (creates new if None)
        """
        self.db = db_manager or DatabaseManager()
    
    # ============================================================
    # PROFILE MANAGEMENT
    # ============================================================
    
    def create_profile(
        self,
        name: str,
        team_name: str = None,
        jersey_number: int = None,
        face_encoding: np.ndarray = None,
        face_image_path: str = None,
        **kwargs
    ) -> int:
        """
        Create a new player profile.
        
        Args:
            name: Player name
            team_name: Team name
            jersey_number: Jersey number
            face_encoding: Face embedding for recognition
            face_image_path: Path to face image file
            **kwargs: Additional player attributes
            
        Returns:
            profile_id: ID of created profile
        """
        return self.db.create_player_profile(
            name=name,
            team_name=team_name,
            jersey_number=jersey_number,
            face_encoding=face_encoding,
            face_image_path=face_image_path,
            **kwargs
        )
    
    def get_profile(self, profile_id: int) -> Optional[PlayerIdentity]:
        """Get player profile by ID."""
        data = self.db.get_player_profile(profile_id)
        if data:
            return PlayerIdentity(
                profile_id=data.get('profile_id'),
                name=data.get('name'),
                team_name=data.get('team_name'),
                jersey_number=data.get('jersey_number'),
                face_encoding=data.get('face_encoding'),
                position=data.get('position_default'),
                nationality=data.get('nationality'),
                height_cm=data.get('height_cm'),
                weight_kg=data.get('weight_kg')
            )
        return None
    
    def find_or_create_profile(
        self,
        name: str = None,
        team_name: str = None,
        jersey_number: int = None,
        face_encoding: np.ndarray = None,
        threshold: float = 0.6
    ) -> Tuple[int, bool]:
        """
        Find existing profile or create new one.
        
        Tries to match by:
        1. Face encoding (if provided)
        2. Name + team + jersey number
        
        Args:
            name: Player name
            team_name: Team name
            jersey_number: Jersey number
            face_encoding: Face embedding
            threshold: Face matching threshold
            
        Returns:
            Tuple of (profile_id, is_new)
        """
        # Try face matching first
        if face_encoding is not None:
            match = self.db.find_player_by_face(face_encoding, threshold)
            if match:
                return match['profile_id'], False
        
        # Try name + team + jersey matching
        if name and team_name:
            profiles = self.db.get_all_player_profiles()
            for profile in profiles:
                if (profile['name'].lower() == name.lower() and 
                    profile['team_name'] and 
                    profile['team_name'].lower() == team_name.lower() and
                    (jersey_number is None or profile['jersey_number'] == jersey_number)):
                    return profile['profile_id'], False
        
        # Create new profile
        profile_id = self.create_profile(
            name=name or f"Unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            team_name=team_name,
            jersey_number=jersey_number,
            face_encoding=face_encoding
        )
        return profile_id, True
    
    def update_profile(self, profile_id: int, **kwargs):
        """Update player profile."""
        self.db.update_player_profile(profile_id, **kwargs)
    
    def list_all_profiles(self) -> List[PlayerIdentity]:
        """List all player profiles."""
        profiles = self.db.get_all_player_profiles()
        return [
            PlayerIdentity(
                profile_id=p['profile_id'],
                name=p['name'],
                team_name=p['team_name'],
                jersey_number=p['jersey_number'],
                face_encoding=p.get('face_encoding'),
                position=p.get('position_default'),
                nationality=p.get('nationality'),
                height_cm=p.get('height_cm'),
                weight_kg=p.get('weight_kg')
            )
            for p in profiles
        ]
    
    def search_profiles(
        self,
        name_query: str = None,
        team_name: str = None,
        position: str = None
    ) -> List[PlayerIdentity]:
        """
        Search player profiles.
        
        Args:
            name_query: Name search string (partial match)
            team_name: Exact team name match
            position: Position filter
            
        Returns:
            List of matching profiles
        """
        profiles = self.list_all_profiles()
        results = []
        
        for profile in profiles:
            match = True
            
            if name_query and name_query.lower() not in profile.name.lower():
                match = False
            
            if team_name and profile.team_name != team_name:
                match = False
            
            if position and profile.position != position:
                match = False
            
            if match:
                results.append(profile)
        
        return results
    
    # ============================================================
    # PERFORMANCE TRACKING
    # ============================================================
    
    def get_match_history(self, profile_id: int) -> List[PlayerMatchPerformance]:
        """
        Get player's match history with performance data.
        
        Args:
            profile_id: Player profile ID
            
        Returns:
            List of match performances
        """
        stats = self.db.get_player_stats_across_matches(profile_id)
        performances = []
        
        for stat in stats:
            # Calculate pass accuracy
            pass_accuracy = 0
            if stat.get('passes_attempted', 0) > 0:
                pass_accuracy = (stat['passes_completed'] / stat['passes_attempted']) * 100
            
            perf = PlayerMatchPerformance(
                match_id=stat['match_id'],
                match_date=stat.get('match_date'),
                opponent=f"{stat.get('team_a')} vs {stat.get('team_b')}",
                result="N/A",  # Could calculate from scores
                
                total_distance_m=stat.get('total_distance_m', 0),
                avg_speed_mps=stat.get('avg_speed_mps', 0),
                max_speed_mps=stat.get('max_speed_mps', 0),
                sprints=stat.get('sprints', 0),
                minutes_played=stat.get('minutes_played', 0),
                workload_score=stat.get('workload_score', 0),
                
                passes_attempted=stat.get('passes_attempted', 0),
                passes_completed=stat.get('passes_completed', 0),
                pass_accuracy=pass_accuracy,
                shots=stat.get('shots', 0),
                shots_on_target=stat.get('shots_on_target', 0),
                goals=stat.get('goals', 0),
                assists=stat.get('assists', 0),
                tackles=stat.get('tackles', 0),
                interceptions=stat.get('interceptions', 0),
                
                touches=stat.get('touches', 0),
                possession_count=stat.get('possession_count', 0),
                possession_duration_s=stat.get('possession_duration_s', 0)
            )
            performances.append(perf)
        
        return performances
    
    def get_career_stats(self, profile_id: int) -> Dict:
        """
        Get aggregated career statistics.
        
        Args:
            profile_id: Player profile ID
            
        Returns:
            Dictionary with career statistics
        """
        return self.db.get_player_career_stats(profile_id)
    
    def calculate_form(
        self,
        profile_id: int,
        num_matches: int = 5
    ) -> PlayerFormMetrics:
        """
        Calculate player form over recent matches.
        
        Args:
            profile_id: Player profile ID
            num_matches: Number of recent matches to analyze
            
        Returns:
            PlayerFormMetrics object
        """
        performances = self.get_match_history(profile_id)[:num_matches]
        
        if not performances:
            return PlayerFormMetrics()
        
        # Calculate averages
        avg_distance = np.mean([p.total_distance_m for p in performances])
        avg_speed = np.mean([p.avg_speed_mps for p in performances])
        avg_pass_accuracy = np.mean([p.pass_accuracy for p in performances])
        avg_touches = np.mean([p.touches for p in performances])
        avg_workload = np.mean([p.workload_score for p in performances])
        
        # Calculate trends (compare first half to second half)
        half = len(performances) // 2
        if half > 0:
            first_half = performances[-half:]
            second_half = performances[:half]
            
            first_dist = np.mean([p.total_distance_m for p in first_half])
            second_dist = np.mean([p.total_distance_m for p in second_half])
            distance_trend = ((second_dist - first_dist) / first_dist * 100) if first_dist > 0 else 0
            
            first_speed = np.mean([p.avg_speed_mps for p in first_half])
            second_speed = np.mean([p.avg_speed_mps for p in second_half])
            speed_trend = ((second_speed - first_speed) / first_speed * 100) if first_speed > 0 else 0
            
            first_acc = np.mean([p.pass_accuracy for p in first_half])
            second_acc = np.mean([p.pass_accuracy for p in second_half])
            pass_trend = ((second_acc - first_acc) / first_acc * 100) if first_acc > 0 else 0
            
            first_work = np.mean([p.workload_score for p in first_half])
            second_work = np.mean([p.workload_score for p in second_half])
            work_trend = ((second_work - first_work) / first_work * 100) if first_work > 0 else 0
        else:
            distance_trend = speed_trend = pass_trend = work_trend = 0
        
        # Calculate consistency (coefficient of variation)
        distances = [p.total_distance_m for p in performances]
        consistency = (np.std(distances) / np.mean(distances) * 100) if np.mean(distances) > 0 else 0
        
        # Calculate form rating (0-100)
        # Based on: distance (30%), speed (20%), pass accuracy (30%), workload (20%)
        form_rating = min(100, max(0,
            (avg_distance / 12000 * 30) +  # Normalize to ~12km max
            (avg_speed / 7 * 20) +          # Normalize to ~7 m/s max
            (avg_pass_accuracy * 0.3) +
            (avg_workload / 100 * 20)
        ))
        
        # Determine trend direction
        avg_trend = (distance_trend + speed_trend + pass_trend + work_trend) / 4
        if avg_trend > 5:
            trend_direction = "improving"
        elif avg_trend < -5:
            trend_direction = "declining"
        else:
            trend_direction = "stable"
        
        return PlayerFormMetrics(
            matches_analyzed=len(performances),
            avg_distance_m=avg_distance,
            avg_speed_mps=avg_speed,
            avg_pass_accuracy=avg_pass_accuracy,
            avg_touches=avg_touches,
            avg_workload=avg_workload,
            distance_trend=distance_trend,
            speed_trend=speed_trend,
            pass_accuracy_trend=pass_trend,
            workload_trend=work_trend,
            form_rating=form_rating,
            trend_direction=trend_direction,
            consistency_score=consistency
        )
    
    def compare_players(
        self,
        profile_ids: List[int],
        metric: str = 'total_distance_m'
    ) -> Dict:
        """
        Compare multiple players on a specific metric.
        
        Args:
            profile_ids: List of player profile IDs
            metric: Metric to compare
            
        Returns:
            Comparison data
        """
        comparison = {
            'metric': metric,
            'players': [],
            'rankings': []
        }
        
        for profile_id in profile_ids:
            profile = self.get_profile(profile_id)
            if not profile:
                continue
            
            career_stats = self.get_career_stats(profile_id)
            value = career_stats.get(metric, 0)
            matches = career_stats.get('matches_played', 0)
            
            comparison['players'].append({
                'profile_id': profile_id,
                'name': profile.name,
                'team': profile.team_name,
                'value': value,
                'matches': matches,
                'per_match': value / matches if matches > 0 else 0
            })
        
        # Sort by value
        comparison['rankings'] = sorted(
            comparison['players'],
            key=lambda x: x['value'],
            reverse=True
        )
        
        return comparison
    
    def get_performance_trend(
        self,
        profile_id: int,
        metric: str = 'total_distance_m',
        num_matches: int = 10
    ) -> List[Dict]:
        """
        Get performance trend for a specific metric over time.
        
        Args:
            profile_id: Player profile ID
            metric: Metric to track
            num_matches: Number of matches to include
            
        Returns:
            List of {match_id, date, value} dictionaries
        """
        performances = self.get_match_history(profile_id)[:num_matches]
        trend = []
        
        for perf in reversed(performances):  # Oldest first
            value = getattr(perf, metric, 0)
            trend.append({
                'match_id': perf.match_id,
                'date': perf.match_date.isoformat() if isinstance(perf.match_date, datetime) else perf.match_date,
                'opponent': perf.opponent,
                'value': value
            })
        
        return trend
    
    # ============================================================
    # FACE RECOGNITION
    # ============================================================
    
    def register_face(
        self,
        profile_id: int,
        face_encoding: np.ndarray,
        face_image_path: str = None
    ):
        """
        Register face encoding for a player.
        
        Args:
            profile_id: Player profile ID
            face_encoding: Face embedding array
            face_image_path: Optional path to face image
        """
        self.db.update_player_profile(
            profile_id,
            face_encoding=face_encoding,
            face_image_path=face_image_path
        )
    
    def recognize_player(
        self,
        face_encoding: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[PlayerIdentity]:
        """
        Recognize player from face encoding.
        
        Args:
            face_encoding: Face embedding to match
            threshold: Similarity threshold
            
        Returns:
            PlayerIdentity if match found, None otherwise
        """
        match = self.db.find_player_by_face(face_encoding, threshold)
        if match:
            return PlayerIdentity(
                profile_id=match['profile_id'],
                name=match['name'],
                team_name=match['team_name'],
                jersey_number=match['jersey_number'],
                face_encoding=match.get('face_encoding'),
                position=match.get('position_default'),
                nationality=match.get('nationality'),
                height_cm=match.get('height_cm'),
                weight_kg=match.get('weight_kg')
            )
        return None
    
    def get_face_gallery(self) -> Dict[int, np.ndarray]:
        """
        Get all registered faces as a gallery.
        
        Returns:
            Dictionary mapping profile_id to face encoding
        """
        profiles = self.db.get_all_player_profiles()
        gallery = {}
        for profile in profiles:
            if profile.get('face_encoding') is not None:
                gallery[profile['profile_id']] = pickle.loads(profile['face_encoding'])
        return gallery
    
    # ============================================================
    # EXPORT
    # ============================================================
    
    def export_player_report(
        self,
        profile_id: int,
        output_path: str = None
    ) -> str:
        """
        Generate comprehensive player report.
        
        Args:
            profile_id: Player profile ID
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        
        career_stats = self.get_career_stats(profile_id)
        form = self.calculate_form(profile_id)
        match_history = self.get_match_history(profile_id)
        
        report = {
            'player_info': profile.to_dict(),
            'career_summary': career_stats,
            'current_form': {
                'matches_analyzed': form.matches_analyzed,
                'form_rating': round(form.form_rating, 1),
                'trend_direction': form.trend_direction,
                'consistency_score': round(form.consistency_score, 1),
                'averages': {
                    'distance_m': round(form.avg_distance_m, 1),
                    'speed_mps': round(form.avg_speed_mps, 2),
                    'pass_accuracy': round(form.avg_pass_accuracy, 1),
                    'touches': round(form.avg_touches, 1),
                    'workload': round(form.avg_workload, 1)
                },
                'trends': {
                    'distance': round(form.distance_trend, 1),
                    'speed': round(form.speed_trend, 1),
                    'pass_accuracy': round(form.pass_accuracy_trend, 1),
                    'workload': round(form.workload_trend, 1)
                }
            },
            'match_history': [p.to_dict() for p in match_history[:20]]
        }
        
        if not output_path:
            safe_name = profile.name.replace(' ', '_')
            output_path = f"{safe_name}_report_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Player report exported: {output_path}")
        return output_path


# Convenience function for quick access
def get_player_profile_system(db_path: str = "matches.db") -> PlayerProfileSystem:
    """Get initialized player profile system."""
    db_manager = DatabaseManager(db_path)
    return PlayerProfileSystem(db_manager)
