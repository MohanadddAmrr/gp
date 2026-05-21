"""
API Connector Module

Handles connections to external football data APIs including StatsBomb, Opta,
Understat, and Football-data.org for comprehensive match data integration.
"""

import json
import time
import logging
import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API connections."""
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 10  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3
    cache_duration: int = 3600  # seconds


@dataclass
class MatchData:
    """Standardized match data structure."""
    match_id: str
    competition: str
    season: str
    match_date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    events: List[Dict] = field(default_factory=list)
    lineups: Dict = field(default_factory=dict)
    statistics: Dict = field(default_factory=dict)
    source: str = ""


class BaseAPIConnector(ABC):
    """Abstract base class for API connectors."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=config.retry_attempts))
        self.last_request_time: Optional[datetime] = None
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        if self.last_request_time:
            min_interval = 60.0 / self.config.rate_limit
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_request_time = datetime.now()
        
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.cache and key in self.cache_timestamps:
            age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
            if age < self.config.cache_duration:
                return self.cache[key]
        return None
        
    def _set_cached(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request with rate limiting and error handling."""
        self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
            
    @abstractmethod
    def get_matches(self, **kwargs) -> List[MatchData]:
        """Fetch matches from API."""
        pass
        
    @abstractmethod
    def get_match_events(self, match_id: str) -> List[Dict]:
        """Fetch match events."""
        pass


class StatsBombConnector(BaseAPIConnector):
    """Connector for StatsBomb API (open data and commercial)."""
    
    def __init__(self, api_key: Optional[str] = None):
        # StatsBomb has open data and commercial API
        config = APIConfig(
            base_url="https://raw.githubusercontent.com/statsbomb/open-data/master/data/",
            api_key=api_key,
            rate_limit=100  # Generous for open data
        )
        super().__init__(config)
        self.commercial_base = "https://data.statsbombservices.com/api/"
        self.use_commercial = api_key is not None
        
    def get_competitions(self) -> List[Dict]:
        """Get available competitions."""
        cache_key = "statsbomb_competitions"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        data = self._make_request("competitions.json")
        if data:
            self._set_cached(cache_key, data)
        return data or []
        
    def get_matches(self, competition_id: Optional[int] = None, 
                   season_id: Optional[int] = None) -> List[MatchData]:
        """Fetch matches from StatsBomb."""
        cache_key = f"statsbomb_matches_{competition_id}_{season_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        if competition_id and season_id:
            endpoint = f"matches/{competition_id}/{season_id}.json"
        else:
            # Get all matches from all competitions
            competitions = self.get_competitions()
            all_matches = []
            for comp in competitions:
                matches = self.get_matches(comp['competition_id'], comp['season_id'])
                all_matches.extend(matches)
            return all_matches
            
        data = self._make_request(endpoint)
        matches = []
        
        if data:
            for match in data:
                match_data = MatchData(
                    match_id=str(match.get('match_id')),
                    competition=match.get('competition', {}).get('competition_name', ''),
                    season=match.get('season', {}).get('season_name', ''),
                    match_date=datetime.fromisoformat(match.get('match_date', '').replace('Z', '+00:00')),
                    home_team=match.get('home_team', {}).get('home_team_name', ''),
                    away_team=match.get('away_team', {}).get('away_team_name', ''),
                    home_score=match.get('home_score', 0),
                    away_score=match.get('away_score', 0),
                    source="StatsBomb"
                )
                matches.append(match_data)
                
        self._set_cached(cache_key, matches)
        return matches
        
    def get_match_events(self, match_id: str) -> List[Dict]:
        """Fetch match events from StatsBomb."""
        cache_key = f"statsbomb_events_{match_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        endpoint = f"events/{match_id}.json"
        data = self._make_request(endpoint)
        
        if data:
            events = data if isinstance(data, list) else data.get('events', [])
            self._set_cached(cache_key, events)
            return events
        return []
        
    def get_lineups(self, match_id: str) -> Dict:
        """Fetch match lineups."""
        cache_key = f"statsbomb_lineups_{match_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        endpoint = f"lineups/{match_id}.json"
        data = self._make_request(endpoint)
        
        if data:
            self._set_cached(cache_key, data)
        return data or {}
        
    def get_360_data(self, match_id: str) -> List[Dict]:
        """Fetch 360 freeze frame data if available."""
        cache_key = f"statsbomb_360_{match_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        endpoint = f"three-sixty/{match_id}.json"
        data = self._make_request(endpoint)
        
        if data:
            self._set_cached(cache_key, data)
            return data if isinstance(data, list) else []
        return []


class OptaConnector(BaseAPIConnector):
    """Connector for Opta/Feedsy API."""
    
    def __init__(self, api_key: str, api_secret: str):
        config = APIConfig(
            base_url="https://api.performfeeds.com/",
            api_key=api_key,
            rate_limit=30
        )
        super().__init__(config)
        self.api_secret = api_secret
        self.auth_token: Optional[str] = None
        
    def _authenticate(self) -> bool:
        """Authenticate with Opta API."""
        if self.auth_token:
            return True
            
        auth_url = f"{self.config.base_url}oauth/token"
        
        try:
            response = self.session.post(
                auth_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': self.config.api_key,
                    'client_secret': self.api_secret
                }
            )
            response.raise_for_status()
            data = response.json()
            self.auth_token = data.get('access_token')
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Opta authentication failed: {e}")
            return False
            
    def _make_authenticated_request(self, endpoint: str, 
                                   params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to Opta API."""
        if not self._authenticate():
            return None
            
        headers = {
            'Authorization': f'Bearer {self.auth_token}',
            'Accept': 'application/json'
        }
        
        return self._make_request(endpoint, params, headers)
        
    def get_matches(self, competition_id: Optional[str] = None,
                   date_from: Optional[datetime] = None,
                   date_to: Optional[datetime] = None) -> List[MatchData]:
        """Fetch matches from Opta."""
        params = {}
        if competition_id:
            params['competition'] = competition_id
        if date_from:
            params['date_from'] = date_from.strftime('%Y-%m-%d')
        if date_to:
            params['date_to'] = date_to.strftime('%Y-%m-%d')
            
        data = self._make_authenticated_request("soccer/matches", params)
        matches = []
        
        if data and 'matches' in data:
            for match in data['matches']:
                match_data = MatchData(
                    match_id=str(match.get('id')),
                    competition=match.get('competition', {}).get('name', ''),
                    season=match.get('season', {}).get('name', ''),
                    match_date=datetime.fromisoformat(match.get('date', '').replace('Z', '+00:00')),
                    home_team=match.get('homeTeam', {}).get('name', ''),
                    away_team=match.get('awayTeam', {}).get('name', ''),
                    home_score=match.get('homeScore', 0),
                    away_score=match.get('awayScore', 0),
                    source="Opta"
                )
                matches.append(match_data)
                
        return matches
        
    def get_match_events(self, match_id: str) -> List[Dict]:
        """Fetch match events from Opta."""
        data = self._make_authenticated_request(f"soccer/matches/{match_id}/events")
        
        if data and 'events' in data:
            return data['events']
        return []
        
    def get_match_statistics(self, match_id: str) -> Dict:
        """Fetch match statistics from Opta."""
        data = self._make_authenticated_request(f"soccer/matches/{match_id}/stats")
        return data or {}


class UnderstatConnector(BaseAPIConnector):
    """Connector for Understat API (xG data)."""
    
    def __init__(self):
        config = APIConfig(
            base_url="https://understat.com/",
            rate_limit=20
        )
        super().__init__(config)
        
    def _extract_json_from_html(self, html: str, var_name: str) -> Optional[Dict]:
        """Extract JSON data from Understat's JavaScript variables."""
        import re
        
        pattern = rf"var {var_name} = JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        
        if match:
            json_str = match.group(1).encode('utf-8').decode('unicode_escape')
            return json.loads(json_str)
        return None
        
    def get_matches(self, league: str = "EPL", 
                   season: int = datetime.now().year) -> List[MatchData]:
        """Fetch matches from Understat."""
        cache_key = f"understat_matches_{league}_{season}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        # Understat uses HTML with embedded JSON
        response = self._make_request(f"league/{league}/{season}")
        
        if not response or isinstance(response, dict):
            return []
            
        # Extract matches data from HTML
        matches_data = self._extract_json_from_html(response.text, 'datesData')
        matches = []
        
        if matches_data:
            for match in matches_data:
                match_data = MatchData(
                    match_id=str(match.get('id')),
                    competition=league,
                    season=str(season),
                    match_date=datetime.fromtimestamp(match.get('datetime', 0)),
                    home_team=match.get('h', {}).get('title', ''),
                    away_team=match.get('a', {}).get('title', ''),
                    home_score=int(match.get('goals', {}).get('h', 0)),
                    away_score=int(match.get('goals', {}).get('a', 0)),
                    source="Understat"
                )
                matches.append(match_data)
                
        self._set_cached(cache_key, matches)
        return matches
        
    def get_match_events(self, match_id: str) -> List[Dict]:
        """Fetch match shots and xG data from Understat."""
        cache_key = f"understat_shots_{match_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        response = self._make_request(f"match/{match_id}")
        
        if not response or isinstance(response, dict):
            return []
            
        shots_data = self._extract_json_from_html(response.text, 'shotsData')
        
        if shots_data:
            events = []
            for team, shots in shots_data.items():
                for shot in shots:
                    event = {
                        'minute': shot.get('minute'),
                        'team': team,
                        'player': shot.get('player'),
                        'xG': float(shot.get('xG', 0)),
                        'shot_type': shot.get('shotType'),
                        'result': shot.get('result'),
                        'x': shot.get('X'),
                        'y': shot.get('Y')
                    }
                    events.append(event)
                    
            self._set_cached(cache_key, events)
            return events
        return []
        
    def get_player_xg(self, player_id: str) -> Dict:
        """Get player xG statistics."""
        cache_key = f"understat_player_{player_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        response = self._make_request(f"player/{player_id}")
        
        if not response or isinstance(response, dict):
            return {}
            
        player_data = self._extract_json_from_html(response.text, 'playersData')
        
        if player_data:
            self._set_cached(cache_key, player_data)
            return player_data
        return {}


class FootballDataConnector(BaseAPIConnector):
    """Connector for Football-data.org API."""
    
    def __init__(self, api_key: str):
        config = APIConfig(
            base_url="https://api.football-data.org/v4/",
            api_key=api_key,
            rate_limit=10
        )
        super().__init__(config)
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                     headers: Optional[Dict] = None) -> Optional[Dict]:
        """Override to add API key header."""
        if headers is None:
            headers = {}
        headers['X-Auth-Token'] = self.config.api_key
        
        return super()._make_request(endpoint, params, headers)
        
    def get_competitions(self) -> List[Dict]:
        """Get available competitions."""
        cache_key = "football_data_competitions"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        data = self._make_request("competitions")
        
        if data and 'competitions' in data:
            self._set_cached(cache_key, data['competitions'])
            return data['competitions']
        return []
        
    def get_matches(self, competition_id: Optional[str] = None,
                   date_from: Optional[datetime] = None,
                   date_to: Optional[datetime] = None,
                   matchday: Optional[int] = None) -> List[MatchData]:
        """Fetch matches from Football-data.org."""
        params = {}
        if date_from:
            params['dateFrom'] = date_from.strftime('%Y-%m-%d')
        if date_to:
            params['dateTo'] = date_to.strftime('%Y-%m-%d')
        if matchday:
            params['matchday'] = matchday
            
        if competition_id:
            endpoint = f"competitions/{competition_id}/matches"
        else:
            endpoint = "matches"
            
        cache_key = f"football_data_matches_{competition_id}_{date_from}_{date_to}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        data = self._make_request(endpoint, params)
        matches = []
        
        if data and 'matches' in data:
            for match in data['matches']:
                match_data = MatchData(
                    match_id=str(match.get('id')),
                    competition=match.get('competition', {}).get('name', ''),
                    season=match.get('season', {}).get('startDate', '')[:4],
                    match_date=datetime.fromisoformat(match.get('utcDate', '').replace('Z', '+00:00')),
                    home_team=match.get('homeTeam', {}).get('name', ''),
                    away_team=match.get('awayTeam', {}).get('name', ''),
                    home_score=match.get('score', {}).get('fullTime', {}).get('home', 0),
                    away_score=match.get('score', {}).get('fullTime', {}).get('away', 0),
                    source="Football-data.org"
                )
                matches.append(match_data)
                
        self._set_cached(cache_key, matches)
        return matches
        
    def get_match_events(self, match_id: str) -> List[Dict]:
        """Fetch match events (limited in free tier)."""
        data = self._make_request(f"matches/{match_id}")
        
        if data:
            # Football-data.org doesn't provide detailed events in free tier
            # Return basic match info
            return [{
                'type': 'match_info',
                'home_team': data.get('homeTeam', {}).get('name'),
                'away_team': data.get('awayTeam', {}).get('name'),
                'score': data.get('score', {}).get('fullTime', {})
            }]
        return []
        
    def get_standings(self, competition_id: str) -> List[Dict]:
        """Get league standings."""
        cache_key = f"football_data_standings_{competition_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        data = self._make_request(f"competitions/{competition_id}/standings")
        
        if data and 'standings' in data:
            self._set_cached(cache_key, data['standings'])
            return data['standings']
        return []
        
    def get_team_matches(self, team_id: str,
                        date_from: Optional[datetime] = None,
                        date_to: Optional[datetime] = None) -> List[MatchData]:
        """Get matches for a specific team."""
        params = {}
        if date_from:
            params['dateFrom'] = date_from.strftime('%Y-%m-%d')
        if date_to:
            params['dateTo'] = date_to.strftime('%Y-%m-%d')

        data = self._make_request(f"teams/{team_id}/matches", params)
        matches = []

        if data and 'matches' in data:
            for match in data['matches']:
                match_data = MatchData(
                    match_id=str(match.get('id')),
                    competition=match.get('competition', {}).get('name', ''),
                    season=match.get('season', {}).get('startDate', '')[:4],
                    match_date=datetime.fromisoformat(match.get('utcDate', '').replace('Z', '+00:00')),
                    home_team=match.get('homeTeam', {}).get('name', ''),
                    away_team=match.get('awayTeam', {}).get('name', ''),
                    home_score=match.get('score', {}).get('fullTime', {}).get('home', 0),
                    away_score=match.get('score', {}).get('fullTime', {}).get('away', 0),
                    source="Football-data.org"
                )
                matches.append(match_data)

        return matches

    # ================================================================
    # Disk-based caching
    # ================================================================

    def _get_cache_dir(self) -> Path:
        """Get or create the disk cache directory."""
        cache_dir = Path("cache/api_responses")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _disk_cache_key(self, label: str) -> str:
        """Create a safe filename hash from a label."""
        return hashlib.md5(label.encode()).hexdigest()

    def _get_disk_cached(self, label: str, max_age_seconds: int = 86400) -> Optional[Any]:
        """Load JSON from disk cache if fresh enough (default 24 h)."""
        path = self._get_cache_dir() / f"{self._disk_cache_key(label)}.json"
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > max_age_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _set_disk_cached(self, label: str, data: Any):
        """Write JSON data to disk cache."""
        path = self._get_cache_dir() / f"{self._disk_cache_key(label)}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================================================================
    # Squad fetching
    # ================================================================

    def fetch_squad(self, team_name_or_id) -> List[Dict]:
        """
        Fetch squad roster from football-data.org.

        Args:
            team_name_or_id: Team name (str) to search, or numeric team ID (int).

        Returns:
            List of dicts: [{name, jersey_number, position}, ...]
        """
        # --- resolve team ID ---------------------------------------------------
        if isinstance(team_name_or_id, int):
            team_id = team_name_or_id
        else:
            # Check disk cache for the search result
            search_label = f"team_search_{team_name_or_id.lower().strip()}"
            cached_id = self._get_disk_cached(search_label)
            if cached_id is not None:
                team_id = cached_id
            else:
                data = self._make_request("teams", params={"name": team_name_or_id.strip()})
                if not data or not data.get("teams"):
                    logger.warning("No team found for query: %s", team_name_or_id)
                    return []
                team_id = data["teams"][0]["id"]
                self._set_disk_cached(search_label, team_id)

        # --- fetch squad -------------------------------------------------------
        squad_label = f"squad_{team_id}"
        cached_squad = self._get_disk_cached(squad_label)
        if cached_squad is not None:
            logger.info("Loaded squad for team %s from disk cache", team_id)
            return cached_squad

        data = self._make_request(f"teams/{team_id}")
        if not data or "squad" not in data:
            logger.warning("No squad data returned for team ID %s", team_id)
            return []

        squad = []
        for p in data["squad"]:
            squad.append({
                "name": p.get("name", ""),
                "jersey_number": p.get("shirtNumber"),
                "position": p.get("position", ""),
            })

        self._set_disk_cached(squad_label, squad)
        logger.info("Fetched %d players for %s (ID %s)", len(squad), data.get("name", ""), team_id)
        return squad

    def fetch_team_squad(self, external_id: int) -> List[Dict]:
        """
        Squad by football-data.org team id (C2 / Member E sync hook).

        Delegates to :meth:`fetch_squad` which already resolves ``teams/{id}``.
        """
        return self.fetch_squad(external_id)

    def fetch_and_save_squad(self, team_name: str, db_manager) -> List[Dict]:
        """
        Fetch squad from football-data.org and persist to the database.

        Args:
            team_name: Team name to search.
            db_manager: DatabaseManager instance.

        Returns:
            The squad list (same format as fetch_squad).
        """
        squad = self.fetch_squad(team_name)
        if not squad:
            return []

        saved = 0
        for player in squad:
            try:
                db_manager.create_player_profile(
                    name=player["name"],
                    team_name=team_name,
                    jersey_number=player.get("jersey_number"),
                    position_default=player.get("position"),
                )
                saved += 1
            except Exception as e:
                logger.warning("Could not save player %s: %s", player.get("name"), e)

        logger.info("Saved %d/%d players for %s to database", saved, len(squad), team_name)
        return squad


class APIDataExporter:
    """Export data from APIs to various formats."""
    
    def __init__(self):
        self.export_path = Path("exports")
        self.export_path.mkdir(exist_ok=True)
        
    def export_to_json(self, data: Any, filename: str):
        """Export data to JSON file."""
        filepath = self.export_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported to {filepath}")
        
    def export_to_csv(self, data: List[Dict], filename: str):
        """Export data to CSV file."""
        if not data:
            return
            
        filepath = self.export_path / filename
        keys = data[0].keys()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Exported to {filepath}")
        
    def export_match_report(self, match_data: MatchData, filename: Optional[str] = None):
        """Export comprehensive match report."""
        if not filename:
            filename = f"match_report_{match_data.match_id}.json"
            
        report = {
            'match_info': {
                'id': match_data.match_id,
                'competition': match_data.competition,
                'season': match_data.season,
                'date': match_data.match_date.isoformat(),
                'home_team': match_data.home_team,
                'away_team': match_data.away_team,
                'score': f"{match_data.home_score}-{match_data.away_score}"
            },
            'events': match_data.events,
            'lineups': match_data.lineups,
            'statistics': match_data.statistics
        }
        
        self.export_to_json(report, filename)


class APIManager:
    """Manager class for all API connections."""
    
    def __init__(self):
        self.connectors: Dict[str, BaseAPIConnector] = {}
        self.exporter = APIDataExporter()
        
    def add_statsbomb(self, api_key: Optional[str] = None):
        """Add StatsBomb connector."""
        self.connectors['statsbomb'] = StatsBombConnector(api_key)
        logger.info("Added StatsBomb connector")
        
    def add_opta(self, api_key: str, api_secret: str):
        """Add Opta connector."""
        self.connectors['opta'] = OptaConnector(api_key, api_secret)
        logger.info("Added Opta connector")
        
    def add_understat(self):
        """Add Understat connector."""
        self.connectors['understat'] = UnderstatConnector()
        logger.info("Added Understat connector")
        
    def add_football_data(self, api_key: str):
        """Add Football-data.org connector."""
        self.connectors['football_data'] = FootballDataConnector(api_key)
        logger.info("Added Football-data.org connector")
        
    def get_all_matches(self, **kwargs) -> Dict[str, List[MatchData]]:
        """Get matches from all configured APIs."""
        results = {}
        
        for name, connector in self.connectors.items():
            try:
                matches = connector.get_matches(**kwargs)
                results[name] = matches
                logger.info(f"Retrieved {len(matches)} matches from {name}")
            except Exception as e:
                logger.error(f"Error getting matches from {name}: {e}")
                results[name] = []
                
        return results
        
    def aggregate_match_data(self, match_id: str, 
                            sources: Optional[List[str]] = None) -> MatchData:
        """Aggregate match data from multiple sources."""
        if sources is None:
            sources = list(self.connectors.keys())
            
        aggregated = MatchData(
            match_id=match_id,
            competition="",
            season="",
            match_date=datetime.now(),
            home_team="",
            away_team="",
            home_score=0,
            away_score=0,
            source="aggregated"
        )
        
        all_events = []
        
        for source in sources:
            if source in self.connectors:
                connector = self.connectors[source]
                try:
                    events = connector.get_match_events(match_id)
                    all_events.extend(events)
                except Exception as e:
                    logger.warning(f"Could not get events from {source}: {e}")
                    
        aggregated.events = all_events
        return aggregated
        
    def export_to_third_party(self, data: Any, destination: str, 
                             format: str = "json") -> bool:
        """
        Export data to third-party systems.
        
        Args:
            data: Data to export
            destination: Target system identifier
            format: Export format (json, csv, xml)
            
        Returns:
            True if export successful
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"{destination}_{timestamp}.json"
            self.exporter.export_to_json(data, filename)
        elif format == "csv":
            filename = f"{destination}_{timestamp}.csv"
            if isinstance(data, list):
                self.exporter.export_to_csv(data, filename)
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
            
        return True


# Convenience function
def create_api_manager() -> APIManager:
    """Create and return a configured API manager."""
    return APIManager()
