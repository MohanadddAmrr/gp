import pytest
from pathlib import Path

from services.database_manager import DatabaseManager
from services.dynamic_roster_manager import DynamicRosterManager

import tempfile
import os

@pytest.fixture
def db():
    # Use a temp file instead of :memory: because DatabaseManager opens/closes connections
    fd, temp_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    db_manager = DatabaseManager(temp_path)
    db_manager.initialize_database()
    
    yield db_manager
    
    os.remove(temp_path)

@pytest.fixture
def roster_mgr(db):
    return DynamicRosterManager(db)

def test_ensure_roster(roster_mgr):
    roster_id = roster_mgr.ensure_roster("Test Team", "A", 123)
    assert roster_id > 0
    
    active = roster_mgr.fetch_active_roster("Test Team", "A", 123)
    assert active == roster_id

def test_add_and_update_player(roster_mgr):
    roster_id = roster_mgr.ensure_roster("Test Team", "A")
    player_id = roster_mgr.add_player_to_roster(roster_id, "John Doe", 10, "FW", True)
    
    players = roster_mgr.get_roster_players(roster_id)
    assert len(players) == 1
    p = players[0]
    assert p["full_name"] == "John Doe"
    assert p["jersey_number"] == 10
    
    roster_mgr.update_roster_player(p["roster_player_id"], "John Doe", 11, "MF", False)
    players = roster_mgr.get_roster_players(roster_id)
    p = players[0]
    assert p["jersey_number"] == 11
    assert p["position"] == "MF"
    assert p["is_starting"] == False

def test_quick_update(roster_mgr):
    roster_id = roster_mgr.ensure_roster("Test Team", "B")
    player_id = roster_mgr.add_player_to_roster(roster_id, "Jane Doe", 7, "MF", True)
    
    players = roster_mgr.get_roster_players(roster_id)
    rp_id = players[0]["roster_player_id"]
    
    roster_mgr.quick_update_roster_player(rp_id, 8, None, None)
    
    players = roster_mgr.get_roster_players(roster_id)
    assert players[0]["jersey_number"] == 8
    assert players[0]["position"] == "MF"  # Unchanged
    assert players[0]["is_starting"] == True  # Unchanged

def test_remove_player(roster_mgr):
    roster_id = roster_mgr.ensure_roster("Test Team", "A")
    roster_mgr.add_player_to_roster(roster_id, "To Delete", 99, "GK", False)
    
    players = roster_mgr.get_roster_players(roster_id)
    assert len(players) == 1
    
    roster_mgr.remove_player_from_roster(players[0]["roster_player_id"])
    
    players = roster_mgr.get_roster_players(roster_id)
    assert len(players) == 0

def test_get_all_rosters(roster_mgr):
    r1 = roster_mgr.ensure_roster("Team1", "A", 1)
    r2 = roster_mgr.ensure_roster("Team1", "B", 2)
    r3 = roster_mgr.ensure_roster("Team2", "A", 1)
    
    rosters = roster_mgr.get_all_rosters("Team1")
    assert len(rosters) == 2
    
    rosters_match = roster_mgr.get_all_rosters("Team1", 2)
    assert len(rosters_match) == 1
    assert rosters_match[0]["roster_id"] == r2
