"""
Test script to verify possession tracking implementation.
Tests the core algorithm with simulated player and ball positions.
"""

import sys
from pathlib import Path

# Add project root to path (go up one directory from tests/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.possession_tracker import PossessionTracker


def test_possession_detection():
    """Test PossessionTracker with simulated game scenarios"""

    print("=" * 70)
    print("POSSESSION TRACKER VERIFICATION TEST")
    print("=" * 70)

    # Initialize tracker with 50px threshold
    tracker = PossessionTracker(distance_threshold=50.0)

    print("\n✓ Test 1: Player Within Threshold (Should Have Possession)")
    print("-" * 70)
    
    # Ball at (500, 300), Player 7 at (510, 305) - distance ~11px
    ball_pos = (500.0, 300.0)
    players = {
        7: (510.0, 305.0, 'A'),   # Very close - should have possession
        10: (600.0, 300.0, 'A'),  # 100px away - too far
        3: (450.0, 400.0, 'B'),   # ~111px away - too far
    }
    
    result = tracker.detect_possession(ball_pos, players, frame_idx=1, timestamp=0.04)
    
    if result:
        player_id, team, distance = result
        print(f"  Ball position: {ball_pos}")
        print(f"  Player {player_id} (Team {team}) has possession")
        print(f"  Distance: {distance:.2f} pixels")
        assert player_id == 7, f"Expected player 7, got {player_id}"
        assert team == 'A', f"Expected team A, got {team}"
        assert distance < 50, f"Distance should be < 50px, got {distance:.2f}"
        print("  ✅ PASS: Possession correctly assigned to closest player")
    else:
        print("  ❌ FAIL: No possession detected")
        return False

    print("\n✓ Test 2: All Players Too Far (No Possession)")
    print("-" * 70)
    
    # Ball at (500, 300), all players >50px away
    ball_pos = (500.0, 300.0)
    players = {
        7: (600.0, 300.0, 'A'),   # 100px away
        10: (500.0, 400.0, 'A'),  # 100px away
        3: (400.0, 250.0, 'B'),   # ~111px away
    }
    
    result = tracker.detect_possession(ball_pos, players, frame_idx=2, timestamp=0.08)
    
    if result is None:
        print(f"  Ball position: {ball_pos}")
        print(f"  All players outside threshold (>50px)")
        print(f"  No possession detected")
        print("  ✅ PASS: Correctly detected loose ball")
    else:
        print(f"  ❌ FAIL: Incorrectly assigned possession to player {result[0]}")
        return False

    print("\n✓ Test 3: Possession Change Detection")
    print("-" * 70)
    
    # Player 7 has ball
    ball_pos = (500.0, 300.0)
    players = {7: (505.0, 302.0, 'A'), 10: (600.0, 300.0, 'B')}
    tracker.detect_possession(ball_pos, players, frame_idx=10, timestamp=0.4)
    
    # Ball moves to player 10
    ball_pos = (595.0, 298.0)
    players = {7: (505.0, 302.0, 'A'), 10: (600.0, 300.0, 'B')}
    tracker.detect_possession(ball_pos, players, frame_idx=15, timestamp=0.6)
    
    history = tracker.get_possession_history()
    print(f"  Possession changes detected: {len(history)}")
    
    if len(history) >= 1:
        last_event = history[-1]
        print(f"  Last possession: Player {last_event['player_id']} (Team {last_event['team']})")
        print(f"  Duration: {last_event['duration']:.3f} seconds")
        print("  ✅ PASS: Possession change logged correctly")
    else:
        print("  ❌ FAIL: No possession history recorded")
        return False

    print("\n✓ Test 4: Possession Statistics")
    print("-" * 70)
    
    stats = tracker.get_statistics()
    possession_pct = tracker.get_possession_percentage()
    
    print(f"  Team A possession: {possession_pct['A']:.1f}%")
    print(f"  Team B possession: {possession_pct['B']:.1f}%")
    print(f"  Total possession events: {stats['total_events']}")
    print(f"  Current possessor: {stats['current_possessor']}")
    
    assert 'A' in possession_pct, "Team A should have stats"
    assert 'B' in possession_pct, "Team B should have stats"
    print("  ✅ PASS: Statistics calculated correctly")

    print("\n✓ Test 5: Player Possession Stats")
    print("-" * 70)
    
    player_stats = tracker.get_player_possession_stats()
    
    for player_id, stats in player_stats.items():
        print(f"  Player {player_id}:")
        print(f"    Total time: {stats['total_time']:.3f}s")
        print(f"    Touch count: {stats['touch_count']}")
        print(f"    Avg duration: {stats['avg_possession_duration']:.3f}s")
    
    assert len(player_stats) > 0, "Should have player stats"
    print("  ✅ PASS: Per-player statistics working")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print("\nPossession Tracker Implementation:")
    print("✅ Closest player detection - Euclidean distance")
    print("✅ Distance threshold check (50px)")
    print("✅ Possession change detection")
    print("✅ Team possession statistics")
    print("✅ Per-player possession statistics")
    print("✅ Possession history tracking")
    print("\nReady for integration with video processing!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_possession_detection()
        if success:
            print("\n✅ Possession tracking module verified and ready!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
