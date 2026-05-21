"""
Possession Validation Tests - Real Video Data

Tests possession tracking on actual processed video data to ensure:
1. Data integrity (percentages add up, no negative values)
2. Tactical metrics are calculated correctly
3. Enhanced features work properly (zones, pressure, duration)

This validates the entire possession tracking pipeline end-to-end.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.possession_tracker import PossessionTracker


def test_possession_data_integrity():
    """
    Test 1: Data Integrity
    
    Validates that possession data from processed videos is valid:
    - Team percentages add up to 100% (or 0% if no possession)
    - No negative durations
    - Possession changes are reasonable
    """
    print("=" * 70)
    print("TEST 1: POSSESSION DATA INTEGRITY")
    print("=" * 70)
    
    demo_outputs = project_root / "demo" / "demo_outputs"
    
    if not demo_outputs.exists():
        print("❌ No demo outputs found. Run demo/run_demo.py first.")
        return False
    
    videos = list(demo_outputs.iterdir())
    if not videos:
        print("❌ No processed videos found.")
        return False
    
    all_passed = True
    
    for video_dir in videos:
        if not video_dir.is_dir():
            continue
        
        metrics_file = video_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        print(f"\n✓ Checking: {video_dir.name}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        possession = metrics.get('possession', {})
        
        # Check 1: Team percentages
        team_poss = possession.get('team_possession_percentage', {})
        team_a_pct = team_poss.get('A', 0)
        team_b_pct = team_poss.get('B', 0)
        total_pct = team_a_pct + team_b_pct
        
        print(f"  Team A: {team_a_pct:.1f}%, Team B: {team_b_pct:.1f}%")
        
        if total_pct > 0:
            if abs(total_pct - 100.0) > 0.1:  # Allow small floating point error
                print(f"  ❌ FAIL: Percentages don't add up to 100% (got {total_pct:.1f}%)")
                all_passed = False
            else:
                print(f"  ✅ PASS: Percentages add up correctly")
        else:
            print(f"  ⚠️  WARNING: No possession data (0%)")
        
        # Check 2: Possession history durations
        poss_history = possession.get('possession_history', [])
        negative_durations = [e for e in poss_history if e.get('duration', 0) < 0]
        
        if negative_durations:
            print(f"  ❌ FAIL: Found {len(negative_durations)} negative durations")
            all_passed = False
        else:
            print(f"  ✅ PASS: No negative durations (checked {len(poss_history)} events)")
        
        # Check 3: Reasonable possession changes
        poss_changes = possession.get('total_possession_changes', 0)
        total_frames = metrics.get('frame', 1)
        
        if poss_changes > total_frames:
            print(f"  ❌ FAIL: More possession changes ({poss_changes}) than frames ({total_frames})")
            all_passed = False
        else:
            print(f"  ✅ PASS: Reasonable possession changes ({poss_changes} in {total_frames} frames)")
    
    return all_passed


def test_zone_statistics():
    """
    Test 2: Zone Statistics
    
    Validates that zone possession data is calculated correctly:
    - Zones add up to 100% per team (or 0%)
    - Zone names are valid
    """
    print("\n" + "=" * 70)
    print("TEST 2: ZONE STATISTICS")
    print("=" * 70)
    
    demo_outputs = project_root / "demo" / "demo_outputs"
    
    if not demo_outputs.exists():
        print("❌ No demo outputs found.")
        return False
    
    all_passed = True
    
    for video_dir in demo_outputs.iterdir():
        if not video_dir.is_dir():
            continue
        
        metrics_file = video_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        print(f"\n✓ Checking: {video_dir.name}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        possession = metrics.get('possession', {})
        zone_stats = possession.get('zone_stats', {})
        
        if not zone_stats:
            print("  ⚠️  No zone statistics available")
            continue
        
        # Check each team's zones
        for team in ['A', 'B']:
            if team not in zone_stats:
                continue
            
            team_zones = zone_stats[team]
            zone_total = sum(team_zones.values())
            
            print(f"  Team {team} zones:")
            print(f"    Defensive: {team_zones.get('Defensive', 0):.1f}%")
            print(f"    Midfield: {team_zones.get('Midfield', 0):.1f}%")
            print(f"    Attacking: {team_zones.get('Attacking', 0):.1f}%")
            print(f"    Total: {zone_total:.1f}%")
            
            # Validate zone percentages
            if zone_total > 0:
                if abs(zone_total - 100.0) > 0.1:
                    print(f"  ❌ FAIL: Team {team} zones don't add up to 100% (got {zone_total:.1f}%)")
                    all_passed = False
                else:
                    print(f"  ✅ PASS: Team {team} zones add up correctly")
            
            # Check valid zone names
            valid_zones = {'Defensive', 'Midfield', 'Attacking'}
            for zone_name in team_zones.keys():
                if zone_name not in valid_zones:
                    print(f"  ❌ FAIL: Invalid zone name '{zone_name}'")
                    all_passed = False
    
    return all_passed


def test_pressure_statistics():
    """
    Test 3: Pressure Statistics
    
    Validates pressure calculations:
    - Pressure counts are reasonable (0-11 opponents)
    - High pressure events are tracked
    """
    print("\n" + "=" * 70)
    print("TEST 3: PRESSURE STATISTICS")
    print("=" * 70)
    
    demo_outputs = project_root / "demo" / "demo_outputs"
    
    if not demo_outputs.exists():
        print("❌ No demo outputs found.")
        return False
    
    all_passed = True
    
    for video_dir in demo_outputs.iterdir():
        if not video_dir.is_dir():
            continue
        
        metrics_file = video_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        print(f"\n✓ Checking: {video_dir.name}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        possession = metrics.get('possession', {})
        pressure_stats = possession.get('pressure_stats', {})
        
        if not pressure_stats or pressure_stats.get('total_high_pressure_events', 0) == 0:
            print("  ⚠️  No pressure events recorded")
            continue
        
        total_events = pressure_stats.get('total_high_pressure_events', 0)
        avg_pressure = pressure_stats.get('avg_pressure_count', 0)
        max_pressure = pressure_stats.get('max_pressure_count', 0)
        
        print(f"  High pressure events: {total_events}")
        print(f"  Average pressure: {avg_pressure:.1f} opponents")
        print(f"  Max pressure: {max_pressure} opponents")
        
        # Validate pressure counts are reasonable (0-11 opponents)
        if max_pressure > 11:
            print(f"  ❌ FAIL: Max pressure ({max_pressure}) exceeds 11 opponents")
            all_passed = False
        elif max_pressure < 0:
            print(f"  ❌ FAIL: Negative pressure count ({max_pressure})")
            all_passed = False
        else:
            print(f"  ✅ PASS: Pressure counts are reasonable")
        
        if avg_pressure < 0 or avg_pressure > 11:
            print(f"  ❌ FAIL: Average pressure ({avg_pressure:.1f}) is out of range")
            all_passed = False
        else:
            print(f"  ✅ PASS: Average pressure is reasonable")
    
    return all_passed


def test_duration_statistics():
    """
    Test 4: Duration Statistics
    
    Validates possession duration analysis:
    - Durations are positive
    - Short/Medium/Long percentages add up to 100%
    - Categories make sense
    """
    print("\n" + "=" * 70)
    print("TEST 4: DURATION STATISTICS")
    print("=" * 70)
    
    demo_outputs = project_root / "demo" / "demo_outputs"
    
    if not demo_outputs.exists():
        print("❌ No demo outputs found.")
        return False
    
    all_passed = True
    
    for video_dir in demo_outputs.iterdir():
        if not video_dir.is_dir():
            continue
        
        metrics_file = video_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        print(f"\n✓ Checking: {video_dir.name}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        possession = metrics.get('possession', {})
        duration_stats = possession.get('duration_stats', {})
        
        if not duration_stats or duration_stats.get('total_possessions', 0) == 0:
            print("  ⚠️  No duration statistics available")
            continue
        
        avg_duration = duration_stats.get('avg_duration', 0)
        max_duration = duration_stats.get('max_duration', 0)
        min_duration = duration_stats.get('min_duration', 0)
        total_poss = duration_stats.get('total_possessions', 0)
        
        print(f"  Total possessions: {total_poss}")
        print(f"  Avg duration: {avg_duration:.2f}s")
        print(f"  Min duration: {min_duration:.2f}s")
        print(f"  Max duration: {max_duration:.2f}s")
        
        # Check for negative durations
        if min_duration < 0:
            print(f"  ❌ FAIL: Negative minimum duration ({min_duration:.2f}s)")
            all_passed = False
        else:
            print(f"  ✅ PASS: All durations are positive")
        
        # Check duration categories
        short_pct = duration_stats.get('short_pct', 0)
        medium_pct = duration_stats.get('medium_pct', 0)
        long_pct = duration_stats.get('long_pct', 0)
        total_pct = short_pct + medium_pct + long_pct
        
        print(f"  Duration breakdown:")
        print(f"    Short (<2s): {short_pct:.1f}%")
        print(f"    Medium (2-5s): {medium_pct:.1f}%")
        print(f"    Long (>5s): {long_pct:.1f}%")
        print(f"    Total: {total_pct:.1f}%")
        
        if abs(total_pct - 100.0) > 0.1:
            print(f"  ❌ FAIL: Duration percentages don't add up to 100% (got {total_pct:.1f}%)")
            all_passed = False
        else:
            print(f"  ✅ PASS: Duration percentages add up correctly")
    
    return all_passed


def run_all_tests():
    """Run all validation tests"""
    print("\n")
    print("=" * 70)
    print("POSSESSION TRACKING VALIDATION - REAL VIDEO DATA")
    print("=" * 70)
    print("\nThese tests validate possession tracking on actual processed videos.")
    print("Make sure you've run demo/run_demo.py first to generate test data.\n")
    
    results = []
    
    # Test 1: Data Integrity
    results.append(("Data Integrity", test_possession_data_integrity()))
    
    # Test 2: Zone Statistics
    results.append(("Zone Statistics", test_zone_statistics()))
    
    # Test 3: Pressure Statistics
    results.append(("Pressure Statistics", test_pressure_statistics()))
    
    # Test 4: Duration Statistics
    results.append(("Duration Statistics", test_duration_statistics()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Possession tracking is working correctly.\n")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED. Check the output above for details.\n")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
