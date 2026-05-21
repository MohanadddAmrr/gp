"""
TactiVision to Database Integration Script

This script takes processed video analytics from demo/demo_outputs
and loads them into Ahmed's SQLite database (matches.db).

Usage:
    python integrate_to_database.py

What it does:
1. Reads metrics.json from processed videos
2. Creates match record in database
3. Inserts teams and players
4. Loads possession events (with zones, pressure, duration)
5. Loads ball tracking data
6. Loads player statistics
7. Links heatmap files

Author: TactiVision Team
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services.database import TactiVisionDB


def integrate_video_to_database(video_dir: Path, db: TactiVisionDB):
    """
    Integrate one processed video into the database.
    
    Args:
        video_dir: Path to demo_outputs/video_name directory
        db: Database connection object
    """
    print("\n" + "="*80)
    print(f"INTEGRATING: {video_dir.name}")
    print("="*80)
    
    # Load metrics.json
    metrics_file = video_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"❌ No metrics.json found in {video_dir}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract video metadata
    video_name = video_dir.name
    fps = 25.0  # Default, can be extracted from metrics if available
    width = 1280  # Default
    height = 720  # Default
    
    # If you have frame dimensions in metrics, extract them
    # (You might need to add this to your run_demo.py output)
    
    print(f"\n📹 Video: {video_name}")
    print(f"   FPS: {fps}, Size: {width}x{height}")
    
    # Step 1: Create match record
    print("\n1️⃣ Creating match record...")
    match_id = db.insert_match(
        video_name=video_name,
        team_a_name="Team A",
        team_b_name="Team B",
        venue="Demo Stadium",
        fps=fps,
        width=width,
        height=height
    )
    
    # Step 2: Create team records
    print("\n2️⃣ Creating team records...")
    team_ids = db.insert_teams(match_id, "Team A", "Team B")
    
    # Step 3: Create player records from tracking data
    print("\n3️⃣ Creating player records...")
    tracks_data = metrics.get('tracks', [])
    
    if not tracks_data:
        print("⚠️  No player tracking data found")
        return
    
    player_id_map = db.insert_players_from_tracks(match_id, team_ids, tracks_data)
    
    # Step 4: Insert possession events
    print("\n4️⃣ Inserting possession events...")
    possession_data = metrics.get('possession', {})
    possession_history = possession_data.get('possession_history', [])
    
    if possession_history:
        db.insert_possession_history(match_id, player_id_map, possession_history)
        
        # Also insert possession summary as metadata
        possession_summary = {
            'team_possession_percentage': possession_data.get('team_possession_percentage', {}),
            'total_possession_changes': possession_data.get('total_possession_changes', 0),
            'zone_stats': possession_data.get('zone_stats', {}),
            'pressure_stats': possession_data.get('pressure_stats', {}),
            'duration_stats': possession_data.get('duration_stats', {})
        }
        
        db.cursor.execute("""
            INSERT INTO events (match_id, event_type, timestamp, metadata)
            VALUES (?, ?, ?, ?)
        """, (match_id, 'possession_summary', 0.0, json.dumps(possession_summary)))
        db.commit()
        print(f"✅ Inserted possession summary with tactical metrics")
    else:
        print("⚠️  No possession history found")
    
    # Step 5: Insert ball tracking data
    print("\n5️⃣ Inserting ball tracking data...")
    ball_tracking = metrics.get('ball_tracking', {})
    
    if ball_tracking:
        db.insert_ball_tracking_events(match_id, ball_tracking)
    else:
        print("⚠️  No ball tracking data found")
    
    # Step 6: Insert player statistics
    print("\n6️⃣ Inserting player statistics...")
    possession_player_stats = possession_data.get('player_stats', {})
    db.insert_all_player_statistics(match_id, player_id_map, tracks_data, possession_player_stats)
    
    # Step 7: Insert heatmap references
    print("\n7️⃣ Inserting heatmap references...")
    db.insert_all_heatmaps(match_id, player_id_map, video_dir)
    
    print("\n✅ INTEGRATION COMPLETE!")
    print(f"   Match ID: {match_id}")
    print(f"   Players: {len(player_id_map)}")
    print(f"   Possession events: {len(possession_history)}")
    print("="*80)


def main():
    """Main integration function."""
    print("\n" + "="*80)
    print("TACTIVISION DATABASE INTEGRATION")
    print("="*80)
    print("\nThis script integrates processed video analytics into Ahmed's database.")
    
    # Connect to database
    db = TactiVisionDB("matches.db")
    db.connect()
    
    # Find all processed videos
    demo_outputs = Path("demo/demo_outputs")
    
    if not demo_outputs.exists():
        print(f"\n❌ No demo outputs found at {demo_outputs}")
        print("   Run demo/run_demo.py first to process videos!")
        db.disconnect()
        return
    
    video_dirs = [d for d in demo_outputs.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print(f"\n❌ No processed videos found in {demo_outputs}")
        db.disconnect()
        return
    
    print(f"\nFound {len(video_dirs)} processed video(s):")
    for vdir in video_dirs:
        print(f"  - {vdir.name}")
    
    # Integrate each video
    for video_dir in video_dirs:
        try:
            integrate_video_to_database(video_dir, db)
        except Exception as e:
            print(f"\n❌ Error integrating {video_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Show summary
    print("\n" + "="*80)
    print("DATABASE SUMMARY")
    print("="*80)
    
    all_matches = db.get_all_matches()
    print(f"\nTotal matches in database: {len(all_matches)}")
    
    for match in all_matches[-5:]:  # Show last 5 matches
        print(f"\n  Match ID: {match['id']}")
        print(f"  Date: {match['date']}")
        print(f"  Teams: {match['team_a']} vs {match['team_b']}")
        print(f"  Venue: {match['venue']}")
        
        # Count possession events
        poss_events = db.get_possession_events(match['id'])
        print(f"  Possession events: {len(poss_events)}")
    
    db.disconnect()
    
    print("\n" + "="*80)
    print("✅ INTEGRATION COMPLETE!")
    print("="*80)
    print("\nYour TactiVision data is now in Ahmed's database!")
    print("You can query it, visualize it, or export it as needed.")
    print("\nDatabase file: matches.db")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
