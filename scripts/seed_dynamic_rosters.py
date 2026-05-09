"""
Seed dynamic player rosters into the database from JSON roster files.

Usage:
    python scripts/seed_dynamic_rosters.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.database_manager import DatabaseManager
from services.roster_sync_service import RosterSyncService

def main():
    roster_dir = Path("rosters")
    db = DatabaseManager()
    sync_service = RosterSyncService(db)

    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]
    else:
        files = sorted(roster_dir.glob("*.json"))

    total_players = 0
    total_rosters = 0
    
    for f in files:
        if f.name == "test.json":
            continue
            
        print(f"Processing {f.name}...")
        results = sync_service.sync_from_json(f)
        
        for side, result in results.items():
            print(f"  Side {side}: Roster ID {result.roster_id}, {result.players_upserted} players added/updated.")
            total_players += result.players_upserted
            total_rosters += 1

    print(f"\nDone! Seeded {total_rosters} rosters and {total_players} players into dynamic roster tables.")

if __name__ == "__main__":
    main()
