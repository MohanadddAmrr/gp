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
from services.dynamic_roster_manager import DynamicRosterManager

def main():
    roster_dir = Path("rosters")
    db = DatabaseManager()
    db.initialize_database()
    dyn = DynamicRosterManager(Path(db.db_path))

    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]
    else:
        files = sorted(roster_dir.glob("*.json"))

    total_players = 0
    total_files = 0

    for f in files:
        print(f"Processing {f.name}...")
        summary = dyn.bulk_import_from_json(f)
        total_players += int(summary.get("players_touched", 0))
        total_files += 1
        print(f"  teams_touched={summary.get('teams_touched')}, players_touched={summary.get('players_touched')}")

    print(f"\nDone! Imported {total_files} roster file(s); player row operations: {total_players} (C2 lineup tables).")

if __name__ == "__main__":
    main()
