"""
Seed player rosters into the database from JSON roster files.

Usage:
    python scripts/seed_rosters.py                  # seeds all rosters in rosters/
    python scripts/seed_rosters.py rosters/file.json # seeds a specific file
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.database_manager import DatabaseManager


def main():
    roster_dir = Path("rosters")
    db = DatabaseManager()

    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]
    else:
        files = sorted(roster_dir.glob("*.json"))

    total = 0
    for f in files:
        if f.name == "test.json":
            continue
        n = db.seed_rosters_from_json(f)
        total += n
        print(f"  {f.name}: {n} new profiles")

    print(f"Done — {total} new profiles created")

    # Show summary
    for team in ["Liverpool", "Bournemouth", "Manchester City", "Arsenal"]:
        roster = db.get_roster_for_team(team)
        if roster:
            print(f"\n  {team}: {len(roster)} players in DB")
            for p in roster[:3]:
                print(f"    #{p['jersey_number']} {p['name']} ({p['position_default'] or '-'})")
            if len(roster) > 3:
                print(f"    ... and {len(roster)-3} more")


if __name__ == "__main__":
    main()
