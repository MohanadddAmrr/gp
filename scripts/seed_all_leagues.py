"""
One-time setup: seed all top-5-league squads into the database.

After running this, the pipeline can look up any team roster from the DB
without needing a JSON roster file or live API access.

Usage:
    python scripts/seed_all_leagues.py              # seed all leagues
    python scripts/seed_all_leagues.py --league "Premier League"  # one league only
    python scripts/seed_all_leagues.py --team "Liverpool"         # one team only
    python scripts/seed_all_leagues.py --stats                    # show DB stats only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.database_manager import DatabaseManager

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "squads_top5.json"


def load_squad_data() -> dict:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_team(db: DatabaseManager, team: dict) -> int:
    """Seed one team's squad into the DB. Returns number of new profiles."""
    team_name = team["name"]
    # Load existing roster once per team
    existing = db.get_roster_for_team(team_name)
    existing_names = {p["name"] for p in existing}

    created = 0
    for number, name, position in team["squad"]:
        if name in existing_names:
            continue
        db.create_player_profile(
            name=name,
            team_name=team_name,
            jersey_number=number,
            position_default=position,
        )
        created += 1
    return created


def seed_all(db: DatabaseManager, data: dict, league_filter: str = None, team_filter: str = None):
    """Seed squads into the DB."""
    total_teams = 0
    total_players = 0

    for league in data["leagues"]:
        if league_filter and league["name"].lower() != league_filter.lower():
            continue

        print(f"\n  {league['name']} ({league['country']})")
        print(f"  {'=' * 40}")

        for team in league["teams"]:
            if team_filter and team["name"].lower() != team_filter.lower():
                continue

            created = seed_team(db, team)
            total_teams += 1
            total_players += created
            marker = f"+{created}" if created > 0 else "up-to-date"
            print(f"    {team['name']:30s} {len(team['squad']):3d} players  ({marker})")

    print(f"\n  TOTAL: {total_teams} teams, {total_players} new player profiles created")


def update_team_colors(data: dict):
    """Update config.yaml with team colors from squad data."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    colors = config.get("team_colors", {})
    added = 0
    for league in data["leagues"]:
        for team in league["teams"]:
            home_color = team.get("colors", {}).get("home")
            if home_color and team["name"] not in colors:
                colors[team["name"]] = home_color
                added += 1

    config["team_colors"] = colors
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Updated config.yaml: {added} new team colors added ({len(colors)} total)")


def show_stats(db: DatabaseManager):
    """Show DB roster statistics."""
    profiles = db.get_all_player_profiles()
    teams = {}
    for p in profiles:
        t = p.get("team_name", "Unknown")
        teams.setdefault(t, 0)
        teams[t] += 1

    print(f"\n  Database roster stats:")
    print(f"  {'=' * 40}")
    print(f"  Total player profiles: {len(profiles)}")
    print(f"  Teams represented: {len(teams)}")
    print()
    for team_name in sorted(teams.keys()):
        print(f"    {team_name:30s} {teams[team_name]:3d} players")


def main():
    parser = argparse.ArgumentParser(description="Seed squad data into the database")
    parser.add_argument("--league", help="Only seed a specific league")
    parser.add_argument("--team", help="Only seed a specific team")
    parser.add_argument("--stats", action="store_true", help="Show DB stats only")
    parser.add_argument("--no-colors", action="store_true", help="Skip updating config.yaml colors")
    args = parser.parse_args()

    db = DatabaseManager()
    data = load_squad_data()

    season = data.get("metadata", {}).get("season", "unknown")
    leagues = [l["name"] for l in data["leagues"]]
    team_count = sum(len(l["teams"]) for l in data["leagues"])
    player_count = sum(len(t["squad"]) for l in data["leagues"] for t in l["teams"])

    print(f"\n  TactiVision Pro - Squad Database Seeder")
    print(f"  Season: {season}")
    print(f"  Source: {DATA_FILE.name} ({team_count} teams, {player_count} players)")
    print(f"  Leagues: {', '.join(leagues)}")

    if args.stats:
        show_stats(db)
        return

    seed_all(db, data, league_filter=args.league, team_filter=args.team)

    if not args.no_colors:
        print()
        update_team_colors(data)

    show_stats(db)
    print("\n  Done. Rosters are cached locally in the database.")
    print("  No API access needed during match tracking.")


if __name__ == "__main__":
    main()
