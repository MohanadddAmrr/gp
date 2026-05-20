"""
Live Match Data Dashboard Tab
Owner: Ahmed Khaled (Member E) — Task E2

Streamlit page that pulls real fixtures, standings, and recent results
from football-data.org and displays them in an interactive tab.

Usage (inside the main dashboard):
    from demo.dashboard_pages import live_match_data
    live_match_data.render()
"""

import os
import streamlit as st
from datetime import datetime
from typing import List, Dict
from pathlib import Path

# Auto-load .env file so API key is always available
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(_env)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Lazy imports — only load heavy services when the tab is actually rendered
# ---------------------------------------------------------------------------

def _load_sync():
    """Return a configured MatchDataSync, or None on failure."""
    try:
        from services.api_connector import APIManager
        from services.database_manager import DatabaseManager
        from services.match_data_sync import MatchDataSync

        api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
        if not api_key:
            return None, "⚠️ FOOTBALL_DATA_API_KEY not set in environment."

        api = APIManager()
        api.add_football_data(api_key)
        db = DatabaseManager()
        sync = MatchDataSync(api, db)
        return sync, None
    except Exception as exc:
        return None, f"❌ Failed to load sync service: {exc}"


# ---------------------------------------------------------------------------
# Competition map (football-data.org IDs)
# ---------------------------------------------------------------------------

COMPETITIONS = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League":    "2021",
    "🇩🇪 Bundesliga":              "2002",
    "🇪🇸 La Liga":                 "2014",
    "🇮🇹 Serie A":                 "2019",
    "🇫🇷 Ligue 1":                 "2015",
    "🏆 UEFA Champions League":    "2001",
}

# ---------------------------------------------------------------------------
# Known team IDs (football-data.org numeric IDs)
# The free tier ignores the ?name= query param on the /teams endpoint, so we
# resolve team names to IDs locally and call fetch_squad(id) directly.
# ---------------------------------------------------------------------------

KNOWN_TEAMS: dict = {
    # Premier League 2024-25
    "Arsenal FC":                    57,
    "Aston Villa FC":                58,
    "AFC Bournemouth":             1044,
    "Brentford FC":                 402,
    "Brighton & Hove Albion FC":    397,
    "Chelsea FC":                    61,
    "Crystal Palace FC":            354,
    "Everton FC":                    62,
    "Fulham FC":                     63,
    "Ipswich Town FC":              394,
    "Leicester City FC":            338,
    "Liverpool FC":                  64,
    "Manchester City FC":            65,
    "Manchester United FC":          66,
    "Newcastle United FC":           67,
    "Nottingham Forest FC":         351,
    "Southampton FC":               340,
    "Tottenham Hotspur FC":          73,
    "West Ham United FC":           563,
    "Wolverhampton Wanderers FC":    76,
    # Bundesliga
    "FC Bayern München":              5,
    "Borussia Dortmund":              4,
    "Bayer 04 Leverkusen":            3,
    "RB Leipzig":                   721,
    "Eintracht Frankfurt":           19,
    "VfB Stuttgart":                 10,
    # La Liga
    "Real Madrid CF":                86,
    "FC Barcelona":                  81,
    "Club Atlético de Madrid":       78,
    "Sevilla FC":                    90,
    "Real Sociedad de Fútbol":       92,
    "Villarreal CF":                 94,
    # Serie A
    "Juventus FC":                  109,
    "FC Internazionale Milano":     108,
    "AC Milan":                     98,
    "SSC Napoli":                   113,
    "AS Roma":                      100,
    "SS Lazio":                     110,
    # Ligue 1
    "Paris Saint-Germain FC":        524,
    "Olympique de Marseille":        516,
    "Olympique Lyonnais":            523,
    "AS Monaco FC":                  548,
    "Stade Rennais FC 1901":         511,
}


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _render_standings(standings: List[Dict]):
    """Render a league table from API standings data."""
    if not standings:
        st.info("No standings available for this competition.")
        return

    table = standings[0].get("table", [])
    if not table:
        st.info("Standings table is empty.")
        return

    rows = []
    for row in table:
        team_name = row.get("team", {}).get("name", "—")
        rows.append({
            "Pos":  row.get("position", ""),
            "Team": team_name,
            "P":    row.get("playedGames", 0),
            "W":    row.get("won", 0),
            "D":    row.get("draw", 0),
            "L":    row.get("lost", 0),
            "GF":   row.get("goalsFor", 0),
            "GA":   row.get("goalsAgainst", 0),
            "GD":   row.get("goalDifference", 0),
            "Pts":  row.get("points", 0),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_matches(matches: List[Dict]):
    """Render a results table from raw match dicts."""
    if not matches:
        st.info("No recent matches found.")
        return

    rows = []
    for m in matches:
        date_str = m.get("match_date", "")
        try:
            date_fmt = datetime.fromisoformat(date_str).strftime("%d %b %Y")
        except Exception:
            date_fmt = date_str[:10] if date_str else "—"

        home_score = m.get("home_score")
        away_score = m.get("away_score")
        score = f"{home_score} – {away_score}" if home_score is not None else "vs"

        rows.append({
            "Date":      date_fmt,
            "Home":      m.get("home_team", "—"),
            "Score":     score,
            "Away":      m.get("away_team", "—"),
            "Competition": m.get("competition", "—"),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render():
    """
    Render the Live Match Data tab.
    Call this from the main dashboard file inside the correct st.tab() block.
    """
    st.header("🌐 Live Match Data")
    st.caption("Powered by football-data.org — data refreshes on each page load.")

    # Load sync service
    sync, error = _load_sync()
    if error:
        st.error(error)
        st.info(
            "Make sure your `.env` file contains:\n```\nFOOTBALL_DATA_API_KEY=your_key_here\n```"
        )
        return

    # Competition selector
    comp_label = st.selectbox(
        "Select Competition",
        options=list(COMPETITIONS.keys()),
        index=0,
    )
    comp_id = COMPETITIONS[comp_label]

    # ---- Tabs inside this page ----
    tab_standings, tab_results, tab_squad = st.tabs(
        ["📊 Standings", "📅 Recent Results", "👥 Squad Lookup"]
    )

    # ---- STANDINGS ----
    with tab_standings:
        st.subheader(f"{comp_label} — Current Standings")
        with st.spinner("Fetching standings…"):
            try:
                standings = sync.get_standings(comp_id)
                _render_standings(standings)
            except Exception as exc:
                st.error(f"Could not load standings: {exc}")

    # ---- RECENT RESULTS ----
    with tab_results:
        st.subheader(f"{comp_label} — Match Results")

        # Auto-detect current matchday on first load
        with st.spinner("Detecting current gameweek…"):
            try:
                current_gw = sync.get_current_matchday(comp_id)
            except Exception:
                current_gw = 1

        gw_col, _ = st.columns([1, 2])
        with gw_col:
            gameweek = st.number_input(
                "📅 Gameweek",
                min_value=1,
                max_value=38,
                value=current_gw,
                step=1,
                help="Use the arrows or type a number to browse any gameweek (1–38).",
            )

        with st.spinner(f"Fetching Gameweek {int(gameweek)} matches…"):
            try:
                matches = sync.get_matches_by_gameweek(comp_id, matchday=int(gameweek))
                if matches:
                    st.caption(f"Gameweek **{int(gameweek)}** — {len(matches)} matches")
                    _render_matches(matches)
                else:
                    st.info(f"No matches found for Gameweek {int(gameweek)}.")
            except Exception as exc:
                st.error(f"Could not load matches: {exc}")

    # ---- SQUAD LOOKUP ----
    with tab_squad:
        st.subheader("Team Squad Lookup")
        st.caption(
            "Select a team from the list below. "
            "Squads are fetched directly by team ID (football-data.org free tier)."
        )

        team_options = sorted(KNOWN_TEAMS.keys())
        selected_team = st.selectbox(
            "Select a team",
            options=team_options,
            index=team_options.index("Arsenal FC") if "Arsenal FC" in team_options else 0,
        )

        if st.button("🔍 Fetch Squad"):
            team_id = KNOWN_TEAMS[selected_team]
            with st.spinner(f"Fetching squad for {selected_team}…"):
                try:
                    # Get the raw connector so we can call fetch_squad(id) directly
                    connector = sync._get_connector()
                    squad = connector.fetch_squad(team_id)
                    if squad:
                        st.success(f"Found **{len(squad)}** players for **{selected_team}**")
                        rows = [
                            {
                                "#":        p.get("jersey_number") or "—",
                                "Name":     p.get("name", "—"),
                                "Position": p.get("position", "—"),
                            }
                            for p in squad
                        ]
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.warning(
                            f"No squad data returned for {selected_team}. "
                            "The API may be rate-limiting — try again in a moment."
                        )
                except Exception as exc:
                    st.error(f"Error fetching squad: {exc}")

    # ---- Footer ----
    st.divider()
    st.caption(
        "Data source: [football-data.org](https://www.football-data.org) | "
        "Owner: Ahmed Khaled (Member E)"
    )
