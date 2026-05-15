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
        st.subheader(f"{comp_label} — Recent Matches")
        limit = st.slider("How many matches to show?", 5, 30, 10)
        with st.spinner("Fetching matches…"):
            try:
                matches = sync.get_recent_matches(comp_id, limit=limit)
                _render_matches(matches)
            except Exception as exc:
                st.error(f"Could not load matches: {exc}")

    # ---- SQUAD LOOKUP ----
    with tab_squad:
        st.subheader("Team Squad Lookup")
        team_name = st.text_input(
            "Enter team name (e.g. Arsenal FC, Liverpool FC)",
            placeholder="Arsenal FC",
        )
        if st.button("🔍 Fetch Squad") and team_name.strip():
            with st.spinner(f"Fetching squad for {team_name}…"):
                try:
                    squad = sync.sync_team_squad(team_name.strip())
                    if squad:
                        st.success(f"Found {len(squad)} players for **{team_name}**")
                        rows = [
                            {
                                "#": p.get("jersey_number", "—"),
                                "Name": p.get("name", "—"),
                                "Position": p.get("position", "—"),
                            }
                            for p in squad
                        ]
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.warning(
                            f"No squad data found for '{team_name}'. "
                            "Try the full name, e.g. 'Arsenal FC'."
                        )
                except Exception as exc:
                    st.error(f"Error fetching squad: {exc}")

    # ---- Footer ----
    st.divider()
    st.caption(
        "Data source: [football-data.org](https://www.football-data.org) | "
        "Owner: Ahmed Khaled (Member E)"
    )
