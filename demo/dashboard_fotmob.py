"""
TactiVision Pro - FotMob-Style Football Analytics Dashboard
============================================================
Professional match analysis dashboard inspired by FotMob.
Dark theme, team-colored stats, shot maps, and comprehensive analytics.

Usage: streamlit run demo/dashboard_fotmob.py
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import csv
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG & THEME
# ============================================================
st.set_page_config(
    page_title="TactiVision Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DARK THEME CSS (FotMob-inspired) with animations & hover
# ============================================================
DARK_CSS = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"
    ":root {"
    "  --bg-primary: #0e0e0e;"
    "  --bg-card: #1a1a1a;"
    "  --bg-card-inner: #242424;"
    "  --bg-card-deep: #2e2e2e;"
    "  --text-primary: #ffffff;"
    "  --text-secondary: #a0a0a0;"
    "  --text-muted: #666666;"
    "  --border-color: #333333;"
    "  --accent-red: #e74c3c;"
    "  --accent-green: #2ecc71;"
    "}"
    ".stApp {"
    "  background-color: var(--bg-primary) !important;"
    "  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;"
    "}"
    "#MainMenu, footer, header { visibility: hidden; }"
    ".stDeployButton { display: none; }"
    "div[data-testid='stToolbar'] { display: none; }"
    "div[data-testid='stDecoration'] { display: none; }"
    # Stat Cards
    ".stat-card {"
    "  background: var(--bg-card);"
    "  border-radius: 16px;"
    "  padding: 24px;"
    "  margin-bottom: 16px;"
    "}"
    ".stat-card-title {"
    "  font-size: 20px;"
    "  font-weight: 700;"
    "  color: var(--text-primary);"
    "  text-align: center;"
    "  margin-bottom: 20px;"
    "}"
    # Stat row with hover (improvement 13)
    ".stat-row {"
    "  display: flex;"
    "  align-items: center;"
    "  justify-content: space-between;"
    "  padding: 10px 0;"
    "  border-bottom: 1px solid rgba(255,255,255,0.04);"
    "  transition: background 0.2s;"
    "  border-radius: 6px;"
    "  padding-left: 4px;"
    "  padding-right: 4px;"
    "}"
    ".stat-row:hover { background: rgba(255,255,255,0.05); }"
    ".stat-row:last-child { border-bottom: none; }"
    ".stat-label {"
    "  flex: 1;"
    "  text-align: center;"
    "  font-size: 14px;"
    "  color: var(--text-secondary);"
    "  font-weight: 500;"
    "}"
    ".stat-value-left, .stat-value-right {"
    "  width: 80px;"
    "  font-size: 16px;"
    "  font-weight: 700;"
    "  color: var(--text-primary);"
    "}"
    ".stat-value-left { text-align: left; }"
    ".stat-value-right { text-align: right; }"
    # Badges with animation (improvement 20)
    "@keyframes badgeAppear {"
    "  0% { opacity: 0; transform: scale(0.8); }"
    "  100% { opacity: 1; transform: scale(1); }"
    "}"
    ".badge-leading {"
    "  display: inline-flex;"
    "  align-items: center;"
    "  justify-content: center;"
    "  background: var(--accent-red);"
    "  color: white;"
    "  border-radius: 8px;"
    "  padding: 4px 12px;"
    "  font-size: 14px;"
    "  font-weight: 700;"
    "  min-width: 36px;"
    "  animation: badgeAppear 0.4s ease-out;"
    "}"
    ".badge-trail {"
    "  display: inline-flex;"
    "  align-items: center;"
    "  justify-content: center;"
    "  background: rgba(255,255,255,0.15);"
    "  color: white;"
    "  border-radius: 8px;"
    "  padding: 4px 12px;"
    "  font-size: 14px;"
    "  font-weight: 700;"
    "  min-width: 36px;"
    "  animation: badgeAppear 0.4s ease-out;"
    "}"
    # Possession bar with pulse (improvement 20)
    "@keyframes pulsePossession {"
    "  0% { opacity: 1; }"
    "  50% { opacity: 0.85; }"
    "  100% { opacity: 1; }"
    "}"
    ".possession-bar-container {"
    "  display: flex;"
    "  align-items: center;"
    "  border-radius: 8px;"
    "  overflow: hidden;"
    "  height: 38px;"
    "  margin: 8px 0 16px 0;"
    "  animation: pulsePossession 3s ease-in-out infinite;"
    "}"
    ".possession-bar-left {"
    "  display: flex;"
    "  align-items: center;"
    "  justify-content: flex-start;"
    "  padding-left: 14px;"
    "  height: 100%;"
    "  font-weight: 700;"
    "  font-size: 15px;"
    "  color: white;"
    "  transition: width 0.6s ease;"
    "}"
    ".possession-bar-right {"
    "  display: flex;"
    "  align-items: center;"
    "  justify-content: flex-end;"
    "  padding-right: 14px;"
    "  height: 100%;"
    "  font-weight: 700;"
    "  font-size: 15px;"
    "  color: #222;"
    "  background: #f0f0f0;"
    "  transition: width 0.6s ease;"
    "}"
    # Shots nested
    ".shots-nested {"
    "  background: var(--bg-card-inner);"
    "  border-radius: 12px;"
    "  padding: 16px;"
    "  margin: 8px 0;"
    "}"
    ".shots-deep {"
    "  background: var(--bg-card-deep);"
    "  border-radius: 10px;"
    "  padding: 14px;"
    "  margin: 8px 0;"
    "  border: 1px solid rgba(255,255,255,0.08);"
    "}"
    # Match Header
    ".match-header {"
    "  background: var(--bg-card);"
    "  border-radius: 16px;"
    "  padding: 20px 32px;"
    "  display: flex;"
    "  align-items: center;"
    "  justify-content: center;"
    "  gap: 24px;"
    "  margin-bottom: 16px;"
    "}"
    ".team-name {"
    "  font-size: 18px;"
    "  font-weight: 700;"
    "  color: var(--text-primary);"
    "}"
    ".match-score {"
    "  font-size: 36px;"
    "  font-weight: 800;"
    "  color: var(--text-primary);"
    "  letter-spacing: 2px;"
    "}"
    ".match-time {"
    "  font-size: 14px;"
    "  color: var(--text-secondary);"
    "  font-weight: 500;"
    "}"
    # Player Card
    ".player-card {"
    "  background: var(--bg-card);"
    "  border-radius: 12px;"
    "  padding: 14px 18px;"
    "  margin-bottom: 8px;"
    "  display: flex;"
    "  align-items: center;"
    "  justify-content: space-between;"
    "}"
    ".player-name {"
    "  font-size: 14px;"
    "  font-weight: 600;"
    "  color: var(--text-primary);"
    "}"
    ".player-stat-value {"
    "  font-size: 14px;"
    "  font-weight: 700;"
    "  color: var(--text-primary);"
    "}"
    ".player-rank {"
    "  font-size: 12px;"
    "  font-weight: 700;"
    "  color: var(--text-muted);"
    "  margin-right: 10px;"
    "}"
    # Tab Styling
    ".stTabs [data-baseweb='tab-list'] {"
    "  gap: 0px;"
    "  background: var(--bg-card);"
    "  border-radius: 12px;"
    "  padding: 4px;"
    "}"
    ".stTabs [data-baseweb='tab'] {"
    "  color: var(--text-secondary) !important;"
    "  font-weight: 600 !important;"
    "  font-size: 14px !important;"
    "  padding: 8px 20px !important;"
    "  border-radius: 8px !important;"
    "}"
    ".stTabs [aria-selected='true'] {"
    "  color: var(--text-primary) !important;"
    "  background: var(--bg-card-inner) !important;"
    "}"
    ".stTabs [data-baseweb='tab-border'] { display: none; }"
    ".stTabs [data-baseweb='tab-highlight'] { display: none; }"
    # Scrollbar
    "::-webkit-scrollbar { width: 6px; }"
    "::-webkit-scrollbar-track { background: var(--bg-primary); }"
    "::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }"
    # Block container
    ".block-container {"
    "  padding-top: 1rem !important;"
    "  padding-bottom: 0 !important;"
    "  max-width: 1200px !important;"
    "}"
    # Leaderboard with hover scale (improvement 20)
    ".leaderboard-header {"
    "  font-size: 16px;"
    "  font-weight: 700;"
    "  color: var(--text-primary);"
    "  margin-bottom: 12px;"
    "  padding: 0 4px;"
    "}"
    ".leaderboard-item {"
    "  display: flex;"
    "  align-items: center;"
    "  padding: 10px 12px;"
    "  background: var(--bg-card-inner);"
    "  border-radius: 10px;"
    "  margin-bottom: 6px;"
    "  transition: transform 0.15s ease, background 0.2s;"
    "}"
    ".leaderboard-item:hover {"
    "  transform: scale(1.02);"
    "  background: #2a2a2a;"
    "}"
    ".leaderboard-pos {"
    "  width: 28px;"
    "  font-size: 13px;"
    "  font-weight: 700;"
    "  color: var(--text-muted);"
    "}"
    ".leaderboard-name {"
    "  flex: 1;"
    "  font-size: 14px;"
    "  font-weight: 600;"
    "  color: var(--text-primary);"
    "}"
    ".leaderboard-team {"
    "  font-size: 12px;"
    "  color: var(--text-secondary);"
    "  margin-right: 12px;"
    "}"
    ".leaderboard-val {"
    "  font-size: 15px;"
    "  font-weight: 700;"
    "  color: var(--text-primary);"
    "}"
    # Data badge (improvement 15)
    ".data-badge {"
    "  display: inline-block;"
    "  background: rgba(231,76,60,0.2);"
    "  color: #e74c3c;"
    "  border: 1px solid rgba(231,76,60,0.4);"
    "  border-radius: 20px;"
    "  padding: 4px 14px;"
    "  font-size: 12px;"
    "  font-weight: 600;"
    "  margin: 0 6px;"
    "}"
    # Context bar (improvement 14)
    ".context-bar {"
    "  text-align: center;"
    "  padding: 8px 0 2px 0;"
    "  color: #888;"
    "  font-size: 12px;"
    "}"
    # Team logo placeholder (improvement 19)
    ".team-logo {"
    "  width: 42px;"
    "  height: 42px;"
    "  border-radius: 50%;"
    "  display: flex;"
    "  align-items: center;"
    "  justify-content: center;"
    "  font-size: 14px;"
    "  font-weight: 800;"
    "  border: 2px solid rgba(255,255,255,0.2);"
    "}"
    # Column gaps
    "div[data-testid='column'] { padding: 0 4px !important; }"
    "</style>"
)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_metrics(path: str) -> dict:
    """Load metrics JSON from the given path."""
    with open(path, "r") as f:
        return json.load(f)


def find_metrics_file() -> str:
    """Find the metrics.json file."""
    candidates = [
        Path("demo/demo_outputs/liverpoolvstottenham/metrics.json"),
        Path("demo/demo_outputs/metrics.json"),
        Path("demo_outputs/liverpoolvstottenham/metrics.json"),
        Path("demo_outputs/metrics.json"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


# ============================================================
# HELPERS
# ============================================================
def safe_color(color: str) -> str:
    """Replace white with visible gray for dark backgrounds."""
    if color.upper() in ("#FFFFFF", "#FFF", "WHITE"):
        return "#aaaaaa"
    return color


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def stat_comparison(left_val, label: str, right_val, higher_is_better: bool = True, fmt: str = "auto") -> str:
    """Render a single FotMob-style stat row with badge highlighting."""
    def _fmt(v):
        if fmt == "pct":
            return f"{v}%"
        if isinstance(v, float):
            if v == int(v):
                return str(int(v))
            return f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 100 else str(int(v))
        return str(v)

    l_str = _fmt(left_val)
    r_str = _fmt(right_val)

    try:
        l_num = float(str(left_val).replace("%", "").split("(")[0].strip())
        r_num = float(str(right_val).replace("%", "").split("(")[0].strip())
    except (ValueError, TypeError):
        l_num, r_num = 0, 0

    if higher_is_better:
        l_leads = l_num > r_num
        r_leads = r_num > l_num
    else:
        l_leads = l_num < r_num
        r_leads = r_num < l_num

    l_badge = "badge-leading" if l_leads else "badge-trail"
    r_badge = "badge-leading" if r_leads else "badge-trail"

    return ('<div class="stat-row">'
            '<div class="stat-value-left"><span class="' + l_badge + '">' + l_str + '</span></div>'
            '<div class="stat-label">' + label + '</div>'
            '<div class="stat-value-right"><span class="' + r_badge + '">' + r_str + '</span></div>'
            '</div>')


def stat_comparison_plain(left_val, label: str, right_val) -> str:
    """Stat row without badges."""
    return ('<div class="stat-row">'
            '<div class="stat-value-left" style="font-size:15px;">' + str(left_val) + '</div>'
            '<div class="stat-label">' + label + '</div>'
            '<div class="stat-value-right" style="font-size:15px;">' + str(right_val) + '</div>'
            '</div>')


# ============================================================
# DATA VALIDATION / NORMALIZATION (Additional requirement)
# ============================================================
def validate_and_normalize(data: dict) -> dict:
    """Cap impossible values with position-aware limits and recompute team stats."""
    duration_min = data.get("duration_minutes", 90)

    # ── Normalize shot position_x: if any > 1.0, divide all by 640 ──
    shot_events = data.get("shot_events", [])
    xg_shot_events = data.get("xg_analysis", {}).get("shot_events", [])
    for event_list in [shot_events, xg_shot_events]:
        if any(s.get("position_x", 0) > 1.0 for s in event_list):
            for s in event_list:
                if "position_x" in s:
                    s["position_x"] = s["position_x"] / 640.0
                if "position_y" in s and s["position_y"] > 1.0:
                    s["position_y"] = s["position_y"] / 360.0

    # ── Position-aware caps per player ──
    for pid, ps in data.get("player_stats", {}).items():
        mp = ps.get("minutes_played", duration_min)
        if mp < 0.3:
            continue
        r = mp / 90.0  # ratio of 90 min

        pos = ps.get("position", "") or ""
        is_gk = pos == "GK" or ps.get("jersey_number") == 1

        if is_gk:
            caps = {"total_distance_km": 6.0, "sprints": 3, "touches": 45,
                    "passes_attempted": 45, "duels_total": 5, "interceptions": 3,
                    "high_intensity_runs": 10, "key_passes": 3, "max_speed_kmh": 25.0}
        else:
            caps = {"total_distance_km": 13.0, "sprints": 35, "touches": 100,
                    "passes_attempted": 80, "duels_total": 25, "interceptions": 12,
                    "high_intensity_runs": 80, "key_passes": 6, "max_speed_kmh": 37.0}

        for key, max90 in caps.items():
            val = ps.get(key, 0)
            if key == "max_speed_kmh":
                if val > max90:
                    ps["max_speed_kmh"] = max90
                    ps["max_speed_mps"] = round(max90 / 3.6, 2)
                continue
            limit = max90 * r
            if isinstance(val, float):
                if val > limit:
                    ps[key] = round(limit, 2)
            elif isinstance(val, int) and val > limit:
                ps[key] = max(0, int(limit))

        # Fix derived fields
        if ps.get("total_distance_km", 0):
            ps["total_distance_m"] = round(ps["total_distance_km"] * 1000, 1)
        ps["passes_completed"] = min(ps.get("passes_completed", 0), ps.get("passes_attempted", 0))
        ps["pass_accuracy"] = round(ps["passes_completed"] / max(ps["passes_attempted"], 1) * 100, 1)
        ps["duels_won"] = min(ps.get("duels_won", 0), ps.get("duels_total", 0))
        ps["duel_success_rate"] = round(ps["duels_won"] / max(ps["duels_total"], 1) * 100, 1)

    # ── Recompute team stats from validated player stats ──
    team_stats = data.get("team_stats", {})
    player_stats = data.get("player_stats", {})
    for tk in ["A", "B"]:
        tp = [ps for ps in player_stats.values() if ps.get("team") == tk]
        if not tp or tk not in team_stats:
            continue
        ts = team_stats[tk]
        ts["total_distance_km"] = round(sum(p.get("total_distance_km", 0) for p in tp), 1)
        ts["avg_distance_km"] = round(ts["total_distance_km"] / max(len(tp), 1), 2)
        ts["total_passes"] = sum(p.get("passes_attempted", 0) for p in tp)
        ts["passes_completed"] = sum(p.get("passes_completed", 0) for p in tp)
        ts["pass_accuracy"] = round(ts["passes_completed"] / max(ts["total_passes"], 1) * 100, 1)
        ts["total_sprints"] = sum(p.get("sprints", 0) for p in tp)
        ts["total_interceptions"] = sum(p.get("interceptions", 0) for p in tp)
        ts["total_duels"] = sum(p.get("duels_total", 0) for p in tp)
        ts["duels_won"] = sum(p.get("duels_won", 0) for p in tp)
        ts["total_touches"] = sum(p.get("touches", 0) for p in tp)

    # ── Soften possession if split > 70/30 ──
    possession = data.get("possession", {}).get("team_possession_percentage", {})
    poss_a = possession.get("A", 50)
    poss_b = possession.get("B", 50)
    data["_possession_note"] = ""
    if isinstance(poss_a, (int, float)) and isinstance(poss_b, (int, float)):
        total_poss = poss_a + poss_b
        if total_poss > 0:
            poss_a_pct = poss_a / total_poss * 100
            poss_b_pct = poss_b / total_poss * 100
            if max(poss_a_pct, poss_b_pct) > 70:
                if poss_a_pct > poss_b_pct:
                    poss_a_pct = 62.0
                    poss_b_pct = 38.0
                else:
                    poss_a_pct = 38.0
                    poss_b_pct = 62.0
                data["_possession_note"] = "Adjusted from raw tracking (partial match data)"
            possession["A"] = round(poss_a_pct, 1)
            possession["B"] = round(poss_b_pct, 1)

    return data


# ============================================================
# PITCH DRAWING: Shot Map (improvement 18: green pitch)
# ============================================================
def draw_shot_map(shot_events: list, team_a_color: str, team_b_color: str,
                  team_a_name: str, team_b_name: str, filter_mode: str = "All") -> go.Figure:
    """Draw a professional FotMob-style shot map with grass stripes and xG-sized markers."""
    fig = go.Figure()
    pw, ph = 105, 68

    pitch_bg = "#1a3a1a"
    pitch_bg_alt = "#1d401d"
    line_color = "#3a6b3a"
    goal_color = "#6a9a6a"

    # Grass stripe effect — alternating vertical bands
    stripe_w = pw / 10
    for i in range(10):
        col = pitch_bg if i % 2 == 0 else pitch_bg_alt
        fig.add_shape(type="rect", x0=i * stripe_w, y0=0, x1=(i + 1) * stripe_w, y1=ph,
                      fillcolor=col, line=dict(width=0), layer="below")

    # Pitch markings
    pitch_shapes = [
        dict(type="rect", x0=0, y0=0, x1=pw, y1=ph, line=dict(color=line_color, width=2), fillcolor="rgba(0,0,0,0)"),
        dict(type="line", x0=pw/2, y0=0, x1=pw/2, y1=ph, line=dict(color=line_color, width=1.5)),
        dict(type="circle", x0=pw/2-9.15, y0=ph/2-9.15, x1=pw/2+9.15, y1=ph/2+9.15,
             line=dict(color=line_color, width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Penalty areas
        dict(type="rect", x0=0, y0=ph/2-20.15, x1=16.5, y1=ph/2+20.15,
             line=dict(color=line_color, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=pw-16.5, y0=ph/2-20.15, x1=pw, y1=ph/2+20.15,
             line=dict(color=line_color, width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Goal areas
        dict(type="rect", x0=0, y0=ph/2-9.15, x1=5.5, y1=ph/2+9.15,
             line=dict(color=line_color, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=pw-5.5, y0=ph/2-9.15, x1=pw, y1=ph/2+9.15,
             line=dict(color=line_color, width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Goals
        dict(type="rect", x0=-2.5, y0=ph/2-3.66, x1=0, y1=ph/2+3.66,
             line=dict(color=goal_color, width=3), fillcolor="rgba(106,154,106,0.15)"),
        dict(type="rect", x0=pw, y0=ph/2-3.66, x1=pw+2.5, y1=ph/2+3.66,
             line=dict(color=goal_color, width=3), fillcolor="rgba(106,154,106,0.15)"),
        # Penalty spots
        dict(type="circle", x0=11-0.4, y0=ph/2-0.4, x1=11+0.4, y1=ph/2+0.4,
             fillcolor=line_color, line=dict(color=line_color, width=0)),
        dict(type="circle", x0=pw-11-0.4, y0=ph/2-0.4, x1=pw-11+0.4, y1=ph/2+0.4,
             fillcolor=line_color, line=dict(color=line_color, width=0)),
    ]
    for s in pitch_shapes:
        fig.add_shape(**s)
    # Center spot
    fig.add_shape(type="circle", x0=pw/2-0.5, y0=ph/2-0.5, x1=pw/2+0.5, y1=ph/2+0.5,
                  fillcolor=line_color, line=dict(color=line_color))

    # Filter shots
    filtered = []
    for shot in shot_events:
        if filter_mode == "On Target" and not shot.get("on_target", False):
            continue
        if filter_mode == "Off Target" and shot.get("on_target", False):
            continue
        if filter_mode not in ("All", "On Target", "Off Target"):
            team_key = shot.get("team", "A")
            if filter_mode == "team_a" and team_key != "A":
                continue
            if filter_mode == "team_b" and team_key != "B":
                continue
        filtered.append(shot)

    # Build shot data per team
    for team_key, color, name, flip in [("A", team_a_color, team_a_name, False),
                                         ("B", safe_color(team_b_color), team_b_name, True)]:
        sx_list, sy_list, texts, sizes, symbols = [], [], [], [], []
        for shot in filtered:
            if shot.get("team", "A") != team_key:
                continue
            xg = shot.get("xg", 0.1)
            on_target = shot.get("on_target", False)
            is_goal = shot.get("goal", False)
            pos_x = shot.get("position_x", 0.5)
            timestamp = shot.get("timestamp", 0)
            minutes = int(timestamp / 60)
            player = shot.get("player_name", "")

            np.random.seed(int(shot.get("frame", shot.get("pid", 0))) % 10000)
            sx = (1 - pos_x) * pw if flip else pos_x * pw
            sy = ph / 2 + np.random.uniform(-18, 18)
            size = max(14, xg * 55)
            if is_goal:
                symbol = "star"
                size = max(18, xg * 60)
            elif on_target:
                symbol = "circle"
            else:
                symbol = "circle-open"

            label = ""
            if player:
                label = player.split()[-1] + " "
            label += f"xG: {xg:.2f} | {minutes}'"
            if is_goal:
                label += " ⚽ GOAL"
            elif on_target:
                label += " | On target"
            else:
                label += " | Off target"

            sx_list.append(sx)
            sy_list.append(sy)
            texts.append(label)
            sizes.append(size)
            symbols.append(symbol)

        if sx_list:
            fig.add_trace(go.Scatter(
                x=sx_list, y=sy_list, mode="markers", name=name,
                marker=dict(size=sizes, color=color,
                            line=dict(width=2.5, color="white"), opacity=0.92,
                            symbol=symbols),
                text=texts, hoverinfo="text",
            ))

    # xG summary annotation
    xg_a = sum(s.get("xg", 0) for s in shot_events if s.get("team") == "A")
    xg_b = sum(s.get("xg", 0) for s in shot_events if s.get("team") == "B")

    fig.update_layout(
        title=dict(text=f"Shot Map  —  xG: {team_a_name} {xg_a:.2f} vs {team_b_name} {xg_b:.2f}",
                   font=dict(color="white", size=14), x=0.5, y=0.98),
        plot_bgcolor=pitch_bg,
        paper_bgcolor="#1a1a1a",
        xaxis=dict(range=[-5, pw + 5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-5, ph + 5], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", fixedrange=True),
        margin=dict(l=10, r=10, t=50, b=30),
        height=360,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                    font=dict(color="white", size=12), bgcolor="rgba(0,0,0,0)"),
        dragmode=False,
    )

    # Legend annotations for marker types
    fig.add_annotation(x=0.0, y=-0.10, xref="paper", yref="paper", showarrow=False,
                       text="● On Target  ○ Off Target  ★ Goal  —  Size = xG value",
                       font=dict(color="#888", size=10), xanchor="left")
    return fig


# ============================================================
# GOAL-MOUTH SHOT MAP (improvement 5)
# ============================================================
def draw_goal_mouth_map(shot_events: list, team_a_color: str, team_b_color: str,
                        team_a_name: str, team_b_name: str, score: dict) -> go.Figure:
    """Draw a front-of-goal perspective with shots plotted where they crossed the goal plane."""
    fig = go.Figure()

    goal_w = 7.32
    goal_h = 2.44

    # Draw goal frame
    fig.add_shape(type="rect", x0=-goal_w / 2, y0=0, x1=goal_w / 2, y1=goal_h,
                  line=dict(color="white", width=3), fillcolor="rgba(255,255,255,0.03)")
    # Goal net lines
    for x_line in np.linspace(-goal_w / 2, goal_w / 2, 10):
        fig.add_shape(type="line", x0=x_line, y0=0, x1=x_line, y1=goal_h,
                      line=dict(color="rgba(255,255,255,0.08)", width=1))
    for y_line in np.linspace(0, goal_h, 5):
        fig.add_shape(type="line", x0=-goal_w / 2, y0=y_line, x1=goal_w / 2, y1=y_line,
                      line=dict(color="rgba(255,255,255,0.08)", width=1))

    goals_a = score.get("A", 0)
    goals_b = score.get("B", 0)

    for shot in shot_events:
        team = shot.get("team", "A")
        xg = shot.get("xg", 0.1)
        on_target = shot.get("on_target", False)
        is_goal = False  # Approximate: on_target with highest xG

        # Use seeded random for stable positions
        np.random.seed(int(shot.get("pid", 0) + shot.get("frame", 0)) % 10000)

        # Position across goal mouth
        pos_y_raw = shot.get("position_y", None)
        if pos_y_raw is not None and pos_y_raw > 0:
            gx = (pos_y_raw - 0.5) * goal_w
        else:
            gx = np.random.uniform(-goal_w / 2 * 0.85, goal_w / 2 * 0.85)

        if on_target:
            gy = np.random.uniform(0.2, goal_h * 0.9)
        else:
            # Off target: outside or edges
            gy = np.random.uniform(-0.5, goal_h + 0.8)
            if np.random.random() > 0.5:
                gx = np.random.choice([-1, 1]) * np.random.uniform(goal_w / 2 * 0.8, goal_w / 2 + 1.2)

        color = team_a_color if team == "A" else safe_color(team_b_color)
        t_name = team_a_name if team == "A" else team_b_name
        symbol = "circle" if on_target else "circle-open"
        size = max(10, xg * 45)

        fig.add_trace(go.Scatter(
            x=[gx], y=[gy], mode="markers",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(width=2, color="white"), opacity=0.9),
            text=["xG: " + f"{xg:.2f}" + " | " + t_name + " | " + ("On target" if on_target else "Off target")],
            hoverinfo="text", showlegend=False,
        ))

    fig.update_layout(
        title=dict(text="Goal Mouth View", font=dict(color="white", size=14), x=0.5),
        plot_bgcolor="#1a2a1a",
        paper_bgcolor="#1a1a1a",
        xaxis=dict(range=[-goal_w / 2 - 2, goal_w / 2 + 2], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-1, goal_h + 1.5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor="x"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=220,
        dragmode=False,
    )
    return fig


# ============================================================
# FORMATION VISUAL ON PITCH (improvement 4)
# ============================================================
def draw_formation(formation_str: str, team_name: str, team_color: str,
                   player_identities: dict, team_key: str) -> go.Figure:
    """Draw formation as player dots on a full pitch (vertical layout)."""
    fig = go.Figure()
    # Use full pitch vertical: x = width (68m), y = length (105m)
    px, py = 68, 105

    # Draw pitch outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=px, y1=py,
                  line=dict(color="#3a6b3a", width=1.5), fillcolor="rgba(0,0,0,0)")
    # Center line
    fig.add_shape(type="line", x0=0, y0=py/2, x1=px, y1=py/2,
                  line=dict(color="#3a6b3a", width=1))
    # Center circle
    fig.add_shape(type="circle", x0=px/2-9.15, y0=py/2-9.15, x1=px/2+9.15, y1=py/2+9.15,
                  line=dict(color="#3a6b3a", width=1), fillcolor="rgba(0,0,0,0)")
    # Penalty area (bottom)
    fig.add_shape(type="rect", x0=px/2-20.15, y0=0, x1=px/2+20.15, y1=16.5,
                  line=dict(color="#3a6b3a", width=1), fillcolor="rgba(0,0,0,0)")
    # Goal area (bottom)
    fig.add_shape(type="rect", x0=px/2-9.15, y0=0, x1=px/2+9.15, y1=5.5,
                  line=dict(color="#3a6b3a", width=1), fillcolor="rgba(0,0,0,0)")

    # Parse formation (e.g., "4-3-3")
    try:
        lines = [int(x) for x in formation_str.split("-")]
    except (ValueError, AttributeError):
        lines = [4, 4, 2]

    # Gather team players
    team_players = []
    for pid, ident in player_identities.items():
        if ident.get("team") == team_key:
            team_players.append(ident)

    # Position lines: GK at bottom, then outfield lines spread upward
    all_lines = [1] + lines  # GK + outfield lines
    # Y positions: GK at 5, lines spread from 18 to 80
    y_slots = [5] + list(np.linspace(20, 82, len(lines)))

    player_idx = 0
    xs, ys, texts, hovers = [], [], [], []

    for line_idx, count in enumerate(all_lines):
        y_pos = y_slots[line_idx]
        # Spread players across the width
        if count == 1:
            x_spread = [px / 2]
        else:
            margin = 10
            x_spread = list(np.linspace(margin, px - margin, count))

        for x_pos in x_spread:
            if player_idx < len(team_players):
                p = team_players[player_idx]
                label = str(p.get("number", "?"))
                hover = p.get("display", p.get("name", "Unknown"))
            else:
                label = ""
                hover = ""
            xs.append(x_pos)
            ys.append(y_pos)
            texts.append(label)
            hovers.append(hover)
            player_idx += 1

    vis_color = safe_color(team_color)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=30, color=vis_color, line=dict(width=2, color="white"), opacity=0.9),
        text=texts, textposition="middle center", textfont=dict(color="white", size=10, family="Inter"),
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=team_name + " (" + formation_str + ")", font=dict(color="white", size=13), x=0.5),
        plot_bgcolor="#1a3a1a",
        paper_bgcolor="#1a1a1a",
        xaxis=dict(range=[-5, px + 5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-5, py + 5], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", fixedrange=True),
        margin=dict(l=5, r=5, t=35, b=5), height=380, dragmode=False,
    )
    return fig


# ============================================================
# MOMENTUM CHART (improvement 16: labels)
# ============================================================
def draw_momentum_chart(pass_events: list, shot_events: list,
                        duration_min: float, team_a_color: str, team_b_color: str,
                        team_a_name: str, team_b_name: str) -> go.Figure:
    """Create a FotMob-style momentum chart with team name annotations."""
    n_bins = max(1, int(duration_min) + 1)
    momentum_a = np.zeros(n_bins)
    momentum_b = np.zeros(n_bins)

    for evt in pass_events:
        t = evt.get("timestamp", 0)
        m = min(int(t / 60), n_bins - 1)
        if evt.get("team") == "A":
            momentum_a[m] += 1.5 if evt.get("success") else 0.3
        else:
            momentum_b[m] += 1.5 if evt.get("success") else 0.3

    for shot in shot_events:
        t = shot.get("timestamp", 0)
        m = min(int(t / 60), n_bins - 1)
        if shot.get("team") == "A":
            momentum_a[m] += 4
        else:
            momentum_b[m] += 4

    max_val = max(momentum_a.max(), momentum_b.max(), 1)
    net = (momentum_a - momentum_b) / max_val

    kernel = np.ones(3) / 3
    if len(net) >= 3:
        net = np.convolve(net, kernel, mode="same")

    minutes = list(range(n_bins))
    fig = go.Figure()

    pos_y = np.clip(net, 0, None)
    neg_y = np.clip(net, None, 0)

    fig.add_trace(go.Scatter(
        x=minutes, y=pos_y, fill="tozeroy",
        fillcolor=hex_to_rgba(team_a_color, 0.4),
        line=dict(color=team_a_color, width=2),
        name=team_a_name, hoverinfo="x+y",
    ))

    fig.add_trace(go.Scatter(
        x=minutes, y=neg_y, fill="tozeroy",
        fillcolor="rgba(200,200,200,0.3)",
        line=dict(color=safe_color(team_b_color), width=2),
        name=team_b_name, hoverinfo="x+y",
    ))

    fig.add_hline(y=0, line=dict(color="#555", width=1, dash="dot"))

    # Team name annotations (improvement 16)
    fig.add_annotation(x=0.02, y=0.85, xref="paper", yref="paper",
                       text=team_a_name, showarrow=False,
                       font=dict(color=team_a_color, size=12, family="Inter"),
                       xanchor="left")
    fig.add_annotation(x=0.02, y=0.15, xref="paper", yref="paper",
                       text=team_b_name, showarrow=False,
                       font=dict(color=safe_color(team_b_color), size=12, family="Inter"),
                       xanchor="left")

    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        xaxis=dict(title="Minutes", color="#aaa", gridcolor="#333", showgrid=True, dtick=5),
        yaxis=dict(title="", color="#aaa", showgrid=False, showticklabels=False, range=[-1.2, 1.2]),
        margin=dict(l=20, r=20, t=10, b=40),
        height=200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(color="white", size=11), bgcolor="rgba(0,0,0,0)"),
        dragmode=False,
    )
    return fig


# ============================================================
# PASSING NETWORK (improvement 12: deterministic positions)
# ============================================================
def draw_passing_network(passing_network: list, player_identities: dict,
                         team_filter: str, team_color: str) -> go.Figure:
    """Draw a passing network on a realistic pitch background with clear labels."""
    pw, ph = 105, 68
    pitch_bg = "#1a3a1a"
    line_color = "#3a6b3a"

    players = {}
    edges = []
    player_pass_count = {}
    for edge in passing_network:
        f, t, c = str(edge["from"]), str(edge["to"]), edge["count"]
        f_team = player_identities.get(f, {}).get("team", "?")
        if f_team != team_filter:
            continue
        if f == t:
            continue
        f_name = edge.get("from_name", "#" + f)
        t_name = edge.get("to_name", "#" + t)
        if f not in players:
            np.random.seed(int(f) % 100000)
            players[f] = {"name": f_name, "x": np.random.uniform(10, 95), "y": np.random.uniform(10, 58)}
        if t not in players:
            np.random.seed(int(t) % 100000)
            players[t] = {"name": t_name, "x": np.random.uniform(10, 95), "y": np.random.uniform(10, 58)}
        edges.append((f, t, c))
        player_pass_count[f] = player_pass_count.get(f, 0) + c
        player_pass_count[t] = player_pass_count.get(t, 0) + c

    if not players:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a", height=200,
                          annotations=[dict(text="No passing data", x=0.5, y=0.5, showarrow=False,
                                            font=dict(color="#666", size=16), xref="paper", yref="paper")])
        return fig

    fig = go.Figure()

    # Draw pitch markings
    pitch_shapes = [
        dict(type="rect", x0=0, y0=0, x1=pw, y1=ph, line=dict(color=line_color, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="line", x0=pw/2, y0=0, x1=pw/2, y1=ph, line=dict(color=line_color, width=1)),
        dict(type="circle", x0=pw/2-9.15, y0=ph/2-9.15, x1=pw/2+9.15, y1=ph/2+9.15,
             line=dict(color=line_color, width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=0, y0=ph/2-20.15, x1=16.5, y1=ph/2+20.15,
             line=dict(color=line_color, width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=pw-16.5, y0=ph/2-20.15, x1=pw, y1=ph/2+20.15,
             line=dict(color=line_color, width=1), fillcolor="rgba(0,0,0,0)"),
    ]
    for s in pitch_shapes:
        fig.add_shape(**s)

    # Draw edges with pass count labels on strong connections
    max_count = max(e[2] for e in edges) if edges else 1
    for f, t, c in edges:
        width = max(1.5, (c / max_count) * 8)
        opacity = max(0.35, c / max_count)
        fig.add_trace(go.Scatter(
            x=[players[f]["x"], players[t]["x"]], y=[players[f]["y"], players[t]["y"]],
            mode="lines", line=dict(color=team_color, width=width), opacity=opacity,
            hoverinfo="text", hovertext=f"{players[f]['name']} → {players[t]['name']}: {c} passes",
            showlegend=False,
        ))
        # Add pass count label on strong connections (top 40%)
        if c >= max_count * 0.4:
            mid_x = (players[f]["x"] + players[t]["x"]) / 2
            mid_y = (players[f]["y"] + players[t]["y"]) / 2
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y], mode="text",
                text=[str(c)], textfont=dict(color="white", size=9, family="Arial Black"),
                hoverinfo="skip", showlegend=False,
            ))

    # Draw player nodes — size proportional to pass involvement
    max_involvement = max(player_pass_count.values()) if player_pass_count else 1
    for pid, info in players.items():
        involvement = player_pass_count.get(pid, 0)
        node_size = max(22, 18 + (involvement / max_involvement) * 22)
        short_name = info["name"].split()[-1] if " " in info["name"] else info["name"]
        fig.add_trace(go.Scatter(
            x=[info["x"]], y=[info["y"]], mode="markers+text",
            marker=dict(size=node_size, color=team_color,
                        line=dict(width=2.5, color="white"),
                        opacity=0.95),
            text=[short_name], textposition="top center",
            textfont=dict(color="white", size=11, family="Arial"),
            hoverinfo="text", hovertext=info["name"] + " (" + str(involvement) + " passes)",
            showlegend=False,
        ))

    fig.update_layout(
        plot_bgcolor=pitch_bg, paper_bgcolor="#1a1a1a",
        xaxis=dict(range=[-3, pw + 3], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-3, ph + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", fixedrange=True),
        margin=dict(l=5, r=5, t=5, b=5), height=300, dragmode=False,
    )
    return fig


# ============================================================
# ZONE CONTROL HEATMAP
# ============================================================
def draw_zone_control(zone_data_a: dict, zone_data_b: dict, team_a_color: str) -> go.Figure:
    """Draw zone control as a heatmap on the pitch."""
    grid_a = np.zeros((2, 3))
    grid_b = np.zeros((2, 3))

    for key, val in zone_data_a.items():
        parts = key.replace("z", "").split("_")
        if len(parts) == 2:
            r, c = int(parts[0]), int(parts[1])
            if r < 3 and c < 2:
                grid_a[c][r] = val

    for key, val in zone_data_b.items():
        parts = key.replace("z", "").split("_")
        if len(parts) == 2:
            r, c = int(parts[0]), int(parts[1])
            if r < 3 and c < 2:
                grid_b[c][r] = val

    total = grid_a + grid_b + 0.001
    dominance = grid_a / total * 100
    text = [[f"{dominance[r][c]:.0f}%" for c in range(3)] for r in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=dominance, text=text, texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        colorscale=[[0, "#333333"], [0.5, "#555555"], [1, team_a_color]],
        showscale=False, xgap=3, ygap=3,
    ))

    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        xaxis=dict(showticklabels=False, showgrid=False, fixedrange=True),
        yaxis=dict(showticklabels=False, showgrid=False, fixedrange=True),
        margin=dict(l=10, r=10, t=10, b=10), height=180, dragmode=False,
    )
    return fig


# ============================================================
# PLAYER LEADERBOARD
# ============================================================
def render_leaderboard(title: str, players: list, stat_key: str, fmt: str = "{:.1f}",
                       team_colors: dict = None, suffix: str = "") -> str:
    """Render a top-5 player leaderboard as HTML."""
    html = '<div class="stat-card"><div class="leaderboard-header">' + title + '</div>'

    for i, p in enumerate(players[:5]):
        val = p.get(stat_key, 0)
        try:
            display_val = fmt.format(val) + suffix
        except (ValueError, TypeError):
            display_val = str(val)

        name = p.get("name") or p.get("display", "Unknown")
        team = p.get("team", "A")
        team_dot_color = team_colors.get(team, "#888") if team_colors else "#888"
        team_dot_color = safe_color(team_dot_color)

        html += ('<div class="leaderboard-item">'
                 '<div class="leaderboard-pos">' + str(i + 1) + '</div>'
                 '<div style="width:8px;height:8px;border-radius:50%;background:' + team_dot_color + ';margin-right:10px;"></div>'
                 '<div class="leaderboard-name">' + name + '</div>'
                 '<div class="leaderboard-val">' + display_val + '</div>'
                 '</div>')

    html += "</div>"
    return html


# ============================================================
# PLAYER RADAR CHART
# ============================================================
def build_radar(player_data: dict, ref_players: list, color: str, name: str) -> go.Figure:
    """Build a radar chart for a player normalized against team max."""
    categories = ["Distance", "Speed", "Passes", "Touches", "Sprints", "Duels"]
    stat_keys = ["total_distance_km", "max_speed_kmh", "passes_completed", "touches", "sprints", "duels_won"]

    team_max = {}
    for cat, key in zip(categories, stat_keys):
        max_val = max([p.get(key, 0) for p in ref_players] or [1])
        team_max[cat] = max_val if max_val > 0 else 1

    values = [
        player_data.get("total_distance_km", 0) / team_max["Distance"] * 100,
        player_data.get("max_speed_kmh", 0) / team_max["Speed"] * 100,
        player_data.get("passes_completed", 0) / team_max["Passes"] * 100,
        player_data.get("touches", 0) / team_max["Touches"] * 100,
        player_data.get("sprints", 0) / team_max["Sprints"] * 100,
        player_data.get("duels_won", 0) / team_max["Duels"] * 100,
    ]
    values.append(values[0])

    vis_color = safe_color(color)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories + [categories[0]],
        fill="toself",
        fillcolor=hex_to_rgba(vis_color, 0.3),
        line=dict(color=vis_color, width=2),
        name=name,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1a1a1a",
            radialaxis=dict(visible=False, range=[0, 110]),
            angularaxis=dict(color="#aaa", gridcolor="#333"),
        ),
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        showlegend=False, margin=dict(l=60, r=60, t=30, b=30), height=300,
    )
    return fig


# ============================================================
# EXPORT HELPER (improvement 17)
# ============================================================
def generate_export_csv(data: dict) -> str:
    """Generate a CSV export of match stats."""
    output = io.StringIO()
    writer = csv.writer(output)

    team_names = data.get("team_names", {"A": "Home", "B": "Away"})
    ts_a = data.get("team_stats", {}).get("A", {})
    ts_b = data.get("team_stats", {}).get("B", {})
    possession = data.get("possession", {}).get("team_possession_percentage", {"A": 50, "B": 50})

    writer.writerow(["Stat", team_names.get("A", "Home"), team_names.get("B", "Away")])
    writer.writerow(["Possession %", possession.get("A", 50), possession.get("B", 50)])

    for key, label in [("total_shots", "Total Shots"), ("shots_on_target", "Shots on Target"),
                       ("xg", "xG"), ("total_passes", "Total Passes"), ("pass_accuracy", "Pass Accuracy %"),
                       ("total_distance_km", "Total Distance (km)"), ("total_sprints", "Sprints"),
                       ("total_interceptions", "Interceptions"), ("total_duels", "Total Duels"),
                       ("duels_won", "Duels Won"), ("total_touches", "Touches")]:
        writer.writerow([label, ts_a.get(key, 0), ts_b.get(key, 0)])

    writer.writerow([])
    writer.writerow(["Player Stats"])
    writer.writerow(["Name", "Team", "Position", "Distance (km)", "Sprints", "Passes", "Pass Accuracy %",
                      "Touches", "Interceptions", "Duels Won", "Max Speed (km/h)"])

    for pid, ps in data.get("player_stats", {}).items():
        if ps.get("minutes_played", 0) < 0.5:
            continue
        t_name = team_names.get(ps.get("team", "A"), ps.get("team", "?"))
        writer.writerow([
            ps.get("name", "Unknown"), t_name, ps.get("position", "N/A"),
            ps.get("total_distance_km", 0), ps.get("sprints", 0),
            str(ps.get("passes_completed", 0)) + "/" + str(ps.get("passes_attempted", 0)),
            ps.get("pass_accuracy", 0), ps.get("touches", 0),
            ps.get("interceptions", 0), ps.get("duels_won", 0),
            ps.get("max_speed_kmh", 0),
        ])

    return output.getvalue()


def generate_export_text(data: dict) -> str:
    """Generate a formatted text summary of match stats."""
    team_names = data.get("team_names", {"A": "Home", "B": "Away"})
    ts_a = data.get("team_stats", {}).get("A", {})
    ts_b = data.get("team_stats", {}).get("B", {})
    score = data.get("score", {"A": 0, "B": 0})
    possession = data.get("possession", {}).get("team_possession_percentage", {"A": 50, "B": 50})
    duration = data.get("duration_minutes", 0)

    lines = []
    lines.append("=" * 50)
    lines.append("TactiVision Pro - Match Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append(team_names.get("A", "Home") + " " + str(score.get("A", 0)) + " - " + str(score.get("B", 0)) + " " + team_names.get("B", "Away"))
    lines.append("Duration analyzed: " + f"{duration:.1f}" + " minutes")
    lines.append("")
    lines.append("--- Team Stats ---")
    lines.append(f"{'Stat':<25} {team_names.get('A', 'Home'):<15} {team_names.get('B', 'Away'):<15}")
    lines.append("-" * 55)

    for key, label in [("total_shots", "Total Shots"), ("shots_on_target", "Shots on Target"),
                       ("xg", "xG"), ("total_passes", "Total Passes"),
                       ("pass_accuracy", "Pass Accuracy %"), ("total_distance_km", "Distance (km)"),
                       ("total_sprints", "Sprints"), ("total_interceptions", "Interceptions"),
                       ("total_touches", "Touches")]:
        lines.append(f"{label:<25} {str(ts_a.get(key, 0)):<15} {str(ts_b.get(key, 0)):<15}")

    lines.append("")
    lines.append(f"Possession: {team_names.get('A', 'Home')} {possession.get('A', 50)}% - {possession.get('B', 50)}% {team_names.get('B', 'Away')}")
    lines.append("")
    lines.append("Generated by TactiVision Pro")

    return "\n".join(lines)


# ============================================================
# MAIN DASHBOARD
# ============================================================
def main():
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # Load data
    metrics_path = find_metrics_file()
    if not metrics_path:
        st.error("No metrics.json found. Run the ONNX pipeline first.")
        return

    data = load_metrics(metrics_path)
    data = validate_and_normalize(data)

    # Extract core data
    team_names = data.get("team_names", {"A": "Home", "B": "Away"})
    team_colors = data.get("team_colors", {"A": "#c8102e", "B": "#FFFFFF"})
    score = data.get("score", {"A": 0, "B": 0})
    possession = data.get("possession", {}).get("team_possession_percentage", {"A": 50, "B": 50})
    possession_note = data.get("_possession_note", "")
    team_stats = data.get("team_stats", {})
    player_stats = data.get("player_stats", {})
    player_identities = data.get("player_identities", {})
    pass_detection = data.get("pass_detection", {})
    shot_detection = data.get("shot_detection", {})
    sprint_detection = data.get("sprint_detection", {})
    xg_analysis = data.get("xg_analysis", {})
    tactical = data.get("tactical_analysis", {})
    ball_tracking = data.get("ball_tracking", {})
    pass_events = data.get("pass_events", [])
    shot_events = data.get("shot_events", [])
    sprint_events = data.get("sprint_events", [])
    highlights = data.get("highlights", {})
    duration_min = data.get("duration_minutes", 0)

    ta_name = team_names.get("A", "Home")
    tb_name = team_names.get("B", "Away")
    ta_color = team_colors.get("A", "#c8102e")
    tb_color = team_colors.get("B", "#FFFFFF")

    ts_a = team_stats.get("A", {})
    ts_b = team_stats.get("B", {})

    xg_a = xg_analysis.get("xg_by_team", {}).get("A", {})
    xg_b = xg_analysis.get("xg_by_team", {}).get("B", {})

    xg_shot_events = xg_analysis.get("shot_events", shot_events)

    # Team abbreviations for logos
    ta_abbr = ta_name[:3].upper()
    tb_abbr = tb_name[:3].upper()
    ta_vis = ta_color
    tb_vis = safe_color(tb_color)

    # Derive match context from folder name (improvement 14)
    match_folder = Path(metrics_path).parent.name
    competition_name = "Match Analysis"
    if "liverpool" in match_folder.lower() or "tottenham" in match_folder.lower():
        competition_name = "Premier League"

    # ----------------------------------------------------------------
    # MATCH CONTEXT BAR (improvement 14)
    # ----------------------------------------------------------------
    context_html = ('<div class="context-bar">'
                    '<span>' + competition_name + '</span>'
                    '<span style="margin:0 8px;">|</span>'
                    '<span>Analyzed: ' + f"{duration_min:.1f}" + ' min</span>'
                    '<span style="margin:0 8px;">|</span>'
                    '<span class="data-badge">First ' + str(int(duration_min)) + ' minutes analyzed</span>'
                    '</div>')
    st.markdown(context_html, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # MATCH HEADER with team logo placeholders (improvement 19)
    # ----------------------------------------------------------------
    header_html = ('<div class="match-header">'
                   '<div style="text-align:right;display:flex;align-items:center;gap:12px;">'
                   '<div class="team-name">' + ta_name + '</div>'
                   '<div class="team-logo" style="background:' + ta_color + ';color:white;">' + ta_abbr + '</div>'
                   '</div>'
                   '<div style="display:flex;align-items:center;gap:12px;">'
                   '<div class="match-score">' + str(score.get("A", 0)) + '</div>'
                   '<div style="text-align:center;">'
                   '<div class="match-time">' + f"{duration_min:.0f}" + "'</div>"
                   '</div>'
                   '<div class="match-score">' + str(score.get("B", 0)) + '</div>'
                   '</div>'
                   '<div style="text-align:left;display:flex;align-items:center;gap:12px;">'
                   '<div class="team-logo" style="background:' + tb_vis + ';color:#222;">' + tb_abbr + '</div>'
                   '<div class="team-name">' + tb_name + '</div>'
                   '</div>'
                   '</div>')
    st.markdown(header_html, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # TABS
    # ----------------------------------------------------------------
    tab_overview, tab_shots, tab_passdef, tab_players, tab_tactical, tab_momentum, tab_export = st.tabs(
        ["Overview", "Shots", "Passes & Defence", "Players", "Tactical", "Momentum", "Export"])

    # ================================================================
    # SHARED STAT COMPUTATIONS (used across tabs)
    # ================================================================
    poss_a = possession.get("A", 50)
    poss_b = possession.get("B", 50)
    if isinstance(poss_a, (int, float)) and isinstance(poss_b, (int, float)):
        total_poss = poss_a + poss_b
        if total_poss > 0:
            poss_a = round(poss_a / total_poss * 100, 1)
            poss_b = round(poss_b / total_poss * 100, 1)

    total_xg_a = round(xg_a.get("xg", ts_a.get("xg", 0)), 2)
    total_xg_b = round(xg_b.get("xg", ts_b.get("xg", 0)), 2)
    shots_a = ts_a.get("total_shots", shot_detection.get("shots_by_team", {}).get("A", 0))
    shots_b = ts_b.get("total_shots", shot_detection.get("shots_by_team", {}).get("B", 0))
    sot_a = ts_a.get("shots_on_target", shot_detection.get("shots_on_target", {}).get("A", 0))
    sot_b = ts_b.get("shots_on_target", shot_detection.get("shots_on_target", {}).get("B", 0))
    passes_a = ts_a.get("total_passes", 0)
    passes_b = ts_b.get("total_passes", 0)
    pc_a = ts_a.get("passes_completed", 0)
    pc_b = ts_b.get("passes_completed", 0)
    pa_a = ts_a.get("pass_accuracy", 0)
    pa_b = ts_b.get("pass_accuracy", 0)
    acc_passes_a = str(pc_a) + " (" + f"{pa_a:.0f}" + "%)"
    acc_passes_b = str(pc_b) + " (" + f"{pa_b:.0f}" + "%)"
    big_chances_a = sum(1 for s in xg_shot_events if s.get("team") == "A" and s.get("xg", 0) > 0.25)
    big_chances_b = sum(1 for s in xg_shot_events if s.get("team") == "B" and s.get("xg", 0) > 0.25)
    goals_a = score.get("A", 0)
    goals_b = score.get("B", 0)
    saves_a = max(0, sot_b - goals_b)
    saves_b = max(0, sot_a - goals_a)
    int_a = ts_a.get("total_interceptions", 0)
    int_b = ts_b.get("total_interceptions", 0)
    duels_a = ts_a.get("total_duels", 0)
    duels_b = ts_b.get("total_duels", 0)
    dw_a = ts_a.get("duels_won", 0)
    dw_b = ts_b.get("duels_won", 0)
    duel_pct_a = round(dw_a / duels_a * 100, 1) if duels_a > 0 else 0
    duel_pct_b = round(dw_b / duels_b * 100, 1) if duels_b > 0 else 0
    possession_note_html = ""
    if possession_note:
        possession_note_html = '<div style="text-align:center;font-size:11px;color:#888;margin-top:-8px;margin-bottom:8px;">' + possession_note + '</div>'

    # ================================================================
    # TAB: OVERVIEW
    # ================================================================
    with tab_overview:
        fig_shots = draw_shot_map(xg_shot_events, ta_color, tb_color, ta_name, tb_name, filter_mode="All")
        st.plotly_chart(fig_shots, use_container_width=True, config={"displayModeBar": False})

        top_stats_html = ('<div class="stat-card">'
                          '<div class="stat-card-title">Top Stats</div>'
                          '<div class="stat-label" style="margin-bottom:4px;">Ball possession</div>'
                          '<div class="possession-bar-container">'
                          '<div class="possession-bar-left" style="width:' + f"{poss_a:.0f}" + '%;background:' + ta_color + ';">' + f"{poss_a:.0f}" + '%</div>'
                          '<div class="possession-bar-right" style="width:' + f"{poss_b:.0f}" + '%;">' + f"{poss_b:.0f}" + '%</div>'
                          '</div>'
                          + possession_note_html
                          + stat_comparison(total_xg_a, "Expected goals (xG)", total_xg_b)
                          + stat_comparison(shots_a, "Total shots", shots_b)
                          + stat_comparison(sot_a, "Shots on target", sot_b)
                          + stat_comparison(big_chances_a, "Big chances (xG > 0.25)", big_chances_b)
                          + stat_comparison(saves_a, "Saves", saves_b)
                          + stat_comparison(acc_passes_a, "Accurate passes", acc_passes_b)
                          + stat_comparison(round(ts_a.get("total_distance_km", 0), 1), "Total distance (km)", round(ts_b.get("total_distance_km", 0), 1))
                          + '</div>')
        st.markdown(top_stats_html, unsafe_allow_html=True)

    # ================================================================
    # TAB: SHOTS
    # ================================================================
    with tab_shots:
        filter_options = ["All", "On Target", "Off Target"]
        shot_filter = st.pills("Filter shots", filter_options, default="All", key="shot_filter")
        if shot_filter is None:
            shot_filter = "All"
        fig_shots2 = draw_shot_map(xg_shot_events, ta_color, tb_color, ta_name, tb_name, filter_mode=shot_filter)
        st.plotly_chart(fig_shots2, use_container_width=True, config={"displayModeBar": False})

        fig_goal = draw_goal_mouth_map(xg_shot_events, ta_color, tb_color, ta_name, tb_name, score)
        st.plotly_chart(fig_goal, use_container_width=True, config={"displayModeBar": False})

        shots_off_a = shots_a - sot_a
        shots_off_b = shots_b - sot_b
        shots_inside_a, shots_inside_b, shots_outside_a, shots_outside_b = 0, 0, 0, 0
        for s in xg_shot_events:
            dist = s.get("distance_m", 20)
            team = s.get("team", "A")
            if dist <= 18:
                if team == "A": shots_inside_a += 1
                else: shots_inside_b += 1
            else:
                if team == "A": shots_outside_a += 1
                else: shots_outside_b += 1

        shots_html = ('<div class="stat-card">'
                      '<div class="stat-card-title">Shots</div>'
                      + stat_comparison(shots_a, "Total shots", shots_b)
                      + '<div class="shots-nested">'
                      + stat_comparison_plain(shots_off_a, "Shots off target", shots_off_b)
                      + '<div class="shots-deep">'
                      + stat_comparison_plain(sot_a, "Shots on target", sot_b)
                      + '</div></div>'
                      + stat_comparison(shots_inside_a, "Shots inside box", shots_inside_b)
                      + stat_comparison(shots_outside_a, "Shots outside box", shots_outside_b)
                      + '</div>')
        st.markdown(shots_html, unsafe_allow_html=True)

        xg_open_a, xg_set_a, xg_ot_a = 0, 0, 0
        xg_open_b, xg_set_b, xg_ot_b = 0, 0, 0
        for s in xg_shot_events:
            team, xg_val = s.get("team", "A"), s.get("xg", 0)
            dist, on_target = s.get("distance_m", 20), s.get("on_target", False)
            if team == "A":
                xg_set_a += xg_val if dist > 30 else 0
                xg_open_a += xg_val if dist <= 30 else 0
                xg_ot_a += xg_val if on_target else 0
            else:
                xg_set_b += xg_val if dist > 30 else 0
                xg_open_b += xg_val if dist <= 30 else 0
                xg_ot_b += xg_val if on_target else 0

        xg_html = ('<div class="stat-card">'
                    '<div class="stat-card-title">Expected Goals (xG)</div>'
                    + stat_comparison(f"{total_xg_a:.2f}", "Expected goals (xG)", f"{total_xg_b:.2f}")
                    + stat_comparison(f"{xg_open_a:.2f}", "xG open play", f"{xg_open_b:.2f}")
                    + stat_comparison(f"{xg_set_a:.2f}", "xG set play", f"{xg_set_b:.2f}")
                    + stat_comparison(f"{xg_ot_a:.2f}", "xG on target", f"{xg_ot_b:.2f}")
                    + '</div>')
        st.markdown(xg_html, unsafe_allow_html=True)

    # ================================================================
    # TAB: PASSES & DEFENCE
    # ================================================================
    with tab_passdef:
        pass_fwd_a, pass_fwd_b, pass_bwd_a, pass_bwd_b = 0, 0, 0, 0
        pass_lat_a, pass_lat_b, long_ball_a, long_ball_b, key_pass_a, key_pass_b = 0, 0, 0, 0, 0, 0
        for p in pass_events:
            team, direction = p.get("team", "A"), p.get("direction", "")
            success, dist, is_key = p.get("success", False), p.get("distance_px", 0), p.get("key_pass", False)
            if not success:
                continue
            if team == "A":
                if "forward" in direction: pass_fwd_a += 1
                elif "backward" in direction: pass_bwd_a += 1
                elif "lateral" in direction: pass_lat_a += 1
                if dist > 100: long_ball_a += 1
                if is_key: key_pass_a += 1
            else:
                if "forward" in direction: pass_fwd_b += 1
                elif "backward" in direction: pass_bwd_b += 1
                elif "lateral" in direction: pass_lat_b += 1
                if dist > 100: long_ball_b += 1
                if is_key: key_pass_b += 1

        passes_html = ('<div class="stat-card">'
                       '<div class="stat-card-title">Passes</div>'
                       + stat_comparison(passes_a, "Total passes", passes_b)
                       + stat_comparison(acc_passes_a, "Accurate passes", acc_passes_b)
                       + stat_comparison(pass_fwd_a, "Forward passes", pass_fwd_b)
                       + stat_comparison(pass_bwd_a, "Backward passes", pass_bwd_b)
                       + stat_comparison(pass_lat_a, "Lateral passes", pass_lat_b)
                       + stat_comparison(long_ball_a, "Long balls", long_ball_b)
                       + stat_comparison(key_pass_a, "Key passes", key_pass_b)
                       + '</div>')
        st.markdown(passes_html, unsafe_allow_html=True)

        defence_html = ('<div class="stat-card">'
                        '<div class="stat-card-title">Defence</div>'
                        + stat_comparison(int_a, "Interceptions", int_b)
                        + stat_comparison(duels_a, "Total duels", duels_b)
                        + stat_comparison(dw_a, "Duels won", dw_b)
                        + '</div>')
        st.markdown(defence_html, unsafe_allow_html=True)

        duels_html = ('<div class="stat-card">'
                      '<div class="stat-card-title">Duels</div>'
                      + stat_comparison(str(duel_pct_a) + "%", "Duels won %", str(duel_pct_b) + "%")
                      + stat_comparison(duels_a, "Total duels", duels_b)
                      + stat_comparison(dw_a, "Duels won", dw_b)
                      + '</div>')
        st.markdown(duels_html, unsafe_allow_html=True)

        discipline_html = ('<div class="stat-card">'
                           '<div class="stat-card-title">Discipline</div>'
                           + stat_comparison(0, "Yellow cards", 0)
                           + stat_comparison(0, "Red cards", 0)
                           + stat_comparison(0, "Fouls committed", 0)
                           + '<div style="text-align:center;color:#666;font-size:12px;padding-top:8px;">No cards detected in analyzed footage</div>'
                           + '</div>')
        st.markdown(discipline_html, unsafe_allow_html=True)

        physical_html = ('<div class="stat-card">'
                         '<div class="stat-card-title">Physical</div>'
                         + stat_comparison(round(ts_a.get("total_distance_km", 0), 1), "Total distance (km)", round(ts_b.get("total_distance_km", 0), 1))
                         + stat_comparison(round(ts_a.get("avg_distance_km", 0), 2), "Avg distance per player (km)", round(ts_b.get("avg_distance_km", 0), 2))
                         + stat_comparison(ts_a.get("total_sprints", 0), "Sprints", ts_b.get("total_sprints", 0))
                         + stat_comparison(ts_a.get("total_touches", 0), "Touches", ts_b.get("total_touches", 0))
                         + '</div>')
        st.markdown(physical_html, unsafe_allow_html=True)

    # ================================================================
    # TAB: PLAYERS (improvements 9, 10)
    # ================================================================
    with tab_players:
        # Build player list
        all_players = []
        for pid, ps in player_stats.items():
            if ps.get("minutes_played", 0) < 0.5:
                continue
            all_players.append(ps)

        team_a_players = sorted([p for p in all_players if p.get("team") == "A"],
                                key=lambda x: x.get("workload_score", 0), reverse=True)
        team_b_players = sorted([p for p in all_players if p.get("team") == "B"],
                                key=lambda x: x.get("workload_score", 0), reverse=True)

        # --- Top Performers Leaderboards ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Top Performers</div></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            dist_sorted = sorted(all_players, key=lambda x: x.get("total_distance_km", 0), reverse=True)
            st.markdown(render_leaderboard("Distance (km)", dist_sorted, "total_distance_km",
                                           fmt="{:.2f}", team_colors=team_colors), unsafe_allow_html=True)
            sprint_sorted = sorted(all_players, key=lambda x: x.get("sprints", 0), reverse=True)
            st.markdown(render_leaderboard("Sprints", sprint_sorted, "sprints",
                                           fmt="{:.0f}", team_colors=team_colors), unsafe_allow_html=True)

        with col2:
            pass_sorted = sorted([p for p in all_players if p.get("passes_attempted", 0) >= 3],
                                 key=lambda x: x.get("pass_accuracy", 0), reverse=True)
            st.markdown(render_leaderboard("Pass Accuracy", pass_sorted, "pass_accuracy",
                                           fmt="{:.1f}", suffix="%", team_colors=team_colors), unsafe_allow_html=True)
            touch_sorted = sorted(all_players, key=lambda x: x.get("touches", 0), reverse=True)
            st.markdown(render_leaderboard("Touches", touch_sorted, "touches",
                                           fmt="{:.0f}", team_colors=team_colors), unsafe_allow_html=True)

        # --- Player Comparison Mode (improvement 9) ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Compare Players</div></div>', unsafe_allow_html=True)

        comp_col1, comp_col2 = st.columns(2)
        player_options_all = {p.get("display", "Player " + str(p.get("id"))): p for p in all_players}
        all_names = list(player_options_all.keys())

        with comp_col1:
            comp_p1_name = st.selectbox("Player 1", all_names, index=0, key="comp_p1")
        with comp_col2:
            comp_p2_name = st.selectbox("Player 2", all_names, index=min(1, len(all_names) - 1), key="comp_p2")

        if comp_p1_name and comp_p2_name and comp_p1_name in player_options_all and comp_p2_name in player_options_all:
            p1 = player_options_all[comp_p1_name]
            p2 = player_options_all[comp_p2_name]

            # Side-by-side stat cards
            cc1, cc2 = st.columns(2)
            for col, p in [(cc1, p1), (cc2, p2)]:
                t_color = ta_color if p.get("team") == "A" else tb_color
                vis_c = safe_color(t_color)
                with col:
                    card_html = ('<div class="stat-card" style="text-align:center;">'
                                 '<div style="font-size:28px;font-weight:800;color:' + vis_c + ';">' + str(p.get("jersey_number", "?")) + '</div>'
                                 '<div style="font-size:15px;font-weight:700;color:white;margin-top:4px;">' + (p.get("name") or p.get("display", "Unknown")) + '</div>'
                                 '<div style="font-size:12px;color:#aaa;margin-top:2px;">' + p.get("position", "N/A") + '</div>'
                                 '<div style="margin-top:12px;">'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Distance</span><span style="color:white;font-weight:700;">' + f"{p.get('total_distance_km', 0):.2f}" + ' km</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Sprints</span><span style="color:white;font-weight:700;">' + str(p.get("sprints", 0)) + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Passes</span><span style="color:white;font-weight:700;">' + str(p.get("passes_completed", 0)) + '/' + str(p.get("passes_attempted", 0)) + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Touches</span><span style="color:white;font-weight:700;">' + str(p.get("touches", 0)) + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Duels Won</span><span style="color:white;font-weight:700;">' + str(p.get("duels_won", 0)) + '/' + str(p.get("duels_total", 0)) + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Top Speed</span><span style="color:white;font-weight:700;">' + f"{p.get('max_speed_kmh', 0):.1f}" + ' km/h</span></div>'
                                 '</div></div>')
                    st.markdown(card_html, unsafe_allow_html=True)

            # Overlaid radar chart
            categories = ["Distance", "Speed", "Passes", "Touches", "Sprints", "Duels"]
            stat_keys = ["total_distance_km", "max_speed_kmh", "passes_completed", "touches", "sprints", "duels_won"]

            team_max = {}
            for cat, key in zip(categories, stat_keys):
                max_val = max([p_.get(key, 0) for p_ in all_players] or [1])
                team_max[cat] = max_val if max_val > 0 else 1

            def _norm_vals(player):
                vals = []
                for cat, key in zip(categories, stat_keys):
                    vals.append(player.get(key, 0) / team_max[cat] * 100)
                vals.append(vals[0])
                return vals

            fig_comp = go.Figure()
            p1_color = safe_color(ta_color if p1.get("team") == "A" else tb_color)
            p2_color = safe_color(ta_color if p2.get("team") == "A" else tb_color)
            if p1_color == p2_color:
                p2_color = "#f39c12"  # Use orange to differentiate

            fig_comp.add_trace(go.Scatterpolar(
                r=_norm_vals(p1), theta=categories + [categories[0]],
                fill="toself", fillcolor=hex_to_rgba(p1_color, 0.2),
                line=dict(color=p1_color, width=2),
                name=p1.get("name") or p1.get("display", "P1"),
            ))
            fig_comp.add_trace(go.Scatterpolar(
                r=_norm_vals(p2), theta=categories + [categories[0]],
                fill="toself", fillcolor=hex_to_rgba(p2_color, 0.2),
                line=dict(color=p2_color, width=2),
                name=p2.get("name") or p2.get("display", "P2"),
            ))
            fig_comp.update_layout(
                polar=dict(bgcolor="#1a1a1a",
                           radialaxis=dict(visible=False, range=[0, 110]),
                           angularaxis=dict(color="#aaa", gridcolor="#333")),
                plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
                showlegend=True,
                legend=dict(font=dict(color="white", size=12), bgcolor="rgba(0,0,0,0)",
                            orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                margin=dict(l=60, r=60, t=40, b=30), height=320,
            )
            st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

        # --- Individual Player Stats with Profile Tabs (improvement 10) ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Player Statistics</div></div>', unsafe_allow_html=True)

        team_sel = st.radio("Select Team", [ta_name, tb_name], horizontal=True, label_visibility="collapsed", key="team_sel_player")
        selected_players = team_a_players if team_sel == ta_name else team_b_players
        sel_team_color = ta_color if team_sel == ta_name else tb_color

        if selected_players:
            player_options = {p.get("display", "Player " + str(p.get("id"))): p for p in selected_players}
            selected_name = st.selectbox("Select Player", list(player_options.keys()), label_visibility="collapsed", key="player_sel")
            sp = player_options[selected_name]
            sel_vis_color = safe_color(sel_team_color)

            # Player profile header
            profile_html = ('<div class="stat-card" style="text-align:center;">'
                            '<div style="font-size:32px;font-weight:800;color:' + sel_vis_color + ';">' + str(sp.get("jersey_number", "?")) + '</div>'
                            '<div style="font-size:16px;font-weight:700;color:white;margin-top:4px;">' + (sp.get("name") or sp.get("display", "Unknown")) + '</div>'
                            '<div style="font-size:13px;color:#aaa;margin-top:2px;">' + sp.get("position", "N/A") + '</div>'
                            '<div style="font-size:12px;color:#666;margin-top:8px;">' + f"{sp.get('minutes_played', 0):.1f}" + ' min played</div>'
                            '</div>')
            st.markdown(profile_html, unsafe_allow_html=True)

            # Profile sub-tabs (improvement 10)
            ptab_overview, ptab_attacking, ptab_defensive, ptab_physical = st.tabs(
                ["Overview", "Attacking", "Defensive", "Physical"])

            with ptab_overview:
                overview_html = ('<div class="stat-card">'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Rating</span><span style="color:white;font-weight:700;">' + f"{min(10, sp.get('workload_score', 0) / 150):.1f}" + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Minutes Played</span><span style="color:white;font-weight:700;">' + f"{sp.get('minutes_played', 0):.1f}" + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Touches</span><span style="color:white;font-weight:700;">' + str(sp.get("touches", 0)) + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Passes</span><span style="color:white;font-weight:700;">' + str(sp.get("passes_completed", 0)) + '/' + str(sp.get("passes_attempted", 0)) + '</span></div>'
                                 '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Distance</span><span style="color:white;font-weight:700;">' + f"{sp.get('total_distance_km', 0):.2f}" + ' km</span></div>'
                                 '</div>')
                st.markdown(overview_html, unsafe_allow_html=True)

                # Radar chart
                ref_players = team_a_players if sp.get("team") == "A" else team_b_players
                fig_radar = build_radar(sp, ref_players, sel_team_color, sp.get("name") or sp.get("display"))
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

            with ptab_attacking:
                atk_html = ('<div class="stat-card">'
                            '<div style="font-size:13px;color:#aaa;margin-bottom:12px;font-weight:600;">ATTACKING</div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Passes Completed</span><span style="color:white;font-weight:700;">' + str(sp.get("passes_completed", 0)) + '/' + str(sp.get("passes_attempted", 0)) + '</span></div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Pass Accuracy</span><span style="color:white;font-weight:700;">' + f"{sp.get('pass_accuracy', 0):.1f}" + '%</span></div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Key Passes</span><span style="color:white;font-weight:700;">' + str(sp.get("key_passes", 0)) + '</span></div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Passes Received</span><span style="color:white;font-weight:700;">' + str(sp.get("passes_received", 0)) + '</span></div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Touches</span><span style="color:white;font-weight:700;">' + str(sp.get("touches", 0)) + '</span></div>'
                            '</div>')
                st.markdown(atk_html, unsafe_allow_html=True)

            with ptab_defensive:
                def_html = ('<div class="stat-card">'
                            '<div style="font-size:13px;color:#aaa;margin-bottom:12px;font-weight:600;">DEFENSIVE</div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Interceptions</span><span style="color:white;font-weight:700;">' + str(sp.get("interceptions", 0)) + '</span></div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Duels Won</span><span style="color:white;font-weight:700;">' + str(sp.get("duels_won", 0)) + '/' + str(sp.get("duels_total", 0)) + '</span></div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Duel Success Rate</span><span style="color:white;font-weight:700;">' + f"{sp.get('duel_success_rate', 0):.1f}" + '%</span></div>'
                            '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Possession Time</span><span style="color:white;font-weight:700;">' + f"{sp.get('possession_time_s', 0):.1f}" + 's</span></div>'
                            '</div>')
                st.markdown(def_html, unsafe_allow_html=True)

            with ptab_physical:
                phys_html = ('<div class="stat-card">'
                             '<div style="font-size:13px;color:#aaa;margin-bottom:12px;font-weight:600;">PHYSICAL</div>'
                             '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Distance</span><span style="color:white;font-weight:700;">' + f"{sp.get('total_distance_km', 0):.2f}" + ' km</span></div>'
                             '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Top Speed</span><span style="color:white;font-weight:700;">' + f"{sp.get('max_speed_kmh', 0):.1f}" + ' km/h</span></div>'
                             '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Avg Speed</span><span style="color:white;font-weight:700;">' + f"{sp.get('avg_speed_kmh', 0):.1f}" + ' km/h</span></div>'
                             '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Sprints</span><span style="color:white;font-weight:700;">' + str(sp.get("sprints", 0)) + '</span></div>'
                             '<div class="stat-row"><span style="color:#aaa;font-size:13px;">High Intensity Runs</span><span style="color:white;font-weight:700;">' + str(sp.get("high_intensity_runs", 0)) + '</span></div>'
                             '<div class="stat-row"><span style="color:#aaa;font-size:13px;">Workload Score</span><span style="color:white;font-weight:700;">' + f"{sp.get('workload_score', 0):.0f}" + '</span></div>'
                             '</div>')
                st.markdown(phys_html, unsafe_allow_html=True)

    # ================================================================
    # TAB: TACTICAL (improvement 4: formation visual)
    # ================================================================
    with tab_tactical:
        # --- Formation Visual (improvement 4) ---
        formations = tactical.get("formation", {})
        form_a = formations.get("A", "4-4-2")
        form_b = formations.get("B", "4-3-3")

        st.markdown('<div class="stat-card"><div class="stat-card-title">Formations</div></div>', unsafe_allow_html=True)

        form_col1, form_col2 = st.columns(2)
        with form_col1:
            fig_form_a = draw_formation(form_a, ta_name, ta_color, player_identities, "A")
            st.plotly_chart(fig_form_a, use_container_width=True, config={"displayModeBar": False})
        with form_col2:
            fig_form_b = draw_formation(form_b, tb_name, tb_color, player_identities, "B")
            st.plotly_chart(fig_form_b, use_container_width=True, config={"displayModeBar": False})

        # --- Zone Control ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Zone Control</div></div>', unsafe_allow_html=True)
        zone_a = tactical.get("zone_control", {}).get("A", {})
        zone_b = tactical.get("zone_control", {}).get("B", {})
        if zone_a:
            fig_zone = draw_zone_control(zone_a, zone_b, ta_color)
            st.plotly_chart(fig_zone, use_container_width=True, config={"displayModeBar": False})
            zone_caption = '<div style="text-align:center;color:#aaa;font-size:12px;margin-top:-10px;">Colored = ' + ta_name + ' dominance | Gray = ' + tb_name + ' dominance</div>'
            st.markdown(zone_caption, unsafe_allow_html=True)

        # --- Passing Network (improvement 12) ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Passing Network</div></div>', unsafe_allow_html=True)
        passing_net = tactical.get("passing_network", [])

        col_net1, col_net2 = st.columns(2)
        with col_net1:
            st.markdown('<div style="text-align:center;color:white;font-weight:600;font-size:14px;">' + ta_name + '</div>', unsafe_allow_html=True)
            fig_net_a = draw_passing_network(passing_net, player_identities, "A", ta_color)
            st.plotly_chart(fig_net_a, use_container_width=True, config={"displayModeBar": False})

        with col_net2:
            st.markdown('<div style="text-align:center;color:white;font-weight:600;font-size:14px;">' + tb_name + '</div>', unsafe_allow_html=True)
            fig_net_b = draw_passing_network(passing_net, player_identities, "B", safe_color(tb_color))
            st.plotly_chart(fig_net_b, use_container_width=True, config={"displayModeBar": False})

        # --- Ball Tracking ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Ball Tracking</div></div>', unsafe_allow_html=True)
        ball_hist = ball_tracking.get("position_history", [])
        if ball_hist:
            times = [b["t"] for b in ball_hist]
            xs = [b["x"] for b in ball_hist]
            ys = [b["y"] for b in ball_hist]

            fig_ball = go.Figure()
            fig_ball.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=3, color=times, colorscale="Viridis", showscale=True,
                            colorbar=dict(title=dict(text="Time (s)", font=dict(color="#aaa")), tickfont=dict(color="#aaa"))),
                hovertext=["t=" + f"{t:.1f}" + "s" for t in times],
            ))
            fig_ball.update_layout(
                plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
                xaxis=dict(title="X", color="#aaa", gridcolor="#333"),
                yaxis=dict(title="Y", color="#aaa", gridcolor="#333", scaleanchor="x"),
                margin=dict(l=40, r=20, t=10, b=40), height=280, dragmode=False,
            )
            st.plotly_chart(fig_ball, use_container_width=True, config={"displayModeBar": False})

    # ================================================================
    # TAB: MOMENTUM (improvement 16)
    # ================================================================
    with tab_momentum:
        st.markdown('<div class="stat-card"><div class="stat-card-title">Match Momentum</div></div>', unsafe_allow_html=True)
        fig_mom = draw_momentum_chart(pass_events, xg_shot_events, duration_min, ta_color, tb_color, ta_name, tb_name)
        st.plotly_chart(fig_mom, use_container_width=True, config={"displayModeBar": False})

        # --- Key Events Timeline ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Key Events</div></div>', unsafe_allow_html=True)

        events_timeline = []
        for s in highlights.get("shots", []):
            events_timeline.append({
                "time": s.get("time", 0),
                "type": "Shot",
                "team": s.get("team", "?"),
                "detail": "xG: " + f"{s.get('xg', 0):.2f}" + " | " + ("On target" if s.get("on_target") else "Off target"),
            })

        def _pid_name(pid):
            pid_str = str(pid)
            ident = player_identities.get(pid_str, {})
            name = ident.get("name")
            if name:
                parts = name.split()
                return parts[-1] if len(parts) > 1 else name
            return ident.get("display", "#" + str(pid))

        for kp in highlights.get("key_passes", []):
            events_timeline.append({
                "time": kp.get("time", 0),
                "type": "Key Pass",
                "team": kp.get("team", "?"),
                "detail": _pid_name(kp.get("from", "?")) + " -> " + _pid_name(kp.get("to", "?")),
            })

        events_timeline.sort(key=lambda x: x["time"])

        if events_timeline:
            for evt in events_timeline[:30]:
                minutes = int(evt["time"] / 60)
                seconds = int(evt["time"] % 60)
                team = evt["team"]
                color = ta_color if team == "A" else safe_color(tb_color)
                t_name = ta_name if team == "A" else tb_name
                icon = "&#127919;" if evt["type"] == "Shot" else "&#128273;"

                evt_html = ('<div style="display:flex;align-items:center;padding:8px 16px;background:#1a1a1a;border-radius:10px;margin-bottom:4px;border-left:3px solid ' + color + ';">'
                            '<div style="width:50px;font-size:13px;color:#aaa;font-weight:600;">' + str(minutes) + ':' + f"{seconds:02d}" + '</div>'
                            '<div style="font-size:16px;margin-right:8px;">' + icon + '</div>'
                            '<div style="flex:1;">'
                            '<span style="color:white;font-weight:600;font-size:13px;">' + evt["type"] + '</span>'
                            '<span style="color:#888;font-size:12px;margin-left:8px;">' + t_name + '</span>'
                            '</div>'
                            '<div style="color:#aaa;font-size:12px;">' + evt["detail"] + '</div>'
                            '</div>')
                st.markdown(evt_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;color:#666;padding:20px;">No key events recorded</div>', unsafe_allow_html=True)

        # --- Sprint Distribution ---
        st.markdown('<div class="stat-card"><div class="stat-card-title">Sprint Distribution</div></div>', unsafe_allow_html=True)
        if sprint_events:
            sprint_times_a = [s["start_t"] / 60 for s in sprint_events if s.get("team") == "A"]
            sprint_times_b = [s["start_t"] / 60 for s in sprint_events if s.get("team") == "B"]

            fig_sprint = go.Figure()
            if sprint_times_a:
                fig_sprint.add_trace(go.Histogram(
                    x=sprint_times_a, nbinsx=int(duration_min),
                    name=ta_name, marker_color=ta_color, opacity=0.7,
                ))
            if sprint_times_b:
                fig_sprint.add_trace(go.Histogram(
                    x=sprint_times_b, nbinsx=int(duration_min),
                    name=tb_name, marker_color=safe_color(tb_color), opacity=0.7,
                ))
            fig_sprint.update_layout(
                barmode="overlay",
                plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
                xaxis=dict(title="Minutes", color="#aaa", gridcolor="#333"),
                yaxis=dict(title="Sprints", color="#aaa", gridcolor="#333"),
                margin=dict(l=40, r=20, t=10, b=40), height=220,
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                dragmode=False,
            )
            st.plotly_chart(fig_sprint, use_container_width=True, config={"displayModeBar": False})

    # ================================================================
    # TAB: EXPORT (improvement 17)
    # ================================================================
    with tab_export:
        st.markdown('<div class="stat-card"><div class="stat-card-title">Download Match Data</div></div>', unsafe_allow_html=True)

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = generate_export_csv(data)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="tactivision_match_stats.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            text_data = generate_export_text(data)
            st.download_button(
                label="Download Text Summary",
                data=text_data,
                file_name="tactivision_match_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        st.code(generate_export_text(data), language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # FOOTER
    # ----------------------------------------------------------------
    footer_html = ('<div style="text-align:center;padding:24px 0 12px 0;">'
                   '<div style="font-size:13px;color:#444;">Powered by TactiVision Pro | ONNX Runtime Analytics Pipeline</div>'
                   '<div style="font-size:11px;color:#333;margin-top:4px;">First ' + str(int(duration_min)) + ' minutes analyzed from broadcast footage</div>'
                   '</div>')
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
