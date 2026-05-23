"""
Generate a clean ER-diagram PNG for TactiVision Pro's SQLite database.
Output:  C:/Users/hp/Desktop/gp/docs/figures/db_schema.png

Run:
    python scripts/generate_db_schema.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Output path ───────────────────────────────────────────────────────────────
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
OUT_FILE = os.path.join(OUT_DIR, "db_schema.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_HEADER   = "#1a3a5c"   # dark navy — table header
C_PK       = "#2e6da4"   # medium blue — PK row
C_ROW_ODD  = "#f0f4f8"   # very light blue-grey
C_ROW_EVEN = "#ffffff"   # white
C_BORDER   = "#1a3a5c"
C_TEXT_H   = "#ffffff"
C_TEXT     = "#1a1a2e"
C_PK_TEXT  = "#ffffff"
C_CONN     = "#e05c2d"   # orange — relationship line

# ── Table definitions ─────────────────────────────────────────────────────────
# Each column: (name, type, constraints, description)
MATCHES_COLS = [
    ("id",          "INTEGER", "PK  AUTOINCREMENT", "Surrogate key"),
    ("video_path",  "TEXT",    "NOT NULL",           "Path to uploaded video file"),
    ("team_a",      "TEXT",    "NOT NULL",           "Home team name"),
    ("team_b",      "TEXT",    "NOT NULL",           "Away team name"),
    ("score_a",     "INTEGER", "NULL allowed",       "Home score (NULL if unplayed)"),
    ("score_b",     "INTEGER", "NULL allowed",       "Away score (NULL if unplayed)"),
    ("created_at",  "DATETIME","DEFAULT NOW",        "Record creation timestamp"),
]

PLAYERS_COLS = [
    ("id",               "INTEGER", "PK  AUTOINCREMENT", "Surrogate key"),
    ("name",             "TEXT",    "NOT NULL",           "Player full name"),
    ("team_name",        "TEXT",    "NULL allowed",       "Club name from API"),
    ("jersey_number",    "INTEGER", "NULL allowed",       "Squad number"),
    ("position_default", "TEXT",    "NULL allowed",       "GK / DEF / MID / FWD"),
    ("created_at",       "DATETIME","DEFAULT NOW",        "Record creation timestamp"),
]

# ── Drawing helper ────────────────────────────────────────────────────────────
ROW_H   = 0.38   # height of each data row
HEAD_H  = 0.50   # height of header row
PAD     = 0.08   # internal horizontal padding
COL_W   = [1.9, 1.3, 1.8, 2.9]   # column widths: name | type | constraint | desc
TABLE_W = sum(COL_W)

def draw_table(ax, x, y_top, title, columns):
    """
    Draw an ER-style table at position (x, y_top).
    Returns the y coordinate of the bottom of the table.
    """
    n_rows = len(columns)
    table_h = HEAD_H + n_rows * ROW_H

    # ── Outer border ──────────────────────────────────────────────────────
    rect = FancyBboxPatch(
        (x, y_top - table_h), TABLE_W, table_h,
        boxstyle="round,pad=0.04",
        linewidth=1.8, edgecolor=C_BORDER, facecolor=C_ROW_EVEN,
        zorder=2,
    )
    ax.add_patch(rect)

    # ── Header ────────────────────────────────────────────────────────────
    hdr = FancyBboxPatch(
        (x, y_top - HEAD_H), TABLE_W, HEAD_H,
        boxstyle="round,pad=0.04",
        linewidth=0, facecolor=C_HEADER, zorder=3,
    )
    ax.add_patch(hdr)
    ax.text(
        x + TABLE_W / 2, y_top - HEAD_H / 2, title,
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=C_TEXT_H, zorder=4,
    )

    # ── Column header labels (small, italic) ──────────────────────────────
    col_labels = ["Column", "Type", "Constraints", "Description"]
    cx = x
    for lbl, cw in zip(col_labels, COL_W):
        ax.text(
            cx + PAD, y_top - HEAD_H + 0.07, lbl,
            ha="left", va="bottom", fontsize=6.5, fontstyle="italic",
            color="#aac8e0", zorder=4,
        )
        cx += cw

    # ── Data rows ─────────────────────────────────────────────────────────
    for i, (col_name, col_type, constraint, desc) in enumerate(columns):
        row_y = y_top - HEAD_H - (i + 1) * ROW_H
        is_pk = "PK" in constraint

        # Row background
        bg = C_PK if is_pk else (C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN)
        row_rect = plt.Rectangle(
            (x, row_y), TABLE_W, ROW_H,
            linewidth=0, facecolor=bg, zorder=2,
        )
        ax.add_patch(row_rect)

        # Subtle row divider
        ax.plot(
            [x, x + TABLE_W], [row_y + ROW_H, row_y + ROW_H],
            color="#d0dce8", linewidth=0.5, zorder=3,
        )

        text_color = C_PK_TEXT if is_pk else C_TEXT
        cell_data  = [col_name, col_type, constraint, desc]
        cx = x
        fontsizes  = [9, 8.5, 8, 8]
        fontweights = ["bold" if is_pk else "normal", "normal", "normal", "normal"]

        for val, cw, fs, fw in zip(cell_data, COL_W, fontsizes, fontweights):
            ax.text(
                cx + PAD, row_y + ROW_H / 2, val,
                ha="left", va="center", fontsize=fs, fontweight=fw,
                color=text_color, zorder=4, clip_on=True,
            )
            cx += cw

        # PK badge
        if is_pk:
            ax.text(
                x + TABLE_W - 0.12, row_y + ROW_H / 2, "[PK]",
                ha="right", va="center", fontsize=7, fontweight="bold",
                color="#ffe08a", zorder=4,
            )

    return y_top - table_h

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(17, 7))
ax.set_xlim(0, 16.8)
ax.set_ylim(-1, 8)
ax.axis("off")
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#f7f9fc")

# ── Main title ────────────────────────────────────────────────────────────────
ax.text(
    8.4, 7.7, "TactiVision Pro — Database Schema (ER Diagram)",
    ha="center", va="center", fontsize=14, fontweight="bold",
    color=C_HEADER,
)
ax.text(
    8.4, 7.3, "Storage: SQLite  •  File: matches.db  •  Managed by: DatabaseManager",
    ha="center", va="center", fontsize=9, color="#555577", style="italic",
)

# ── Draw tables ───────────────────────────────────────────────────────────────
X_MATCHES  = 0.3
X_PLAYERS  = 8.5
Y_TOP      = 7.0

bot_matches = draw_table(ax, X_MATCHES, Y_TOP, "matches",        MATCHES_COLS)
bot_players = draw_table(ax, X_PLAYERS, Y_TOP, "player_profiles", PLAYERS_COLS)

# ── "No FK" annotation between tables ────────────────────────────────────────
mid_y    = (Y_TOP + bot_matches) / 2
right_x  = X_MATCHES + TABLE_W
left_x   = X_PLAYERS
mid_x    = (right_x + left_x) / 2

ax.annotate(
    "", xy=(left_x, mid_y), xytext=(right_x, mid_y),
    arrowprops=dict(
        arrowstyle="<->",
        color=C_CONN,
        lw=1.6,
        connectionstyle="arc3,rad=0.0",
        linestyle="dashed",
    ),
    zorder=5,
)
ax.text(
    mid_x, mid_y + 0.22,
    "No FK relationship\n(independent tables)",
    ha="center", va="bottom", fontsize=8.5, color=C_CONN,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4ee", edgecolor=C_CONN, lw=1),
    zorder=6,
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_x = 0.4
legend_y = bot_matches - 0.3

patches = [
    mpatches.Patch(color=C_HEADER,  label="Table header"),
    mpatches.Patch(color=C_PK,      label="Primary key (PK)"),
    mpatches.Patch(color=C_ROW_ODD, label="Regular column"),
]
ax.legend(
    handles=patches,
    loc="lower left",
    bbox_to_anchor=(legend_x / 15, (legend_y + 1) / 9),
    fontsize=8.5,
    framealpha=0.9,
    edgecolor="#c0c8d8",
)

# ── Footer ────────────────────────────────────────────────────────────────────
ax.text(
    8.4, -0.7,
    "Both tables are created automatically by DatabaseManager.initialize_database()  •  "
    "No manual migration required",
    ha="center", va="center", fontsize=8, color="#888899", style="italic",
)

plt.tight_layout(pad=0.5)
plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print("Saved: " + os.path.abspath(OUT_FILE))
