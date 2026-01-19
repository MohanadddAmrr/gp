import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------- Paths & basic UI ----------------
ROOT = Path(__file__).resolve().parent
DEMO_BASE = ROOT / "demo_outputs"

st.set_page_config(page_title="TactiVision", layout="wide")
st.title("TactiVision")


# ---------------- Helpers ----------------
def load_pil_image(path: Path):
    """Safely load an image, or return None if not ready."""
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB").copy()
    except Exception:
        return None


def load_metrics(video_dir: Path):
    p = video_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_player_heatmaps(video_dir: Path):
    """Return {player_id: Path} for all per-player heatmaps."""
    result = {}
    if not video_dir.exists():
        return result
    for p in video_dir.glob("heatmap_player_*.png"):
        try:
            pid = int(p.stem.split("_")[-1])
            result[pid] = p
        except Exception:
            continue
    return dict(sorted(result.items()))


def list_videos():
    """Return {video_id: Path} for all processed videos."""
    vids = {}
    if DEMO_BASE.exists():
        for d in DEMO_BASE.iterdir():
            if d.is_dir() and (d / "metrics.json").exists():
                vids[d.name] = d
    return dict(sorted(vids.items()))


def build_tracks_df(metrics):
    """Convert metrics['tracks'] to DataFrame and add some safe defaults."""
    if not metrics or not metrics.get("tracks"):
        return None

    df = pd.DataFrame(metrics["tracks"])

    # Ensure essential columns exist with defaults
    for col, default in [
        ("id", -1),
        ("team", "A"),  # or 0/1, but we map later
        ("total_distance_m", 0.0),
        ("avg_speed_mps", 0.0),
        ("max_speed_mps", 0.0),
        ("workload_score", 0.0),
        ("involvement_index", 0.0),
        ("attacking_ratio", 0.0),
        ("central_attacking_ratio", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Clean team labels: map 0/1 -> A/B if numeric
    if pd.api.types.is_numeric_dtype(df["team"]):
        df["team_label"] = df["team"].map({0: "Team A", 1: "Team B"}).fillna("Team ?")
    else:
        df["team_label"] = df["team"].astype(str)
    df["player_label"] = "P" + df["id"].astype(int).astype(str)

    # ---- NEW: workload band (Low / Medium / High) ----
    if "workload_score" in df.columns and df["workload_score"].max() > 0:
        low_thr = df["workload_score"].quantile(0.33)
        high_thr = df["workload_score"].quantile(0.66)

        def classify_workload(x):
            if x < low_thr:
                return "Low"
            elif x < high_thr:
                return "Medium"
            else:
                return "High"

        df["workload_band"] = df["workload_score"].apply(classify_workload)
    else:
        df["workload_band"] = "Unknown"

    return df


def build_team_summary(df: pd.DataFrame):
    """Aggregate physical stats per team."""
    grouped = df.groupby("team_label", dropna=False)

    rows = []
    for team_name, g in grouped:
        rows.append(
            {
                "Team": team_name,
                "Players (tracked)": len(g),
                "Total distance (m)": g["total_distance_m"].sum(),
                "Avg workload": g["workload_score"].mean(),
                "Avg involvement": g["involvement_index"].mean(),
            }
        )
    return pd.DataFrame(rows)


# ---------------- Main UI ----------------
def main():
    videos = list_videos()
    if not videos:
        st.info("No processed videos found in demo_outputs/. Run run_demo.py first.")
        return

    # Sidebar: choose video
    video_ids = list(videos.keys())
    selected_video_id = st.sidebar.selectbox("Video", video_ids, index=0)
    current_dir = videos[selected_video_id]
    st.caption(f"Video ID: {selected_video_id}")

    metrics = load_metrics(current_dir)
    df_tracks = build_tracks_df(metrics)

    # Tabs to keep layout clean
    tab_overview, tab_player = st.tabs(["Match Overview", "Player Details"])

    # --------- TAB 1: Match Overview ----------
    with tab_overview:
        st.subheader("Match Status")

        if metrics is None:
            st.info("Waiting for data...")
        else:
            st.text(
                f"Frame: {metrics['frame']} | "
                f"Players: {metrics['num_players']} | "
                f"Avg step(px): {metrics['avg_step_px']:.1f} | "
                f"Ball: {'Yes' if metrics['ball_detected'] else 'No'}"
            )

        # Main players table + rankings
        if df_tracks is not None and not df_tracks.empty:
            st.subheader("Players (per-frame summary)")

            # Show full table
            show_cols = [
                "player_label",
                "team_label",
                "total_distance_m",
                "avg_speed_mps",
                "max_speed_mps",
                "workload_score",
                "workload_band",       # <-- NEW
                "involvement_index",
            ]
            existing_cols = [c for c in show_cols if c in df_tracks.columns]
            st.dataframe(
                df_tracks[existing_cols].sort_values("total_distance_m", ascending=False),
                use_container_width=True,
            )

            st.divider()

            # Rankings + team summary in 3 columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Top 3 Workload")
                if "workload_score" in df_tracks.columns:
                    top_w = (
                        df_tracks.sort_values("workload_score", ascending=False)
                        .head(3)[
                            [
                                "player_label",
                                "team_label",
                                "workload_score",
                                "total_distance_m",
                                "workload_band",  # show band here too
                            ]
                        ]
                    )
                    top_w.rename(
                        columns={
                            "player_label": "Player",
                            "team_label": "Team",
                            "workload_score": "Workload",
                            "total_distance_m": "Dist (m)",
                            "workload_band": "Band",
                        },
                        inplace=True,
                    )
                    st.table(top_w)
                else:
                    st.info("Workload score not available.")

            with col2:
                st.subheader("Top 3 Involvement")
                if "involvement_index" in df_tracks.columns:
                    top_i = (
                        df_tracks.sort_values("involvement_index", ascending=False)
                        .head(3)[["player_label", "team_label", "involvement_index"]]
                    )
                    top_i.rename(
                        columns={
                            "player_label": "Player",
                            "team_label": "Team",
                            "involvement_index": "Involvement",
                        },
                        inplace=True,
                    )
                    st.table(top_i)
                else:
                    st.info("Involvement index not available.")

            with col3:
                st.subheader("Team Physical Summary")
                team_df = build_team_summary(df_tracks)
                st.table(team_df)

        st.divider()

        # Heatmaps
        st.subheader("Heatmaps")

        col_g, col_a, col_b = st.columns(3)

        with col_g:
            st.caption("Global")
            img = load_pil_image(current_dir / "heatmap_global.png")
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info("Global heatmap not available yet.")

        with col_a:
            st.caption("Team A")
            img = load_pil_image(current_dir / "heatmap_team_A.png")
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info("Team A heatmap not available yet.")

        with col_b:
            st.caption("Team B")
            img = load_pil_image(current_dir / "heatmap_team_B.png")
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info("Team B heatmap not available yet.")

    # --------- TAB 2: Player Details ----------
    with tab_player:
        st.subheader("Player Heatmap & Profile")

        player_maps = list_player_heatmaps(current_dir)

        # Build a list of player IDs from heatmaps and from metrics, then merge
        ids_from_maps = set(player_maps.keys())
        ids_from_df = set(df_tracks["id"].astype(int).tolist()) if df_tracks is not None else set()
        all_ids = sorted(ids_from_maps.union(ids_from_df))

        if not all_ids:
            st.info("No player data yet.")
        else:
            labels = [f"P{pid}" for pid in all_ids]
            selected_label = st.selectbox("Player", labels)
            selected_id = int(selected_label[1:])

            # Player profile (numeric stats)
            if df_tracks is not None:
                row = df_tracks[df_tracks["id"].astype(int) == selected_id]
                if not row.empty:
                    r = row.iloc[0]
                    st.markdown(f"### {selected_label} – Profile")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Total distance (m)",
                            f"{r['total_distance_m']:.1f}",
                        )
                        st.metric(
                            "Avg speed (m/s)",
                            f"{r['avg_speed_mps']:.2f}",
                        )
                    with c2:
                        st.metric(
                            "Max speed (m/s)",
                            f"{r['max_speed_mps']:.2f}",
                        )
                        st.metric(
                            "Workload score",
                            f"{r['workload_score']:.2f}",
                        )
                    with c3:
                        st.metric(
                            "Involvement index",
                            f"{r['involvement_index']:.2f}",
                        )
                        st.metric(
                            "Attacking-zone ratio",
                            f"{r['attacking_ratio']:.2f}",
                        )
                        # NEW: show workload band as a quick label
                        st.metric(
                            "Workload band",
                            str(r["workload_band"]),
                        )

                    st.divider()

            # Player heatmap image
            st.markdown(f"#### {selected_label} – Heatmap")
            if selected_id in player_maps:
                img = load_pil_image(player_maps[selected_id])
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.info("Heatmap image still generating...")
            else:
                st.info("No heatmap available for this player.")


if __name__ == "__main__":
    main()
