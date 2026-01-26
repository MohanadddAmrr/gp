"""
TactiVision Dashboard - Enhanced Assistant Manager Analytics

Professional dashboard for coaches with FULL tactical insights:
- Team positioning and movement
- Player performance metrics
- Ball tracking with prediction stats
- Possession analytics with zones, pressure, duration
- Pass detection and passing networks
- Tactical insights and recommendations
"""

import streamlit as st
from pathlib import Path
import json
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TactiVision - Assistant Manager",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================
OUTPUTS_DIR = Path(__file__).parent / "demo_outputs"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_pil_image(path: Path):
    """Safely load an image, or return None if not ready."""
    if not path.exists():
        return None
    try:
        return Image.open(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None


def load_metrics(video_dir: Path):
    p = video_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None


def list_player_heatmaps(video_dir: Path):
    """Return {player_id: Path} for all per-player heatmaps."""
    heatmaps = {}
    for p in video_dir.glob("heatmap_player_*.png"):
        try:
            pid = int(p.stem.replace("heatmap_player_", ""))
            heatmaps[pid] = p
        except:
            continue
    return heatmaps


def list_videos():
    """Return {video_id: Path} for all processed videos."""
    if not OUTPUTS_DIR.exists():
        return {}
    return {d.name: d for d in OUTPUTS_DIR.iterdir() if d.is_dir()}


def build_tracks_df(metrics):
    """Build DataFrame from tracks data."""
    if "tracks" not in metrics:
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics["tracks"])
    if df.empty:
        return df
    
    # Round for display
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    
    return df


def format_time(seconds):
    """Format seconds as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def build_pass_events_df(metrics):
    """Build DataFrame from pass events."""
    pass_events = metrics.get("pass_events", [])
    if not pass_events:
        return pd.DataFrame()
    
    df = pd.DataFrame(pass_events)
    if df.empty:
        return df
    
    # Select and rename columns for display
    display_cols = ['timestamp', 'passer_id', 'receiver_id', 'passer_team', 
                    'outcome', 'distance_m', 'direction', 'velocity_mps']
    available_cols = [c for c in display_cols if c in df.columns]
    df = df[available_cols]
    
    # Round numeric columns
    if 'distance_m' in df.columns:
        df['distance_m'] = df['distance_m'].round(1)
    if 'velocity_mps' in df.columns:
        df['velocity_mps'] = df['velocity_mps'].round(1)
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].round(1)
    
    return df


def build_player_passing_df(metrics):
    """Build DataFrame for player passing stats."""
    pass_stats = metrics.get("pass_detection", {})
    player_passes = pass_stats.get("player_passes", {})
    
    if not player_passes:
        return pd.DataFrame()
    
    rows = []
    for pid, stats in player_passes.items():
        rows.append({
            'player_id': int(pid),
            'attempted': stats.get('attempted', 0),
            'completed': stats.get('completed', 0),
            'accuracy': stats.get('accuracy', 0.0)
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('attempted', ascending=False)
    
    return df


# ============================================================
# MAIN APP
# ============================================================

st.title("⚽ TactiVision - Assistant Manager Dashboard")
st.markdown("**Professional football analytics for tactical insights**")

# Sidebar: Video Selection
with st.sidebar:
    st.header("📹 Video Selection")
    
    videos = list_videos()
    if not videos:
        st.error("No processed videos found!")
        st.info(f"Run `demo/run_demo.py` first to process videos")
        st.stop()
    
    video_names = list(videos.keys())
    selected = st.selectbox("Select Match", video_names, index=0)
    video_dir = videos[selected]
    
    st.markdown("---")
    st.markdown(f"**Selected:** {selected}")

# Load metrics
metrics = load_metrics(video_dir)
if not metrics:
    st.error(f"No metrics.json found in {video_dir}")
    st.stop()

# Extract key data
ball_tracking = metrics.get("ball_tracking", {})
possession_data = metrics.get("possession", {})
pass_detection = metrics.get("pass_detection", {})
ball_detected = ball_tracking.get("total_detections", 0) > 0
detection_rate = ball_tracking.get("detection_rate", 0) * 100

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔥 Team Heatmaps",
    "👤 Player Analysis",
    "⚽ Ball Analytics",
    "🎯 Possession Analytics",
    "🎯 Pass Analytics",
    "📈 Statistics",
    "🏃 Movement Metrics"
])

# ============================================================
# TAB 1: TEAM HEATMAPS
# ============================================================
with tab1:
    st.header("🔥 Team Position Heatmaps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Global (All Players)")
        img_global = load_pil_image(video_dir / "heatmap_global.png")
        if img_global:
            st.image(img_global, use_container_width=True)
        else:
            st.info("Heatmap not yet generated")
    
    with col2:
        st.subheader("Team A (Left Side)")
        img_a = load_pil_image(video_dir / "heatmap_team_A.png")
        if img_a:
            st.image(img_a, use_container_width=True)
        else:
            st.info("Heatmap not yet generated")
    
    with col3:
        st.subheader("Team B (Right Side)")
        img_b = load_pil_image(video_dir / "heatmap_team_B.png")
        if img_b:
            st.image(img_b, use_container_width=True)
        else:
            st.info("Heatmap not yet generated")

# ============================================================
# TAB 2: PLAYER ANALYSIS
# ============================================================
with tab2:
    st.header("👤 Individual Player Analysis")
    
    player_heatmaps = list_player_heatmaps(video_dir)
    
    if not player_heatmaps:
        st.warning("No player heatmaps available")
    else:
        tracks_df = build_tracks_df(metrics)
        
        player_ids = sorted(player_heatmaps.keys())
        selected_player = st.selectbox("Select Player", player_ids)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Player {selected_player} Heatmap")
            img = load_pil_image(player_heatmaps[selected_player])
            if img:
                st.image(img, use_container_width=True)
        
        with col2:
            st.subheader(f"Player {selected_player} Performance")
            
            if not tracks_df.empty and selected_player in tracks_df['player_id'].values:
                player_row = tracks_df[tracks_df['player_id'] == selected_player].iloc[0]
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Team", player_row.get('team', 'N/A'))
                m2.metric("Distance", f"{player_row.get('total_distance_m', 0):.0f}m")
                m3.metric("Avg Speed", f"{player_row.get('avg_speed_mps', 0):.1f} m/s")
                m4.metric("Max Speed", f"{player_row.get('max_speed_mps', 0):.1f} m/s")
                
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Workload", f"{player_row.get('workload_score', 0):.0f}")
                m6.metric("Attack Zone %", f"{player_row.get('attacking_third_ratio', 0)*100:.0f}%")
                m7.metric("Central Attack %", f"{player_row.get('central_attacking_ratio', 0)*100:.0f}%")
                m8.metric("Involvement", f"{player_row.get('involvement_index', 0):.2f}")
            else:
                st.info("No performance data for this player")

# ============================================================
# TAB 3: BALL ANALYTICS (ENHANCED)
# ============================================================
with tab3:
    st.header("⚽ Ball Tracking & Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Detection Rate", f"{detection_rate:.1f}%")
    col2.metric("Total Detections", ball_tracking.get("total_detections", 0))
    col3.metric("Avg Velocity", f"{ball_tracking.get('avg_velocity_px_s', 0):.0f} px/s")
    col4.metric("Max Velocity", f"{ball_tracking.get('max_velocity_px_s', 0):.0f} px/s")
    
    st.markdown("---")
    
    # Detection method breakdown
    st.subheader("Detection Method Breakdown")
    
    yolo = ball_tracking.get("yolo_detections", 0)
    color = ball_tracking.get("color_detections", 0)
    predicted = ball_tracking.get("predicted_detections", 0)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("YOLO Detections", yolo, help="Direct neural network detection")
        st.metric("Color Detections", color, help="HSV color-based fallback")
        st.metric("Predicted", predicted, help="Trajectory prediction when detection fails")
    
    with col2:
        if yolo + color + predicted > 0:
            fig = px.pie(
                values=[yolo, color, predicted],
                names=['YOLO (Neural Net)', 'Color (HSV)', 'Predicted (Trajectory)'],
                title="Detection Method Distribution",
                color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Ball heatmap
    st.subheader("Ball Position Heatmap")
    ball_heatmap = load_pil_image(video_dir / "heatmap_ball.png")
    if ball_heatmap:
        st.image(ball_heatmap, use_container_width=True)
    else:
        st.info("Ball heatmap not available")

# ============================================================
# TAB 4: POSSESSION ANALYTICS (FULLY ENHANCED)
# ============================================================
with tab4:
    st.header("🎯 Possession Analytics - Tactical Insights")
    st.markdown("**Understanding ball control, pressure, zones, and playing style**")
    
    # Team possession percentages
    poss_pct = possession_data.get("team_possession_percentage", {"A": 0, "B": 0})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Team A Possession", f"{poss_pct.get('A', 0):.1f}%")
    col2.metric("Team B Possession", f"{poss_pct.get('B', 0):.1f}%")
    col3.metric("Possession Changes", possession_data.get("total_possession_changes", 0))
    
    st.markdown("---")
    
    # Zone statistics
    st.subheader("⚡ Possession by Zone")
    zone_stats = possession_data.get("zone_stats", {})
    
    if zone_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Team A Zones**")
            zone_a = zone_stats.get("A", {})
            for zone, pct in zone_a.items():
                st.progress(pct / 100 if pct <= 100 else 1.0, text=f"{zone}: {pct:.1f}%")
        
        with col2:
            st.markdown("**Team B Zones**")
            zone_b = zone_stats.get("B", {})
            for zone, pct in zone_b.items():
                st.progress(pct / 100 if pct <= 100 else 1.0, text=f"{zone}: {pct:.1f}%")
    else:
        st.info("Zone statistics not available")
    
    st.markdown("---")
    
    # Pressure statistics
    st.subheader("💪 Pressure Analysis")
    pressure_stats = possession_data.get("pressure_stats", {})
    
    if pressure_stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("High Pressure Events", pressure_stats.get("total_high_pressure_events", 0))
        col2.metric("Avg Pressure", f"{pressure_stats.get('avg_pressure_count', 0):.1f} opponents")
        col3.metric("Max Pressure", f"{pressure_stats.get('max_pressure_count', 0)} opponents")
    else:
        st.info("Pressure statistics not available")
    
    st.markdown("---")
    
    # Duration statistics
    st.subheader("⏱️ Possession Duration Analysis")
    duration_stats = possession_data.get("duration_stats", {})
    
    if duration_stats and duration_stats.get("total_possessions", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Possessions", duration_stats.get("total_possessions", 0))
        col2.metric("Avg Duration", f"{duration_stats.get('avg_duration', 0):.1f}s")
        col3.metric("Max Duration", f"{duration_stats.get('max_duration', 0):.1f}s")
        col4.metric("Zone Changes", possession_data.get("zone_changes", 0))
        
        # Duration breakdown
        st.markdown("**Possession Style Breakdown:**")
        short_pct = duration_stats.get("short_pct", 0)
        medium_pct = duration_stats.get("medium_pct", 0)
        long_pct = duration_stats.get("long_pct", 0)
        
        fig = px.pie(
            values=[short_pct, medium_pct, long_pct],
            names=['Short (<2s)', 'Medium (2-5s)', 'Long (>5s)'],
            title="Possession Duration Distribution",
            color_discrete_sequence=['#FF5722', '#FFC107', '#4CAF50']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tactical insight
        if short_pct > 70:
            st.info("📊 **Tactical Insight:** High percentage of short possessions indicates a direct, fast-paced playing style.")
        elif long_pct > 30:
            st.info("📊 **Tactical Insight:** Significant long possessions suggest patient build-up play and ball retention.")
    else:
        st.info("Duration statistics not available")

# ============================================================
# TAB 5: PASS ANALYTICS (NEW)
# ============================================================
with tab5:
    st.header("🎯 Pass Detection & Analysis")
    st.markdown("**Understanding passing patterns, accuracy, and team connectivity**")
    
    if not pass_detection or pass_detection.get("total_passes", 0) == 0:
        st.warning("No passes detected in this video. This could be due to:")
        st.markdown("""
        - Low ball velocity during possession changes
        - Short distances between players (< 5m counted as dribbles)
        - Limited possession changes in the footage
        """)
    else:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Passes", pass_detection.get("total_passes", 0))
        col2.metric("Completed", pass_detection.get("completed_passes", 0), 
                   delta=f"{pass_detection.get('pass_accuracy', 0):.0f}% accuracy")
        col3.metric("Intercepted", pass_detection.get("intercepted_passes", 0))
        col4.metric("Avg Distance", f"{pass_detection.get('distance', {}).get('avg_m', 0):.1f}m")
        
        st.markdown("---")
        
        # Pass direction breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Pass Direction")
            direction = pass_detection.get("direction", {})
            
            if direction:
                forward = direction.get("forward", 0)
                backward = direction.get("backward", 0)
                lateral = direction.get("lateral", 0)
                
                fig = px.pie(
                    values=[forward, backward, lateral],
                    names=['Forward', 'Backward', 'Lateral'],
                    title="Pass Direction Distribution",
                    color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Direction metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Forward", f"{direction.get('forward_pct', 0):.0f}%")
                m2.metric("Backward", f"{direction.get('backward_pct', 0):.0f}%")
                m3.metric("Lateral", f"{direction.get('lateral_pct', 0):.0f}%")
        
        with col2:
            st.subheader("📊 Team Comparison")
            team_passes = pass_detection.get("team_passes", {})
            team_accuracy = pass_detection.get("team_accuracy", {})
            
            if team_passes:
                team_a = team_passes.get("A", {})
                team_b = team_passes.get("B", {})
                
                fig = go.Figure(data=[
                    go.Bar(name='Attempted', x=['Team A', 'Team B'], 
                           y=[team_a.get('attempted', 0), team_b.get('attempted', 0)],
                           marker_color='#2196F3'),
                    go.Bar(name='Completed', x=['Team A', 'Team B'], 
                           y=[team_a.get('completed', 0), team_b.get('completed', 0)],
                           marker_color='#4CAF50')
                ])
                fig.update_layout(
                    barmode='group',
                    title="Passes by Team",
                    yaxis_title="Number of Passes"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Team accuracy
                m1, m2 = st.columns(2)
                m1.metric("Team A Accuracy", f"{team_accuracy.get('A', 0):.1f}%")
                m2.metric("Team B Accuracy", f"{team_accuracy.get('B', 0):.1f}%")
        
        st.markdown("---")
        
        # Pass distance analysis
        st.subheader("📏 Pass Distance Analysis")
        distance_stats = pass_detection.get("distance", {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average", f"{distance_stats.get('avg_m', 0):.1f}m")
        col2.metric("Shortest", f"{distance_stats.get('min_m', 0):.1f}m")
        col3.metric("Longest", f"{distance_stats.get('max_m', 0):.1f}m")
        
        # Pass events table
        st.markdown("---")
        st.subheader("📋 Recent Passes")
        
        pass_df = build_pass_events_df(metrics)
        if not pass_df.empty:
            # Style the dataframe
            st.dataframe(
                pass_df.tail(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No pass event details available")
        
        st.markdown("---")
        
        # Player passing stats
        st.subheader("👤 Player Passing Statistics")
        
        player_pass_df = build_player_passing_df(metrics)
        if not player_pass_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    player_pass_df.head(10),
                    x='player_id',
                    y=['completed', 'attempted'],
                    title="Top 10 Passers",
                    barmode='overlay',
                    labels={'value': 'Passes', 'player_id': 'Player ID'},
                    color_discrete_map={'completed': '#4CAF50', 'attempted': '#E0E0E0'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Passing Leaderboard**")
                st.dataframe(
                    player_pass_df[['player_id', 'attempted', 'accuracy']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No player passing data available")
        
        # Passing network
        st.markdown("---")
        st.subheader("🔗 Passing Network")
        
        passing_network = metrics.get("passing_network", {})
        top_combinations = passing_network.get("most_common_combinations", [])
        
        if top_combinations:
            st.markdown("**Most Frequent Passing Combinations:**")
            
            for i, combo in enumerate(top_combinations[:5], 1):
                col1, col2, col3 = st.columns([1, 2, 1])
                col1.write(f"**#{i}**")
                col2.write(f"Player {combo['from']} → Player {combo['to']}")
                col3.write(f"**{combo['count']} passes**")
        else:
            st.info("No passing network data available")
        
        # Tactical insight
        st.markdown("---")
        if pass_detection.get("pass_accuracy", 0) > 80:
            st.success("📊 **Tactical Insight:** Excellent passing accuracy (>80%) indicates strong ball retention and controlled play.")
        elif pass_detection.get("pass_accuracy", 0) > 60:
            st.info("📊 **Tactical Insight:** Good passing accuracy. Consider focusing on reducing turnovers in attacking third.")
        else:
            st.warning("📊 **Tactical Insight:** Low passing accuracy suggests the team is under pressure or attempting risky passes.")
        
        direction = pass_detection.get("direction", {})
        if direction.get("forward_pct", 0) > 60:
            st.info("📊 **Style:** Direct, attacking style with majority forward passes.")
        elif direction.get("backward_pct", 0) > 40:
            st.info("📊 **Style:** Patient build-up play with significant backward/recycling passes.")

# ============================================================
# TAB 6: STATISTICS
# ============================================================
with tab6:
    st.header("📈 Overall Match Statistics")
    
    tracks_df = build_tracks_df(metrics)
    
    if tracks_df.empty:
        st.warning("No player statistics available")
    else:
        # Team comparison
        st.subheader("Team Comparison")
        
        team_stats = tracks_df.groupby('team').agg({
            'total_distance_m': 'sum',
            'avg_speed_mps': 'mean',
            'max_speed_mps': 'max',
            'workload_score': 'mean'
        }).round(2)
        
        st.dataframe(team_stats, use_container_width=True)
        
        # Top players
        st.subheader("Top Performers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Distance Covered**")
            top_distance = tracks_df.nlargest(5, 'total_distance_m')[['player_id', 'team', 'total_distance_m']]
            st.dataframe(top_distance, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Highest Involvement**")
            top_involvement = tracks_df.nlargest(5, 'involvement_index')[['player_id', 'team', 'involvement_index']]
            st.dataframe(top_involvement, use_container_width=True, hide_index=True)

# ============================================================
# TAB 7: MOVEMENT METRICS
# ============================================================
with tab7:
    st.header("🏃 Movement & Workload Analysis")
    
    tracks_df = build_tracks_df(metrics)
    
    if tracks_df.empty:
        st.warning("No movement data available")
    else:
        # Distance chart
        fig = px.bar(
            tracks_df.sort_values('total_distance_m', ascending=False),
            x='player_id',
            y='total_distance_m',
            color='team',
            color_discrete_map={'A': '#4CAF50', 'B': '#F44336'},
            title="Total Distance Covered by Player",
            labels={'total_distance_m': 'Distance (m)', 'player_id': 'Player ID'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Speed distribution
        fig2 = px.box(
            tracks_df,
            x='team',
            y='avg_speed_mps',
            color='team',
            color_discrete_map={'A': '#4CAF50', 'B': '#F44336'},
            title="Speed Distribution by Team",
            labels={'avg_speed_mps': 'Avg Speed (m/s)', 'team': 'Team'}
        )
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**TactiVision** - Professional Football Analytics | Assistant Manager Dashboard v3.0 (Pass Detection)")
