"""
TactiVision Dashboard - Enhanced Assistant Manager Analytics

Professional dashboard for coaches with FULL tactical insights:
- Team positioning and movement
- Player performance metrics
- Ball tracking with prediction stats
- Possession analytics with zones, pressure, duration
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
ball_detected = ball_tracking.get("total_detections", 0) > 0
detection_rate = ball_tracking.get("detection_rate", 0) * 100

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔥 Team Heatmaps",
    "👤 Player Analysis",
    "⚽ Ball Analytics",
    "🎯 Possession Analytics",
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
        st.subheader("🌍 Global Heatmap")
        img_global = load_pil_image(video_dir / "heatmap_global.png")
        if img_global:
            st.image(img_global, use_container_width=True)
        else:
            st.warning("Heatmap not found")
    
    with col2:
        st.subheader("🔵 Team A")
        img_a = load_pil_image(video_dir / "heatmap_team_A.png")
        if img_a:
            st.image(img_a, use_container_width=True)
        else:
            st.warning("Team A heatmap not found")
    
    with col3:
        st.subheader("🔴 Team B")
        img_b = load_pil_image(video_dir / "heatmap_team_B.png")
        if img_b:
            st.image(img_b, use_container_width=True)
        else:
            st.warning("Team B heatmap not found")

# ============================================================
# TAB 2: PLAYER ANALYSIS
# ============================================================
with tab2:
    st.header("👤 Individual Player Analysis")
    
    player_heatmaps = list_player_heatmaps(video_dir)
    
    if not player_heatmaps:
        st.warning("No player heatmaps found")
    else:
        player_ids = sorted(player_heatmaps.keys())
        
        # Get player stats for filtering
        tracks_df = build_tracks_df(metrics)
        
        if not tracks_df.empty:
            # Filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_player = st.selectbox(
                    "Select Player",
                    player_ids,
                    format_func=lambda x: f"Player {x}"
                )
            with col2:
                team = tracks_df[tracks_df['player_id'] == selected_player]['team'].values
                team_label = team[0] if len(team) > 0 else "Unknown"
                st.metric("Team", team_label)
            
            # Display heatmap and stats side by side
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"🔥 Position Heatmap - Player {selected_player}")
                img = load_pil_image(player_heatmaps[selected_player])
                if img:
                    st.image(img, use_container_width=True)
            
            with col2:
                st.subheader("📊 Performance Stats")
                player_data = tracks_df[tracks_df['player_id'] == selected_player]
                
                if not player_data.empty:
                    row = player_data.iloc[0]
                    
                    st.metric("Total Distance", f"{row['total_distance_m']:.1f} m")
                    st.metric("Avg Speed", f"{row['avg_speed_mps']:.2f} m/s")
                    st.metric("Max Speed", f"{row['max_speed_mps']:.2f} m/s")
                    st.metric("Workload Score", f"{row['workload_score']:.1f}")
                    st.metric("Involvement Index", f"{row['involvement_index']:.2f}")
                    
                    # Possession stats if available
                    poss_stats = possession_data.get('player_stats', {})
                    if str(selected_player) in poss_stats:
                        player_poss = poss_stats[str(selected_player)]
                        st.markdown("---")
                        st.markdown("**⚽ Possession Stats**")
                        st.metric("Touches", int(player_poss.get('touch_count', 0)))
                        st.metric("Possession Time", format_time(player_poss.get('total_time', 0)))
                        st.metric("Avg Touch Duration", f"{player_poss.get('avg_possession_duration', 0):.2f}s")

# ============================================================
# TAB 3: BALL ANALYTICS (ENHANCED)
# ============================================================
with tab3:
    st.header("⚽ Ball Tracking & Analytics")
    
    if not ball_detected:
        st.warning("⚠️ No ball detected in this video")
        st.info("💡 Ball detection works best with:\n- Clear white/colored ball\n- Good lighting\n- Ball visible in frame\n- Hybrid detection enabled (YOLO + Color + Prediction)")
    else:
        # Detection summary with prediction stats
        yolo_count = ball_tracking.get('yolo_detections', 0)
        color_count = ball_tracking.get('color_detections', 0)
        predicted_count = ball_tracking.get('predicted_detections', 0)
        total_detections = ball_tracking.get('total_detections', 0)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Detections", f"{total_detections}/{metrics['frame']}")
        with col2:
            st.metric("Detection Rate", f"{detection_rate:.1f}%")
        with col3:
            st.metric("YOLO", yolo_count, help="Direct YOLO detections")
        with col4:
            st.metric("Color", color_count, help="Color-based fallback detections")
        with col5:
            st.metric("Predicted", predicted_count, help="Trajectory predictions (gap filling)")
        
        # Detection improvement insight
        if predicted_count > 0:
            improvement = (predicted_count / total_detections) * 100
            st.success(f"✨ Trajectory prediction improved tracking by {improvement:.1f}% (filled {predicted_count} gaps)")
        
        # Ball heatmap
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔥 Ball Position Heatmap")
            ball_heatmap = load_pil_image(video_dir / "heatmap_ball.png")
            if ball_heatmap:
                st.image(ball_heatmap, use_container_width=True)
                st.caption("Shows where the ball spent most time during the match")
            else:
                st.warning("Ball heatmap not generated")
        
        with col2:
            st.subheader("📊 Ball Movement Stats")
            avg_vel = ball_tracking.get('avg_velocity_px_s', 0)
            max_vel = ball_tracking.get('max_velocity_px_s', 0)
            
            st.metric("Avg Velocity", f"{avg_vel:.1f} px/s")
            st.metric("Max Velocity", f"{max_vel:.1f} px/s")
            
            # Detection method breakdown
            if yolo_count > 0 or color_count > 0 or predicted_count > 0:
                st.markdown("---")
                st.markdown("**Detection Methods**")
                fig = go.Figure(data=[go.Pie(
                    labels=['YOLO', 'Color', 'Predicted'],
                    values=[yolo_count, color_count, predicted_count],
                    marker=dict(colors=['#FFD700', '#FF00FF', '#FFA500'])
                )])
                fig.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: POSSESSION ANALYTICS (FULLY ENHANCED)
# ============================================================
with tab4:
    st.header("🎯 Possession Analytics - Tactical Insights")
    st.markdown("**Understanding ball control, pressure, zones, and playing style**")
    
    if not possession_data or possession_data.get('total_possession_changes', 0) == 0:
        st.warning("⚠️ No possession data available")
        st.info("Possession tracking requires:\n- Ball detection\n- Player tracking\n- Both must be active simultaneously")
    else:
        # Key metrics row
        team_poss = possession_data.get('team_possession_percentage', {})
        poss_changes = possession_data.get('total_possession_changes', 0)
        zone_changes = possession_data.get('zone_changes', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Team A Possession", f"{team_poss.get('A', 0):.1f}%", 
                     delta=None if team_poss.get('A', 0) == 50 else f"{team_poss.get('A', 0) - 50:+.1f}%")
        with col2:
            st.metric("Team B Possession", f"{team_poss.get('B', 0):.1f}%",
                     delta=None if team_poss.get('B', 0) == 50 else f"{team_poss.get('B', 0) - 50:+.1f}%")
        with col3:
            st.metric("Possession Changes", poss_changes)
        with col4:
            st.metric("Zone Changes", zone_changes, help="Transitions between defensive/midfield/attacking zones")
        
        st.markdown("---")
        
        # Row 1: Team possession + Zone breakdown
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Team Possession Breakdown")
            
            if team_poss.get('A', 0) > 0 or team_poss.get('B', 0) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Team A', 'Team B'],
                    values=[team_poss.get('A', 0), team_poss.get('B', 0)],
                    marker=dict(colors=['#4CAF50', '#F44336']),
                    hole=0.4,
                    textinfo='label+percent',
                    textfont_size=16
                )])
                fig.update_layout(
                    height=350,
                    showlegend=True,
                    annotations=[dict(text='Possession', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tactical insight
                team_a_poss = team_poss.get('A', 0)
                if team_a_poss > 60:
                    st.success("✅ Team A dominated possession - controlling the game")
                elif team_a_poss < 40:
                    st.info("🔄 Team B dominated possession - Team A playing counter-attacking style")
                else:
                    st.info("⚖️ Balanced possession - competitive match")
            else:
                st.warning("No possession data available for visualization")
        
        with col2:
            st.subheader("🗺️ Possession by Zone")
            
            zone_stats = possession_data.get('zone_stats', {})
            
            if zone_stats:
                # Create stacked bar chart for zones
                zone_data = []
                for team in ['A', 'B']:
                    if team in zone_stats:
                        for zone, pct in zone_stats[team].items():
                            zone_data.append({
                                'Team': f'Team {team}',
                                'Zone': zone,
                                'Percentage': pct
                            })
                
                if zone_data:
                    zone_df = pd.DataFrame(zone_data)
                    fig = px.bar(
                        zone_df,
                        x='Team',
                        y='Percentage',
                        color='Zone',
                        title='Possession Distribution by Tactical Zone',
                        color_discrete_map={
                            'Defensive': '#2196F3',
                            'Midfield': '#FFC107',
                            'Attacking': '#F44336'
                        },
                        text='Percentage'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                    fig.update_layout(height=350, yaxis_title='Possession %')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tactical insight
                    team_a_zones = zone_stats.get('A', {})
                    if team_a_zones.get('Attacking', 0) > 40:
                        st.success("⚔️ Team A: Aggressive attacking style - high possession in final third")
                    elif team_a_zones.get('Defensive', 0) > 40:
                        st.info("🛡️ Team A: Defensive style - building from the back")
            else:
                st.warning("No zone statistics available")
        
        st.markdown("---")
        
        # Row 2: Pressure stats + Duration analysis
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💪 Pressure Statistics")
            
            pressure_stats = possession_data.get('pressure_stats', {})
            
            if pressure_stats and pressure_stats.get('total_high_pressure_events', 0) > 0:
                high_press_events = pressure_stats.get('total_high_pressure_events', 0)
                avg_pressure = pressure_stats.get('avg_pressure_count', 0)
                max_pressure = pressure_stats.get('max_pressure_count', 0)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("High Pressure Events", high_press_events, help="Possessions with 3+ opponents nearby")
                with col_b:
                    st.metric("Avg Opponents", f"{avg_pressure:.1f}")
                with col_c:
                    st.metric("Max Opponents", max_pressure)
                
                # Pressure insight
                if high_press_events > 10:
                    st.warning("🔥 High defensive pressure throughout the match - physical, intense game")
                else:
                    st.info("💨 Low defensive pressure - teams playing with space")
            else:
                st.info("No high-pressure events recorded (< 3 opponents pressing)")
        
        with col2:
            st.subheader("⏱️ Possession Duration Analysis")
            
            duration_stats = possession_data.get('duration_stats', {})
            
            if duration_stats and duration_stats.get('total_possessions', 0) > 0:
                avg_duration = duration_stats.get('avg_duration', 0)
                short_pct = duration_stats.get('short_pct', 0)
                medium_pct = duration_stats.get('medium_pct', 0)
                long_pct = duration_stats.get('long_pct', 0)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Avg Duration", f"{avg_duration:.2f}s")
                with col_b:
                    st.metric("Total Possessions", duration_stats.get('total_possessions', 0))
                
                # Duration breakdown chart
                duration_data = pd.DataFrame({
                    'Category': ['Short (<2s)', 'Medium (2-5s)', 'Long (>5s)'],
                    'Percentage': [short_pct, medium_pct, long_pct]
                })
                
                fig = px.bar(
                    duration_data,
                    x='Category',
                    y='Percentage',
                    color='Category',
                    color_discrete_map={
                        'Short (<2s)': '#FF5722',
                        'Medium (2-5s)': '#FFC107',
                        'Long (>5s)': '#4CAF50'
                    },
                    text='Percentage'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=250, showlegend=False, yaxis_title='% of Possessions')
                st.plotly_chart(fig, use_container_width=True)
                
                # Playing style insight
                if short_pct > 50:
                    st.info("⚡ Direct playing style - quick, fast-paced transitions")
                elif long_pct > 30:
                    st.success("🧠 Patient build-up play - controlled possession style")
                else:
                    st.info("⚖️ Balanced playing style - mix of direct and build-up play")
            else:
                st.info("Not enough possession data for duration analysis")
        
        st.markdown("---")
        
        # Row 3: Player possession leaders
        st.subheader("👥 Player Possession Leaders")
        
        player_poss_stats = possession_data.get('player_stats', {})
        
        if player_poss_stats:
            # Build dataframe
            poss_rows = []
            for pid_str, stats in player_poss_stats.items():
                pid = int(pid_str)
                # Get team from tracks
                tracks_df = build_tracks_df(metrics)
                team = "?"
                if not tracks_df.empty:
                    player_team = tracks_df[tracks_df['player_id'] == pid]['team'].values
                    team = player_team[0] if len(player_team) > 0 else "?"
                
                poss_rows.append({
                    'Player': f"P{pid}",
                    'Team': team,
                    'Touches': stats.get('touch_count', 0),
                    'Time (s)': stats.get('total_time', 0),
                    'Avg Duration (s)': stats.get('avg_possession_duration', 0)
                })
            
            poss_df = pd.DataFrame(poss_rows)
            poss_df = poss_df.sort_values('Touches', ascending=False).head(10)
            
            # Format for display
            poss_df['Time (s)'] = poss_df['Time (s)'].round(1)
            poss_df['Avg Duration (s)'] = poss_df['Avg Duration (s)'].round(2)
            
            st.dataframe(poss_df, use_container_width=True, height=400)
            
            # Top player insight
            if not poss_df.empty:
                top_player = poss_df.iloc[0]
                st.info(f"🌟 **Key Player:** {top_player['Player']} (Team {top_player['Team']}) - {int(top_player['Touches'])} touches, {top_player['Time (s)']:.1f}s possession time")
        else:
            st.warning("No player possession stats available")

# ============================================================
# TAB 5: STATISTICS
# ============================================================
with tab5:
    st.header("📈 Overall Match Statistics")
    
    tracks_df = build_tracks_df(metrics)
    
    if tracks_df.empty:
        st.warning("No tracking data available")
    else:
        # Team comparison
        st.subheader("🏆 Team Performance Comparison")
        
        team_stats = tracks_df.groupby('team').agg({
            'total_distance_m': 'sum',
            'avg_speed_mps': 'mean',
            'max_speed_mps': 'max',
            'workload_score': 'mean',
            'involvement_index': 'mean'
        }).round(2)
        
        st.dataframe(team_stats, use_container_width=True)
        
        # Player rankings
        st.subheader("🥇 Player Rankings")
        
        metric_choice = st.selectbox(
            "Rank by",
            ['total_distance_m', 'avg_speed_mps', 'max_speed_mps', 'workload_score', 'involvement_index']
        )
        
        ranked = tracks_df.sort_values(metric_choice, ascending=False)[
            ['player_id', 'team', metric_choice]
        ].head(10)
        
        st.dataframe(ranked, use_container_width=True)

# ============================================================
# TAB 6: MOVEMENT METRICS
# ============================================================
with tab6:
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
st.markdown("**TactiVision** - Professional Football Analytics | Assistant Manager Dashboard v2.0 (Enhanced)")
