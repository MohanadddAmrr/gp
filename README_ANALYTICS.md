# AI Tactical Recommendations & Player Performance Analytics

This document describes the new AI-powered analytics features added to TactiVision Pro.

## Overview

Two major features have been implemented to enhance the tactical analysis capabilities:

1. **AI Tactical Recommendations** - Real-time intelligent suggestions based on video data
2. **Player Performance Analytics** - Individual player tracking and comprehensive performance analysis

## Features

### 1. AI Tactical Recommendations (`services/ai_tactical_recommendations.py`)

An intelligent engine that analyzes match data in real-time to provide actionable tactical recommendations.

#### Key Capabilities:
- **Formation Recommendations**: Suggests formation changes based on match state and opponent setup
- **Pressing Strategy**: Analyzes PPDA (Passes Per Defensive Action) to recommend pressing intensity
- **Opponent Weakness Exploitation**: Identifies and suggests how to exploit opponent vulnerabilities
- **Substitution Suggestions**: Recommends substitutions based on player fatigue and performance
- **Game State Awareness**: Context-aware recommendations considering score, time, and momentum

#### Recommendation Categories:
- `FORMATION` - Formation adjustments
- `PRESSING` - Pressing strategy changes
- `ATTACKING` - Attacking approach modifications
- `DEFENSIVE` - Defensive organization improvements
- `TRANSITIONS` - Transition play adjustments
- `SET_PIECES` - Set piece strategies
- `PLAYER_POSITIONING` - Individual positioning guidance
- `SUBSTITUTION` - Player substitution recommendations
- `OPPONENT_EXPLOIT` - Exploiting opponent weaknesses

#### Priority Levels:
- `CRITICAL` (5) - Immediate action required
- `HIGH` (4) - Strong recommendation
- `MEDIUM` (3) - Consider implementing
- `LOW` (2) - Optional adjustment
- `INFO` (1) - Informational only

#### Usage Example:
```python
from services.ai_tactical_recommendations import AITacticalRecommendations

# Initialize
ai_rec = AITacticalRecommendations()

# Update match context
ai_rec.update_match_context(
    score_a=1, score_b=0,
    possession_a=55.0, possession_b=45.0,
    match_minute=35.5
)

# Update player performance data
ai_rec.update_player_performance(
    player_id=10,
    team='A',
    fatigue_score=75.0,
    form_rating=65.0
)

# Generate recommendations
recommendations = ai_rec.generate_all_recommendations(
    current_formations={'A': '4-3-3', 'B': '4-2-3-1'}
)

# Get top recommendations
top_recs = ai_rec.get_top_recommendations(n=5, min_confidence=0.7)
for rec in top_recs:
    print(f"{rec.priority.name}: {rec.title}")
    print(f"  {rec.description}")
    print(f"  Confidence: {rec.confidence_score:.0%}")
```

### 2. Player Performance Analytics (`services/player_performance_analytics.py`)

Comprehensive player performance tracking and analysis system.

#### Key Capabilities:
- **Physical Metrics**: Distance covered, speed, sprints, workload, fatigue
- **Technical Metrics**: Passes, shots, tackles, interceptions, ball control
- **Tactical Metrics**: Positioning, pressing, zone time, defensive actions
- **Performance Ratings**: Overall and component ratings (0-10 scale)
- **Form Tracking**: Historical performance trends
- **Comparative Analysis**: Player-to-player comparisons

#### Physical Metrics Tracked:
- Total distance covered (meters)
- Sprint distance (meters)
- High-intensity distance (meters)
- Maximum speed (m/s)
- Average speed (m/s)
- Number of sprints
- Workload score (0-100)
- Fatigue index (0-100)

#### Technical Metrics Tracked:
- Passes attempted/completed
- Pass accuracy (%)
- Shots/shots on target
- Goals scored
- Tackles/interceptions
- Dribbles attempted/successful
- Ball touches

#### Tactical Metrics Tracked:
- Average position (x, y)
- Time in each third (defensive/middle/attacking)
- Pressing actions and success rate
- Defensive recoveries
- Runs in behind/in box

#### Usage Example:
```python
from services.player_performance_analytics import PlayerPerformanceAnalytics

# Initialize
analytics = PlayerPerformanceAnalytics()

# Register player
analytics.register_player(
    player_id=10,
    team='A',
    name="Mohamed Salah",
    jersey_number=11,
    position="FWD"
)

# Update during video processing
analytics.update_position(player_id=10, x=0.8, y=0.5, timestamp=35.5)
analytics.update_speed(player_id=10, speed_mps=8.5, timestamp=35.5)
analytics.record_event(player_id=10, event_type='pass', success=True)
analytics.record_event(player_id=10, event_type='shot', success=True)

# Get player summary
summary = analytics.get_player_summary(10)
print(f"Overall Rating: {summary['rating']['overall']}/10")
print(f"Distance: {summary['physical']['total_distance_m']:.1f}m")
print(f"Pass Accuracy: {summary['technical']['pass_accuracy']:.1f}%")

# Get team summary
team_summary = analytics.get_team_summary('A')
print(f"Team Average Rating: {team_summary['average_rating']:.1f}")

# Compare players
comparison = analytics.compare_players([10, 11, 9])
```

### 3. Match Analytics Integration (`services/match_analytics_integration.py`)

Unified interface that combines both analytics modules for easy integration with the video processing pipeline.

#### Features:
- Single initialization point for all analytics
- Automatic data synchronization between modules
- Periodic recommendation generation
- Export and database integration
- Live dashboard data formatting

#### Usage Example:
```python
from services.match_analytics_integration import (
    MatchAnalyticsIntegration,
    AnalyticsConfig
)

# Configure
config = AnalyticsConfig(
    enable_ai_recommendations=True,
    enable_player_performance=True,
    recommendation_interval=30,  # seconds
    min_confidence_threshold=0.6
)

# Initialize
analytics = MatchAnalyticsIntegration(config=config)

# Initialize match
analytics.initialize_match(
    match_id="match_001",
    team_a_name="Liverpool",
    team_b_name="Man City",
    players_a=[{'id': 1, 'name': 'Alisson', 'number': 1, 'position': 'GK'}, ...],
    players_b=[...]
)

# During video processing loop
for frame_idx, frame in enumerate(video_stream):
    timestamp = frame_idx / fps
    
    # Update player positions from tracking
    for player_id, (x, y) in player_positions.items():
        analytics.update_player_position(player_id, x, y, timestamp)
    
    # Update speeds
    for player_id, speed in player_speeds.items():
        analytics.update_player_speed(player_id, speed, timestamp)
    
    # Record events
    if pass_detected:
        analytics.record_player_event(
            passer_id, 'pass',
            success=(receiver_team == passer_team)
        )
    
    # Update match context periodically
    if frame_idx % 300 == 0:  # Every 10 seconds at 30fps
        analytics.update_match_context(
            score_a=current_score_a,
            score_b=current_score_b,
            possession_a=possession_pct_a,
            possession_b=possession_pct_b,
            match_minute=timestamp / 60
        )

# Get recommendations
recommendations = analytics.get_current_recommendations(n=3)

# Get player performance
player_data = analytics.get_player_summary(player_id=10)

# Finalize match
report = analytics.finalize_match()
```

## Demo Script

A demo script is provided to showcase the features:

```bash
# Run the demo
python demo/demo_analytics.py

# Run with specific mode
python demo/demo_analytics.py --mode simulation
python demo/demo_analytics.py --mode integration
python demo/demo_analytics.py --mode both
```

The demo simulates a match between Liverpool and Manchester City, generating:
- Real-time tactical recommendations
- Player performance summaries
- Team statistics
- Comprehensive match report

## Testing

Comprehensive test suites are provided:

```bash
# Run AI tactical recommendations tests
python -m pytest tests/test_ai_tactical_recommendations.py -v

# Run player performance analytics tests
python -m pytest tests/test_player_performance_analytics.py -v

# Run all tests
python -m pytest tests/ -v
```

## Integration with Existing Pipeline

To integrate with the existing `demo/run_demo.py` video processing pipeline:

1. **Import the integration module:**
```python
from services.match_analytics_integration import (
    MatchAnalyticsIntegration,
    AnalyticsConfig
)
```

2. **Initialize in `process_video()`:**
```python
# Initialize analytics
analytics_config = AnalyticsConfig(
    enable_ai_recommendations=True,
    enable_player_performance=True
)
match_analytics = MatchAnalyticsIntegration(
    config=analytics_config,
    db_manager=db_manager
)

# Initialize match
match_analytics.initialize_match(
    match_id=str(match_id),
    team_a_name=team_a,
    team_b_name=team_b
)
```

3. **Update during frame processing:**
```python
# In the main processing loop
for res in results:
    frame_idx += 1
    t = frame_idx / fps
    
    # ... existing processing ...
    
    # Update analytics
    for pid, (cx, cy, team) in frame_player_positions.items():
        match_analytics.update_player_position(pid, cx/frame_width, cy/frame_height, t)
        match_analytics.update_player_speed(pid, smooth_speed, t)
    
    # Record events
    if pass_event:
        match_analytics.record_player_event(
            pass_event['passer_id'], 'pass',
            success=(pass_event['outcome'] == 'complete')
        )
```

4. **Generate reports at end:**
```python
# Finalize analytics
analytics_report = match_analytics.finalize_match()
```

## Output Files

Analytics reports are exported to `analytics_output/` directory:
- JSON format with full match data
- Player performance summaries
- Tactical recommendations history
- Team statistics

## Configuration

Configure behavior via `AnalyticsConfig`:

```python
config = AnalyticsConfig(
    enable_ai_recommendations=True,      # Enable/disable AI recommendations
    enable_player_performance=True,      # Enable/disable player tracking
    enable_tactical_analysis=True,       # Enable/disable tactical analysis
    recommendation_interval=30,          # Seconds between recommendation updates
    min_confidence_threshold=0.6,        # Minimum confidence for recommendations
    max_recommendations=10,              # Max recommendations to store
    track_all_players=True,              # Track all players or specific ones
    update_frequency=10,                 # Frames between updates
    auto_export=True,                    # Auto-export reports
    export_format="json",                # Export format
    export_path="analytics_output"       # Export directory
)
```

## Future Enhancements

Potential future improvements:
- Machine learning models for pattern recognition
- Predictive analytics for match outcomes
- Integration with wearable device data
- Advanced opponent modeling
- Automated highlight generation based on analytics
- Real-time dashboard with WebSocket updates

## Support

For questions or issues:
- Check the test files for usage examples
- Review the demo script for integration patterns
- Examine the docstrings for detailed API documentation
