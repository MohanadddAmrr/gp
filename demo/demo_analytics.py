"""
Demo script for AI Tactical Recommendations and Player Performance Analytics

This script demonstrates the new analytics features:
1. AI Tactical Recommendations - Real-time tactical suggestions based on video data
2. Player Performance Analytics - Individual player tracking and analysis

Usage:
    python demo/demo_analytics.py --video input_videos/match.mp4
    python demo/demo_analytics.py --video input_videos/match.mp4 --team-a "Liverpool" --team-b "Man City"
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.match_analytics_integration import (
    MatchAnalyticsIntegration,
    AnalyticsConfig
)
from services.ai_tactical_recommendations import (
    AITacticalRecommendations,
    RecommendationPriority
)
from services.player_performance_analytics import PlayerPerformanceAnalytics


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_recommendation(rec, index: int = 1):
    """Print a recommendation in formatted way."""
    priority_colors = {
        RecommendationPriority.CRITICAL: "🔴",
        RecommendationPriority.HIGH: "🟠",
        RecommendationPriority.MEDIUM: "🟡",
        RecommendationPriority.LOW: "🟢",
        RecommendationPriority.INFO: "⚪"
    }
    
    icon = priority_colors.get(rec.priority, "⚪")
    
    print(f"{icon} Recommendation #{index}: {rec.title}")
    print(f"   Priority: {rec.priority.name} | Category: {rec.category.value}")
    print(f"   Confidence: {rec.confidence_score:.0%}")
    print(f"   Description: {rec.description}")
    print(f"   Reasoning: {rec.reasoning}")
    print(f"   Expected Outcome: {rec.expected_outcome}")
    
    if rec.implementation_steps:
        print("   Implementation Steps:")
        for i, step in enumerate(rec.implementation_steps, 1):
            print(f"      {i}. {step}")
    print()


def print_player_summary(summary: dict, index: int = 1):
    """Print player performance summary."""
    info = summary.get('player_info', {})
    physical = summary.get('physical', {})
    technical = summary.get('technical', {})
    tactical = summary.get('tactical', {})
    rating = summary.get('rating', {})
    
    print(f"👤 Player #{index}: {info.get('name', 'Unknown')} (#{info.get('jersey_number', 0)})")
    print(f"   Position: {info.get('position', 'Unknown')} | Team: {info.get('team', 'Unknown')}")
    print()
    print(f"   📊 Overall Rating: {rating.get('overall', 0)}/10")
    print(f"      Physical: {rating.get('physical', 0)}/10 | Technical: {rating.get('technical', 0)}/10")
    print(f"      Tactical: {rating.get('tactical', 0)}/10")
    print()
    print(f"   🏃 Physical Metrics:")
    print(f"      Distance: {physical.get('total_distance_m', 0):.1f}m")
    print(f"      Sprint Distance: {physical.get('sprint_distance_m', 0):.1f}m")
    print(f"      Max Speed: {physical.get('max_speed_mps', 0):.2f} m/s")
    print(f"      Sprints: {physical.get('sprints', 0)}")
    print(f"      Workload: {physical.get('workload_score', 0):.1f}/100")
    print()
    print(f"   ⚽ Technical Metrics:")
    print(f"      Passes: {technical.get('passes', '0/0')} ({technical.get('pass_accuracy', 0):.1f}%)")
    print(f"      Shots: {technical.get('shots', 0)} (On target: {technical.get('shots_on_target', 0)})")
    print(f"      Goals: {technical.get('goals', 0)}")
    print(f"      Tackles: {technical.get('tackles', 0)} | Interceptions: {technical.get('interceptions', 0)}")
    print()
    print(f"   🎯 Tactical Metrics:")
    print(f"      Avg Position: {tactical.get('avg_position', (0, 0))}")
    print(f"      Time in Attacking Third: {tactical.get('time_in_attacking_third', 0):.1f}s")
    print()


def run_simulation_demo():
    """Run a simulation demo with synthetic data."""
    print_header("TactiVision Pro - AI Analytics Demo")
    print("Running simulation with synthetic match data...\n")
    
    # Initialize analytics
    config = AnalyticsConfig(
        enable_ai_recommendations=True,
        enable_player_performance=True,
        recommendation_interval=30
    )
    
    analytics = MatchAnalyticsIntegration(config=config)
    
    # Initialize match
    match_id = f"demo_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    players_a = [
        {'id': 1, 'name': 'Alisson', 'number': 1, 'position': 'GK'},
        {'id': 2, 'name': 'Alexander-Arnold', 'number': 66, 'position': 'DEF'},
        {'id': 3, 'name': 'Van Dijk', 'number': 4, 'position': 'DEF'},
        {'id': 4, 'name': 'Konate', 'number': 5, 'position': 'DEF'},
        {'id': 5, 'name': 'Robertson', 'number': 26, 'position': 'DEF'},
        {'id': 6, 'name': 'Fabinho', 'number': 3, 'position': 'MID'},
        {'id': 7, 'name': 'Henderson', 'number': 14, 'position': 'MID'},
        {'id': 8, 'name': 'Thiago', 'number': 6, 'position': 'MID'},
        {'id': 9, 'name': 'Salah', 'number': 11, 'position': 'FWD'},
        {'id': 10, 'name': 'Nunez', 'number': 9, 'position': 'FWD'},
        {'id': 11, 'name': 'Diaz', 'number': 23, 'position': 'FWD'},
    ]
    
    players_b = [
        {'id': 12, 'name': 'Ederson', 'number': 31, 'position': 'GK'},
        {'id': 13, 'name': 'Walker', 'number': 2, 'position': 'DEF'},
        {'id': 14, 'name': 'Dias', 'number': 3, 'position': 'DEF'},
        {'id': 15, 'name': 'Ake', 'number': 6, 'position': 'DEF'},
        {'id': 16, 'name': 'Cancelo', 'number': 7, 'position': 'DEF'},
        {'id': 17, 'name': 'Rodri', 'number': 16, 'position': 'MID'},
        {'id': 18, 'name': 'De Bruyne', 'number': 17, 'position': 'MID'},
        {'id': 19, 'name': 'Silva', 'number': 20, 'position': 'MID'},
        {'id': 20, 'name': 'Mahrez', 'number': 26, 'position': 'FWD'},
        {'id': 21, 'name': 'Haaland', 'number': 9, 'position': 'FWD'},
        {'id': 22, 'name': 'Foden', 'number': 47, 'position': 'FWD'},
    ]
    
    analytics.initialize_match(
        match_id=match_id,
        team_a_name="Liverpool",
        team_b_name="Manchester City",
        players_a=players_a,
        players_b=players_b
    )
    
    print("✅ Match initialized with 22 players\n")
    
    # Simulate match events
    print_header("Simulating Match Events")
    
    # Simulate some player movements and events
    import random
    
    # Simulate first 15 minutes
    for minute in range(0, 16, 5):
        print(f"⏱️  Minute {minute}:")
        
        # Update match context
        score_a = 0 if minute < 10 else 1
        score_b = 0
        possession_a = 55.0
        possession_b = 45.0
        
        analytics.update_match_context(
            score_a=score_a,
            score_b=score_b,
            possession_a=possession_a,
            possession_b=possession_b,
            match_minute=minute,
            period="first_half"
        )
        
        # Simulate player positions and metrics
        for player_id in range(1, 23):
            # Simulate distance covered
            distance = random.uniform(1000, 3000) + (minute * 100)
            speed = random.uniform(2.0, 8.5)
            
            # Update performance data
            analytics.update_player_performance_data(
                player_id=player_id,
                distance_covered=distance,
                current_speed=speed,
                pass_accuracy=random.uniform(70, 95),
                touches=random.randint(10, 50),
                fatigue_score=random.uniform(20, 60),
                form_rating=random.uniform(60, 90)
            )
            
            # Record some events
            if random.random() > 0.7:
                event_type = random.choice(['pass', 'touch', 'tackle'])
                analytics.record_player_event(
                    player_id=player_id,
                    event_type=event_type,
                    success=random.random() > 0.3
                )
        
        # Generate recommendations at minute 15
        if minute == 15:
            print("\n🤖 Generating tactical recommendations...\n")
            
            # Force recommendation generation
            analytics._generate_recommendations()
            
            recommendations = analytics.get_current_recommendations(n=5)
            
            print_header("AI Tactical Recommendations")
            for i, rec in enumerate(recommendations, 1):
                print_recommendation(rec, i)
    
    # Print player performance summaries
    print_header("Player Performance Summaries (Top 5)")
    
    # Get top performers from team A
    team_a_summary = analytics.get_team_summary('A')
    top_performers = team_a_summary.get('top_performers', [])
    
    player_count = 0
    for performer in top_performers[:3]:
        player_id = performer.get('player_id')
        if player_id:
            summary = analytics.get_player_summary(player_id)
            if summary:
                player_count += 1
                print_player_summary(summary, player_count)
    
    # Print team summary
    print_header("Team Performance Summary")
    
    for team in ['A', 'B']:
        summary = analytics.get_team_summary(team)
        team_name = "Liverpool" if team == 'A' else "Manchester City"
        
        print(f"📊 {team_name} (Team {team}):")
        print(f"   Players Tracked: {summary.get('players_tracked', 0)}")
        print(f"   Total Distance: {summary.get('aggregate_stats', {}).get('total_distance_m', 0):.1f}m")
        print(f"   Total Passes: {summary.get('aggregate_stats', {}).get('total_passes', 0)}")
        print(f"   Total Shots: {summary.get('aggregate_stats', {}).get('total_shots', 0)}")
        print(f"   Average Rating: {summary.get('average_rating', 0):.1f}/10")
        print()
    
    # Finalize and export
    print_header("Finalizing Match Analytics")
    
    report = analytics.finalize_match()
    
    print(f"✅ Match report generated")
    print(f"   Total recommendations: {report.get('tactical_recommendations', {}).get('total_generated', 0)}")
    print(f"   Players analyzed: {len(report.get('player_performance', {}).get('players', {}))}")
    
    # Export to file
    output_file = Path("demo_outputs") / f"analytics_demo_{match_id}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Report exported to: {output_file}")
    
    print_header("Demo Complete")
    print("The AI Tactical Recommendations and Player Performance Analytics")
    print("modules are now ready for integration with the video processing pipeline.")
    print()
    print("Key Features Demonstrated:")
    print("  • Real-time tactical recommendation generation")
    print("  • Player performance tracking and rating")
    print("  • Team and individual player summaries")
    print("  • Comprehensive match analytics export")
    print()


def run_integration_example():
    """Show how to integrate with existing video processing."""
    print_header("Integration Example")
    
    print("""
To integrate these new features into the existing video processing pipeline:

1. Import the analytics module:
   from services.match_analytics_integration import MatchAnalyticsIntegration, AnalyticsConfig

2. Initialize analytics in your video processing script:
   config = AnalyticsConfig(
       enable_ai_recommendations=True,
       enable_player_performance=True
   )
   analytics = MatchAnalyticsIntegration(config=config)

3. Initialize match with player rosters:
   analytics.initialize_match(
       match_id="match_001",
       team_a_name="Liverpool",
       team_b_name="Man City",
       players_a=[{'id': 1, 'name': 'Player 1', 'number': 10, 'position': 'FWD'}, ...],
       players_b=[...]
   )

4. During frame processing, update player data:
   analytics.update_player_position(player_id, x, y, timestamp, team)
   analytics.update_player_speed(player_id, speed_mps, timestamp)
   analytics.record_player_event(player_id, 'pass', success=True)

5. Update match context periodically:
   analytics.update_match_context(
       score_a=1, score_b=0,
       possession_a=55.0, possession_b=45.0,
       match_minute=35.5
   )

6. Get real-time recommendations:
   recommendations = analytics.get_current_recommendations(n=3)
   for rec in recommendations:
       print(f"{rec.priority.name}: {rec.title} - {rec.description}")

7. Get player performance data:
   player_summary = analytics.get_player_summary(player_id)
   print(f"Rating: {player_summary['rating']['overall']}/10")

8. At end of match, generate report:
   report = analytics.finalize_match()
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demo for AI Tactical Recommendations and Player Performance Analytics"
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "integration", "both"],
        default="both",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    if args.mode in ["simulation", "both"]:
        run_simulation_demo()
    
    if args.mode in ["integration", "both"]:
        run_integration_example()


if __name__ == "__main__":
    main()
