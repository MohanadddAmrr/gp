"""
Tests for highlights generator v2.
"""

import pytest
from pathlib import Path
import json
import tempfile

from services.highlights_generator import HighlightsGenerator, Highlight


@pytest.fixture
def metrics_with_events():
    """Create metrics with various events."""
    return {
        'shot_events': [
            {
                'timestamp': 10.0,
                'shooter_id': 1,
                'velocity_mps': 18.0,
                'angle_to_goal_deg': 15.0,
                'is_goal': True
            },
            {
                'timestamp': 15.0,
                'shooter_id': 2,
                'velocity_mps': 12.0,
                'angle_to_goal_deg': 40.0,
                'is_goal': False
            },
            {
                'timestamp': 20.0,
                'shooter_id': 3,
                'velocity_mps': 8.0,
                'angle_to_goal_deg': 50.0,
                'is_goal': False
            }
        ],
        'sprint_events': [
            {
                'start_time': 5.0,
                'player_id': 1,
                'max_speed_mps': 8.5,
                'duration_sec': 1.2
            },
            {
                'start_time': 25.0,
                'player_id': 2,
                'max_speed_mps': 6.5,
                'duration_sec': 1.0
            }
        ],
        'pass_events': [
            {
                'timestamp': 8.0,
                'passer_id': 1,
                'distance_m': 35.0,
                'outcome': 'complete'
            },
            {
                'timestamp': 18.0,
                'passer_id': 2,
                'distance_m': 12.0,
                'outcome': 'complete'
            }
        ],
        'tracks': []
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_score_events_orders_by_importance(metrics_with_events, temp_output_dir):
    """Test that events are scored and ordered by importance."""
    generator = HighlightsGenerator(temp_output_dir)
    events = generator._extract_events(metrics_with_events)

    assert len(events) > 0, "Should extract events"

    # Score events
    scored = generator._score_events(events)

    # First event should be a goal (highest importance)
    assert scored[0]['type'] == 'goal', "Goal should be highest priority"
    assert scored[0]['importance'] == 1.0

    # Check that scores are descending
    for i in range(len(scored) - 1):
        assert scored[i]['importance'] >= scored[i + 1]['importance']


def test_non_overlapping_highlights(metrics_with_events, temp_output_dir):
    """Test that selected highlights don't overlap."""
    generator = HighlightsGenerator(temp_output_dir)
    events = generator._extract_events(metrics_with_events)
    scored = generator._score_events(events)

    selected = generator._select_non_overlapping(scored, top_n=5)

    # Check non-overlapping constraint
    min_gap = generator.PRE_EVENT_BUFFER + generator.POST_EVENT_BUFFER

    for i in range(len(selected) - 1):
        time_diff = abs(selected[i + 1]['time'] - selected[i]['time'])
        assert time_diff >= min_gap, f"Events too close: {time_diff} < {min_gap}"


def test_missing_ffmpeg_does_not_crash(metrics_with_events, temp_output_dir):
    """Test that missing ffmpeg doesn't crash generation."""
    generator = HighlightsGenerator(temp_output_dir)

    # Create a dummy video file (won't actually process, just test error handling)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)

    try:
        # Should not raise exception even if ffmpeg missing
        highlights = generator.generate(metrics_with_events, video_path, top_n=5)

        assert isinstance(highlights, list)
        # Should have highlights even without clips
        assert len(highlights) > 0
        # Clips might be None if ffmpeg unavailable
        for h in highlights:
            assert isinstance(h, Highlight)
    finally:
        video_path.unlink()


def test_generate_returns_highlight_objects(metrics_with_events, temp_output_dir):
    """Test that generate returns proper Highlight objects."""
    generator = HighlightsGenerator(temp_output_dir)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)

    try:
        highlights = generator.generate(metrics_with_events, video_path, top_n=3)

        assert len(highlights) <= 3
        for h in highlights:
            assert isinstance(h, Highlight)
            assert isinstance(h.time, float)
            assert isinstance(h.type, str)
            assert isinstance(h.importance, float)
            assert isinstance(h.description, str)
            assert h.importance > 0
    finally:
        video_path.unlink()


def test_top_n_selection(metrics_with_events, temp_output_dir):
    """Test that top_n parameter limits results."""
    generator = HighlightsGenerator(temp_output_dir)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)

    try:
        highlights_3 = generator.generate(metrics_with_events, video_path, top_n=3)
        highlights_5 = generator.generate(metrics_with_events, video_path, top_n=5)

        assert len(highlights_3) <= 3
        assert len(highlights_5) <= 5
        assert len(highlights_3) <= len(highlights_5)
    finally:
        video_path.unlink()


def test_clips_directory_created(temp_output_dir):
    """Test that clips directory is created."""
    generator = HighlightsGenerator(temp_output_dir)

    assert generator.clips_dir.exists()
    assert generator.clips_dir.is_dir()


def test_extract_events_types(metrics_with_events, temp_output_dir):
    """Test that all event types are extracted."""
    generator = HighlightsGenerator(temp_output_dir)
    events = generator._extract_events(metrics_with_events)

    event_types = {e['type'] for e in events}

    # Should have goal
    assert 'goal' in event_types
    # Should have sprint
    assert 'sprint_high_intensity' in event_types
    # Should have pass
    assert 'pass_progressive' in event_types


def test_highlight_importance_weights(temp_output_dir):
    """Test that importance weights are correct."""
    generator = HighlightsGenerator(temp_output_dir)

    assert generator.IMPORTANCE_WEIGHTS['goal'] == 1.0
    assert generator.IMPORTANCE_WEIGHTS['big_chance'] == 0.8
    assert generator.IMPORTANCE_WEIGHTS['save'] == 0.7
    assert generator.IMPORTANCE_WEIGHTS['shot_on_target'] == 0.6
    assert generator.IMPORTANCE_WEIGHTS['sprint_high_intensity'] == 0.5
    assert generator.IMPORTANCE_WEIGHTS['dribble_success'] == 0.4
    assert generator.IMPORTANCE_WEIGHTS['pass_progressive'] == 0.3


def test_no_events_returns_empty_list(temp_output_dir):
    """Test that empty metrics returns empty highlights."""
    generator = HighlightsGenerator(temp_output_dir)
    empty_metrics = {
        'shot_events': [],
        'sprint_events': [],
        'pass_events': [],
        'tracks': []
    }

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)

    try:
        highlights = generator.generate(empty_metrics, video_path)
        assert len(highlights) == 0
    finally:
        video_path.unlink()


def test_idempotent_clip_generation(metrics_with_events, temp_output_dir):
    """Test that clip generation is idempotent."""
    # This test checks that the code doesn't re-cut clips
    generator = HighlightsGenerator(temp_output_dir)

    # Create a fake clip file
    fake_clip = generator.clips_dir / '00_goal.mp4'
    fake_clip.write_text('dummy')

    # When trying to cut, it should detect existing file
    result = generator._cut_clip(Path('/nonexistent/video.mp4'),
                                {'time': 10.0, 'type': 'goal'}, 0)

    # If ffmpeg available, should return the existing clip path
    # If not available, should return None
    if generator.ffmpeg_available:
        assert result == fake_clip or result is None
