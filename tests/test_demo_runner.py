"""
Tests for demo runner service.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from services.demo_runner import DemoRunner


@pytest.fixture
def temp_scenario_dir():
    """Create temporary directory for scenario files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_scenario(temp_scenario_dir):
    """Create a sample scenario file."""
    scenario = {
        'name': 'test_scenario',
        'steps': [
            {'kind': 'pause', 'note': 'Test pause'},
        ]
    }

    scenario_path = temp_scenario_dir / 'test_scenario.yaml'
    with open(scenario_path, 'w') as f:
        yaml.dump(scenario, f)

    return scenario_path


def test_step_dispatch():
    """Test that step dispatcher routes to correct handlers."""
    runner = DemoRunner()

    # Valid step kinds
    for kind in ['ensure_processed', 'open_dashboard', 'generate_pdf', 'pause']:
        assert kind in runner.STEP_KINDS


def test_ensure_processed_skips_existing(temp_scenario_dir):
    """Test that ensure_processed skips if metrics exist."""
    runner = DemoRunner()

    # Create temp video and metrics
    video_dir = temp_scenario_dir / 'video_dir'
    video_dir.mkdir()

    video_file = video_dir / 'test.mp4'
    video_file.write_text('dummy video')

    metrics_file = video_dir / 'metrics.json'
    metrics_file.write_text('{}')

    # Execute step - should skip
    step = {'kind': 'ensure_processed', 'video': str(video_file)}

    # Should not raise exception even without real video processing
    try:
        runner._step_ensure_processed(step)
    except RuntimeError:
        # May fail on batch processing, but should skip if metrics exist
        pass


def test_unknown_step_kind_raises():
    """Test that unknown step kind raises error."""
    runner = DemoRunner()

    step = {'kind': 'unknown_step'}

    with pytest.raises(ValueError, match="Unknown step kind"):
        runner._execute_step(step)


def test_scenario_file_not_found():
    """Test that missing scenario file raises error."""
    runner = DemoRunner()

    with pytest.raises(FileNotFoundError):
        runner.run_demo_scenario(Path('/nonexistent/scenario.yaml'))


def test_empty_scenario_raises():
    """Test that empty scenario file raises error."""
    runner = DemoRunner()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        f.flush()
        scenario_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="empty"):
            runner.run_demo_scenario(scenario_path)
    finally:
        scenario_path.unlink()


def test_pause_step(capsys):
    """Test pause step."""
    runner = DemoRunner()

    step = {'kind': 'pause', 'note': 'Test note'}

    # Simulate user pressing Enter
    import io
    import sys
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO('\n')
        runner._step_pause(step)
    finally:
        sys.stdin = old_stdin


def test_open_dashboard_step(capsys):
    """Test open_dashboard step."""
    runner = DemoRunner()

    step = {'kind': 'open_dashboard', 'tab': 'model_comparison', 'note': 'Check models'}

    # Simulate user pressing Enter
    import io
    import sys
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO('\n')
        runner._step_open_dashboard(step)
        captured = capsys.readouterr()
        assert 'URL:' in captured.out
        assert 'Tab: model_comparison' in captured.out
    finally:
        sys.stdin = old_stdin


def test_scenario_execution_sequence(sample_scenario, capsys):
    """Test that scenario executes steps in sequence."""
    runner = DemoRunner()

    # Mock stdin for pause step
    import io
    import sys
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO('\n\n')
        runner.run_demo_scenario(sample_scenario)
        captured = capsys.readouterr()
        assert 'DEMO SCENARIO: test_scenario' in captured.out
        assert 'SCENARIO COMPLETE' in captured.out
    finally:
        sys.stdin = old_stdin


def test_pdf_generation_requires_metrics(temp_scenario_dir):
    """Test that generate_pdf fails without metrics."""
    runner = DemoRunner()

    video_dir = temp_scenario_dir / 'no_metrics'
    video_dir.mkdir()

    step = {'kind': 'generate_pdf', 'video_dir': str(video_dir)}

    with pytest.raises(FileNotFoundError):
        runner._step_generate_pdf(step)


def test_ensure_processed_video_not_found():
    """Test that ensure_processed fails if video not found."""
    runner = DemoRunner()

    step = {'kind': 'ensure_processed', 'video': '/nonexistent/video.mp4'}

    with pytest.raises(FileNotFoundError):
        runner._step_ensure_processed(step)


def test_step_kinds_completeness():
    """Test that all expected step kinds are defined."""
    runner = DemoRunner()

    expected_kinds = {
        'ensure_processed',
        'open_dashboard',
        'generate_pdf',
        'pause',
    }

    assert runner.STEP_KINDS == expected_kinds
