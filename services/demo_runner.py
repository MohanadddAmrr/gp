"""
Demo Runner - Orchestrates scripted demo scenarios end-to-end.

Reads YAML scenario files and executes steps like processing videos,
generating reports, and guiding user interactions.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from services.batch_processor import BatchProcessor
from services.pdf_report_generator import generate as generate_pdf

logger = logging.getLogger(__name__)


class DemoRunner:
    """Executes demo scenarios step by step."""

    STEP_KINDS = {
        'ensure_processed',
        'open_dashboard',
        'generate_pdf',
        'pause',
    }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize demo runner."""
        self.config = config or {}
        self.batch_processor = BatchProcessor(config)
        self.dashboard_url = self.config.get('dashboard', {}).get('url', 'http://localhost:8501')

    def run_demo_scenario(self, scenario_path: Path) -> None:
        """
        Execute a demo scenario from a YAML file.

        Args:
            scenario_path: Path to YAML scenario file

        Raises:
            ValueError: If scenario is invalid
            FileNotFoundError: If scenario file not found
        """
        scenario_path = Path(scenario_path)

        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        with open(scenario_path) as f:
            scenario = yaml.safe_load(f)

        if not scenario:
            raise ValueError("Scenario file is empty")

        scenario_name = scenario.get('name', 'unnamed')
        steps = scenario.get('steps', [])

        print("\n" + "=" * 70)
        print(f"DEMO SCENARIO: {scenario_name}")
        print("=" * 70)

        for idx, step in enumerate(steps, 1):
            print(f"\n[Step {idx}/{len(steps)}]")
            try:
                self._execute_step(step)
            except Exception as e:
                logger.error(f"Step {idx} failed: {e}")
                raise

        print("\n" + "=" * 70)
        print(f"SCENARIO COMPLETE: {scenario_name}")
        print("=" * 70)

    def _execute_step(self, step: Dict[str, Any]) -> None:
        """Execute a single step."""
        step_kind = step.get('kind')

        if step_kind not in self.STEP_KINDS:
            raise ValueError(f"Unknown step kind: {step_kind}. "
                           f"Expected one of {self.STEP_KINDS}")

        if step_kind == 'ensure_processed':
            self._step_ensure_processed(step)
        elif step_kind == 'open_dashboard':
            self._step_open_dashboard(step)
        elif step_kind == 'generate_pdf':
            self._step_generate_pdf(step)
        elif step_kind == 'pause':
            self._step_pause(step)

    def _step_ensure_processed(self, step: Dict[str, Any]) -> None:
        """
        Ensure video is processed. Skip if metrics.json exists.

        Args:
            step: Step config with 'video' key
        """
        video_path = Path(step.get('video'))

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Determine output directory (same as video)
        output_dir = video_path.parent

        # Check if metrics.json already exists
        metrics_path = output_dir / 'metrics.json'

        if metrics_path.exists():
            print(f"  Metrics already exist: {metrics_path}")
            print(f"  Skipping processing.")
            return

        print(f"  Processing video: {video_path.name}")
        print(f"  Output directory: {output_dir}")

        # Process single video
        try:
            # For a single video, we need to invoke the chunked processor
            # The BatchProcessor works on directories, but we can process a single video
            from services.batch_processor import BatchProcessor
            from main import initialize_pipeline

            # Initialize pipeline components
            config_path = Path('config.yaml')
            if config_path.exists():
                with open(config_path) as f:
                    import yaml
                    config = yaml.safe_load(f)
            else:
                config = {}

            # Create a temporary directory with just this video
            # Actually, just process the directory containing the video
            processor = BatchProcessor(config)

            # For single video, call process_batch on parent directory
            print(f"  Running batch processor on {video_path.parent}...")
            result = processor.process_batch(str(video_path.parent))

            if result.total_failed > 0:
                raise RuntimeError(f"Video processing failed: {result.errors}")

            print(f"  Processing complete. Generated metrics.json")

        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            raise

    def _step_open_dashboard(self, step: Dict[str, Any]) -> None:
        """
        Instruct user to open dashboard tab.

        Args:
            step: Step config with 'tab' key and optional 'note'
        """
        tab = step.get('tab', 'home')
        note = step.get('note', '')

        print(f"  Open dashboard in your browser:")
        print(f"  URL: {self.dashboard_url}")
        print(f"  Tab: {tab}")

        if note:
            print(f"  Note: {note}")

        print(f"\n  Press Enter when ready...")
        input()

    def _step_generate_pdf(self, step: Dict[str, Any]) -> None:
        """
        Generate PDF report from metrics.

        Args:
            step: Step config with 'video_dir' key
        """
        video_dir = Path(step.get('video_dir'))

        if not video_dir.exists():
            raise FileNotFoundError(f"Directory not found: {video_dir}")

        metrics_path = video_dir / 'metrics.json'

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics not found: {metrics_path}")

        print(f"  Generating PDF report...")
        print(f"  Input directory: {video_dir}")

        try:
            pdf_path = generate_pdf(video_dir)
            print(f"  PDF generated: {pdf_path.name}")
            print(f"  Location: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise

    def _step_pause(self, step: Dict[str, Any]) -> None:
        """
        Pause for manual interaction.

        Args:
            step: Step config with optional 'seconds' and 'note'
        """
        note = step.get('note', '')

        if note:
            print(f"  {note}")

        print(f"\n  Press Enter to continue...")
        input()


def run_demo_scenario(scenario_path: Path) -> None:
    """
    Public API to run a demo scenario.

    Args:
        scenario_path: Path to YAML scenario file
    """
    runner = DemoRunner()
    runner.run_demo_scenario(scenario_path)
