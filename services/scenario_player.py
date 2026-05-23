"""
Scenario Player CLI - Command-line interface for running demo scenarios.

Usage:
    python -m services.scenario_player demo/scenarios/prototype_review.yaml
"""

import sys
from pathlib import Path
import logging

from services.demo_runner import run_demo_scenario

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for scenario player CLI."""
    if len(sys.argv) < 2:
        print("Usage: python -m services.scenario_player <scenario.yaml>")
        print("\nExample:")
        print("  python -m services.scenario_player demo/scenarios/prototype_review.yaml")
        sys.exit(1)

    scenario_path = Path(sys.argv[1])

    try:
        run_demo_scenario(scenario_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
