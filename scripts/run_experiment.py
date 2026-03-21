"""Run experiment pipeline from a YAML configuration.

Usage:
    python scripts/run_experiment.py --config configs/nllb_int8.yaml --steps all
    python scripts/run_experiment.py --config configs/nllb_baseline.yaml --steps baseline
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Map step names to experiment script paths (relative to project root)
STEP_SCRIPTS = {
    "baseline": "experiments/nllb_optimization/01_baseline.py",
    "quantize": "experiments/nllb_optimization/02_quantize.py",
    "benchmark": "experiments/nllb_optimization/03_benchmark.py",
    "analysis": "experiments/nllb_optimization/04_analysis.py",
}

STEP_ORDER = ["baseline", "quantize", "benchmark", "analysis"]


def main(config_path: str, steps: list[str]) -> None:
    """Run experiment steps sequentially.

    Args:
        config_path: Path to the YAML configuration file.
        steps: List of step names to run.
    """
    project_root = Path(__file__).resolve().parents[1]

    if "all" in steps:
        steps = STEP_ORDER

    logger.info("Running steps: %s", steps)
    logger.info("Config: %s", config_path)

    for step in steps:
        if step not in STEP_SCRIPTS:
            logger.error("Unknown step: %s. Available: %s", step, list(STEP_SCRIPTS.keys()))
            sys.exit(1)

        script_path = project_root / STEP_SCRIPTS[step]
        if not script_path.exists():
            logger.error("Script not found: %s", script_path)
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("Running step: %s", step)
        logger.info("=" * 60)

        result = subprocess.run(
            [sys.executable, str(script_path), "--config", config_path],
            cwd=str(project_root),
        )

        if result.returncode != 0:
            logger.error("Step '%s' failed with exit code %d", step, result.returncode)
            sys.exit(result.returncode)

        logger.info("Step '%s' complete.", step)

    logger.info("All steps complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ALM experiment pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help=f"Steps to run: {list(STEP_SCRIPTS.keys())} or 'all' (default: all)",
    )
    args = parser.parse_args()
    main(args.config, args.steps)
