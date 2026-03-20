"""Run experiment pipeline from a YAML configuration.

Usage:
    python scripts/run_experiment.py --config configs/nllb_int8.yaml --steps all
    python scripts/run_experiment.py --config configs/nllb_baseline.yaml --steps baseline
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Map step names to experiment scripts
STEP_MODULES = {
    "baseline": "experiments.nllb_optimization.01_baseline",
    "quantize": "experiments.nllb_optimization.02_quantize",
    "benchmark": "experiments.nllb_optimization.03_benchmark",
    "analysis": "experiments.nllb_optimization.04_analysis",
}

STEP_ORDER = ["baseline", "quantize", "benchmark", "analysis"]


def main(config_path: str, steps: list[str]) -> None:
    """Run experiment steps sequentially.

    Args:
        config_path: Path to the YAML configuration file.
        steps: List of step names to run.
    """
    # Ensure project root is on path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    if "all" in steps:
        steps = STEP_ORDER

    logger.info("Running steps: %s", steps)
    logger.info("Config: %s", config_path)

    for step in steps:
        if step not in STEP_MODULES:
            logger.error("Unknown step: %s. Available: %s", step, list(STEP_MODULES.keys()))
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("Running step: %s", step)
        logger.info("=" * 60)

        module = importlib.import_module(STEP_MODULES[step])
        module.main(config_path)

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
        help=f"Steps to run: {list(STEP_MODULES.keys())} or 'all' (default: all)",
    )
    args = parser.parse_args()
    main(args.config, args.steps)
