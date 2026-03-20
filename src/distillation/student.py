"""Student model for Knowledge Distillation.

Defines student model architecture and training with KD loss.
Full implementation planned for Phase 3.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class StudentModel:
    """Student model that learns from teacher's soft labels.

    Trains using a combination of KD loss (from teacher) and
    cross-entropy loss (from hard labels).

    Args:
        config: Student model configuration.

    Note: Full implementation planned for Phase 3.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        logger.info("StudentModel initialized (Phase 3 placeholder).")

    def train(self, soft_labels: list[Any], hard_labels: list[Any]) -> dict:
        """Train the student model.

        Args:
            soft_labels: Teacher's soft label outputs.
            hard_labels: Ground truth labels.

        Returns:
            Training metrics dictionary.

        Raises:
            NotImplementedError: Phase 3 feature.
        """
        raise NotImplementedError("StudentModel training is planned for Phase 3.")
