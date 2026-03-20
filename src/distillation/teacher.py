"""Teacher model wrapper for Knowledge Distillation.

Handles teacher model inference, soft label generation, and output caching.
Full implementation planned for Phase 3.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TeacherModel:
    """Wrapper around a teacher model for distillation.

    Supports local models and API-based models (GPT-4, Claude, etc.).
    Implements output caching to prevent duplicate inference/API calls.

    Args:
        config: Teacher model configuration.

    Note: Full implementation planned for Phase 3.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._cache: dict[str, Any] = {}
        logger.info("TeacherModel initialized (Phase 3 placeholder).")

    def generate_soft_labels(self, inputs: list[str]) -> list[Any]:
        """Generate soft labels (logits/probabilities) for given inputs.

        Args:
            inputs: List of source sentences.

        Returns:
            List of soft label tensors.

        Raises:
            NotImplementedError: Phase 3 feature.
        """
        raise NotImplementedError("TeacherModel is planned for Phase 3.")
