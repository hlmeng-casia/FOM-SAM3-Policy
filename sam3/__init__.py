"""Release-side interface to an external official SAM3 checkout."""

from .interface import FrozenSAM3, LocalizationResult, joint_candidate_scores

__all__ = ["FrozenSAM3", "LocalizationResult", "joint_candidate_scores"]
