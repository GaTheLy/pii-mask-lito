"""pii_mask_lito -- format-preserving PII/PHI masking for documents."""

from .detect import Detector
from .pipeline import MaskingError, Report, mask
from .policy import GENERAL_PII, PROFILES, SAFE_HARBOR
from .registry import TagRegistry

__version__ = "0.1.0"
__all__ = [
    "Detector",
    "GENERAL_PII",
    "MaskingError",
    "PROFILES",
    "Report",
    "SAFE_HARBOR",
    "TagRegistry",
    "mask",
]
