"""
Golden Thread protocol implementation.

The Golden Thread provides cryptographic verification of approval chains,
ensuring that AI assets can be traced back to their approval decisions.
"""

from aigrc.golden_thread.hash import (
    compute_golden_thread_hash,
    compute_canonical_string,
)
from aigrc.golden_thread.verify import verify_golden_thread_hash

__all__ = [
    "compute_golden_thread_hash",
    "compute_canonical_string",
    "verify_golden_thread_hash",
]
