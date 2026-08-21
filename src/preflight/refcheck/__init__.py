"""A fast, type-aware reference checker.

Built for wall-clock speed and for not manufacturing accusations: every source
runs concurrently, non-paper citations are verified against what they actually
are, and live web search clears the real-but-unindexed citations that a
database-only checker reports as missing.
"""

from .core import Kind, MatchPolicy, Reference, Status, Verdict
from .engine import RefCheckConfig, RefCheckReport, verify
from .parse import bibliography_lines, parse_entries, segment

__all__ = [
    "Kind",
    "MatchPolicy",
    "RefCheckConfig",
    "RefCheckReport",
    "Reference",
    "Status",
    "Verdict",
    "bibliography_lines",
    "parse_entries",
    "segment",
    "verify",
]
