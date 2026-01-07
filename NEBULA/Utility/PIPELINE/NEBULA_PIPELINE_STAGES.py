"""
NEBULA_PIPELINE_STAGES
======================

Defines the canonical NEBULA upstream pipeline stages (through WINDOW_SOURCES) and
small utilities for parsing/validating stage identifiers.

This module is intentionally:
- standard-library only,
- free of project imports (so it can be imported early and safely),
- the single source of truth for stage ordering and stage-ID normalization.

Design notes
------------
We use `class Stage(str, Enum)` (instead of Python 3.11+ `StrEnum`) so this
remains compatible with Python 3.9 while still giving string-like stage values.
Mix-in types (here: `str`) must appear before `Enum` in the base class list. 
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple, Union


class Stage(str, Enum):
    """
    Canonical pipeline stages (strict chain) for NEBULA upstream through WINDOW_SOURCES.

    The enum values are the canonical stage IDs used by:
      - RECOMPUTE_FROM_STAGE
      - STOP_AFTER_STAGE
      - dry-run plan output

    NOTE: Do not reorder these members; stage ordering is defined separately
    via STAGE_ORDER to keep the enum stable even if future stages are inserted.
    """

    BASE = "BASE"
    LOS = "LOS"
    ILLUM = "ILLUM"
    FLUX = "FLUX"
    LOS_FLUX = "LOS_FLUX"
    POINTING = "POINTING"
    ICRS = "ICRS"
    PIXELS = "PIXELS"
    TARGET_FRAMES = "TARGET_FRAMES"
    TARGET_PHOTONS = "TARGET_PHOTONS"
    TRACKING_MODE = "TRACKING_MODE"           
    SKY = "SKY"
    GAIA = "GAIA"
    STAR_PROJECTION = "STAR_PROJECTION"
    STAR_PHOTONS = "STAR_PHOTONS"    
    WINDOW_SOURCES = "WINDOW_SOURCES"
    ZODIACAL_LIGHT = "ZODIACAL_LIGHT"

    def __str__(self) -> str:
        # Keep the printed form aligned with canonical stage IDs.
        return self.value


# -----------------------------------------------------------------------------
# Canonical ordering (strict chain) through STAR_PHOTONS
# -----------------------------------------------------------------------------
STAGE_ORDER: Tuple[Stage, ...] = (
    Stage.BASE,
    Stage.LOS,
    Stage.ILLUM,
    Stage.FLUX,
    Stage.LOS_FLUX,
    Stage.POINTING,
    Stage.ICRS,
    Stage.PIXELS,
    Stage.TARGET_FRAMES,
    Stage.TARGET_PHOTONS,
    Stage.TRACKING_MODE,
    Stage.SKY,
    Stage.GAIA,
    Stage.STAR_PROJECTION,
    Stage.STAR_PHOTONS,
    Stage.WINDOW_SOURCES,
    Stage.ZODIACAL_LIGHT,
)

# Convenience maps for ordering and lookups.
STAGE_INDEX: Dict[Stage, int] = {s: i for i, s in enumerate(STAGE_ORDER)}
STAGE_BY_ID: Dict[str, Stage] = {s.value: s for s in STAGE_ORDER}

# Aliases accepted by parsing utilities (case-insensitive).
# Keep this small and conservative; it exists to reduce user friction.
_STAGE_ALIASES: Dict[str, str] = {
    # Common variations for LOS_FLUX
    "LOSFLUX": "LOS_FLUX",
    "LOS-FLUX": "LOS_FLUX",
    "LOS FLUX": "LOS_FLUX",
    "LOS_FLUX": "LOS_FLUX",
}


StageLike = Union[Stage, str]


def normalize_stage_id(stage_id: str) -> str:
    """
    Normalize a user-provided stage ID string into a canonical lookup key.

    Rules:
    - strip whitespace
    - uppercase
    - collapse internal whitespace to single spaces (then alias resolution)
    """
    s = stage_id.strip().upper()
    # Preserve hyphens/spaces for alias matching before we replace separators.
    s = " ".join(s.split())
    # First, resolve explicit alias keys.
    if s in _STAGE_ALIASES:
        return _STAGE_ALIASES[s]
    # Then perform a light normalization for separator variants.
    s = s.replace("-", "_").replace(" ", "_")
    if s in _STAGE_ALIASES:
        return _STAGE_ALIASES[s]
    return s


def try_parse_stage(stage: Optional[StageLike]) -> Optional[Stage]:
    """
    Best-effort parse of a stage specifier.

    Returns:
    - Stage if parseable
    - None if input is None or invalid

    This is intentionally non-throwing so the pipeline manager can implement
    your policy: "only stop after stage if it exists; otherwise run full chain."
    """
    if stage is None:
        return None
    if isinstance(stage, Stage):
        return stage
    if isinstance(stage, str):
        key = normalize_stage_id(stage)
        return STAGE_BY_ID.get(key)
    return None


def parse_stage(stage: StageLike) -> Stage:
    """
    Strict parse of a stage specifier.

    Returns:
      Stage

    Raises:
      ValueError if invalid.
    """
    parsed = try_parse_stage(stage)
    if parsed is None:
        valid = ", ".join(s.value for s in STAGE_ORDER)
        raise ValueError(f"Invalid stage '{stage}'. Valid stages: {valid}")
    return parsed


def is_valid_stage(stage: Optional[StageLike]) -> bool:
    """True if the provided value parses to a known Stage."""
    return try_parse_stage(stage) is not None


def upstream_stages(stage: StageLike, inclusive: bool = True) -> Tuple[Stage, ...]:
    """
    Return all upstream stages in canonical order.

    Parameters
    ----------
    stage : StageLike
        Stage or stage-id string.
    inclusive : bool
        If True, include `stage` itself.

    Returns
    -------
    tuple[Stage, ...]
        Ordered upstream stages.
    """
    s = parse_stage(stage)
    idx = STAGE_INDEX[s]
    if inclusive:
        return STAGE_ORDER[: idx + 1]
    return STAGE_ORDER[:idx]


def downstream_stages(stage: StageLike, inclusive: bool = True) -> Tuple[Stage, ...]:
    """
    Return all downstream stages in canonical order.

    Parameters
    ----------
    stage : StageLike
        Stage or stage-id string.
    inclusive : bool
        If True, include `stage` itself.

    Returns
    -------
    tuple[Stage, ...]
        Ordered downstream stages.
    """
    s = parse_stage(stage)
    idx = STAGE_INDEX[s]
    if inclusive:
        return STAGE_ORDER[idx:]
    return STAGE_ORDER[idx + 1 :]


def resolve_run_list(stop_after: Optional[StageLike]) -> Tuple[Stage, ...]:
    """
    Resolve the RUN list given STOP_AFTER semantics:

    - If stop_after is valid: run through that stage (inclusive).
    - If stop_after is None or invalid: run the full chain.

    This function encodes your policy in a reusable, testable way.
    """
    s = try_parse_stage(stop_after)
    if s is None:
        return STAGE_ORDER
    return upstream_stages(s, inclusive=True)


def resolve_invalidate_list(recompute_from: Optional[StageLike]) -> Tuple[Stage, ...]:
    """
    Resolve the INVALIDATE list given RECOMPUTE_FROM semantics:

    - If recompute_from is valid: invalidate from that stage onward (inclusive).
    - If recompute_from is None or invalid: invalidate nothing.

    NOTE: The manager will delete "owned outputs" for these stages, not folders.
    """
    s = try_parse_stage(recompute_from)
    if s is None:
        return tuple()
    return downstream_stages(s, inclusive=True)


def validate_stage_order() -> None:
    """
    Internal sanity check: ensures STAGE_ORDER contains each Stage exactly once.

    Intended for use in unit tests or early startup checks.
    """
    if len(set(STAGE_ORDER)) != len(STAGE_ORDER):
        raise RuntimeError("STAGE_ORDER contains duplicate stages.")
    if set(STAGE_ORDER) != set(Stage):
        missing = set(Stage) - set(STAGE_ORDER)
        extra = set(STAGE_ORDER) - set(Stage)
        raise RuntimeError(
            f"STAGE_ORDER mismatch. Missing={sorted(m.value for m in missing)} "
            f"Extra={sorted(s.value for s in extra)}"
        )


__all__ = [
    "Stage",
    "StageLike",
    "STAGE_ORDER",
    "STAGE_INDEX",
    "STAGE_BY_ID",
    "normalize_stage_id",
    "try_parse_stage",
    "parse_stage",
    "is_valid_stage",
    "upstream_stages",
    "downstream_stages",
    "resolve_run_list",
    "resolve_invalidate_list",
    "validate_stage_order",
]
