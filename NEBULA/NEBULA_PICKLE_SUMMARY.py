# -*- coding: utf-8 -*-
"""
NEBULA_PICKLE_STAGE_INSPECTOR.py

Purpose
-------
Load ONLY the stage pickles you enable in STAGES_TO_LOAD, print a clear
nested-dictionary layout for each, and leave only those loaded pickles
in the Spyder workspace.

How to extend
-------------
To add the next stage, ONLY edit STAGE_SPECS and/or STAGES_TO_LOAD.
No other code changes should be necessary.

Spyder Variable Explorer hygiene
-------------------------------
- All helpers are prefixed with "_" and then deleted at the end.
- You can also hide underscore variables by keeping Spyder's
  "Exclude private references" enabled. :contentReference[oaicite:8]{index=8}

Security
--------
Only unpickle files you trust; pickle.load can execute code. :contentReference[oaicite:9]{index=9}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle

from Configuration.NEBULA_PATH_CONFIG import ensure_output_directory

try:
    import numpy as _np
except Exception:
    _np = None


# =============================================================================
# EDIT HERE: stage specs + which stages to load
# =============================================================================
STAGE_SPECS: Dict[str, Dict[str, Any]] = {
    "BASE": {
        "dir": "BASE_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks.pkl",
            "TARGET": "target_tracks.pkl",
        },
    },
    "LOS": {
        "dir": "LOS_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks_with_los.pkl",
            "TARGET": "target_tracks_with_los.pkl",
        },
    },
    "ILLUM": {
        "dir": "ILLUM_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks_with_los_illum.pkl",
            "TARGET": "target_tracks_with_los_illum.pkl",
        },
    },
    "FLUX": {
        "dir": "FLUX_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks_with_los_illum_flux.pkl",
            "TARGET": "target_tracks_with_los_illum_flux.pkl",
        },
    },
    "FLUX_LOS": {
        "dir": "LOSFLUX_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks_with_los_illum_flux_los.pkl",
            "TARGET": "target_tracks_with_los_illum_flux_los.pkl",
        },
    },
    "POINTING": {
        "dir": "POINT_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks_with_pointing.pkl",
        },
    },
    "ICRS": {
        "dir": "ICRS_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks_with_icrs.pkl",
            "TARGET": "target_tracks_with_icrs_pairs.pkl",
        },
    },
    "PIXELS": {
        "dir": "PIXEL_SatPickles",
        "pickles": {
            "OBSERVER": "observer_tracks_with_pixels.pkl",
            "TARGET": "target_tracks_with_pixels.pkl",
        },
    },
    "TARGET_FRAMES": {
        "dir": "PHOTON_FRAMES",
        "pickles": {
            "OBSERVER": "obs_photon_frame_catalog.pkl",
        },
    },
    "TARGET_PHOTONS": {
        "dir": "TARGET_PHOTON_FRAMES",
        "pickles": {
            "OBSERVER": "obs_target_frames_ranked.pkl",
        },
    }, 
    "TRACKING_MODE": {
        "dir": "TARGET_PHOTON_FRAMES",
        "pickles": {
            "OBSERVER": "obs_target_frames_ranked_with_tracking.pkl",
        },
    },
    "SKY": {
        "dir": "TARGET_PHOTON_FRAMES",
        "pickles": {
            "OBSERVER": "obs_target_frames_ranked_with_sky.pkl",
        },
    },
    "GAIA": {
        "dir": "STARS\GAIA_DR3_G",
        "pickles": {
            "STARS": "obs_gaia_cones.pkl",
        },
    },
    "STAR_PROJECTION": {
        "dir": "STARS\GAIA_DR3_G",
        "pickles": {
            "SIDEREAL": "obs_star_projections.pkl",
            "SLEW": "obs_star_slew_tracks.pkl",
        },
    }, 
    "STAR_PHOTONS": {
        "dir": "STARS\GAIA_DR3_G",
        "pickles": {
            "OBSERVER": "obs_star_photons.pkl",
        },
    },
    "WINDOW_SOURCES": {
        "dir": "SCENE",
        "pickles": {
            "OBSERVER": "obs_window_sources.pkl",
        },
    },
    "ZODIACAL_LIGHT": {"dir": "ZODIACAL_LIGHT", "pickles": {"OBSERVER": "obs_zodiacal_light.pkl"}},
}
'''
"BASE", "LOS", "ILLUM", "FLUX","FLUX_LOS","POINTING","ICRS", "PIXELS","TARGET_FRAMES","TARGET_PHOTONS","TRACKING_MODE","SKY",
                             "GAIA","STAR_PROJECTION","STAR_PHOTONS","WINDOW_SOURCES","ZODIACAL_LIGHT"
'''
STAGES_TO_LOAD: List[str] = ["WINDOW_SOURCES","ZODIACAL_LIGHT"]

# How deep to inspect list-heavy structures without flooding the console
EXAMPLE_WINDOWS_TO_SHOW: int = 1
EXAMPLE_SOURCES_TO_SHOW: int = 1

# NEW: generic recursion controls
MAX_DEPTH: int = 8               # how deep to descend into nested dict/list
MAX_DICT_KEYS: int = 26          # max keys to print per dict (prevents huge dumps)
MAX_LIST_ITEMS: int = 11          # max items to print per list

# =============================================================================
# Helpers (kept private; removed from workspace at end)
# =============================================================================
def _is_ndarray(x: Any) -> bool:
    return (_np is not None) and isinstance(x, _np.ndarray)


def _type_str(x: Any) -> str:
    if _is_ndarray(x):
        return f"np.ndarray[{x.dtype}] shape={tuple(x.shape)}"
    if isinstance(x, list):
        elem = type(x[0]).__name__ if x else "empty"
        return f"list[{elem}] len={len(x)}"
    if isinstance(x, dict):
        return f"dict len={len(x)}"
    return type(x).__name__


def _load_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _example_track_name(tracks: Dict[str, Any]) -> Optional[str]:
    try:
        return next(iter(tracks.keys()))
    except Exception:
        return None

def _short_scalar(x: Any) -> str:
    """Pretty print small scalar-ish values without flooding output."""
    if x is None:
        return "None"
    if isinstance(x, (bool, int, float)):
        return repr(x)
    if isinstance(x, str):
        return repr(x if len(x) <= 120 else x[:117] + "...")
    return None  # type: ignore


def _render_any(
    obj: Any,
    *,
    name: str = "<root>",
    indent: str = "  ",
    depth: int = 0,
    max_depth: int = MAX_DEPTH,
) -> List[str]:
    """
    Recursively render dict/list structures with depth and size limits.
    Numpy arrays are summarized, not expanded.
    """
    lines: List[str] = []

    # Stop condition
    if depth >= max_depth:
        lines.append(f'{indent * depth}{name}: {_type_str(obj)}  # max depth reached')
        return lines

    # Numpy arrays: summarize only
    if _is_ndarray(obj):
        lines.append(f'{indent * depth}{name}: {_type_str(obj)}')
        return lines

    # Small scalars: print value
    scalar = _short_scalar(obj)
    if scalar is not None:
        lines.append(f'{indent * depth}{name}: {scalar}')
        return lines

    # Dict: recurse into keys
    if isinstance(obj, dict):
        lines.append(f'{indent * depth}{name}: {{  # dict len={len(obj)}')
        keys = sorted(obj.keys(), key=lambda k: str(k))
        if len(keys) > MAX_DICT_KEYS:
            keys = keys[:MAX_DICT_KEYS]
            truncated = True
        else:
            truncated = False

        for k in keys:
            v = obj.get(k)
            # render nested with key as name
            lines.extend(
                _render_any(
                    v,
                    name=repr(k),
                    indent=indent,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )

        if truncated:
            lines.append(f'{indent * (depth + 1)}...  # keys truncated to MAX_DICT_KEYS={MAX_DICT_KEYS}')
        lines.append(f'{indent * depth}}}')
        return lines

    # List/tuple: recurse into items
    if isinstance(obj, (list, tuple)):
        lines.append(f'{indent * depth}{name}: [  # {_type_str(obj)}')
        n = len(obj)
        n_show = min(n, MAX_LIST_ITEMS)
        for i in range(n_show):
            lines.extend(
                _render_any(
                    obj[i],
                    name=f"[{i}]",
                    indent=indent,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
        if n > n_show:
            lines.append(f'{indent * (depth + 1)}...  # items truncated to MAX_LIST_ITEMS={MAX_LIST_ITEMS}')
        lines.append(f'{indent * depth}]')
        return lines

    # Fallback: type only
    lines.append(f'{indent * depth}{name}: {_type_str(obj)}')
    return lines


def _render_track_layout(track: Dict[str, Any], indent: str = "  ") -> List[str]:
    """
    Render a single TrackDict recursively.
    """
    return _render_any(track, name="{TrackDict}", indent=indent, depth=0, max_depth=MAX_DEPTH)



def _render_pickle_layout(label: str, obj: Any, path: Path) -> List[str]:
    """
    Render a full pickle layout:
      - top-level summary
      - schema layout for one example track
    """
    lines: List[str] = []
    if obj is None:
        lines.append(f"{label}: exists=False loaded=False path={path}")
        return lines

    if isinstance(obj, dict):
        lines.append(f"{label}: exists=True loaded=True type=dict len={len(obj)} path={path}")
        lines.append("Top-level layout:")
        lines.append("{")
        lines.append('  "<satellite_name>": TrackDict,')
        lines.append("  ...")
        lines.append("}")

        ex_name = _example_track_name(obj)
        if ex_name is None:
            lines.append("TrackDict layout: (no keys)")
            return lines

        ex_track = obj.get(ex_name)
        if not isinstance(ex_track, dict):
            lines.append(f"TrackDict layout (example {ex_name!r}): non-dict track value: {_type_str(ex_track)}")
            return lines

        lines.append(f"TrackDict layout (example: {ex_name!r}):")
        lines.extend(_render_track_layout(ex_track, indent="  "))
        return lines

    # Non-dict top-level object (unexpected for your current stages, but safe)
    lines.append(f"{label}: exists=True loaded=True type={type(obj).__name__} path={path}")
    lines.append(f"Value summary: {_type_str(obj)}")
    return lines


# =============================================================================
# Main: load only enabled stages and print report
# =============================================================================
OUT_DIR: Path = ensure_output_directory()

# Loaded pickles are stored as top-level variables for Spyder browsing.
# Names follow: <STAGE>_<KIND> (e.g., BASE_TARGET, ILLUM_TARGET).
_PICKLE_VARS: List[str] = []

_REPORT_LINES: List[str] = []
_REPORT_LINES.append("=" * 80)
_REPORT_LINES.append("NEBULA_PICKLE_STAGE_INSPECTOR")
_REPORT_LINES.append(f"Output dir: {OUT_DIR}")
_REPORT_LINES.append(f"Stages loaded: {STAGES_TO_LOAD}")
_REPORT_LINES.append("-" * 80)

for _stage in STAGES_TO_LOAD:
    spec = STAGE_SPECS.get(_stage)
    if spec is None:
        _REPORT_LINES.append(f"[WARN] Stage '{_stage}' not found in STAGE_SPECS.")
        _REPORT_LINES.append("-" * 80)
        continue

    stage_dir = OUT_DIR / spec["dir"]
    for _kind, _fname in spec["pickles"].items():
        var_name = f"{_stage}_{_kind}"
        pkl_path = stage_dir / _fname

        globals()[var_name] = _load_pickle(pkl_path)
        _PICKLE_VARS.append(var_name)

        _REPORT_LINES.extend(_render_pickle_layout(var_name, globals()[var_name], pkl_path))
        _REPORT_LINES.append("-" * 80)

_REPORT_LINES.append("=" * 80)

PICKLE_STRUCTURE_REPORT = "\n".join(_REPORT_LINES)
print(PICKLE_STRUCTURE_REPORT)

# =============================================================================
# Workspace cleanup: keep ONLY the loaded pickle variables + report
# =============================================================================
_KEEP = set(_PICKLE_VARS) | {"PICKLE_STRUCTURE_REPORT"}

for _k in list(globals().keys()):
    if _k in _KEEP:
        continue
    if _k.startswith("__"):
        continue
    # leave nothing else visible
    try:
        del globals()[_k]
    except Exception:
        pass
