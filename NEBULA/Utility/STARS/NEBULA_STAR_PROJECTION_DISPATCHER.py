"""
NEBULA_STAR_PROJECTION_DISPATCHER
=================================

Purpose
-------
Pipeline-stage dispatcher for the unified STAR_PROJECTION stage.

This module inspects the authoritative frames-with-sky product
(obs_target_frames_ranked_with_sky.pkl) for per-window `tracking_mode`
annotations and then dispatches to:

  - NEBULA_STAR_PROJECTION        (sidereal windows; typically static WCS)
  - NEBULA_STAR_SLEW_PROJECTION   (slew windows; time-varying WCS sequence)

Pipeline-correct behavior
-------------------------
1) Stage-decoupled I/O:
   - Loads frames_with_sky, gaia_cache, obs_tracks from disk via explicit paths
     (or default resolvers consistent with NEBULA output layout).

2) Strict invariants:
   - This stage assumes TRACKING_MODE has already run.
   - If *any* window is missing `tracking_mode`, this stage fails loudly.
     (No in-place annotation / mutation of upstream pickles.)

3) Deterministic outputs:
   - Always writes both outputs (sidereal + slew), even if one branch has
     no windows to process. This keeps pipeline manifests simple:
       "stage success => owned outputs exist".

Outputs (recommended)
---------------------
- obs_star_projections.pkl     (sidereal)
- obs_star_slew_tracks.pkl     (slew)

Notes
-----
- This module uses signature-compatible invocation so it can tolerate minor
  parameter-name differences across NEBULA_STAR_PROJECTION / NEBULA_STAR_SLEW_PROJECTION
  (e.g., frames_path vs frames_with_sky_path).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Set
import inspect
import logging
import os
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# NEBULA configuration imports
# ---------------------------------------------------------------------------

from Configuration import NEBULA_PATH_CONFIG  # type: ignore[attr-defined]

from Configuration.NEBULA_SENSOR_CONFIG import (  # type: ignore[attr-defined]
    SensorConfig,
    ACTIVE_SENSOR,
)

from Configuration.NEBULA_STAR_CONFIG import (  # type: ignore[attr-defined]
    NEBULA_STAR_CATALOG,
)

from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ObsFramesWithSky = Dict[str, Dict[str, Any]]
ObsGaiaCache = Dict[str, Dict[str, Any]]
ObsTracks = Dict[str, Dict[str, Any]]

ObsStarProjections = Dict[str, Dict[str, Any]]
ObsStarSlewTracks = Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    if logger is not None:
        return logger

    logger = logging.getLogger("NEBULA_STAR_PROJECTION_DISPATCHER")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Default path resolvers (stage-decoupled)
# ---------------------------------------------------------------------------

def _nebula_output_dir() -> Path:
    # NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR may be str or Path across the repo.
    return Path(NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR)

def _catalog_name() -> str:
    return str(getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG"))

def _resolve_default_frames_with_sky_path() -> str:
    return str(_nebula_output_dir() / "TARGET_PHOTON_FRAMES" / "obs_target_frames_ranked_with_sky.pkl")

def _resolve_default_gaia_cache_path() -> str:
    return str(_nebula_output_dir() / "STARS" / _catalog_name() / "obs_gaia_cones.pkl")

def _resolve_default_obs_tracks_path() -> str:
    return str(NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH)

def _resolve_default_sidereal_output_path() -> str:
    return str(_nebula_output_dir() / "STARS" / _catalog_name() / "obs_star_projections.pkl")

def _resolve_default_slew_output_path() -> str:
    # Matches your current NEBULA_STAR_SLEW_PROJECTION default resolver in the code you shared.
    return str(_nebula_output_dir() / "STARS" / _catalog_name() / "obs_star_slew_tracks.pkl")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: str, logger: logging.Logger, label: str) -> Any:
    if not path:
        raise ValueError(f"{label} path must be a non-empty string.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    logger.info("Loading %s from '%s'.", label, path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def _write_pickle(obj: Any, path: str, logger: logging.Logger, label: str) -> str:
    if not path:
        raise ValueError(f"{label} output path must be a non-empty string.")
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %s to '%s'.", label, str(outp))
    with open(str(outp), "wb") as f:
        pickle.dump(obj, f)
    return str(outp)


# ---------------------------------------------------------------------------
# Tracking-mode scan + strict validation
# ---------------------------------------------------------------------------

def _scan_tracking_modes(frames_with_sky: ObsFramesWithSky) -> Tuple[bool, bool, int, Set[str]]:
    """
    Returns
    -------
    has_sidereal : bool
    has_slew     : bool
    n_missing    : int   (# TARGET-bearing windows missing the `tracking_mode` key)
    unknown      : set[str] of non-(sidereal/slew) modes encountered on TARGET-bearing windows
    """
    has_sidereal = False
    has_slew = False
    n_missing = 0
    unknown: Set[str] = set()

    for _obs_name, obs_entry in frames_with_sky.items():
        for w in obs_entry.get("windows", []):
            # Ignore windows with no targets entirely.
            if int(w.get("n_targets", 0)) <= 0:
                continue

            if "tracking_mode" not in w:
                n_missing += 1
                continue

            mode_raw = w.get("tracking_mode")
            if mode_raw is None:
                # Treat explicit None as “missing” for strictness.
                n_missing += 1
                continue

            mode = str(mode_raw).strip().lower()
            if mode == "sidereal":
                has_sidereal = True
            elif mode == "slew":
                has_slew = True
            else:
                unknown.add(mode)

    return has_sidereal, has_slew, n_missing, unknown



def _raise_missing_tracking_mode(frames_with_sky: ObsFramesWithSky) -> None:
    """
    Construct a helpful error message that points the user back to the TRACKING_MODE stage.
    """
    examples = []
    for obs_name, obs_entry in frames_with_sky.items():
        for w in obs_entry.get("windows", []):
            if int(w.get("n_targets", 0)) <= 0:
                continue

            if ("tracking_mode" not in w) or (w.get("tracking_mode") is None):
                examples.append(
                    f"{obs_name}:window_index={w.get('window_index', 'NA')}"
                )
            if len(examples) >= 8:
                break
        if len(examples) >= 8:
            break

    msg = (
        "STAR_PROJECTION dispatcher: frames-with-sky is missing required "
        "'tracking_mode' annotations on one or more TARGET-bearing windows (n_targets > 0).\n"
        "This stage does NOT annotate in-place. Ensure the TRACKING_MODE stage "
        "runs before STAR_PROJECTION.\n"
        f"Examples (first few missing): {examples}"
    )
    raise RuntimeError(msg)



# ---------------------------------------------------------------------------
# Signature-compatible module invocation
# ---------------------------------------------------------------------------

def _call_main_signature_compat(
    module_obj: Any,
    kwargs_logical: Dict[str, Any],
    logger: logging.Logger,
) -> Any:
    """
    Call module_obj.main(...) but only pass kwargs that the callee accepts.

    This protects you against small naming differences like:
      - frames_path vs frames_with_sky_path
      - output_path vs out_path
    """
    if not hasattr(module_obj, "main"):
        raise AttributeError(f"Module {module_obj} has no 'main' function.")

    fn = module_obj.main
    sig = inspect.signature(fn)

    # Build a compat map: for each logical concept, try multiple parameter names.
    alias_groups = {
        "sensor_config": ["sensor_config"],
        "logger": ["logger"],
        "frames_with_sky_path": ["frames_with_sky_path", "frames_path", "frames"],
        "gaia_cache_path": ["gaia_cache_path", "gaia_path"],
        "obs_tracks_path": ["obs_tracks_path", "tracks_path"],
        "output_path": ["output_path", "out_path", "output"],
    }

    filtered: Dict[str, Any] = {}

    for logical_key, aliases in alias_groups.items():
        value = kwargs_logical.get(logical_key, None)
        if value is None:
            continue
        for name in aliases:
            if name in sig.parameters:
                filtered[name] = value
                break

    # If the callee takes **kwargs, we can pass everything as-is.
    takes_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if takes_var_kw:
        # Prefer explicit names if present; otherwise add any remaining logical keys
        # under their logical names.
        for k, v in kwargs_logical.items():
            filtered.setdefault(k, v)

    logger.info("Calling %s.main with kwargs=%s", getattr(module_obj, "__name__", str(module_obj)), list(filtered.keys()))
    return fn(**filtered)


# ---------------------------------------------------------------------------
# Public entry point (pipeline stage)
# ---------------------------------------------------------------------------

def main(
    sensor_config: Optional[SensorConfig] = None,
    frames_with_sky_path: Optional[str] = None,
    gaia_cache_path: Optional[str] = None,
    obs_tracks_path: Optional[str] = None,
    sidereal_output_path: Optional[str] = None,
    slew_output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[ObsStarProjections, ObsStarSlewTracks]:
    """
    STAR_PROJECTION stage entrypoint.

    Loads stage-decoupled inputs, inspects tracking_mode, and runs:
      - NEBULA_STAR_PROJECTION (sidereal) if needed
      - NEBULA_STAR_SLEW_PROJECTION (slew) if needed

    Always writes both outputs (empty dicts if branch is not needed).

    Returns
    -------
    (obs_star_projections, obs_star_slew_tracks)
    """
    logger = _get_logger(logger)

    # Resolve SensorConfig
    sensor_config = sensor_config or ACTIVE_SENSOR
    if sensor_config is None:
        raise RuntimeError(
            "NEBULA_STAR_PROJECTION_DISPATCHER.main: no SensorConfig available "
            "(sensor_config is None and ACTIVE_SENSOR is not defined)."
        )

    # Resolve default paths
    frames_with_sky_path = frames_with_sky_path or _resolve_default_frames_with_sky_path()
    gaia_cache_path = gaia_cache_path or _resolve_default_gaia_cache_path()
    obs_tracks_path = obs_tracks_path or _resolve_default_obs_tracks_path()

    sidereal_output_path = sidereal_output_path or _resolve_default_sidereal_output_path()
    slew_output_path = slew_output_path or _resolve_default_slew_output_path()

    logger.info(
        "STAR_PROJECTION dispatcher starting.\n"
        "  frames_with_sky_path  = %s\n"
        "  gaia_cache_path       = %s\n"
        "  obs_tracks_path       = %s\n"
        "  sidereal_output_path  = %s\n"
        "  slew_output_path      = %s",
        frames_with_sky_path,
        gaia_cache_path,
        obs_tracks_path,
        sidereal_output_path,
        slew_output_path,
    )

    # Load authoritative upstream products
    frames_with_sky: ObsFramesWithSky = _load_pickle(frames_with_sky_path, logger, "frames-with-sky")
    # NOTE: We don’t *need* to load gaia_cache/obs_tracks just to scan modes,
    # but we do load them here to fail fast on missing upstream pickles.
    _ = _load_pickle(gaia_cache_path, logger, "Gaia cache")
    _ = _load_pickle(obs_tracks_path, logger, "observer tracks")

    # Scan and strictly validate tracking_mode
    has_sidereal, has_slew, n_missing, unknown = _scan_tracking_modes(frames_with_sky)

    if n_missing > 0:
        _raise_missing_tracking_mode(frames_with_sky)
    if unknown:
        raise RuntimeError(
            "STAR_PROJECTION dispatcher: encountered non-(sidereal/slew) tracking modes on "
            f"TARGET-bearing windows: {sorted(unknown)}. This is treated as a hard error."
        )


    logger.info(
        "STAR_PROJECTION dispatcher: has_sidereal=%s, has_slew=%s",
        has_sidereal,
        has_slew,
    )

    # Import lazily to avoid import-time costs if a branch is not needed
    obs_star_projections: ObsStarProjections = {}
    obs_star_slew_tracks: ObsStarSlewTracks = {}

    # ------------------------------------------------------------------
    # Sidereal branch
    # ------------------------------------------------------------------
    if has_sidereal:
        from Utility.STARS import NEBULA_STAR_PROJECTION  # type: ignore

        kwargs_logical = {
            "sensor_config": sensor_config,
            "frames_with_sky_path": frames_with_sky_path,
            "gaia_cache_path": gaia_cache_path,
            "obs_tracks_path": obs_tracks_path,
            "output_path": sidereal_output_path,
            "logger": logger,
        }

        result = _call_main_signature_compat(NEBULA_STAR_PROJECTION, kwargs_logical, logger)
        if result is None:
            # Still ensure an on-disk output exists; treat None as “module wrote it, no return”.
            obs_star_projections = {}
        elif isinstance(result, dict):
            obs_star_projections = result
        else:
            raise RuntimeError(
                f"NEBULA_STAR_PROJECTION.main returned unexpected type {type(result)}; expected dict or None."
            )
    else:
        # No sidereal windows: write empty output to satisfy pipeline owned outputs.
        obs_star_projections = {}
        _write_pickle(obs_star_projections, sidereal_output_path, logger, "sidereal star projections (empty)")

    # ------------------------------------------------------------------
    # Slew branch
    # ------------------------------------------------------------------
    if has_slew:
        from Utility.STARS import NEBULA_STAR_SLEW_PROJECTION  # type: ignore

        kwargs_logical = {
            "sensor_config": sensor_config,
            "frames_with_sky_path": frames_with_sky_path,
            "gaia_cache_path": gaia_cache_path,
            "obs_tracks_path": obs_tracks_path,
            "output_path": slew_output_path,
            "logger": logger,
        }

        result = _call_main_signature_compat(NEBULA_STAR_SLEW_PROJECTION, kwargs_logical, logger)
        if result is None:
            obs_star_slew_tracks = {}
        elif isinstance(result, dict):
            obs_star_slew_tracks = result
        else:
            raise RuntimeError(
                f"NEBULA_STAR_SLEW_PROJECTION.main returned unexpected type {type(result)}; expected dict or None."
            )
    else:
        obs_star_slew_tracks = {}
        _write_pickle(obs_star_slew_tracks, slew_output_path, logger, "slewing star tracks (empty)")

    # As a final safety net, ensure both output files exist even if submodules
    # “returned” but didn’t write (or wrote elsewhere).
    if not os.path.exists(sidereal_output_path):
        _write_pickle(obs_star_projections, sidereal_output_path, logger, "sidereal star projections (forced write)")

    if not os.path.exists(slew_output_path):
        _write_pickle(obs_star_slew_tracks, slew_output_path, logger, "slewing star tracks (forced write)")

    logger.info(
        "STAR_PROJECTION dispatcher finished.\n"
        "  sidereal observers=%d\n"
        "  slew observers=%d",
        len(obs_star_projections),
        len(obs_star_slew_tracks),
    )

    return obs_star_projections, obs_star_slew_tracks


if __name__ == "__main__":
    raise SystemExit(
        "NEBULA_STAR_PROJECTION_DISPATCHER is a library module. "
        "Import it and call main(...) from your pipeline/driver."
    )
