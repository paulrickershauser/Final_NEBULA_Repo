"""
NEBULA_STAR_SLEW_PROJECTION
===========================

Purpose
-------
Star track builder for *non-sidereal* (slewing) windows.

This module builds **time-resolved** per-star pixel tracks across each slewing window
by projecting a **per-window reference** sky position through the observer’s **time-varying
WCS sequence** (one WCS snapshot per coarse timestep).

Inputs (disk)
-------------
1) frames-with-sky window metadata:
       obs_target_frames_ranked_with_sky.pkl
   frames_with_sky[obs_name]["windows"][...] entries provide:
       - window_index, start_index, end_index, n_frames
       - tracking_mode (only "slew" windows are processed)

2) Gaia cone cache:
       obs_gaia_cones.pkl
   gaia_cache[obs_name]["windows"][...] entries provide:
       - status, error_message
       - gaia_source_id, ra_deg, dec_deg, mag_G
       - optional astrometry arrays: pm_ra_masyr, pm_dec_masyr, parallax_mas,
                                     radial_velocity_km_s

3) Observer tracks (pixel pipeline):
       observer_tracks_with_pixels.pkl
   obs_tracks[obs_name] must provide:
       - t_mjd_utc (REQUIRED)
       - pointing fields consumed by Utility.SENSOR.NEBULA_WCS.build_wcs_for_observer

4) SensorConfig (active sensor geometry).

Tracking-mode gate
------------------
Only windows with tracking_mode == "slew" (case-insensitive) are processed.

Window reference epoch + optional propagation
---------------------------------------------
For each window, a reference epoch is defined as the window-start time:

    t_ref_mjd_utc = obs_track["t_mjd_utc"][start_index]

If proper-motion propagation is enabled via configuration (see implementation notes),
Gaia catalog astrometry is propagated to t_ref_mjd_utc using available Gaia astrometric
fields. If propagation is disabled or required fields are missing, the catalog RA/Dec
are used directly as (ra_deg_ref, dec_deg_ref).

Projection model
----------------
Within a window, each star is treated as fixed at (ra_deg_ref, dec_deg_ref), and its
motion across the detector is due solely to the changing WCS sequence over time.


Output (disk)
-------------
Writes:
    obs_star_slew_tracks.pkl

Structure:
    obs_star_slew_tracks[obs_name] = {
        "observer_name": ...,
        "rows": ...,
        "cols": ...,
        "catalog_name": ...,
        "catalog_band": ...,
        "run_meta": {...},
        "windows": [
            {
                "window_index": ...,
                "start_index": ...,
                "end_index": ...,
                "n_frames": ...,
                "coarse_indices": np.ndarray[int],
                "t_mjd_utc": np.ndarray[float],
                "t_ref_mjd_utc": float,
                "tracking_mode": str or None,
                "gaia_status": str,
                "gaia_error_message": str or None,
                "stars": {
                    source_id_str: {
                        "gaia_source_id": int,
                        "source_id": str,
                        "source_type": "star",
                        "mag_G": float,
                        "ra_deg_catalog": float,
                        "dec_deg_catalog": float,
                        "ra_deg_ref": float,
                        "dec_deg_ref": float,
                        "coarse_indices": np.ndarray[int],
                        "t_mjd_utc": np.ndarray[float],
                        "x_pix": np.ndarray[float],
                        "y_pix": np.ndarray[float],
                        "on_detector": np.ndarray[bool],
                        # optional astrometry fields if present in Gaia cache:
                        # pm_ra_masyr, pm_dec_masyr, parallax_mas, radial_velocity_km_s
                    },
                    ...
                },
            },
            ...
        ],
    }


Assumptions
-----------
* Star RA/Dec is treated as fixed over the window duration.
* Motion across the detector is due to changing pointing encoded in the WCS sequence.
* If you enable proper motion, stars are optionally propagated to a **single**
  global reference epoch before tracking.

This module is intended to be consumed by downstream photon builders or
visualization/animation tools for slewing-star smear.
"""


from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging
import os
import pickle

import numpy as np

# Astropy time + coordinates for optional proper-motion propagation.
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance

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
    NEBULA_STAR_QUERY,
)

from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER

from Utility.SENSOR.NEBULA_WCS import (  # type: ignore[attr-defined]
    build_wcs_for_observer,
    project_radec_to_pixels,
)




# ---------------------------------------------------------------------------
# Type aliases and version tag
# ---------------------------------------------------------------------------

# Per-window, per-star track dictionary type.
StarSlewWindow = Dict[str, Any]

# Top-level type: obs_star_slew_tracks[obs_name] -> dict
ObsStarSlewTracks = Dict[str, Dict[str, Any]]

# Version tag for this module's run_meta.
STAR_SLEW_PROJECTION_VERSION: str = "0.2"
# ---------------------------------------------------------------------------
# Default path resolvers
# ---------------------------------------------------------------------------


def _resolve_default_obs_tracks_path() -> str:
    """
    Resolve the default path to ``observer_tracks_with_pixels.pkl``.

    Returns
    -------
    str
        Path used by :mod:`Utility.SAT_OBJECTS.NEBULA_PIXEL_PICKLER`.
    """

    return NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH


def _resolve_default_output_path() -> str:
    """
    Resolve the default output path for ``obs_star_slew_tracks.pkl``.

    Returns
    -------
    str
        Absolute path under ``NEBULA_OUTPUT/STARS/<catalog_name>``.
    """
    catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
    return os.path.join(
        NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR,
        "STARS",
        catalog_name,
        "obs_star_slew_tracks.pkl",
    )

def _resolve_default_frames_with_sky_path() -> str:
    """
    Resolve default path to obs_target_frames_ranked_with_sky.pkl.
    """
    return os.path.join(
        NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR,
        "TARGET_PHOTON_FRAMES",
        "obs_target_frames_ranked_with_sky.pkl",
    )


def _resolve_default_gaia_cache_path() -> str:
    """
    Resolve default path to obs_gaia_cones.pkl under STARS/<catalog_name>.
    """
    catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
    return os.path.join(
        NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR,
        "STARS",
        catalog_name,
        "obs_gaia_cones.pkl",
    )

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger for this module, creating a default one if needed.

    Parameters
    ----------
    logger : logging.Logger or None
        Existing logger to reuse. If None, a module-local logger is
        created with a simple StreamHandler.

    Returns
    -------
    logging.Logger
        Logger instance configured for NEBULA_STAR_SLEW_PROJECTION.
    """
    if logger is not None:
        return logger

    logger = logging.getLogger("NEBULA_STAR_SLEW_PROJECTION")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# I/O helpers: star projection cache and observer tracks
# ---------------------------------------------------------------------------

def _load_obs_tracks(
    obs_tracks_path: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Load observer tracks with pointing information from disk.

    Parameters
    ----------
    obs_tracks_path : str
        Absolute path to the observer_tracks_with_pointing (or
        observer_tracks_with_pixels) pickle. This should be passed
        explicitly by sim_test.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    dict
        obs_tracks dict keyed by observer name, each entry containing
        coarse times (t_mjd_utc) and pointing fields.

    Raises
    ------
    ValueError
        If obs_tracks_path is an empty string.
    FileNotFoundError
        If the file does not exist at obs_tracks_path.
    """
    if not obs_tracks_path:
        raise ValueError("obs_tracks_path must be a non-empty string.")

    logger.info("Loading observer tracks from '%s'.", obs_tracks_path)
    with open(obs_tracks_path, "rb") as f:
        obs_tracks = pickle.load(f)

    logger.info("Loaded observer tracks for %d observers.", len(obs_tracks))
    return obs_tracks


def _save_star_slew_tracks(
    obs_star_slew_tracks: ObsStarSlewTracks,
    output_path: str,
    logger: logging.Logger,
) -> str:
    """
    Save obs_star_slew_tracks to disk.

    Parameters
    ----------
    obs_star_slew_tracks : dict
        Dictionary keyed by observer name describing per-window star
        tracks during slewing.
    output_path : str
        Destination path for obs_star_slew_tracks.pkl. Should be
        provided explicitly by sim_test.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    str
        Absolute path where the file was written.

    Raises
    ------
    ValueError
        If output_path is an empty string.
    """
    if not output_path:
        raise ValueError("output_path must be a non-empty string.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Writing obs_star_slew_tracks to '%s'.", output_path)
    with open(output_path, "wb") as f:
        pickle.dump(obs_star_slew_tracks, f)

    logger.info(
        "Saved star slew tracks for %d observers.",
        len(obs_star_slew_tracks),
    )
    return output_path

def _load_frames_with_sky(
    frames_with_sky_path: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    if not frames_with_sky_path:
        raise ValueError("frames_with_sky_path must be a non-empty string.")
    if not os.path.exists(frames_with_sky_path):
        raise FileNotFoundError(frames_with_sky_path)

    logger.info("Loading frames-with-sky from '%s'.", frames_with_sky_path)
    with open(frames_with_sky_path, "rb") as f:
        frames_with_sky = pickle.load(f)

    logger.info("Loaded frames-with-sky for %d observers.", len(frames_with_sky))
    return frames_with_sky

def _load_gaia_cache(
    gaia_cache_path: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    if not gaia_cache_path:
        raise ValueError("gaia_cache_path must be a non-empty string.")
    if not os.path.exists(gaia_cache_path):
        raise FileNotFoundError(gaia_cache_path)

    logger.info("Loading Gaia cache from '%s'.", gaia_cache_path)
    with open(gaia_cache_path, "rb") as f:
        gaia_cache = pickle.load(f)

    logger.info("Loaded Gaia cache for %d observers.", len(gaia_cache))
    return gaia_cache

# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _build_wcs_for_all_observers(
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build NebulaWCS entries for all observers, reusing NEBULA_WCS.

    Parameters
    ----------
    obs_tracks : dict
        Observer tracks keyed by observer name, each entry containing
        pointing fields that NEBULA_WCS.build_wcs_for_observer expects.
    sensor_config : SensorConfig
        Sensor configuration used to define the WCS geometry.
    logger : logging.Logger
        Logger for informational messages.

    Returns
    -------
    dict
        Mapping from observer name -> NebulaWCS or sequence of NebulaWCS,
        exactly as returned by :func:`build_wcs_for_observer`.
    """
    wcs_map: Dict[str, Any] = {}

    for obs_name, obs_track in obs_tracks.items():
        logger.info(
            "Building WCS sequence for observer '%s' (star slew projection).",
            obs_name,
        )
        wcs_entry = build_wcs_for_observer(
            observer_track=obs_track,
            sensor_config=sensor_config,
        )
        wcs_map[obs_name] = wcs_entry

    return wcs_map

def _select_wcs_for_coarse_index(
    nebula_wcs_entry: Any,
    coarse_index: int,
) -> Any:
    """
    Select the NebulaWCS corresponding to a given coarse index.

    Semantics
    ---------
    We assume that:

        - build_wcs_for_observer(...) has returned either:
            * a single WCS (static pointing case), or
            * a sequence (list/tuple/ndarray) of WCS objects aligned
              with the coarse time grid obs_track["t_mjd_utc"].

        - The 'coarse_index' we are given corresponds to the index
          in that same coarse time grid (e.g., the same index used
          for NEBULA_PIXEL_PICKLER / frame building).

    Parameters
    ----------
    nebula_wcs_entry : object or sequence
        Output of build_wcs_for_observer for a single observer.
    coarse_index : int
        Coarse time index for which we want a WCS.

    Returns
    -------
    object
        Selected WCS object for that coarse index.
    """
    # Static WCS: use the same WCS for all indices.
    if not isinstance(nebula_wcs_entry, (list, tuple, np.ndarray)):
        return nebula_wcs_entry

    # Dynamic WCS: index into the list/array.
    if coarse_index < 0 or coarse_index >= len(nebula_wcs_entry):
        raise IndexError(
            f"Coarse index {coarse_index} out of range for WCS entry "
            f"of length {len(nebula_wcs_entry)}."
        )

    return nebula_wcs_entry[coarse_index]

def _build_star_tracks_for_window_slew(
    obs_name: str,
    frames_window: Dict[str, Any],
    stars_in: Dict[str, Dict[str, Any]],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> StarSlewWindow:
    """
    Build per-frame star pixel tracks for a single *slewing* window.

    This function is intentionally **stage-decoupled** from NEBULA_STAR_PROJECTION
    and **schema-decoupled** from any "window_projection" structure.

    Inputs
    ------
    1) frames_window
       A single window entry from frames-with-sky, e.g.
       frames_with_sky[obs_name]["windows"][k]. Must provide:
           - window_index (int)
           - start_index (int, inclusive)
           - end_index   (int, inclusive)
           - n_frames    (int)  [informational; not used for indexing]

    2) stars_in
       A dict keyed by Gaia source-id string (preferred) or any stable string key.
       Each value must provide (schema requirement):
           - gaia_source_id (int)
           - ra_deg_ref (float)
           - dec_deg_ref (float)
       Optional:
           - mag_G (float)
           - source_id (str) [if absent, we use the dict key]

       Critically: this function does **not** use any "*_mid" fields and does
       **not** perform epoch/catalog fallback logic. If you want to propagate
       Gaia astrometry to a specific "window epoch", do it *upstream* when
       constructing stars_in.

    3) obs_track + nebula_wcs_entry
       - obs_track must contain a non-empty 't_mjd_utc' array aligned with the
         same coarse index grid used to construct the WCS sequence.
       - nebula_wcs_entry must be either:
           * a single WCS object (static), or
           * a sequence of WCS objects indexable by coarse_index.

    Geometry model
    --------------
    - Star sky positions are treated as fixed at (ra_deg_ref, dec_deg_ref) for
      the duration of this window.
    - Motion on the detector is entirely due to changing pointing encoded in
      the time-varying WCS.

    Returns
    -------
    StarSlewWindow
        Dictionary with:
            {
                "window_index": int,
                "start_index": int,
                "end_index": int,
                "n_frames": int,
                "coarse_indices": np.ndarray[int],
                "t_mjd_utc": np.ndarray[float],
                "t_ref_mjd_utc": float,   # defined as window-start (not mid-window)
                "stars": {
                    sid_str: {
                        "gaia_source_id": int,
                        "source_id": str,
                        "source_type": "star",
                        "mag_G": float,
                        "ra_deg_ref": float,
                        "dec_deg_ref": float,
                        "coarse_indices": np.ndarray[int],
                        "t_mjd_utc": np.ndarray[float],
                        "x_pix": np.ndarray[float],
                        "y_pix": np.ndarray[float],
                        "on_detector": np.ndarray[bool],
                    },
                    ...
                }
            }

    Raises
    ------
    KeyError
        If required window fields or required per-star fields are missing.
    RuntimeError
        If obs_track['t_mjd_utc'] is missing/empty, if sensor geometry is invalid,
        or if no valid projection method exists.
    IndexError
        If the window index bounds are outside obs_track['t_mjd_utc'].
    """
    # ------------------------------------------------------------------
    # 0) Validate window metadata (frames_with_sky window)
    # ------------------------------------------------------------------
    if "start_index" not in frames_window or "end_index" not in frames_window:
        raise KeyError(
            f"frames_window for obs='{obs_name}' missing required "
            "'start_index' and/or 'end_index'."
        )

    window_index = int(frames_window.get("window_index", -1))
    start_index = int(frames_window["start_index"])
    end_index = int(frames_window["end_index"])
    n_frames_meta = int(frames_window.get("n_frames", (end_index - start_index + 1)))

    # Inclusive coarse index range [start_index, end_index]
    idx_window = np.arange(start_index, end_index + 1, dtype=int)
    if idx_window.size == 0:
        raise RuntimeError(
            f"Observer '{obs_name}' has empty index range [{start_index}, {end_index}] "
            f"for window_index={window_index}."
        )

    # Optional consistency check: frames_with_sky n_frames vs computed length.
    # We do not use n_frames for indexing, but a mismatch is a useful diagnostic.
    if n_frames_meta != idx_window.size:
        logger.warning(
            "Observer '%s' window_index=%d: frames_window['n_frames']=%d but "
            "(end-start+1)=%d. Proceeding with computed index range.",
            obs_name,
            window_index,
            n_frames_meta,
            idx_window.size,
        )

    # ------------------------------------------------------------------
    # 1) Coarse time grid (must exist; fail loudly)
    # ------------------------------------------------------------------
    t_mjd_array = obs_track.get("t_mjd_utc", None)
    if t_mjd_array is None:
        raise RuntimeError(
            f"Observer '{obs_name}' missing required 't_mjd_utc' in obs_track. "
            "This module does not fall back to legacy 't_mjd'."
        )

    t_mjd_utc_all = np.asarray(t_mjd_array, dtype=float)
    if t_mjd_utc_all.size == 0:
        raise RuntimeError(
            f"Observer '{obs_name}' has empty 't_mjd_utc' array; cannot build slew tracks."
        )

    # Bounds check: window indices must lie on the obs_track coarse grid.
    if start_index < 0 or end_index >= t_mjd_utc_all.size:
        raise IndexError(
            f"Observer '{obs_name}' window_index={window_index} indices "
            f"[{start_index}, {end_index}] out of bounds for t_mjd_utc size "
            f"{t_mjd_utc_all.size}."
        )

    # Subset the time grid to just this window.
    t_mjd_window = t_mjd_utc_all[idx_window]

    # Define "window epoch" explicitly as the start of the window.
    # (No mid-window logic anywhere in this function.)
    t_ref_mjd_utc = float(t_mjd_window[0])

    # ------------------------------------------------------------------
    # 2) Sensor geometry (rows, cols)
    # ------------------------------------------------------------------
    # Support both 'n_rows'/'n_cols' and legacy 'rows'/'cols' attribute names.
    n_rows = int(getattr(sensor_config, "n_rows", getattr(sensor_config, "rows", 0)))
    n_cols = int(getattr(sensor_config, "n_cols", getattr(sensor_config, "cols", 0)))

    if n_rows <= 0 or n_cols <= 0:
        raise RuntimeError(
            "SensorConfig must define positive 'n_rows'/'n_cols' (or 'rows'/'cols'); "
            f"got n_rows={n_rows}, n_cols={n_cols}."
        )

    # ------------------------------------------------------------------
    # 3) Build per-star tracks across this window
    # ------------------------------------------------------------------
    # stars_in must already be a dict keyed by a stable source id string.
    # This function does not normalize list/dict inputs; that belongs upstream.
    if not isinstance(stars_in, dict):
        raise TypeError(
            f"stars_in must be a dict keyed by source-id string; got {type(stars_in)}."
        )

    stars_out: Dict[str, Dict[str, Any]] = {}

    for sid_str, star in stars_in.items():
        if not isinstance(star, dict):
            continue  # skip malformed entries quietly

        # --------------------------------------------------------------
        # 3a) Required astrometry (schema requirement)
        # --------------------------------------------------------------
        if "ra_deg_ref" not in star or "dec_deg_ref" not in star:
            raise KeyError(
                f"Star '{sid_str}' in obs='{obs_name}', window_index={window_index} "
                "missing required 'ra_deg_ref'/'dec_deg_ref'."
            )

        ra_ref = float(star["ra_deg_ref"])
        dec_ref = float(star["dec_deg_ref"])

        # Required ID field (Gaia provenance)
        if "gaia_source_id" not in star or star["gaia_source_id"] is None:
            raise KeyError(
                f"Star '{sid_str}' in obs='{obs_name}', window_index={window_index} "
                "missing required 'gaia_source_id'."
            )
        gaia_source_id = int(star["gaia_source_id"])

        # Optional photometry
        mag_G = float(star.get("mag_G", np.nan))

        # Stable display/source id for downstream consumers
        source_id = str(star.get("source_id", sid_str))

        # --------------------------------------------------------------
        # 3b) Allocate arrays for this star's per-frame track
        # --------------------------------------------------------------
        x_pix = np.zeros(idx_window.size, dtype=float)
        y_pix = np.zeros(idx_window.size, dtype=float)
        on_detector = np.zeros(idx_window.size, dtype=bool)

        # --------------------------------------------------------------
        # 3c) Loop over coarse indices and project RA/Dec -> pixels
        # --------------------------------------------------------------
        for i, coarse_idx in enumerate(idx_window):
            wcs_t = _select_wcs_for_coarse_index(
                nebula_wcs_entry=nebula_wcs_entry,
                coarse_index=int(coarse_idx),
            )

            # Prefer shared helper (NEBULA_WCS.project_radec_to_pixels),
            # otherwise require the WCS object to expose world_to_pixel.
            if project_radec_to_pixels is not None:
                x_i, y_i = project_radec_to_pixels(
                    wcs_t,
                    np.array([ra_ref], dtype=float),
                    np.array([dec_ref], dtype=float),
                )
            else:
                if not hasattr(wcs_t, "world_to_pixel"):
                    raise RuntimeError(
                        "No RA/Dec -> pixel projection method available: "
                        "project_radec_to_pixels is None and WCS lacks world_to_pixel."
                    )
                x_i, y_i = wcs_t.world_to_pixel(
                    np.array([ra_ref], dtype=float),
                    np.array([dec_ref], dtype=float),
                )

            # Store scalar pixel positions for this timestep.
            x_val = float(x_i[0])
            y_val = float(y_i[0])
            x_pix[i] = x_val
            y_pix[i] = y_val

            # On-detector if inside [0, n_cols) x [0, n_rows)
            on_detector[i] = (
                (0.0 <= x_val < float(n_cols))
                and (0.0 <= y_val < float(n_rows))
            )

        # --------------------------------------------------------------
        # 3d) Pack per-star track entry
        # --------------------------------------------------------------
        stars_out[sid_str] = {
            "gaia_source_id": gaia_source_id,
            "source_id": source_id,
            "source_type": "star",
            "mag_G": mag_G,
            "ra_deg_ref": ra_ref,
            "dec_deg_ref": dec_ref,
            "coarse_indices": idx_window.copy(),
            "t_mjd_utc": t_mjd_window.copy(),
            "x_pix": x_pix,
            "y_pix": y_pix,
            "on_detector": on_detector,
        }

    # ------------------------------------------------------------------
    # 4) Assemble per-window output structure
    # ------------------------------------------------------------------
    window_out: StarSlewWindow = {
        "window_index": window_index,
        "start_index": start_index,
        "end_index": end_index,
        "n_frames": n_frames_meta,
        "coarse_indices": idx_window,
        "t_mjd_utc": t_mjd_window,
        "t_ref_mjd_utc": t_ref_mjd_utc,

        # Preserve upstream metadata/diagnostics if present on the frames window.
        "tracking_mode": frames_window.get("tracking_mode", None),
        "gaia_status": frames_window.get("gaia_status", None),
        "gaia_error_message": frames_window.get("gaia_error_message", None),

        "stars": stars_out,
    }


    logger.debug(
        "Observer '%s', window_index=%d: built slew tracks for %d stars "
        "(%d coarse frames).",
        obs_name,
        window_index,
        len(stars_out),
        idx_window.size,
    )

    return window_out


def _build_window_star_inputs_from_gaia_cache(
    frames_window: Dict[str, Any],
    gaia_window_or_none: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build a minimal window_projection-like dict from frames_with_sky + Gaia cache.

    This returns only what _build_star_tracks_for_window_slew actually needs:
        - window_index, start_index, end_index, n_frames, tracking_mode (if present)
        - stars: dict keyed by Gaia source_id string, with ra/dec + mag + ids

    Gaia failure modes are converted into empty-star windows with gaia_status.
    """
    out: Dict[str, Any] = {
        "window_index": int(frames_window.get("window_index")),
        "start_index": int(frames_window.get("start_index")),
        "end_index": int(frames_window.get("end_index")),
        "n_frames": int(frames_window.get("n_frames")),
        "tracking_mode": frames_window.get("tracking_mode", None),
        "gaia_status": "missing" if gaia_window_or_none is None else gaia_window_or_none.get("status", "missing"),
        "gaia_error_message": None,
        "stars": {},
    }

    if gaia_window_or_none is None:
        out["gaia_error_message"] = "No Gaia window for this window_index"
        return out

    gaia_window = gaia_window_or_none

    if gaia_window.get("status") != "ok":
        out["gaia_status"] = "error"
        out["gaia_error_message"] = gaia_window.get("error_message") or "Gaia status != 'ok'"
        return out

    n_rows = int(gaia_window.get("n_rows", 0))
    if n_rows == 0:
        out["gaia_status"] = "ok_empty"
        out["gaia_error_message"] = "Gaia query returned 0 rows"
        return out

    # Required arrays (force 64-bit IDs to avoid overflow/collisions)
    gaia_source_id = np.asarray(gaia_window["gaia_source_id"], dtype=np.int64)
    ra_deg = np.asarray(gaia_window["ra_deg"], dtype=float)
    dec_deg = np.asarray(gaia_window["dec_deg"], dtype=float)
    mag_G = np.asarray(gaia_window["mag_G"], dtype=float)

    stars: Dict[str, Dict[str, Any]] = {}
    for i in range(ra_deg.size):
        sid_int = int(np.int64(gaia_source_id[i]))
        sid_str = str(sid_int)

        # Fail loudly if an upstream dtype/overflow issue caused collisions.
        if sid_str in stars:
            raise RuntimeError(
                f"Gaia source_id collision in _build_window_star_inputs_from_gaia_cache: "
                f"sid_str={sid_str} already present. This indicates source_id truncation/overflow upstream."
            )

        stars[sid_str] = {
            "gaia_source_id": sid_int,
            "source_id": sid_str,
            "source_type": "star",
            "mag_G": float(mag_G[i]),
            # Minimal contract required by _build_star_tracks_for_window_slew:
            "ra_deg_ref": float(ra_deg[i]),
            "dec_deg_ref": float(dec_deg[i]),
            # Provenance:
            "ra_deg_catalog": float(ra_deg[i]),
            "dec_deg_catalog": float(dec_deg[i]),
        }



    out["stars"] = stars
    out["gaia_status"] = "ok"
    out["gaia_error_message"] = None
    return out

# ---------------------------------------------------------------------------
# Per-observer and all-observer orchestration
# ---------------------------------------------------------------------------

def build_star_slew_tracks_for_observer(
    obs_name: str,
    obs_star_entry: Dict[str, Any],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build star slew tracks for all *slewing* windows of a single observer.

    This function is **stage-decoupled** from NEBULA_STAR_PROJECTION and assumes
    the caller has already constructed an `obs_star_entry` that contains the
    windows to process and per-window star inputs in a minimal, stable schema.

    Recommended upstream source (your "decoupled" path):
        - windows come from frames-with-sky (obs_target_frames_ranked_with_sky.pkl)
        - stars come from Gaia cache (obs_gaia_cones.pkl), converted upstream into
          a per-window dict keyed by Gaia source id string, where each star provides
          *reference* astrometry fields used by tracking:
              ra_deg_ref, dec_deg_ref  (required)

    Accepted `obs_star_entry` schema (minimal contract)
    --------------------------------------------------
    obs_star_entry must provide:
        - "windows": List[dict]
          Each window dict must include:
              - window_index (int)
              - start_index (int, inclusive)
              - end_index   (int, inclusive)
              - n_frames    (int)  [informational]
              - tracking_mode (str)  (must be "slew" to be processed)
              - stars: Dict[str, Dict[str, Any]]
                    keyed by source-id string; each star dict must contain:
                        - gaia_source_id (int)
                        - ra_deg_ref (float)
                        - dec_deg_ref (float)
                    optional:
                        - mag_G (float)
                        - source_id (str)

    Notes on what this function *does not* do:
        - It does NOT load pickles.
        - It does NOT normalize list-vs-dict star containers.
        - It does NOT compute ra/dec reference epochs (no mid-window logic).
        - It does NOT fall back to legacy time fields (handled in the per-window builder).

    Parameters
    ----------
    obs_name : str
        Observer name (used for logging and output container).
    obs_star_entry : dict
        Per-observer container holding the windows to process. In the decoupled
        pipeline, this is typically an "entry-like" wrapper created by
        build_star_slew_tracks_for_all_observers_from_gaia(...).
    obs_track : dict
        Observer track dictionary used to align coarse indices to times and WCS.
        Must contain at least 't_mjd_utc' (required by the per-window function).
    nebula_wcs_entry : object or sequence
        WCS entry returned by build_wcs_for_observer for this observer.
        Either a single static WCS or a sequence aligned with obs_track['t_mjd_utc'].
    sensor_config : SensorConfig
        Sensor geometry configuration (rows/cols).
    logger : logging.Logger
        Logger for progress and summaries.

    Returns
    -------
    dict
        Per-observer star slew track entry with structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "catalog_name": str,
                "catalog_band": str,
                "run_meta": {
                    "version": str,
                    "n_windows_total": int,
                    "n_windows_slew": int,
                    "n_windows_built": int,
                    "total_star_tracks": int,
                    "skipped_windows": List[dict],
                    "upstream_meta": dict or None,
                },
                "windows": [StarSlewWindow, ...],
            }

        where each StarSlewWindow is produced by _build_star_tracks_for_window_slew
        and includes per-star x_pix/y_pix time series.

    Raises
    ------
    RuntimeError
        If sensor rows/cols cannot be determined, or if per-window building fails
        due to missing/empty t_mjd_utc or invalid sensor geometry.
    KeyError
        If required window fields or required per-star fields are missing
        (propagated from the per-window builder).
    IndexError
        If a window’s index range is out of bounds for obs_track['t_mjd_utc']
        (propagated from the per-window builder).
    """
    # ------------------------------------------------------------------
    # 1) Collect windows (already prepared upstream)
    # ------------------------------------------------------------------
    windows: List[Dict[str, Any]] = list(obs_star_entry.get("windows", []))
    n_windows_total = len(windows)

    # Output containers and counters.
    slew_windows: List[StarSlewWindow] = []
    total_tracks = 0
    n_windows_slew = 0
    skipped_windows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 2) Iterate windows; gate to slewing windows; build tracks
    # ------------------------------------------------------------------
    for w in windows:
        # Basic window identity (used for diagnostics even if malformed).
        widx = int(w.get("window_index", -1))
        mode = w.get("tracking_mode", None)

        # Gate: only process slewing windows.
        if mode is None or str(mode).lower() != "slew":
            skipped_windows.append(
                {
                    "window_index": widx,
                    "reason": "tracking_mode_not_slew",
                    "tracking_mode": mode,
                }
            )
            logger.debug(
                "Observer '%s': skipping window_index=%d (tracking_mode=%r; expected 'slew').",
                obs_name,
                widx,
                mode,
            )
            continue

        n_windows_slew += 1

        # Required star container for the decoupled path.
        # (We fail loudly: if stars aren't present, that is an upstream wiring error.)
        if "stars" not in w or not isinstance(w.get("stars"), dict):
            raise KeyError(
                f"Observer '{obs_name}' window_index={widx} missing required "
                "'stars' dict. Upstream must attach per-window stars before calling "
                "build_star_slew_tracks_for_observer."
            )

        stars_in = w["stars"]

        # Build the per-window slew tracks using the decoupled per-window function.
        logger.info(
            "Observer '%s': building star slew tracks for window_index=%d.",
            obs_name,
            widx,
        )

        window_slew = _build_star_tracks_for_window_slew(
            obs_name=obs_name,
            frames_window=w,
            stars_in=stars_in,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )

        slew_windows.append(window_slew)
        total_tracks += len(window_slew.get("stars", {}))

    # ------------------------------------------------------------------
    # 3) Determine sensor dimensions + catalog metadata (provenance only)
    # ------------------------------------------------------------------
    # Prefer explicit rows/cols in obs_star_entry (if the wrapper provided them),
    # otherwise fall back to SensorConfig.
    rows_raw = obs_star_entry.get("rows", getattr(sensor_config, "rows", None))
    cols_raw = obs_star_entry.get("cols", getattr(sensor_config, "cols", None))

    if rows_raw is None:
        rows_raw = getattr(sensor_config, "n_rows", 0)
    if cols_raw is None:
        cols_raw = getattr(sensor_config, "n_cols", 0)

    rows = int(rows_raw)
    cols = int(cols_raw)

    if rows <= 0 or cols <= 0:
        raise RuntimeError(
            f"Failed to determine positive sensor dimensions for observer '{obs_name}': "
            f"rows={rows}, cols={cols}. Ensure obs_star_entry provides rows/cols or "
            "SensorConfig defines rows/cols (or n_rows/n_cols)."
        )

    # Catalog metadata is provenance only; for Gaia-cone path this should be Gaia DR3 G-band.
    catalog_name = obs_star_entry.get(
        "catalog_name", getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
    )
    catalog_band = obs_star_entry.get(
        "catalog_band", getattr(NEBULA_STAR_CATALOG, "band", "G")
    )

    # ------------------------------------------------------------------
    # 4) Build run_meta summary (explicitly not STAR_PROJECTION-specific)
    # ------------------------------------------------------------------
    run_meta: Dict[str, Any] = {
        "version": STAR_SLEW_PROJECTION_VERSION,
        "n_windows_total": n_windows_total,
        "n_windows_slew": n_windows_slew,
        "n_windows_built": len(slew_windows),
        "total_star_tracks": int(total_tracks),
        "skipped_windows": skipped_windows,
        # Preserve upstream provenance if provided by wrapper (e.g., "frames_with_sky + gaia_cache").
        "upstream_meta": obs_star_entry.get("run_meta", None),
    }

    logger.info(
        "Observer '%s': built star slew tracks for %d/%d slewing windows "
        "(%d total star tracks).",
        obs_name,
        len(slew_windows),
        n_windows_slew,
        total_tracks,
    )

    # ------------------------------------------------------------------
    # 5) Assemble per-observer output structure
    # ------------------------------------------------------------------
    obs_slew_entry: Dict[str, Any] = {
        "observer_name": obs_name,
        "rows": rows,
        "cols": cols,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "run_meta": run_meta,
        "windows": slew_windows,
    }

    return obs_slew_entry


def _should_use_proper_motion() -> bool:
    """
    Decide whether to propagate Gaia catalog positions using proper motion.

    This function does not assume NEBULA_STAR_QUERY exists. If absent, propagation is disabled.
    """
    if NEBULA_STAR_QUERY is None:
        return False
    return bool(getattr(NEBULA_STAR_QUERY, "use_proper_motion", False))


def _catalog_obstime_or_none() -> Optional[Time]:
    """
    Get Gaia catalog reference epoch from NEBULA_STAR_CATALOG if available.

    This function does not assume a specific type for reference_epoch.
    """
    ref_epoch = getattr(NEBULA_STAR_CATALOG, "reference_epoch", None)
    if ref_epoch is None:
        return None
    try:
        if isinstance(ref_epoch, (int, float)):
            return Time(ref_epoch, format="jyear")
        return Time(str(ref_epoch))
    except Exception:
        return None


def _propagate_gaia_window_to_epoch(
    gaia_window: Dict[str, Any],
    t_obs_epoch: Time,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate Gaia RA/Dec to a specific observation epoch (e.g., window-start epoch).

    Key robustness behavior:
      * If some stars have invalid/negative/NaN parallax, we DO NOT fail the whole window.
      * Stars with good parallax propagate with distance; others propagate without distance.
      * Any star that cannot be propagated falls back to its catalog RA/Dec (per-star fallback).
    """
    ra_deg = np.asarray(gaia_window["ra_deg"], dtype=float)
    dec_deg = np.asarray(gaia_window["dec_deg"], dtype=float)

    if not _should_use_proper_motion():
        return ra_deg, dec_deg

    pm_ra_arr = gaia_window.get("pm_ra_masyr", None)
    pm_dec_arr = gaia_window.get("pm_dec_masyr", None)
    if pm_ra_arr is None or pm_dec_arr is None:
        return ra_deg, dec_deg

    obstime = _catalog_obstime_or_none()
    if obstime is None:
        return ra_deg, dec_deg

    pm_ra_masyr = np.asarray(pm_ra_arr, dtype=float)
    pm_dec_masyr = np.asarray(pm_dec_arr, dtype=float)

    # Default outputs: per-star fallback is the catalog position.
    ra_out = ra_deg.copy()
    dec_out = dec_deg.copy()

    # Only attempt propagation where core astrometry is finite.
    astrom_ok = (
        np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(pm_ra_masyr)
        & np.isfinite(pm_dec_masyr)
    )

    # Optional: parallax handling (mask good parallaxes only).
    parallax_arr = gaia_window.get("parallax_mas", None)
    parallax_mas = None
    good_plx = None
    if parallax_arr is not None:
        parallax_mas = np.asarray(parallax_arr, dtype=float) * u.mas
        p = parallax_mas.value
        good_plx = np.isfinite(p) & (p > 0.0)

    # Optional: radial velocity (only include when finite for the subset).
    rv_arr = gaia_window.get("radial_velocity_km_s", None)
    rv_vals_all = None
    if rv_arr is not None:
        rv_vals_all = np.asarray(rv_arr, dtype=float)

    def _kwargs_for(mask: np.ndarray, include_distance: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "ra": ra_deg[mask] * u.deg,
            "dec": dec_deg[mask] * u.deg,
            "pm_ra_cosdec": pm_ra_masyr[mask] * (u.mas / u.yr),
            "pm_dec": pm_dec_masyr[mask] * (u.mas / u.yr),
            "frame": "icrs",
        }

        # Use distance derived from parallax only for stars with good parallax.
        if include_distance and (parallax_mas is not None):
            # Distance rejects negative distances unless allow_negative=True. We only feed p>0 here.
            kw["distance"] = Distance(parallax=parallax_mas[mask], allow_negative=False)

        if rv_vals_all is not None:
            rv_sub = rv_vals_all[mask]
            if np.all(np.isfinite(rv_sub)):
                kw["radial_velocity"] = rv_sub * (u.km / u.s)

        return kw

    def _propagate(mask: np.ndarray, include_distance: bool) -> None:
        if not np.any(mask):
            return
        try:
            coord = SkyCoord(obstime=obstime, **_kwargs_for(mask, include_distance=include_distance))
            coord_epoch = coord.apply_space_motion(new_obstime=t_obs_epoch)
            ra_out[mask] = coord_epoch.ra.deg
            dec_out[mask] = coord_epoch.dec.deg
        except Exception:
            # Per-star subset failed: keep catalog ra/dec for that subset.
            return

    # Two-pass: (1) good parallax -> include distance, (2) everything else -> no distance.
    if good_plx is not None:
        m_good = astrom_ok & good_plx
        m_bad = astrom_ok & (~good_plx)
        _propagate(m_good, include_distance=True)
        _propagate(m_bad, include_distance=False)
    else:
        _propagate(astrom_ok, include_distance=False)

    return ra_out, dec_out




def build_star_slew_tracks_for_all_observers_from_gaia(
    frames_with_sky: Dict[str, Dict[str, Any]],
    gaia_cache: Dict[str, Dict[str, Any]],
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> ObsStarSlewTracks:
    """
    PRIMARY PATH (STAGE-DECOUPLED):
    Build star slew tracks directly from frames-with-sky + Gaia cone cache.

    This is the function you should treat as the canonical all-observers driver
    for slewing-star tracking. It removes the STAR_PROJECTION dependency by:

      - Using frames-with-sky to define the per-observer/per-window index ranges
        (start_index/end_index) and tracking_mode gating.
      - Using gaia_cache to define the star list per window.
      - Using obs_tracks (+ build_wcs_for_observer) to construct WCS sequences.

    Output schema contract
    ----------------------
    This function constructs an obs_star_entry-like dict per observer whose windows
    include a per-window "stars" dict keyed by Gaia source id string, where each
    star provides a single reference RA/Dec for tracking:

        star["ra_deg_ref"], star["dec_deg_ref"]

    IMPORTANT:
    ----------
    This function does not assume any mid-window astrometry. The Gaia cache
    provides catalog RA/Dec arrays; those are used as the reference unless your
    gaia_cache windows already include epoch-propagated RA/Dec fields (in which case
    you can swap the assignment in one place below).

    Parameters
    ----------
    frames_with_sky : dict
        frames-with-sky product keyed by observer name. Each entry must contain:
            - "windows": list of window dicts with start_index/end_index/n_frames
            - "rows"/"cols" (optional; falls back to sensor_config)
    gaia_cache : dict
        Gaia cone cache keyed by observer name, each with:
            - "windows": list of Gaia-window dicts keyed by window_index
              each window typically provides arrays:
                  gaia_source_id, ra_deg, dec_deg, mag_G
              and status metadata.
    obs_tracks : dict
        Observer tracks keyed by observer name, providing:
            - t_mjd_utc array and pointing fields used by build_wcs_for_observer.
    sensor_config : SensorConfig
        Sensor geometry configuration (rows/cols).
    logger : logging.Logger
        Logger for progress and summaries.

    Returns
    -------
    ObsStarSlewTracks
        Dictionary keyed by observer name containing per-observer slew-track results.

    Raises
    ------
    RuntimeError / KeyError / IndexError
        Propagated from WCS construction and per-observer/per-window builders.
    """
    # ------------------------------------------------------------------
    # 1) Build WCS entries for all observers present in obs_tracks
    # ------------------------------------------------------------------
    wcs_map = _build_wcs_for_all_observers(
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
    )

    # Container for all observers' slew outputs.
    obs_star_slew_tracks: ObsStarSlewTracks = {}

    # Counters for a final summary.
    n_obs_processed = 0
    n_windows_built_total = 0
    n_tracks_total = 0

    # ------------------------------------------------------------------
    # 2) Loop over observers from frames_with_sky (defines the windows)
    # ------------------------------------------------------------------
    for obs_name, frames_entry in frames_with_sky.items():
        # Guard: must have Gaia cache for this observer.
        if obs_name not in gaia_cache:
            logger.warning("Observer '%s' missing from gaia_cache; skipping.", obs_name)
            continue

        # Guard: must have observer tracks for this observer.
        if obs_name not in obs_tracks:
            logger.warning("Observer '%s' missing from obs_tracks; skipping.", obs_name)
            continue

        # Guard: must have WCS entry for this observer.
        if obs_name not in wcs_map:
            logger.warning("Observer '%s' missing from WCS map; skipping.", obs_name)
            continue

        gaia_obs_entry = gaia_cache[obs_name]
        obs_track = obs_tracks[obs_name]
        nebula_wcs_entry = wcs_map[obs_name]

        # ------------------------------------------------------------------
        # 2a) Build window_index -> Gaia window lookup
        # ------------------------------------------------------------------
        gaia_by_idx: Dict[int, Dict[str, Any]] = {
            int(w.get("window_index")): w for w in gaia_obs_entry.get("windows", [])
            if w.get("window_index") is not None
        }

        # ------------------------------------------------------------------
        # 2b) Prepare slewing windows with attached stars in decoupled schema
        # ------------------------------------------------------------------
        windows_frames = list(frames_entry.get("windows", []))
        windows_prepared: List[Dict[str, Any]] = []

        for w in windows_frames:
            widx = int(w.get("window_index", -1))
            mode = w.get("tracking_mode", None)

            # Gate: only slewing windows are relevant for this module.
            if mode is None or str(mode).lower() != "slew":
                continue

            gaia_w = gaia_by_idx.get(widx, None)

            # Default per-window Gaia diagnostics (kept for provenance/debug).
            gaia_status = "missing"
            gaia_error_message = None

            # Stars dict we will attach to this window (may be empty).
            stars_in: Dict[str, Dict[str, Any]] = {}

            if gaia_w is None:
                gaia_status = "missing"
                gaia_error_message = "No Gaia window for this window_index"
            else:
                # Interpret Gaia status if present.
                status = gaia_w.get("status", "missing")
                if status != "ok":
                    gaia_status = "error"
                    gaia_error_message = gaia_w.get("error_message") or f"Gaia status={status!r}"
                else:
                    # Gaia query returned ok; may still be empty.
                    n_rows = int(gaia_w.get("n_rows", 0))
                    if n_rows <= 0:
                        gaia_status = "ok_empty"
                        gaia_error_message = "Gaia query returned 0 rows"
                    else:
                        gaia_status = "ok"
                        gaia_error_message = None

                        # Required arrays (force 64-bit IDs to avoid overflow/collisions).
                        gaia_source_id = np.asarray(gaia_w["gaia_source_id"], dtype=np.int64)
                        ra_deg_catalog = np.asarray(gaia_w["ra_deg"], dtype=float)
                        dec_deg_catalog = np.asarray(gaia_w["dec_deg"], dtype=float)
                        mag_G = np.asarray(gaia_w["mag_G"], dtype=float)

                        # Compute per-window epoch = window-start time on the observer coarse grid.
                        if "t_mjd_utc" not in obs_track or obs_track["t_mjd_utc"] is None:
                            raise RuntimeError(
                                f"Observer '{obs_name}' missing required 't_mjd_utc' in obs_track; "
                                "cannot propagate Gaia positions to the window epoch."
                            )

                        t_grid_mjd = np.asarray(obs_track["t_mjd_utc"], dtype=float)
                        if t_grid_mjd.size == 0:
                            raise RuntimeError(
                                f"Observer '{obs_name}' has empty 't_mjd_utc' array; cannot compute window epoch."
                            )

                        start_index = int(w.get("start_index"))
                        if start_index < 0 or start_index >= t_grid_mjd.size:
                            raise IndexError(
                                f"Observer '{obs_name}' window_index={widx} start_index={start_index} "
                                f"out of bounds for t_mjd_utc size {t_grid_mjd.size}."
                            )

                        t_ref_mjd_utc = float(t_grid_mjd[start_index])
                        t_ref_epoch = Time(t_ref_mjd_utc, format="mjd", scale="utc")

                        # Propagate to window epoch if enabled and fields exist; otherwise returns catalog RA/Dec.
                        ra_deg_ref_arr, dec_deg_ref_arr = _propagate_gaia_window_to_epoch(
                            gaia_window=gaia_w,
                            t_obs_epoch=t_ref_epoch,
                        )

                        # Optional Gaia astrometry arrays (pass through if present).
                        pm_ra_arr = gaia_w.get("pm_ra_masyr", None)
                        pm_dec_arr = gaia_w.get("pm_dec_masyr", None)
                        parallax_arr = gaia_w.get("parallax_mas", None)
                        rv_arr = gaia_w.get("radial_velocity_km_s", None)

                        if pm_ra_arr is not None:
                            pm_ra_arr = np.asarray(pm_ra_arr, dtype=float)
                        if pm_dec_arr is not None:
                            pm_dec_arr = np.asarray(pm_dec_arr, dtype=float)
                        if parallax_arr is not None:
                            parallax_arr = np.asarray(parallax_arr, dtype=float)
                        if rv_arr is not None:
                            rv_arr = np.asarray(rv_arr, dtype=float)

                        for i in range(ra_deg_catalog.size):
                            sid_int = int(np.int64(gaia_source_id[i]))
                            sid_str = str(sid_int)

                            # Fail loudly if an upstream dtype/overflow issue caused collisions.
                            if sid_str in stars_in:
                                raise RuntimeError(
                                    f"Gaia source_id collision while building stars_in for "
                                    f"obs='{obs_name}', window_index={widx}: sid_str={sid_str} already present. "
                                    "This indicates source_id truncation/overflow upstream."
                                )

                            star_entry: Dict[str, Any] = {
                                "gaia_source_id": sid_int,
                                "source_id": sid_str,
                                "source_type": "star",
                                "mag_G": float(mag_G[i]),

                                # Provenance (catalog):
                                "ra_deg_catalog": float(ra_deg_catalog[i]),
                                "dec_deg_catalog": float(dec_deg_catalog[i]),

                                # Reference astrometry for tracking in this window:
                                "ra_deg_ref": float(ra_deg_ref_arr[i]),
                                "dec_deg_ref": float(dec_deg_ref_arr[i]),
                            }

                            if pm_ra_arr is not None:
                                star_entry["pm_ra_masyr"] = float(pm_ra_arr[i])
                            if pm_dec_arr is not None:
                                star_entry["pm_dec_masyr"] = float(pm_dec_arr[i])
                            if parallax_arr is not None:
                                star_entry["parallax_mas"] = float(parallax_arr[i])
                            if rv_arr is not None:
                                star_entry["radial_velocity_km_s"] = float(rv_arr[i])

                            stars_in[sid_str] = star_entry


            # Attach stars + Gaia diagnostics directly onto a copy of the frames window.
            w_out = dict(w)
            w_out["stars"] = stars_in
            w_out["gaia_status"] = gaia_status
            w_out["gaia_error_message"] = gaia_error_message

            windows_prepared.append(w_out)

        # ------------------------------------------------------------------
        # 2c) Assemble an obs_star_entry-like container expected by the
        #     per-observer builder (no STAR_PROJECTION fields required).
        # ------------------------------------------------------------------
        rows = int(frames_entry.get("rows", getattr(sensor_config, "rows", getattr(sensor_config, "n_rows", 0))))
        cols = int(frames_entry.get("cols", getattr(sensor_config, "cols", getattr(sensor_config, "n_cols", 0))))

        catalog_name = gaia_obs_entry.get("catalog_name", getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG"))
        catalog_band = gaia_obs_entry.get("band", getattr(NEBULA_STAR_CATALOG, "band", "G"))

        obs_star_entry = {
            "observer_name": obs_name,
            "rows": rows,
            "cols": cols,
            "catalog_name": catalog_name,
            "catalog_band": catalog_band,
            # Provenance: explicitly state this is the decoupled path.
            "run_meta": {
                "source": "frames_with_sky + gaia_cache",
                "note": "Stage-decoupled slew projection (no STAR_PROJECTION dependency).",
            },
            "windows": windows_prepared,
        }

        # ------------------------------------------------------------------
        # 2d) Build per-observer slew tracks
        # ------------------------------------------------------------------
        obs_slew_entry = build_star_slew_tracks_for_observer(
            obs_name=obs_name,
            obs_star_entry=obs_star_entry,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )

        obs_star_slew_tracks[obs_name] = obs_slew_entry
        n_obs_processed += 1

        # Robustly sum counters across old/new run_meta key styles.
        run_meta = obs_slew_entry.get("run_meta", {})
        n_windows_built_total += int(
            run_meta.get(
                "n_windows_built",
                run_meta.get("n_windows", len(obs_slew_entry.get("windows", []))),
            )
        )
        n_tracks_total += int(run_meta.get("total_star_tracks", 0))

    # ------------------------------------------------------------------
    # 3) Final summary / sanity check
    # ------------------------------------------------------------------
    if n_obs_processed == 0:
        logger.warning(
            "build_star_slew_tracks_for_all_observers_from_gaia: no observers processed. "
            "Check frames_with_sky vs gaia_cache vs obs_tracks name matching."
        )
    else:
        logger.info(
            "build_star_slew_tracks_for_all_observers_from_gaia: built star slew tracks "
            "for %d observers (%d windows built, %d total star tracks).",
            n_obs_processed,
            n_windows_built_total,
            n_tracks_total,
        )

    return obs_star_slew_tracks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def main(
    sensor_config: Optional[SensorConfig] = None,
    frames_with_sky_path: Optional[str] = None,
    gaia_cache_path: Optional[str] = None,
    obs_tracks_path: Optional[str] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> ObsStarSlewTracks:
    """
    High-level entry point for building *slewing* star pixel tracks for all observers.

    This `main()` is the **stage-decoupled** pipeline entry point for
    NEBULA_STAR_SLEW_PROJECTION. It does not require NEBULA_STAR_PROJECTION.

    Inputs (disk)
    -------------
    1) frames_with_sky_path
       Default: NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_sky.pkl
       Loaded via `_load_frames_with_sky`.

    2) gaia_cache_path
       Default: NEBULA_OUTPUT/STARS/<catalog_name>/obs_gaia_cones.pkl
       Loaded via `_load_gaia_cache`.

    3) obs_tracks_path
       Default: NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH
       Loaded via `_load_obs_tracks`.

    Output (disk)
    -------------
    - If output_path is None, write to `_resolve_default_output_path()`.
    - The in-memory dictionary is always returned.

    Returns
    -------
    ObsStarSlewTracks
        Dictionary keyed by observer name containing per-window slewing star tracks.
    """
    # ------------------------------------------------------------------
    # 1) Normalize logger and resolve SensorConfig
    # ------------------------------------------------------------------
    logger = _get_logger(logger)

    sensor_config = sensor_config or ACTIVE_SENSOR
    if sensor_config is None:
        raise RuntimeError(
            "NEBULA_STAR_SLEW_PROJECTION.main: no SensorConfig available "
            "(sensor_config is None and ACTIVE_SENSOR is not defined)."
        )

    # ------------------------------------------------------------------
    # 2) Resolve default paths (stage-decoupled inputs)
    # ------------------------------------------------------------------
    frames_with_sky_path = frames_with_sky_path or _resolve_default_frames_with_sky_path()
    gaia_cache_path = gaia_cache_path or _resolve_default_gaia_cache_path()
    obs_tracks_path = obs_tracks_path or _resolve_default_obs_tracks_path()

    if output_path is None:
        output_path = _resolve_default_output_path()

    logger.info(
        "NEBULA_STAR_SLEW_PROJECTION: starting star slew track build.\n"
        "  frames_with_sky_path = %s\n"
        "  gaia_cache_path      = %s\n"
        "  obs_tracks_path      = %s\n"
        "  output_path          = %s",
        frames_with_sky_path,
        gaia_cache_path,
        obs_tracks_path,
        output_path,
    )

    # ------------------------------------------------------------------
    # 3) Load upstream products from disk
    # ------------------------------------------------------------------
    frames_with_sky = _load_frames_with_sky(
        frames_with_sky_path=frames_with_sky_path,
        logger=logger,
    )
    gaia_cache = _load_gaia_cache(
        gaia_cache_path=gaia_cache_path,
        logger=logger,
    )
    obs_tracks = _load_obs_tracks(
        obs_tracks_path=obs_tracks_path,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 4) Build slew tracks for all observers (decoupled canonical driver)
    # ------------------------------------------------------------------
    obs_star_slew_tracks = build_star_slew_tracks_for_all_observers_from_gaia(
        frames_with_sky=frames_with_sky,
        gaia_cache=gaia_cache,
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 5) Save to disk (pipeline default)
    # ------------------------------------------------------------------
    _save_star_slew_tracks(
        obs_star_slew_tracks=obs_star_slew_tracks,
        output_path=output_path,
        logger=logger,
    )

    logger.info(
        "NEBULA_STAR_SLEW_PROJECTION: finished; output saved to '%s'.",
        output_path,
    )

    return obs_star_slew_tracks




if __name__ == "__main__":
    # This module is intended to be called from a driver (e.g., sim_test)
    # with explicit paths. Running it directly without arguments will
    # raise errors. You can either:
    #
    #   * Edit this block to hard-code test paths for ad-hoc experiments, or
    #   * Import NEBULA_STAR_SLEW_PROJECTION in your driver and call main(...)
    #     with explicit star_projection_path / obs_tracks_path / output_path.
    raise SystemExit(
        "NEBULA_STAR_SLEW_PROJECTION is a library module. Import it and call "
        "main(sensor_config=..., frames_with_sky_path=..., gaia_cache_path=..., "
        "obs_tracks_path=..., output_path=...)."
    )


