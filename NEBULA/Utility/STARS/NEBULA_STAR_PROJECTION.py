"""
NEBULA_STAR_PROJECTION
======================

This module implements the *star projection* stage for the NEBULA pipeline.

High-level role
---------------
Given:

    1) Windowed target frames with sky footprints
       (obs_target_frames_ranked_with_sky.pkl),

    2) Gaia cone queries per observer + window (obs_gaia_cones.pkl),

    3) Observer tracks with pointing + pixel-augmented metadata
       (observer_tracks_with_pixels.pkl via NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH),

this module:

    * Optionally propagates Gaia star positions from the catalog reference epoch
      to a single *global observation epoch* using astrometric proper motion
      (when enabled). The default epoch is:

          DEFAULT_TIME_WINDOW.start_utc

      (see _propagate_gaia_to_window_epoch for details and overrides).

    * Builds / selects a NebulaWCS instance representing the sensor pointing
      for each observer, using the *same WCS semantics* as the pixel pipeline
      (NEBULA_WCS / build_wcs_for_observer).

    * For each window, selects a representative WCS snapshot and projects
      Gaia RA/Dec (evaluated at the global observation epoch) into sensor
      pixel coordinates (x_pix_epoch, y_pix_epoch).

    * Applies a simple on-detector mask to determine which stars fall
      inside the active sensor rows × cols under that window’s WCS.

    * Aggregates per-window star metadata into a new pickle:

          obs_star_projections.pkl

      keyed by observer name, with one "StarWindowProjection" dict per window
      that is actually processed (see tracking-mode gate below).

Important constraints
---------------------
* This module is **read-only** with respect to upstream NEBULA stages:

    - It does **not** re-query Gaia.
    - It does **not** recompute LOS, illumination, flux, or pointing.
    - It does **not** rebuild target photon frames.

* It only consumes existing pickles and writes a new one.

* It uses the **same WCS + projection stack** as the rest of your detector
  projection pipeline (NEBULA_WCS / project_radec_to_pixels) so stars and
  targets are aligned in the same pixel coordinate convention.

* Tracking-mode gate:

    - STAR_PROJECTION currently **only processes windows that are sidereal**.
    - If a window has window_entry["tracking_mode"] set and it is not "sidereal",
      that window is **skipped** (it is not included in the output "windows" list).
    - This prevents STAR_PROJECTION from silently producing incorrect “static”
      star projections for non-sidereal / slewing windows.

* Stars are treated as **single-epoch per window**:

    - One propagated RA/Dec per star per window at the *global observation epoch*:
          ra_deg_epoch, dec_deg_epoch
    - One projected pixel position per star per window under the selected WCS snapshot:
          x_pix_epoch, y_pix_epoch
    - A boolean on_detector flag per star.

  Note: The astrometry epoch is global by default (DEFAULT_TIME_WINDOW.start_utc),
  while the WCS snapshot may vary by window (via index-based selection). Therefore,
  pixel positions can still vary window-to-window due to pointing changes.

* WCS selection per window uses a **single snapshot chosen by indices**:

    - For each observer, build WCS either as a single NebulaWCS (static pointing)
      or a sequence aligned with the observer track time grid.
    - For each window, select the WCS at the representative coarse index:

          idx_rep = floor((start_index + end_index) / 2)

      clamped to the valid WCS list range.

Outputs and schema
------------------
This module writes a pickle, typically at:

    NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

with the structure:

    obs_star_projections[obs_name] = {
        "observer_name": str,
        "rows": int,
        "cols": int,
        "catalog_name": str,
        "catalog_band": str,
        "mag_limit_sensor_G": float,
        "run_meta": {...},
        "windows": [StarWindowProjection, ...],
    }

Where each StarWindowProjection is a dict with:

    * window metadata (indices, sky center / radius, etc.)
    * Gaia cone status per window (gaia_status, gaia_error_message)
    * counts (n_stars_input, n_stars_on_detector)
    * density estimate: n_stars_on_detector / (π * query_radius_deg²)
    * a "stars" dict keyed by Gaia source_id (as string), with:

          {
              "gaia_source_id": int,
              "source_id": str,          # string alias of gaia_source_id
              "source_type": "star",
              "ra_deg_catalog": float,
              "dec_deg_catalog": float,
              "ra_deg_epoch": float,
              "dec_deg_epoch": float,
              "mag_G": float,
              "x_pix_epoch": float,
              "y_pix_epoch": float,
              "on_detector": bool,
              # optional: pm_ra_masyr, pm_dec_masyr, parallax_mas,
              #           radial_velocity_km_s
          }

See individual function docstrings for more detail.

NOTE ABOUT IMPORTS / FAIL-FAST BEHAVIOR
---------------------------------------
This module uses explicit imports from your NEBULA codebase (NEBULA_WCS,
NEBULA_QUERY_GAIA, NEBULA_PIXEL_PICKLER, config modules). It is designed
to **fail loudly** if those dependencies are missing, rather than using
best-effort import fallbacks.

Gaia cache loading supports both:
    * standard “STARS/<catalog>/obs_gaia_cones.pkl” layouts via
      NEBULA_QUERY_GAIA.load_gaia_cache, and
    * direct pickle loading when a non-standard gaia_cache_path is provided.
"""


# Typing utilities for type hints and generic container types.
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Standard library modules for logging, file paths, pickling, and timestamps.
import logging
import os
import pickle
from datetime import datetime

# Numerical arrays for vectorized operations.
import numpy as np

# Astropy time and coordinates for proper-motion propagation.
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS  # <-- add this
from astropy.coordinates import Distance  # add at top of file (once)

# NEBULA configuration: base output directory for all products.
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# NEBULA sensor configuration: sensor model and active sensor selection.
from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig, ACTIVE_SENSOR

# NEBULA star configuration: Gaia catalog metadata and query policy.
from Configuration.NEBULA_STAR_CONFIG import NEBULA_STAR_CATALOG, NEBULA_STAR_QUERY

# WCS helpers for building per-observer WCS series and projecting RA/Dec to pixels.
from Utility.SENSOR.NEBULA_WCS import ( 
    NebulaWCS,
    build_wcs_for_observer,
    project_radec_to_pixels,
)

# Gaia cache loader and metadata from the NEBULA_QUERY_GAIA module.
from Utility.STARS import NEBULA_QUERY_GAIA

# Pixel pickler, used for locating observer tracks with pointing + pixel geometry.
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER

# Run-meta version string for this star projection stage.
STAR_PROJECTION_RUN_META_VERSION: str = "0.1"

# Alias for the per-window star projection dictionary type.
StarWindowProjection = Dict[str, Any]



def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger instance for this module.

    Parameters
    ----------
    logger : logging.Logger or None, optional
        If provided, this logger instance will be returned unchanged.
        If None, this function returns ``logging.getLogger(__name__)``
        without modifying handlers or levels.

    Returns
    -------
    logging.Logger
        The logger to use inside this module.
    """
    # If the caller supplied a logger, simply return it.
    if logger is not None:
        return logger

    # Otherwise, obtain a module-level logger using the standard pattern.
    return logging.getLogger(__name__)


def _resolve_default_frames_path() -> str:
    """
    Resolve the default path to ``obs_target_frames_ranked_with_sky.pkl``.

    This helper assumes that the NEBULA pipeline has been run via
    ``sim_test.py`` with both ``BUILD_TARGET_PHOTON_FRAMES`` and
    ``RUN_GAIA_PIPELINE`` (and thus ``NEBULA_SKY_SELECTOR``) enabled.

    In that workflow, ``NEBULA_TARGET_PHOTONS`` writes
    ``obs_target_frames_ranked.pkl`` under::

        NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES"

    and ``NEBULA_SKY_SELECTOR`` reads that file, attaches sky footprints,
    and writes::

        NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES" /
            "obs_target_frames_ranked_with_sky.pkl"

    This function simply reconstructs that default path using the
    configured ``NEBULA_OUTPUT_DIR`` from ``Configuration.NEBULA_PATH_CONFIG``.

    Returns
    -------
    str
        Absolute path to the frames-with-sky pickle
        (``obs_target_frames_ranked_with_sky.pkl``).
    """
    # Construct a default path under the TARGET_PHOTON_FRAMES subdirectory,
    # matching the layout used by NEBULA_TARGET_PHOTONS and NEBULA_SKY_SELECTOR.
    default_path = os.path.join(
        NEBULA_OUTPUT_DIR,
        "TARGET_PHOTON_FRAMES",
        "obs_target_frames_ranked_with_sky.pkl",
    )
    return default_path


def _resolve_default_obs_tracks_path() -> str:
    """
    Resolve the default path to the observer tracks pickle used for WCS.
    
    This helper assumes that the NEBULA pixel pipeline has been run (e.g. via
    sim_test.py), and that the pixel pickler wrote the observer tracks to:
    
        NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH
    
    This path typically resolves to:
    
        NEBULA_OUTPUT_DIR / "PIXEL_SatPickles" / "observer_tracks_with_pixels.pkl"
    
    Although this pickle is pixel-augmented, STAR_PROJECTION uses it primarily
    for the observer time grid and pointing fields required by NEBULA_WCS
    (build_wcs_for_observer). Reusing the pixel pipeline’s observer product
    keeps WCS construction consistent with the rest of the detector pipeline.
    
    Returns
    -------
    str
        Absolute path to the observer tracks pickle used by STAR_PROJECTION
        (observer_tracks_with_pixels.pkl).
    """

    # Use the same path that NEBULA_PIXEL_PICKLER uses when it writes
    # observer tracks with pixel geometry. This guarantees that
    # NEBULA_STAR_PROJECTION is aligned with the upstream pixel pipeline.
    return NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH


def _resolve_default_output_path() -> str:
    """
    Resolve the default path for writing 'obs_star_projections.pkl'.

    This helper uses the global NEBULA_OUTPUT_DIR constant and
    NEBULA_STAR_CATALOG to determine a default location:

        <NEBULA_OUTPUT_DIR>/STARS/<catalog_name>/obs_star_projections.pkl
    """
    # Determine catalog name (e.g., "GaiaDR3_G").
    catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")

    # Build a default path under STARS/<catalog_name>.
    default_path = os.path.join(
        NEBULA_OUTPUT_DIR,
        "STARS",
        catalog_name,
        "obs_star_projections.pkl",
    )

    return default_path


def _load_frames_with_sky_from_disk(
    frames_path: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Load ``obs_target_frames_ranked_with_sky.pkl`` from disk.

    This helper is used after the NEBULA pipeline has been run via
    ``sim_test.py`` with::

        BUILD_TARGET_PHOTON_FRAMES = True
        RUN_GAIA_PIPELINE = True

    In that workflow:

    * :mod:`Utility.FRAMES.NEBULA_TARGET_PHOTONS` writes
      ``obs_target_frames_ranked.pkl`` under
      ``NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES"``.
    * :mod:`Utility.STARS.NEBULA_SKY_SELECTOR` reads that file,
      attaches sky footprints, and writes
      ``obs_target_frames_ranked_with_sky.pkl`` in the same directory.

    This function simply resolves the expected path (if not provided),
    loads the pickle with :mod:`pickle`, and logs a short summary.

    Parameters
    ----------
    frames_path : str or None
        Path to ``obs_target_frames_ranked_with_sky.pkl``. If None, a
        default is computed via :func:`_resolve_default_frames_path`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    dict
        Dictionary keyed by observer name, each entry being the
        frames-with-sky structure produced by NEBULA_SKY_SELECTOR.
    """
    # If no explicit path is provided, use the standard TARGET_PHOTON_FRAMES
    # location where NEBULA_SKY_SELECTOR writes the frames-with-sky pickle.
    if frames_path is None:
        frames_path = _resolve_default_frames_path()

    logger.info("Loading frames-with-sky from '%s'.", frames_path)

    # Directly load the pickle written by NEBULA_SKY_SELECTOR.
    with open(frames_path, "rb") as f:
        frames_with_sky: Dict[str, Dict[str, Any]] = pickle.load(f)

    # Compute basic statistics for logging.
    n_observers = len(frames_with_sky)
    total_windows = sum(
        len(entry.get("windows", [])) for entry in frames_with_sky.values()
    )

    logger.info(
        "Loaded frames-with-sky for %d observers (%d windows total).",
        n_observers,
        total_windows,
    )

    return frames_with_sky


def _load_observer_tracks_with_pointing_from_disk(
    obs_tracks_path: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Load observer tracks with pointing from disk.

    This helper assumes that the NEBULA pixel pipeline has already been
    run via :func:`NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs`
    (for example, through ``sim_test.py``). In that workflow,
    :mod:`Utility.SAT_OBJECTS.NEBULA_PIXEL_PICKLER` writes the observer
    tracks with pointing and pixel geometry to::

        NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH

    The tracks are keyed by observer name and each entry includes, at a
    minimum:

        * ``t_mjd_utc`` : 1D array of float
        * ``pointing_boresight_ra_deg`` : 1D array of float
        * ``pointing_boresight_dec_deg`` : 1D array of float

    STAR projection uses these time series and pointing fields to build
    WCS solutions for each observer.

    Parameters
    ----------
    obs_tracks_path : str or None
        Path to the observer tracks pickle. If None, a default is
        computed via :func:`_resolve_default_obs_tracks_path`, which
        returns :data:`NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    dict
        Dictionary keyed by observer name, each entry being a track dict
        that includes at least the required time and pointing fields.
    """
    # If no explicit path is supplied, use the observer tracks pickle
    # written by the NEBULA pixel pipeline.
    if obs_tracks_path is None:
        obs_tracks_path = _resolve_default_obs_tracks_path()

    logger.info(
        "Loading observer tracks with pointing from '%s'.",
        obs_tracks_path,
    )

    # Directly load the pickle produced by NEBULA_PIXEL_PICKLER.
    with open(obs_tracks_path, "rb") as f:
        obs_tracks: Dict[str, Dict[str, Any]] = pickle.load(f)

    # Compute the number of observers contained in the tracks dictionary.
    n_observers = len(obs_tracks)

    # Determine, for each observer, whether the pointing fields are present.
    for obs_name, track in obs_tracks.items():
        has_pointing = all(
            key in track
            for key in (
                "pointing_boresight_ra_deg",
                "pointing_boresight_dec_deg",
            )
        )
        logger.debug(
            "Observer '%s': pointing fields present = %s.",
            obs_name,
            has_pointing,
        )

    logger.info(
        "Loaded observer tracks with pointing for %d observers.",
        n_observers,
    )

    return obs_tracks


def _save_star_projection_cache(
    obs_star_projections: Dict[str, Dict[str, Any]],
    output_path: Optional[str],
    logger: logging.Logger,
) -> str:
    """
    Save ``obs_star_projections`` to disk.

    This helper is called after the star projection stage has built the
    per-observer star projection cache in memory via
    :func:`build_star_projections_for_all_observers`. It performs three
    main tasks:

      * Resolves a default output path when ``output_path`` is None
        using :func:`_resolve_default_output_path`, which constructs
        a path of the form::

            NEBULA_OUTPUT_DIR / "STARS" / NEBULA_STAR_CATALOG.name /
                "obs_star_projections.pkl"

      * Ensures that the parent directory exists.

      * Serializes the ``obs_star_projections`` dictionary using
        :mod:`pickle` and logs a short summary.

    Parameters
    ----------
    obs_star_projections : dict
        Dictionary keyed by observer name that contains the star
        projection results for all observers. Each per-observer entry
        includes at least a ``"windows"`` list of per-window
        star-projection dicts.
    output_path : str or None
        Path to write ``obs_star_projections.pkl``. If None, a default
        is computed via :func:`_resolve_default_output_path`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    str
        The absolute path where the file was written.
    """
    # If no explicit path is supplied, use the standard STARS/<catalog>
    # location defined by _resolve_default_output_path().
    if output_path is None:
        output_path = _resolve_default_output_path()

    # Ensure that the directory in which we are writing exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Log that we are writing the star projections to disk.
    logger.info("Writing obs_star_projections to '%s'.", output_path)

    # Serialize the obs_star_projections dictionary using pickle.
    with open(output_path, "wb") as f:
        pickle.dump(obs_star_projections, f)

    # Compute the number of observers contained in the dictionary for logging.
    n_observers = len(obs_star_projections)

    # Compute the total number of windows across all observers.
    total_windows = sum(
        len(entry.get("windows", [])) for entry in obs_star_projections.values()
    )

    # Compute the total number of on-detector stars across all windows.
    total_stars_on_detector = 0
    for entry in obs_star_projections.values():
        for w in entry.get("windows", []):
            total_stars_on_detector += int(w.get("n_stars_on_detector", 0))

    # Log a summary of what was saved.
    logger.info(
        "Saved obs_star_projections for %d observers (%d windows, %d stars on detector).",
        n_observers,
        total_windows,
        total_stars_on_detector,
    )

    return output_path


def _build_wcs_for_all_observers(
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build NebulaWCS objects for all observers.

    This function is a thin wrapper around the shared
    :func:`build_wcs_for_observer` helper from
    :mod:`Utility.SENSOR.NEBULA_WCS`. For each observer track, it
    invokes that builder and returns a mapping::

        wcs_map[obs_name] = NebulaWCS or list[NebulaWCS]

    where each entry can be:

      * a single :class:`NebulaWCS` instance for static pointing, or
      * a sequence of :class:`NebulaWCS` objects aligned with
        ``obs_track["t_mjd_utc"]`` for time-varying pointing.

    The subsequent :func:`_select_wcs_for_window` function then chooses
    the appropriate WCS snapshot per window based on the mid-window time.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary keyed by observer name, each value being the observer
        track dict that includes at least:

          * ``"t_mjd_utc"`` : 1D float array of coarse times
          * ``"pointing_boresight_ra_deg"`` : scalar or 1D float array
          * ``"pointing_boresight_dec_deg"`` : scalar or 1D float array
          * ``"roll_deg"`` : scalar or 1D float array

        These fields are typically attached by the scheduling / pointing
        stages (e.g. NEBULA_SCHEDULE_PICKLER and NEBULA_POINTING_ANTISUN_GEO)
        before the pixel pipeline runs.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, FOV, etc.). The exact
        dataclass definition lives in :mod:`Configuration.NEBULA_SENSOR_CONFIG`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    dict
        A mapping from observer name to whatever object (or sequence of
        objects) :func:`build_wcs_for_observer` returns for that observer.
    """
    # Initialize an empty dictionary to hold the WCS objects per observer.
    wcs_map: Dict[str, Any] = {}

    # Loop over each observer track in the provided dictionary.
    for obs_name, obs_track in obs_tracks.items():
        # Log that we are building WCS objects for this observer.
        logger.info("Building WCS for observer '%s'.", obs_name)

        # Call the shared WCS builder with the observer track and sensor config.
        nebula_wcs_entry = build_wcs_for_observer(
            observer_track=obs_track,
            sensor_config=sensor_config,
        )

        # Store the resulting NebulaWCS (or list of NebulaWCS) in the map.
        wcs_map[obs_name] = nebula_wcs_entry

    # Return the mapping from observer name to WCS entry.
    return wcs_map


def _select_wcs_for_window(
    nebula_wcs_entry: Any,
    obs_track: Dict[str, Any],
    window_entry: Dict[str, Any],
) -> Any:
    """
    Select the appropriate WCS snapshot for a given window.
    
    Selection semantics
    -------------------
    This function supports two forms of WCS input:
    
      1) Static pointing:
         If nebula_wcs_entry is already a single NebulaWCS (or astropy.wcs.WCS),
         it is returned unchanged.
    
      2) Time-varying pointing:
         If nebula_wcs_entry is a sequence (list/tuple/ndarray) of WCS objects,
         this function selects a representative snapshot by window indices:
    
             idx_rep = floor((start_index + end_index) / 2)
    
         The selected index is clamped to the valid range of the WCS sequence.
    
    This implementation does not require per-window mid-time fields (e.g.
    t_mid_mjd_utc). It relies purely on the window’s coarse frame indices.
    
    Parameters
    ----------
    nebula_wcs_entry : NebulaWCS or astropy.wcs.WCS or sequence
        WCS information for an observer, either static or time-aligned.
    obs_track : dict
        Observer track dictionary (currently unused by this selector).
    window_entry : dict
        One window entry from frames-with-sky, expected to contain
        'start_index' and 'end_index' when time-varying WCS is used.
    
    Returns
    -------
    NebulaWCS or astropy.wcs.WCS
        The WCS object to use as the representative pointing for this window.
    """


    # ------------------------------------------------------------------
    # 1) If this is already a single static WCS, just return it
    # ------------------------------------------------------------------
    if isinstance(nebula_wcs_entry, (NebulaWCS, WCS)):
        return nebula_wcs_entry

    # ------------------------------------------------------------------
    # 2) Otherwise, interpret as a sequence of WCS objects
    # ------------------------------------------------------------------
    if not isinstance(nebula_wcs_entry, (list, tuple, np.ndarray)):
        raise TypeError(
            "nebula_wcs_entry must be a NebulaWCS/WCS or a sequence of "
            f"NebulaWCS objects; got type={type(nebula_wcs_entry)!r}"
        )

    wcs_list = list(nebula_wcs_entry)
    if len(wcs_list) == 0:
        raise ValueError("nebula_wcs_entry sequence is empty; cannot select WCS.")

    # ------------------------------------------------------------------
    # 3) Select a representative coarse index for this window
    # ------------------------------------------------------------------
    # We use the integer midpoint of [start_index, end_index] as the
    # representative frame index for the window. If one of these is
    # missing, fall back to the other; if both are missing, use index 0.
    start_idx = window_entry.get("start_index", None)
    end_idx = window_entry.get("end_index", None)

    if start_idx is not None and end_idx is not None:
        idx_rep = (int(start_idx) + int(end_idx)) // 2
    elif start_idx is not None:
        idx_rep = int(start_idx)
    elif end_idx is not None:
        idx_rep = int(end_idx)
    else:
        idx_rep = 0

    # Clamp index to valid range of the WCS list
    if idx_rep < 0:
        idx_rep = 0
    if idx_rep >= len(wcs_list):
        idx_rep = len(wcs_list) - 1

    return wcs_list[idx_rep]

'''
# def _propagate_gaia_to_window_epoch(
#     gaia_window: Dict[str, Any],
#     t_obs_epoch: Optional[Time] = None,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Propagate Gaia star positions to a single global observation epoch.

#     This function takes the Gaia cone data for a single window and
#     computes the apparent RA/Dec at a chosen observation epoch using
#     Astropy's :class:`SkyCoord` and :meth:`SkyCoord.apply_space_motion`.

#     Epoch choice
#     ------------
#     By design for NEBULA_STAR_PROJECTION, the default observation epoch
#     is the start of the simulation time window defined in
#     :mod:`NEBULA_TIME_CONFIG`, i.e.::

#         DEFAULT_TIME_WINDOW.start_utc

#     The optional ``t_obs_epoch`` parameter allows you to override this
#     (e.g., for tests), but typical usage should pass ``None`` and let
#     the function pull from the configuration.

#     Proper-motion semantics
#     -----------------------
#     We adopt the following conventions:

#       * Gaia proper motion components are stored (when available) as::

#             pm_ra_masyr   # pmra (already cos(dec)-weighted)
#             pm_dec_masyr  # pmdec

#         These map directly to Astropy's::

#             pm_ra_cosdec = pm_ra_masyr * (u.mas / u.yr)
#             pm_dec       = pm_dec_masyr * (u.mas / u.yr)

#       * Optional parallax and radial velocity fields (if present) are::

#             parallax_mas
#             radial_velocity_km_s

#     Whether proper motion is used is controlled by
#     ``NEBULA_STAR_QUERY.use_proper_motion`` (bool). If
#     ``use_proper_motion`` is False, or the necessary astrometric fields
#     are missing, the function simply returns the catalog RA/Dec as-is.

#     Parameters
#     ----------
#     gaia_window : dict
#         Gaia window entry containing at least:

#           * ``"ra_deg"``, ``"dec_deg"`` : arrays of catalog RA/Dec
#             in degrees

#         and optionally:

#           * ``"pm_ra_masyr"``, ``"pm_dec_masyr"``
#           * ``"parallax_mas"``
#           * ``"radial_velocity_km_s"``
#     t_obs_epoch : astropy.time.Time, optional
#         Observation epoch at which to evaluate the star positions.
#         If ``None``, this function uses
#         ``DEFAULT_TIME_WINDOW.start_utc`` from
#         :mod:`NEBULA_TIME_CONFIG`.

#     Returns
#     -------
#     ra_deg_epoch : np.ndarray
#         Array of RA values at the observation epoch in degrees.
#     dec_deg_epoch : np.ndarray
#         Array of Dec values at the observation epoch in degrees.
#     """
#     # Extract the catalog RA/Dec in degrees as numpy arrays. Let a missing
#     # key raise KeyError rather than silently propagating empty arrays.
#     ra_deg = np.asarray(gaia_window["ra_deg"], dtype=float)
#     dec_deg = np.asarray(gaia_window["dec_deg"], dtype=float)

#     # If the NEBULA_STAR_QUERY configuration disables proper motion,
#     # return the catalog positions directly.
#     use_pm = bool(getattr(NEBULA_STAR_QUERY, "use_proper_motion", False))
#     if not use_pm:
#         return ra_deg, dec_deg

#     # Extract proper motion arrays if present; if missing, fall back
#     # to catalog positions.
#     pm_ra_arr = gaia_window.get("pm_ra_masyr", None)
#     pm_dec_arr = gaia_window.get("pm_dec_masyr", None)
#     if pm_ra_arr is None or pm_dec_arr is None:
#         return ra_deg, dec_deg

#     # Convert proper motions to numpy arrays of floats.
#     pm_ra_masyr = np.asarray(pm_ra_arr, dtype=float)
#     pm_dec_masyr = np.asarray(pm_dec_arr, dtype=float)

#     # Extract parallax and radial velocity arrays if present.
#     parallax_arr = gaia_window.get("parallax_mas", None)
#     rv_arr = gaia_window.get("radial_velocity_km_s", None)

#     # Build keyword arguments for SkyCoord construction. Gaia positions
#     # are in the ICRS frame.
#     coord_kwargs: Dict[str, Any] = {
#         "ra": ra_deg * u.deg,
#         "dec": dec_deg * u.deg,
#         "pm_ra_cosdec": pm_ra_masyr * (u.mas / u.yr),
#         "pm_dec": pm_dec_masyr * (u.mas / u.yr),
#         "frame": "icrs",
#     }

#     # If parallax is available, pass it as a quantity to SkyCoord.
#     # if parallax_arr is not None:
#     #     coord_kwargs["parallax"] = np.asarray(parallax_arr, dtype=float) * u.mas
#     # If parallax is available, convert it to a Distance and pass as `distance=`.
#     if parallax_arr is not None:
#         parallax = np.asarray(parallax_arr, dtype=float) * u.mas
    
#         # Distance(parallax=...) uses Astropy’s parallax↔distance equivalency internally.
#         # allow_negative=True avoids hard-failing on negative Gaia parallaxes; invalid values become NaN.
#         coord_kwargs["distance"] = Distance(parallax=parallax, allow_negative=True)
#     # If radial velocity is available, pass it as a quantity to SkyCoord.
#     if rv_arr is not None:
#         coord_kwargs["radial_velocity"] = (
#             np.asarray(rv_arr, dtype=float) * (u.km / u.s)
#         )

#     # Determine the catalog reference epoch from NEBULA_STAR_CATALOG.
#     ref_epoch = getattr(NEBULA_STAR_CATALOG, "reference_epoch", None)

#     # Attempt to construct an Astropy Time for the reference epoch.
#     try:
#         # If ref_epoch is a numeric year, interpret it as Julian year.
#         if isinstance(ref_epoch, (int, float)):
#             obstime = Time(ref_epoch, format="jyear")
#         else:
#             # Otherwise, treat it as a time string that Time can parse.
#             obstime = Time(str(ref_epoch))
#     except Exception:  # noqa: BLE001
#         obstime = None  # type: ignore[assignment]

#     # If we could not determine a valid reference epoch, fall back to
#     # catalog RA/Dec.
#     if obstime is None:
#         return ra_deg, dec_deg

#     # Decide which observation epoch to use:
#     #   - If the caller supplied t_obs_epoch, use it.
#     #   - Otherwise, use DEFAULT_TIME_WINDOW.start_utc from NEBULA_TIME_CONFIG.
#     if t_obs_epoch is None:
#         # Local import to avoid any potential circular-import issues.
#         from Configuration.NEBULA_TIME_CONFIG import DEFAULT_TIME_WINDOW

#         try:
#             t_obs_epoch = Time(DEFAULT_TIME_WINDOW.start_utc, scale="utc")
#         except Exception:
#             # If for some reason the config start time cannot be parsed,
#             # fall back to catalog positions.
#             return ra_deg, dec_deg

#     # Construct a SkyCoord at the catalog reference epoch with proper motions.
#     coord = SkyCoord(obstime=obstime, **coord_kwargs)

#     # Apply space motion to propagate to the chosen observation epoch.
#     coord_epoch = coord.apply_space_motion(new_obstime=t_obs_epoch)

#     # Extract RA and Dec at the observation epoch in degrees and return them.
#     return coord_epoch.ra.deg, coord_epoch.dec.deg
'''

def _propagate_gaia_to_window_epoch(
    gaia_window: Dict[str, Any],
    t_obs_epoch: Optional[Time] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    ra_deg = np.asarray(gaia_window["ra_deg"], dtype=float)
    dec_deg = np.asarray(gaia_window["dec_deg"], dtype=float)

    use_pm = bool(getattr(NEBULA_STAR_QUERY, "use_proper_motion", False))
    if not use_pm:
        return ra_deg, dec_deg

    pm_ra_arr = gaia_window.get("pm_ra_masyr", None)
    pm_dec_arr = gaia_window.get("pm_dec_masyr", None)
    if pm_ra_arr is None or pm_dec_arr is None:
        return ra_deg, dec_deg

    pm_ra_masyr = np.asarray(pm_ra_arr, dtype=float)
    pm_dec_masyr = np.asarray(pm_dec_arr, dtype=float)

    parallax_arr = gaia_window.get("parallax_mas", None)
    rv_arr = gaia_window.get("radial_velocity_km_s", None)

    # ------------------------------------------------------------
    # 1) Define catalog reference epoch (obstime) BEFORE any SkyCoord use
    # ------------------------------------------------------------
    ref_epoch = getattr(NEBULA_STAR_CATALOG, "reference_epoch", None)
    try:
        if isinstance(ref_epoch, (int, float)):
            obstime = Time(ref_epoch, format="jyear")
        else:
            obstime = Time(str(ref_epoch))
    except Exception:
        return ra_deg, dec_deg

    # ------------------------------------------------------------
    # 2) Define observation epoch (t_obs_epoch) BEFORE propagation
    # ------------------------------------------------------------
    if t_obs_epoch is None:
        from Configuration.NEBULA_TIME_CONFIG import DEFAULT_TIME_WINDOW
        try:
            t_obs_epoch = Time(DEFAULT_TIME_WINDOW.start_utc, scale="utc")
        except Exception:
            return ra_deg, dec_deg

    # ------------------------------------------------------------
    # 3) Build masks
    # ------------------------------------------------------------
    astrom_ok = (
        np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(pm_ra_masyr)
        & np.isfinite(pm_dec_masyr)
    )

    parallax_mas = None
    good_plx = None
    if parallax_arr is not None:
        parallax_mas = np.asarray(parallax_arr, dtype=float) * u.mas
        p = parallax_mas.value
        good_plx = np.isfinite(p) & (p > 0.0)

    # Initialize outputs; default NaN for anything we do not propagate.
    ra_out = np.full_like(ra_deg, np.nan, dtype=float)
    dec_out = np.full_like(dec_deg, np.nan, dtype=float)

    def _kwargs_for(mask: np.ndarray, include_distance: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "ra": ra_deg[mask] * u.deg,
            "dec": dec_deg[mask] * u.deg,
            "pm_ra_cosdec": pm_ra_masyr[mask] * (u.mas / u.yr),
            "pm_dec": pm_dec_masyr[mask] * (u.mas / u.yr),
            "frame": "icrs",
        }

        if include_distance and (parallax_mas is not None):
            # Only called on masks where parallax is finite and > 0
            kw["distance"] = Distance(parallax=parallax_mas[mask], allow_negative=False)

        # Only include radial_velocity if present AND finite for this subset
        if rv_arr is not None:
            rv_vals = np.asarray(rv_arr, dtype=float)[mask]
            if np.all(np.isfinite(rv_vals)):
                kw["radial_velocity"] = rv_vals * (u.km / u.s)

        return kw

    def _propagate(mask: np.ndarray, include_distance: bool) -> None:
        if not np.any(mask):
            return
        coord = SkyCoord(obstime=obstime, **_kwargs_for(mask, include_distance=include_distance))
        coord_epoch = coord.apply_space_motion(new_obstime=t_obs_epoch)
        ra_out[mask] = coord_epoch.ra.deg
        dec_out[mask] = coord_epoch.dec.deg

    # ------------------------------------------------------------
    # 4) Two-pass propagation
    # ------------------------------------------------------------
    if good_plx is not None:
        m_good = astrom_ok & good_plx
        m_bad = astrom_ok & (~good_plx)

        # Good parallaxes -> include distance (enables full 3D effects)
        _propagate(m_good, include_distance=True)

        # Bad/negative/NaN parallaxes -> omit distance (avoid invalid-distance warnings)
        _propagate(m_bad, include_distance=False)
    else:
        # No parallax column -> propagate without distance
        _propagate(astrom_ok, include_distance=False)

    return ra_out, dec_out


def _make_empty_star_window_projection(
    window_entry: Dict[str, Any],
    gaia_status: str,
    reason: str,
) -> StarWindowProjection:
    """
    Construct an empty StarWindowProjection for non-success Gaia cases.

    This helper fills in the per-window metadata from the frames-with-sky
    window entry and sets all star-related counts to zero, with an empty
    ``"stars"`` dict. It is used when Gaia data is missing, in error,
    or when a query succeeds but returns zero rows for a given window.

    Parameters
    ----------
    window_entry : dict
        Window entry from frames-with-sky containing metadata such as
        ``window_index``, ``start_index``, ``end_index``, ``n_frames``,
        ``sky_center_ra_deg``, ``sky_center_dec_deg``, ``sky_radius_deg``,
        and ``sky_selector_status``.
    gaia_status : {"ok_empty", "error", "missing"}
        Status string describing the Gaia data situation for this window.
        This helper is only used for non-success cases; successful windows
        use ``gaia_status="ok"`` and are constructed separately.
    reason : str
        Human-readable explanation for the status (stored in
        ``gaia_error_message``).

    Returns
    -------
    StarWindowProjection
        A StarWindowProjection dict with zero stars and the given status.
    """
    # Optional sanity check: catch accidental misuse with "ok".
    # allowed_statuses = {"ok_empty", "error", "missing"}
    # if gaia_status not in allowed_statuses:
    #     raise ValueError(f"Unexpected gaia_status='{gaia_status}' for empty projection.")

    # Create the base projection dict, copying over basic window metadata.
    projection: StarWindowProjection = {
        # Copy window index from frames-with-sky entry.
        "window_index": int(window_entry.get("window_index")),
        # Copy start index (frame index) from frames-with-sky entry.
        "start_index": int(window_entry.get("start_index")),
        # Copy end index (frame index) from frames-with-sky entry.
        "end_index": int(window_entry.get("end_index")),
        # Copy number of frames in this window.
        "n_frames": int(window_entry.get("n_frames")),
        # Copy sky center RA in degrees.
        "sky_center_ra_deg": float(window_entry.get("sky_center_ra_deg")),
        # Copy sky center Dec in degrees.
        "sky_center_dec_deg": float(window_entry.get("sky_center_dec_deg")),
        # Copy sky radius in degrees.
        "sky_radius_deg": float(window_entry.get("sky_radius_deg")),
        # Copy sky selector status string.
        "sky_selector_status": window_entry.get("sky_selector_status"),
        # Set Gaia status for this window.
        "gaia_status": gaia_status,
        # Attach a human-readable explanation for the Gaia status.
        "gaia_error_message": reason,
        # Initialize the number of Gaia stars in the cone as zero.
        "n_stars_input": 0,
        # Initialize the number of on-detector stars as zero.
        "n_stars_on_detector": 0,
        # No density information is available for empty/error windows.
        "star_density_on_detector_per_deg2": None,
        # Initialize an empty dict for per-star details.
        "stars": {},
    }

    return projection


def project_gaia_stars_for_window(
    obs_name: str,
    window_entry: Dict[str, Any],
    gaia_window_or_none: Optional[Dict[str, Any]],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> StarWindowProjection:
    """
    Project Gaia stars for a single window onto the sensor.

    This function performs the per-window core logic:

        1. Handles missing or error Gaia cones by returning an empty
           StarWindowProjection with an appropriate gaia_status.

        2. For valid Gaia cones:

            * Extracts Gaia positions and magnitudes.
            * Propagates the catalog RA/Dec to a single global
              observation epoch (the start of the default simulation
              window from NEBULA_TIME_CONFIG) using
              :func:`_propagate_gaia_to_window_epoch`.
            * Selects a WCS snapshot for the window via
              :func:`_select_wcs_for_window` (based on the window's
              indices and the observer time grid).
            * Projects RA/Dec at the observation epoch to pixel
              coordinates using project_radec_to_pixels (or
              NebulaWCS.world_to_pixel).
            * Applies a simple FOV mask:

                  0 <= x < sensor_config.n_cols
                  0 <= y < sensor_config.n_rows

            * Counts on-detector stars and computes a star density
              based on the Gaia query cone area:

                  density = n_stars_on_detector / (π * query_radius_deg²)

            * Builds the per-star dictionary keyed by Gaia source ID
              string, with no "mid" fields in the schema.

    Parameters
    ----------
    obs_name : str
        Name of the observer (used for logging only).
    window_entry : dict
        Window entry from frames-with-sky, containing metadata and
        time-indexing information for this window.
    gaia_window_or_none : dict or None
        Matching Gaia window entry from the Gaia cache for this window
        (same window_index), or None if missing entirely.
    obs_track : dict
        Observer track dict containing at least "t_mjd_utc".
    nebula_wcs_entry : object or sequence
        WCS entry returned by :func:`build_wcs_for_observer`.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, etc.).
    logger : logging.Logger
        Logger for emitting debug information.

    Returns
    -------
    StarWindowProjection
        A fully populated StarWindowProjection dict for this window,
        with no 'mid' fields in the schema.
    """

    # ---------------------------------------------------------------------
    # 1) Handle missing Gaia windows
    # ---------------------------------------------------------------------

    if gaia_window_or_none is None:
        # No Gaia window at all for this window_index: structurally valid
        # but empty projection, marked as "missing".
        return _make_empty_star_window_projection(
            window_entry=window_entry,
            gaia_status="missing",
            reason="No Gaia window for this window_index",
        )

    # At this point we know we have *some* Gaia window dict (could be ok/error).
    gaia_window = gaia_window_or_none

    # Non-"ok" Gaia status (e.g. TAP error): treat as "error" at the
    # projection level and return an empty projection.
    if gaia_window.get("status") != "ok":
        return _make_empty_star_window_projection(
            window_entry=window_entry,
            gaia_status="error",
            reason=gaia_window.get("error_message") or "Gaia status != 'ok'",
        )

    # Number of rows reported by Gaia for this window.
    n_rows_gaia = int(gaia_window.get("n_rows", 0))

    # "ok_empty" case: query succeeded but returned zero rows.
    if n_rows_gaia == 0:
        return _make_empty_star_window_projection(
            window_entry=window_entry,
            gaia_status="ok_empty",
            reason="Gaia query returned 0 rows",
        )

    # ---------------------------------------------------------------------
    # 2) Optional consistency check on window indices
    # ---------------------------------------------------------------------

    if int(window_entry.get("window_index")) != int(gaia_window.get("window_index")):
        logger.warning(
            "Observer '%s': frames window_index=%s does not match Gaia window_index=%s.",
            obs_name,
            window_entry.get("window_index"),
            gaia_window.get("window_index"),
        )

    # ---------------------------------------------------------------------
    # 3) Extract required Gaia arrays
    # ---------------------------------------------------------------------

    gaia_source_id = np.asarray(gaia_window["gaia_source_id"], dtype=np.int64)
    ra_deg = np.asarray(gaia_window["ra_deg"], dtype=float)
    dec_deg = np.asarray(gaia_window["dec_deg"], dtype=float)
    mag_G = np.asarray(gaia_window["mag_G"], dtype=float)

    n_input = ra_deg.size

    if n_input != n_rows_gaia:
        logger.warning(
            "Observer '%s', window %s: n_input=%d but Gaia n_rows=%d.",
            obs_name,
            window_entry.get("window_index"),
            n_input,
            n_rows_gaia,
        )

    # ---------------------------------------------------------------------
    # 4) Propagate Gaia astrometry to the global observation epoch
    # ---------------------------------------------------------------------
    # This uses NEBULA_TIME_CONFIG.DEFAULT_TIME_WINDOW.start_utc as the
    # observation epoch (unless explicitly overridden inside the helper).

    ra_deg_epoch, dec_deg_epoch = _propagate_gaia_to_window_epoch(
        gaia_window=gaia_window,
        # t_obs_epoch=None -> use DEFAULT_TIME_WINDOW.start_utc
    )

    # ---------------------------------------------------------------------
    # 5) Select the appropriate WCS snapshot for this window
    # ---------------------------------------------------------------------
    # _select_wcs_for_window may internally use window indices and/or
    # any time metadata present in window_entry + obs_track, but this
    # function no longer computes or stores any "mid" fields itself.

    nebula_wcs_window = _select_wcs_for_window(
        nebula_wcs_entry=nebula_wcs_entry,
        obs_track=obs_track,
        window_entry=window_entry,
    )

    # ---------------------------------------------------------------------
    # 6) Convert RA/Dec at the observation epoch to pixel coordinates
    # ---------------------------------------------------------------------

    if project_radec_to_pixels is not None:
        x_pix_epoch, y_pix_epoch = project_radec_to_pixels(
            nebula_wcs_window,
            ra_deg_epoch,
            dec_deg_epoch,
        )
    else:
        if not hasattr(nebula_wcs_window, "world_to_pixel"):
            raise RuntimeError(
                "Neither 'project_radec_to_pixels' nor 'world_to_pixel' is available "
                "for RA/Dec -> pixel projection."
            )
        x_pix_epoch, y_pix_epoch = nebula_wcs_window.world_to_pixel(
            ra_deg_epoch,
            dec_deg_epoch,
        )

    x_pix_epoch = np.asarray(x_pix_epoch, dtype=float)
    y_pix_epoch = np.asarray(y_pix_epoch, dtype=float)

    # ---------------------------------------------------------------------
    # 7) Apply sensor FOV mask and compute star density
    # ---------------------------------------------------------------------

    n_rows = int(getattr(sensor_config, "n_rows", getattr(sensor_config, "rows", 0)))
    n_cols = int(getattr(sensor_config, "n_cols", getattr(sensor_config, "cols", 0)))

    on_detector = (
        (x_pix_epoch >= 0.0)
        & (x_pix_epoch < float(n_cols))
        & (y_pix_epoch >= 0.0)
        & (y_pix_epoch < float(n_rows))
    )

    n_on = int(on_detector.sum())

    query_radius_deg = float(gaia_window.get("query_radius_deg", 0.0))
    area_deg2 = np.pi * (query_radius_deg ** 2)
    density = (n_on / area_deg2) if area_deg2 > 0.0 else None

    # ---------------------------------------------------------------------
    # 8) Extract optional astrometric arrays once (efficiency)
    # ---------------------------------------------------------------------

    pm_ra_arr = pm_dec_arr = parallax_arr = rv_arr = None

    if "pm_ra_masyr" in gaia_window and gaia_window["pm_ra_masyr"] is not None:
        pm_ra_arr = np.asarray(gaia_window["pm_ra_masyr"], dtype=float)

    if "pm_dec_masyr" in gaia_window and gaia_window["pm_dec_masyr"] is not None:
        pm_dec_arr = np.asarray(gaia_window["pm_dec_masyr"], dtype=float)

    if "parallax_mas" in gaia_window and gaia_window["parallax_mas"] is not None:
        parallax_arr = np.asarray(gaia_window["parallax_mas"], dtype=float)

    if "radial_velocity_km_s" in gaia_window and gaia_window["radial_velocity_km_s"] is not None:
        rv_arr = np.asarray(gaia_window["radial_velocity_km_s"], dtype=float)

    # ---------------------------------------------------------------------
    # 9) Build per-star entries for this window (no 'mid' fields)
    # ---------------------------------------------------------------------

    stars: Dict[str, Dict[str, Any]] = {}
    
    for i in range(n_input):
        # Force 64-bit key material to prevent int32 truncation collisions.
        sid_int = int(np.int64(gaia_source_id[i]))
        sid_str = str(sid_int)
    
        # Fail fast if anything still collides (prevents silent overwrites).
        if sid_str in stars:
            raise RuntimeError(
                f"Duplicate Gaia source_id key '{sid_str}' while building stars dict. "
                "This indicates an upstream truncation (e.g., int32 cast) or duplicate Gaia rows."
            )
    
        star_entry: Dict[str, Any] = {
            "gaia_source_id": sid_int,
            "source_id": sid_str,
            "source_type": "star",
            "ra_deg_catalog": float(ra_deg[i]),
            "dec_deg_catalog": float(dec_deg[i]),
            "ra_deg_epoch": float(ra_deg_epoch[i]),
            "dec_deg_epoch": float(dec_deg_epoch[i]),
            "mag_G": float(mag_G[i]),
            "x_pix_epoch": float(x_pix_epoch[i]),
            "y_pix_epoch": float(y_pix_epoch[i]),
            "on_detector": bool(on_detector[i]),
        }
    
        if pm_ra_arr is not None:
            star_entry["pm_ra_masyr"] = float(pm_ra_arr[i])
    
        if pm_dec_arr is not None:
            star_entry["pm_dec_masyr"] = float(pm_dec_arr[i])
    
        if parallax_arr is not None:
            star_entry["parallax_mas"] = float(parallax_arr[i])
    
        if rv_arr is not None:
            star_entry["radial_velocity_km_s"] = float(rv_arr[i])
    
        stars[sid_str] = star_entry


    logger.debug(
        "Observer '%s', window %s: n_input=%d, n_on_detector=%d.",
        obs_name,
        window_entry.get("window_index"),
        n_input,
        n_on,
    )

    # ---------------------------------------------------------------------
    # 10) Assemble the final StarWindowProjection dict for this window
    # ---------------------------------------------------------------------
    # NOTE: No 'mid' time fields are included in this schema. If you still
    #       want to retain any time-like metadata at the window level
    #       (e.g., a representative epoch), you can add new fields with
    #       non-'mid' names here later (e.g. 'window_epoch_mjd_utc').

    projection: StarWindowProjection = {
        # Window indexing and frame range.
        "window_index": int(window_entry.get("window_index")),
        "start_index": int(window_entry.get("start_index")),
        "end_index": int(window_entry.get("end_index")),
        "n_frames": int(window_entry.get("n_frames")),
        # Sky selector metadata (cone center and radius).
        "sky_center_ra_deg": float(window_entry.get("sky_center_ra_deg")),
        "sky_center_dec_deg": float(window_entry.get("sky_center_dec_deg")),
        "sky_radius_deg": float(window_entry.get("sky_radius_deg")),
        "sky_selector_status": window_entry.get("sky_selector_status"),
        # Gaia status and (lack of) error message for a successful window.
        "gaia_status": "ok",
        "gaia_error_message": None,
        # Star counts and density on the detector.
        "n_stars_input": int(n_input),
        "n_stars_on_detector": int(n_on),
        "star_density_on_detector_per_deg2": density,
        # Per-star data keyed by Gaia source ID string.
        "stars": stars,
    }

    return projection


def build_star_projections_for_observer(
    obs_name: str,
    frames_entry: Dict[str, Any],
    gaia_obs_entry: Dict[str, Any],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build star projections for all windows of a single observer.

    For a given observer, this function:

        * Iterates over all windows in the frames-with-sky structure.

        * Uses a lookup dictionary to associate each window_index with
          its corresponding Gaia cones window.

        * Calls :func:`project_gaia_stars_for_window` for each window,
          collecting the resulting StarWindowProjection objects.

        * Maintains per-observer counters:

              - n_windows_total
              - n_windows_with_gaia
              - n_windows_error
              - total_stars_input
              - total_stars_on_detector

          as well as a list of densities for windows with non-None
          star_density_on_detector_per_deg2.

        * Computes density statistics across windows:

              median, p10, p90
        * Tracking-mode gate:

          - Windows with window_entry["tracking_mode"] present and not equal
            to "sidereal" are counted in n_windows_total but are **skipped**
            for projection, and are not appended to the output "windows" list.
          - n_windows_skipped_non_sidereal records how many windows were skipped.


        * Computes the time range covered by windows that have Gaia data,
          using the observer's coarse time grid and the window start/end
          indices (no per-window "mid" times).

        * Builds the final per-observer entry of the form:

              {
                  "observer_name": ...,
                  "rows": ...,
                  "cols": ...,
                  "catalog_name": ...,
                  "catalog_band": ...,
                  "mag_limit_sensor_G": ...,
                  "run_meta": {...},
                  "windows": [StarWindowProjection, ...],
              }

    Parameters
    ----------
    obs_name : str
        Name of the observer.
    frames_entry : dict
        Entry from frames-with-sky keyed by this observer, containing
        rows, cols, dt_frame_s, and a "windows" list.
    gaia_obs_entry : dict
        Gaia cones cache entry keyed by this observer, containing
        catalog metadata and a "windows" list of Gaia windows.
    obs_track : dict
        Observer track dict including "t_mjd_utc" and pointing fields.
    nebula_wcs_entry : object or sequence
        WCS entry returned by :func:`build_wcs_for_observer` for this observer.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, etc.).
    logger : logging.Logger
        Logger for emitting summaries.

    Returns
    -------
    dict
        Per-observer star projection entry as described above.
    """
    # ------------------------------------------------------------------
    # 1) Extract windows and build Gaia lookup by window_index
    # ------------------------------------------------------------------

    windows_frames: List[Dict[str, Any]] = list(frames_entry.get("windows", []))

    gaia_by_idx: Dict[int, Dict[str, Any]] = {
        int(w.get("window_index")): w for w in gaia_obs_entry.get("windows", [])
    }

    # ------------------------------------------------------------------
    # 2) Initialize counters and collections for statistics
    # ------------------------------------------------------------------

    n_windows_total = 0
    n_windows_with_gaia = 0
    n_windows_error = 0
    total_stars_input = 0
    total_stars_on_detector = 0
    n_windows_skipped_non_sidereal = 0


    densities_on_detector: List[float] = []
    window_projections: List[StarWindowProjection] = []

    # Track the coarse index span covered by windows that have Gaia data.
    min_start_idx_with_gaia: Optional[int] = None
    max_end_idx_with_gaia: Optional[int] = None

    # ------------------------------------------------------------------
    # 3) Loop over frames-with-sky windows and project stars per window
    # ------------------------------------------------------------------

    for window_entry in windows_frames:
        n_windows_total += 1
        
        # Skip windows with no targets: this stage should not create outputs for them.
        if int(window_entry.get("n_targets", 0)) <= 0:
            continue
        
        widx = int(window_entry.get("window_index"))

        # --------------------------------------------------------------
        # Tracking-mode gate: STAR_PROJECTION only handles sidereal windows
        # --------------------------------------------------------------
        mode = window_entry.get("tracking_mode", None)
        if mode is not None and str(mode).lower() != "sidereal":
            n_windows_skipped_non_sidereal += 1
            logger.debug(
                "Observer '%s': window_index=%d has tracking_mode=%r -> skipping STAR_PROJECTION.",
                obs_name,
                widx,
                mode,
            )
            continue

        gaia_window = gaia_by_idx.get(widx)

        if gaia_window is not None:
            n_windows_with_gaia += 1

            # Update global coarse-index coverage for Gaia windows.
            w_start = int(window_entry.get("start_index", 0))
            w_end = int(window_entry.get("end_index", w_start))

            if min_start_idx_with_gaia is None or w_start < min_start_idx_with_gaia:
                min_start_idx_with_gaia = w_start
            if max_end_idx_with_gaia is None or w_end > max_end_idx_with_gaia:
                max_end_idx_with_gaia = w_end

        projection = project_gaia_stars_for_window(
            obs_name=obs_name,
            window_entry=window_entry,
            gaia_window_or_none=gaia_window,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )

        window_projections.append(projection)

        total_stars_input += int(projection.get("n_stars_input", 0))
        total_stars_on_detector += int(projection.get("n_stars_on_detector", 0))

        density_val = projection.get("star_density_on_detector_per_deg2")
        if density_val is not None:
            densities_on_detector.append(float(density_val))

        if projection.get("gaia_status") in {"error", "missing"}:
            n_windows_error += 1


    # ------------------------------------------------------------------
    # 4) Compute density statistics across windows (if available)
    # ------------------------------------------------------------------

    if densities_on_detector:
        dens_arr = np.asarray(densities_on_detector, dtype=float)
        dens_median = float(np.median(dens_arr))
        dens_p10 = float(np.percentile(dens_arr, 10.0))
        dens_p90 = float(np.percentile(dens_arr, 90.0))
    else:
        dens_median = None
        dens_p10 = None
        dens_p90 = None

    # ------------------------------------------------------------------
    # 5) Compute time range for windows that have Gaia entries
    # ------------------------------------------------------------------

    if "t_mjd_utc" in obs_track and obs_track["t_mjd_utc"] is not None:
        t_grid_mjd = obs_track["t_mjd_utc"]
    else:
        t_grid_mjd = obs_track.get("t_mjd", None)
    time_range_utc: Tuple[Optional[str], Optional[str]]

    if (
        n_windows_with_gaia > 0
        and t_grid_mjd is not None
        and min_start_idx_with_gaia is not None
        and max_end_idx_with_gaia is not None
    ):
        t_grid_mjd_arr = np.asarray(t_grid_mjd, dtype=float)

        # Clamp indices to the valid range of the time grid.
        n_times = t_grid_mjd_arr.size
        i_min = max(0, min(min_start_idx_with_gaia, n_times - 1))
        i_max = max(0, min(max_end_idx_with_gaia, n_times - 1))

        t_arr = Time(
            [t_grid_mjd_arr[i_min], t_grid_mjd_arr[i_max]],
            format="mjd",
            scale="utc",
        )
        t_min_str = t_arr[0].iso
        t_max_str = t_arr[1].iso
        time_range_utc = (t_min_str, t_max_str)
    else:
        time_range_utc = (None, None)

    # ------------------------------------------------------------------
    # 6) Extract sensor and catalog metadata for this observer
    # ------------------------------------------------------------------

    rows = int(frames_entry.get("rows", getattr(sensor_config, "rows", 0)))
    cols = int(frames_entry.get("cols", getattr(sensor_config, "cols", 0)))

    catalog_name = gaia_obs_entry.get("catalog_name", NEBULA_STAR_CATALOG.name)
    catalog_band = gaia_obs_entry.get("band", "G")
    mag_limit_sensor_G = float(gaia_obs_entry.get("mag_limit_sensor_G", np.nan))

    # ------------------------------------------------------------------
    # 7) Assemble run_meta summary for this observer
    # ------------------------------------------------------------------

    run_meta = {
        "version": STAR_PROJECTION_RUN_META_VERSION,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frames_source_file": None,
        "gaia_cones_file": None,
        "observer_tracks_file": None,
        "time_range_utc": time_range_utc,
        "n_windows_total": n_windows_total,
        "n_windows_with_gaia": n_windows_with_gaia,
        "n_windows_error": n_windows_error,
        "n_windows_skipped_non_sidereal": n_windows_skipped_non_sidereal,
        "total_stars_input": total_stars_input,
        "total_stars_on_detector": total_stars_on_detector,
        "density_stats_on_detector": {
            "per_deg2_median": dens_median,
            "per_deg2_p10": dens_p10,
            "per_deg2_p90": dens_p90,
        },
    }


    # ------------------------------------------------------------------
    # 8) Build and return the final per-observer entry
    # ------------------------------------------------------------------

    obs_star_entry: Dict[str, Any] = {
        "observer_name": obs_name,
        "rows": rows,
        "cols": cols,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "mag_limit_sensor_G": mag_limit_sensor_G,
        "run_meta": run_meta,
        "windows": window_projections,
    }

    return obs_star_entry


def build_star_projections_for_all_observers(
    frames_with_sky: Dict[str, Dict[str, Any]],
    gaia_cache: Dict[str, Dict[str, Any]],
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
    frames_source_file: Optional[str] = None,
    gaia_cones_file: Optional[str] = None,
    observer_tracks_file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build star projections for all observers.

    This function orchestrates the per-observer star projection flow:

        * Builds WCS entries for all observers using
          :func:`_build_wcs_for_all_observers`.

        * Iterates over all observers present in frames_with_sky.

        * For each observer:

            - Checks whether the observer also exists in gaia_cache
              and obs_tracks.

            - If either is missing, logs a warning and skips the observer.

            - Otherwise, calls :func:`build_star_projections_for_observer`
              to obtain the per-observer star projection entry.

            - Inserts the entry into the obs_star_projections dict.

        * After constructing each per-observer entry, this function also
          fills in the run_meta["frames_source_file"], run_meta["gaia_cones_file"],
          and run_meta["observer_tracks_file"] fields using the paths passed
          from :func:`main`.

    Parameters
    ----------
    frames_with_sky : dict
        Dictionary keyed by observer name, as loaded from
        obs_target_frames_ranked_with_sky.pkl.
    gaia_cache : dict
        Dictionary keyed by observer name, as loaded from obs_gaia_cones.pkl.
    obs_tracks : dict
        Dictionary keyed by observer name, as loaded from the pixel pipeline
        observer tracks product (observer_tracks_with_pixels.pkl by default via
        NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH). The track must contain the
        time grid ('t_mjd_utc' or 't_mjd') and pointing fields required by NEBULA_WCS.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, etc.).
    logger : logging.Logger
        Logger for emitting summaries.
    frames_source_file : str or None, optional
        Path to the frames-with-sky source file for run_meta embedding.
    gaia_cones_file : str or None, optional
        Path to the Gaia cones source file for run_meta embedding.
    observer_tracks_file : str or None, optional
        Path to the observer tracks source file for run_meta embedding.

    Returns
    -------
    dict
        obs_star_projections dict keyed by observer name.
    """
    # ------------------------------------------------------------------
    # 1) Build WCS entries for all observers using their tracks
    # ------------------------------------------------------------------
    # The WCS builder derives NebulaWCS objects (static or time-varying)
    # from each observer's track and the active sensor configuration.
    wcs_map = _build_wcs_for_all_observers(
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
    )

    # Initialize the final mapping from observer name -> per-observer entry.
    obs_star_projections: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 2) Loop over observers that have frames-with-sky entries
    # ------------------------------------------------------------------
    for obs_name, frames_entry in frames_with_sky.items():
        # Ensure this observer has a Gaia cache entry; otherwise we cannot
        # project stars for its windows, so we log and skip.
        if obs_name not in gaia_cache:
            logger.warning(
                "Observer '%s' is present in frames_with_sky but not in gaia_cache; skipping.",
                obs_name,
            )
            continue

        # Ensure this observer has an associated track (and thus pointing).
        if obs_name not in obs_tracks:
            logger.warning(
                "Observer '%s' is present in frames_with_sky but not in obs_tracks; skipping.",
                obs_name,
            )
            continue

        # It is expected that wcs_map has the same keys as obs_tracks, but
        # we guard against mismatches for robustness.
        if obs_name not in wcs_map:
            logger.warning(
                "Observer '%s' is present in obs_tracks but missing from WCS map; skipping.",
                obs_name,
            )
            continue

        # Retrieve the Gaia cones entry, observer track, and WCS entry.
        gaia_obs_entry = gaia_cache[obs_name]
        obs_track = obs_tracks[obs_name]
        nebula_wcs_entry = wcs_map[obs_name]

        # ------------------------------------------------------------------
        # 3) Build per-observer star projections via helper
        # ------------------------------------------------------------------
        # This call iterates over windows, performs per-window projections,
        # and returns a single summary dict for this observer.
        obs_star_entry = build_star_projections_for_observer(
            obs_name=obs_name,
            frames_entry=frames_entry,
            gaia_obs_entry=gaia_obs_entry,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )

        # ------------------------------------------------------------------
        # 4) Patch provenance (source file paths) into run_meta
        # ------------------------------------------------------------------
        # Pull out run_meta (created by build_star_projections_for_observer)
        # and fill in the file paths used to load inputs.
        run_meta = obs_star_entry.get("run_meta", {})
        run_meta["frames_source_file"] = frames_source_file
        run_meta["gaia_cones_file"] = gaia_cones_file
        run_meta["observer_tracks_file"] = observer_tracks_file
        obs_star_entry["run_meta"] = run_meta

        # Store this observer's star projections in the final dict.
        obs_star_projections[obs_name] = obs_star_entry

    # ------------------------------------------------------------------
    # 5) Return star projections for all observers
    # ------------------------------------------------------------------
    return obs_star_projections


def main(
    sensor_config: Optional[SensorConfig] = None,
    frames_path: Optional[str] = None,
    gaia_cache_path: Optional[str] = None,
    obs_tracks_path: Optional[str] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Entry point for NEBULA_STAR_PROJECTION.

    This function ties together the star-projection stage for the entire
    simulation. It:

        1. Resolves the active sensor configuration (defaults to
           Configuration.NEBULA_SENSOR_CONFIG.ACTIVE_SENSOR if not
           provided).

    2) Loads the three required inputs from disk:

          * frames_with_sky:
                obs_target_frames_ranked_with_sky.pkl

          * gaia_cache:
                obs_gaia_cones.pkl

          * obs_tracks:
                observer_tracks_with_pixels.pkl (default via NEBULA_PIXEL_PICKLER)

       using the helper wrappers and NEBULA_QUERY_GAIA where appropriate.

    3) Calls build_star_projections_for_all_observers to:

          * Build WCS objects for each observer (via NEBULA_WCS).
          * Project Gaia stars into pixel coordinates for each observer/window
            using an index-selected WCS snapshot per window.
          * Optionally propagate Gaia astrometry to a single global observation
            epoch (DEFAULT_TIME_WINDOW.start_utc by default).
          * Compute per-window and per-observer star statistics.
          * Embed basic provenance paths into run_meta.

        4. Writes the resulting obs_star_projections dict to disk using
           :func:`_save_star_projection_cache` and returns the same dict.

    Parameters
    ----------
    sensor_config : SensorConfig or None, optional
        Sensor configuration to use for projection. If None, the function
        uses ACTIVE_SENSOR imported from Configuration.NEBULA_SENSOR_CONFIG.
        If both are unavailable, a RuntimeError is raised.
    frames_path : str or None, optional
        Path to obs_target_frames_ranked_with_sky.pkl. If None, a default
        is computed via :func:`_resolve_default_frames_path`. The effective
        path is recorded in run_meta["frames_source_file"].
    gaia_cache_path : str or None, optional
        Location of the Gaia cones cache. Two usage patterns are supported:

            * If None:
                  The function uses NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR and
                  NEBULA_STAR_CATALOG.name to infer the default STARS directory
                  and assumes the file name "obs_gaia_cones.pkl".

            * If a directory:
                  Treated as the STARS directory; the file is assumed to be
                  "<gaia_cache_path>/obs_gaia_cones.pkl".

            * If a file path:
                  Treated as the full path to the Gaia cache pickle. In this
                  case, the file is loaded directly with pickle rather than
                  via NEBULA_QUERY_GAIA.load_gaia_cache (to avoid assumptions
                  about file naming).

        The effective file path is recorded in
        run_meta["gaia_cones_file"].
    obs_tracks_path : str or None, optional
        Path to the observer tracks pickle used for WCS construction. If None,
        a default is computed via _resolve_default_obs_tracks_path, which returns
        NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH (typically observer_tracks_with_pixels.pkl).
        The effective path is recorded in run_meta["observer_tracks_file"].
    output_path : str or None, optional
        Path where obs_star_projections.pkl should be written. If None,
        a default is computed via :func:`_resolve_default_output_path`.
        The chosen path is logged but not embedded into run_meta.
    logger : logging.Logger or None, optional
        Logger to use. If None, a simple module-level logger named
        "NEBULA_STAR_PROJECTION" is created.

    Returns
    -------
    dict
        obs_star_projections dictionary keyed by observer name. Each entry
        is the per-observer structure returned by
        :func:`build_star_projections_for_observer`, with run_meta fields
        patched to include the source file paths.
    """
    # --------------------------------------------------------------
    # 1) Initialize logger if the caller did not supply one
    # --------------------------------------------------------------
    if logger is None:
        # Get/create a logger specific to this module.
        logger = logging.getLogger("NEBULA_STAR_PROJECTION")

        # If no handlers exist yet, configure a basic stream handler.
        if not logger.handlers:
            # Create a handler that writes to stderr.
            handler = logging.StreamHandler()
            # Define a simple log format (time, level, name, message).
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            # Attach the formatter to the handler.
            handler.setFormatter(formatter)
            # Add the handler to the logger.
            logger.addHandler(handler)

        # Set a reasonable default logging level.
        logger.setLevel(logging.INFO)

    # --------------------------------------------------------------
    # 2) Resolve the active sensor configuration
    # --------------------------------------------------------------
    if sensor_config is None:
        # If no explicit sensor_config was passed, fall back to ACTIVE_SENSOR.
        if ACTIVE_SENSOR is None:
            # Without a sensor, we cannot build WCS or project stars.
            raise RuntimeError(
                "NEBULA_STAR_PROJECTION: no sensor_config provided and "
                "ACTIVE_SENSOR is None. Please supply a SensorConfig or "
                "define ACTIVE_SENSOR in NEBULA_SENSOR_CONFIG."
            )
        # Use the globally configured active sensor.
        sensor_config = ACTIVE_SENSOR

    # Log a brief summary of the sensor being used.
    logger.info(
        "NEBULA_STAR_PROJECTION: using sensor '%s' (%d x %d pixels).",
        getattr(sensor_config, "name", "<unknown>"),
        int(getattr(sensor_config, "rows", getattr(sensor_config, "n_rows", 0))),
        int(getattr(sensor_config, "cols", getattr(sensor_config, "n_cols", 0))),
    )

    # --------------------------------------------------------------
    # 3) Resolve and load frames_with_sky
    # --------------------------------------------------------------
    # If no frames_path was provided, compute a default using the helper.
    if frames_path is None:
        frames_path = _resolve_default_frames_path()
    # Normalize to an absolute path for provenance.
    frames_source_file = os.path.abspath(frames_path)

    # Use the existing wrapper (which delegates to NEBULA_TARGET_PHOTONS)
    # to load obs_target_frames_ranked_with_sky.pkl.
    frames_with_sky = _load_frames_with_sky_from_disk(
        frames_path=frames_source_file,
        logger=logger,
    )

    # --------------------------------------------------------------
    # 4) Resolve and load Gaia cones cache (gaia_cache)
    # --------------------------------------------------------------
    use_query_gaia_loader = False  # whether to call NEBULA_QUERY_GAIA.load_gaia_cache

    if gaia_cache_path is None:
        # No path provided: infer default STARS directory directly
        # from NEBULA_OUTPUT_DIR and the catalog name.
        catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
        stars_dir = os.path.join(NEBULA_OUTPUT_DIR, "STARS", catalog_name)

        # The Gaia cache file is assumed to be named obs_gaia_cones.pkl.
        gaia_cones_file = os.path.join(stars_dir, "obs_gaia_cones.pkl")

        # In this default case, we can safely use NEBULA_QUERY_GAIA.load_gaia_cache.
        use_query_gaia_loader = True

    else:
        # A path was provided. It may be a directory (STARS dir) or a file.
        if os.path.isdir(gaia_cache_path):
            # Treat the argument as the STARS directory.
            stars_dir = gaia_cache_path
            # Assume standard file name inside that directory.
            gaia_cones_file = os.path.join(stars_dir, "obs_gaia_cones.pkl")
            # We can use NEBULA_QUERY_GAIA.load_gaia_cache in this case.
            use_query_gaia_loader = True
        else:
            # Treat the argument as the full file path to the Gaia cache.
            gaia_cones_file = gaia_cache_path
            # Derive a directory for logging/debugging only.
            stars_dir = os.path.dirname(gaia_cones_file) or "."
            # Since the file name may be non-standard, we will load it
            # directly with pickle instead of using load_gaia_cache().
            use_query_gaia_loader = False

    # Normalize Gaia cones file path for provenance.
    gaia_cones_file = os.path.abspath(gaia_cones_file)

    # Now load the gaia_cache using NEBULA_QUERY_GAIA if available, or
    # fall back to a direct pickle load.
    if use_query_gaia_loader and (NEBULA_QUERY_GAIA is not None) and hasattr(
        NEBULA_QUERY_GAIA, "load_gaia_cache"
    ):
        # Use the dedicated loader from NEBULA_QUERY_GAIA, which performs
        # schema and catalog consistency checks.
        gaia_cache = NEBULA_QUERY_GAIA.load_gaia_cache(
            stars_dir=stars_dir,
            logger=logger,
        )
    else:
        # Fall back to a simple pickle load for the Gaia cache file.
        if not os.path.exists(gaia_cones_file):
            raise FileNotFoundError(
                f"NEBULA_STAR_PROJECTION: Gaia cache file not found: {gaia_cones_file}"
            )

        logger.info(
            "NEBULA_STAR_PROJECTION: loading Gaia cache directly from '%s'.",
            gaia_cones_file,
        )
        with open(gaia_cones_file, "rb") as f:
            gaia_cache = pickle.load(f)

    # --------------------------------------------------------------
    # 5) Resolve and load observer_tracks_with_pointing
    # --------------------------------------------------------------
    # If no path was provided, compute a default using the helper.
    if obs_tracks_path is None:
        obs_tracks_path = _resolve_default_obs_tracks_path()
    # Normalize to an absolute path for provenance.
    observer_tracks_file = os.path.abspath(obs_tracks_path)

    # Use the existing wrapper (which delegates to your pointing pickler)
    # to load observer_tracks_with_pointing.pkl.
    obs_tracks = _load_observer_tracks_with_pointing_from_disk(
        obs_tracks_path=observer_tracks_file,
        logger=logger,
    )

    # --------------------------------------------------------------
    # 6) Build star projections for all observers
    # --------------------------------------------------------------
    # This call:
    #   * Builds WCS objects via _build_wcs_for_all_observers / NEBULA_WCS.
    #   * Iterates over observers and windows.
    #   * Projects Gaia stars onto the sensor for each window.
    #   * Computes per-window and per-observer star statistics.
    #   * Embeds the three source file paths into run_meta for each observer.
    obs_star_projections = build_star_projections_for_all_observers(
        frames_with_sky=frames_with_sky,
        gaia_cache=gaia_cache,
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
        frames_source_file=frames_source_file,
        gaia_cones_file=gaia_cones_file,
        observer_tracks_file=observer_tracks_file,
    )

    # --------------------------------------------------------------
    # 7) Save obs_star_projections to disk
    # --------------------------------------------------------------
    # Write the obs_star_projections dictionary to disk, computing a
    # default output path if needed. The helper logs a summary including
    # observers, windows, and on-detector star counts.
    final_output_path = _save_star_projection_cache(
        obs_star_projections=obs_star_projections,
        output_path=output_path,
        logger=logger,
    )

    # Log where the final star projection cache was written.
    logger.info(
        "NEBULA_STAR_PROJECTION: star projections written to '%s'.",
        final_output_path,
    )

    # --------------------------------------------------------------
    # 8) Return the in-memory obs_star_projections dict
    # --------------------------------------------------------------
    return obs_star_projections


