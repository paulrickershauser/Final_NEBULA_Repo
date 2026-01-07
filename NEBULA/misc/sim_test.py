# sim_test.py
# ---------------------------------------------------------------------------
# High-level driver script to run the NEBULA pipeline through pixel projection
# and inspect basic summary info about the resulting tracks.
# ---------------------------------------------------------------------------

"""
sim_test
========

This script is a simple entry point for exercising the NEBULA pipeline
up through the pixel layer. It:

    1. Configures a basic logger for console output.

    2. Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(), which
       internally cascades through the upstream picklers:

           - NEBULA_FLUX_PICKLER
           - NEBULA_LOS_FLUX_PICKLER
           - NEBULA_SCHEDULE_PICKLER
           - NEBULA_ICRS_PAIR_PICKLER
           - NEBULA_SENSOR_PROJECTION (via NEBULA_PIXEL_PICKLER)

       and ensures that both observer_tracks_with_pixels.pkl and
       target_tracks_with_pixels.pkl are up to date.

    3. Prints a brief summary of the number of observers and targets,
       and shows example per-observer keys for one target. This gives a
       quick check that the pixel-level fields (pix_x, pix_y,
       on_detector, etc.) are present and shaped correctly.

Usage
-----

From your NEBULA root directory in Spyder or a terminal:

    %run sim_test.py

or

    python sim_test.py

You can control whether to force a full recompute of all upstream and
pixel pickles by toggling the FORCE_RECOMPUTE flag below.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np
import pickle
# ---------------------------------------------------------------------------
# NEBULA imports
# ---------------------------------------------------------------------------


# Import the sensor configuration (if you want to override the default).
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR, SensorConfig
from Configuration import NEBULA_PATH_CONFIG

# Import the pixel pickler that runs the full chain and attaches pixel data.
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER

# Photon-domain per-target time series + frames
from Utility.FRAMES import NEBULA_TARGET_PHOTONS as NTP

# Sky footprints (attach RA/Dec/radius to TARGET_PHOTON_FRAMES windows)
from Utility.STARS import NEBULA_SKY_SELECTOR as NSS

# Gaia cone queries over those sky footprints
from Utility.STARS import NEBULA_QUERY_GAIA as NQG

# ----------------------------------------------------------------------
# Star-field utilities: projections + photon time series
# ----------------------------------------------------------------------
from Utility.STARS import NEBULA_STAR_PROJECTION
from Utility.STARS import NEBULA_STAR_SLEW_PROJECTION
from Utility.STARS import NEBULA_STAR_PHOTONS
# Photon-domain per-frame frame builder (canonical frame grid)
from Utility.FRAMES import NEBULA_PHOTON_FRAME_BUILDER as PFB
from Utility.ANIMATION import NEBULA_OBS_TAR_STAR_ANIMATION as NOA

# ---------------------------------------------------------------------------
# Configuration flags
# ---------------------------------------------------------------------------

# Flag that controls whether to recompute the entire upstream and pixel
# pipeline. Set to True if you have changed core code and want to
# regenerate all pickles. Set to False to reuse existing pickles.
FORCE_RECOMPUTE: bool = False

# Flag that controls whether to build photon-domain per-target time series
# and save obs_target_frames pickles (raw + ranked) via NEBULA_TARGET_PHOTONS.
BUILD_TARGET_PHOTON_FRAMES: bool = True

# Flag that controls whether to run the sky-footprint and Gaia catalog
# pipeline after TARGET_PHOTON_FRAMES are available. If True, this will:
#   1) Call NEBULA_SKY_SELECTOR.main(logger=...)
#   2) Call NEBULA_QUERY_GAIA.main(mag_limit_sensor_G=None, logger=...)
RUN_GAIA_PIPELINE: bool = False
# Control whether to run the star-field pipeline (projection + photons)
# RUN_STAR_PIPELINE: bool = True



# ---------------------------------------------------------------------------
# Helper to configure logging
# ---------------------------------------------------------------------------

def configure_logging() -> logging.Logger:
    """
    Configure and return a logger for the sim_test script.

    Returns
    -------
    logger : logging.Logger
        Logger instance configured to log INFO-level messages to the
        console with a simple timestamped format.
    """
    # Get a logger specific to this script using its module name.
    logger = logging.getLogger("sim_test")

    # If the logger has no handlers yet, configure a basic stream handler.
    if not logger.handlers:
        # Create a stream handler that writes log messages to stderr.
        handler = logging.StreamHandler()
        # Define a simple format with time, name, level, and message.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Attach the formatter to the handler so messages are formatted.
        handler.setFormatter(formatter)
        # Add the handler to the logger so it becomes active.
        logger.addHandler(handler)
        # Set the logger to INFO level to show standard progress messages.
        logger.setLevel(logging.INFO)

    # Return the configured logger to the caller.
    return logger


# ---------------------------------------------------------------------------
# Simple summary helpers
# ---------------------------------------------------------------------------

def summarize_tracks(
    obs_tracks: Dict[str, Any],
    tar_tracks: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Print a brief summary of observer and target tracks, focusing on
    pixel-level fields for a quick sanity check.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary mapping observer names to observer track dictionaries.
    tar_tracks : dict
        Dictionary mapping target names to target track dictionaries.
    logger : logging.Logger
        Logger used to emit summary information.
    """
    # Log the number of observers that were processed.
    logger.info("Number of observers: %d", len(obs_tracks))
    # Log the number of targets that were processed.
    logger.info("Number of targets:   %d", len(tar_tracks))

    # If there are no targets at all, there is nothing more to summarize.
    if not tar_tracks:
        logger.warning("No targets found in tar_tracks; nothing to summarize.")
        return

    # Pick an arbitrary target name (e.g., the first key) for inspection.
    example_tar_name = next(iter(tar_tracks.keys()))
    # Retrieve the corresponding target track dictionary.
    example_tar_track = tar_tracks[example_tar_name]

    # Log which target we are using as an example.
    logger.info("Example target: '%s'", example_tar_name)

    # Extract the per-observer dictionary for this example target, if present.
    by_observer = example_tar_track.get("by_observer", None)

    # If this target has no by_observer entry, log and return early.
    if not by_observer:
        logger.warning(
            "Example target '%s' has no by_observer entry; "
            "no per-observer pixel fields to summarize.",
            example_tar_name,
        )
        return

    # Pick an arbitrary observer name that views this example target.
    example_obs_name = next(iter(by_observer.keys()))
    # Retrieve the corresponding per-observer sub-dictionary.
    example_by_obs = by_observer[example_obs_name]

    # Log which observer we are using for per-observer field inspection.
    logger.info("Example observer for that target: '%s'", example_obs_name)

    # Build a sorted list of keys for this observer–target pair.
    obs_keys = sorted(example_by_obs.keys())
    # Log the available keys so we can verify the presence of pixel fields.
    logger.info(
        "Per-observer keys for [target='%s', observer='%s']:\n  %s",
        example_tar_name,
        example_obs_name,
        ", ".join(obs_keys),
    )

    # Try to log the lengths of the main pixel fields if they exist.
    for field in ("pix_x", "pix_y", "on_detector", "on_detector_visible_sunlit"):
        # Check if this field is present for the example observer–target pair.
        if field in example_by_obs:
            # Attempt to get the length of the array (works for lists/ndarrays).
            try:
                field_len = len(example_by_obs[field])
            except TypeError:
                field_len = -1  # Use -1 if length cannot be determined.
            # Log the field name and its length.
            logger.info("  Field '%s' length: %d", field, field_len)
        else:
            # If the field is not present, log that it is missing.
            logger.info("  Field '%s' not present for this pair.", field)

def annotate_windows_with_tracking_mode(
    obs_tracks: Dict[str, Any],
    ranked_target_frames: Dict[str, Any],
    pixel_scale_rad: float,
    pix_threshold: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Annotate each window in ranked_target_frames with a tracking mode.

    This is a STRICT / FAIL-HARD helper.

    For each observer and each window, this function:
        1. Reads boresight pointing from obs_tracks (preferring
           'pointing_boresight_ra_deg' / 'pointing_boresight_dec_deg',
           falling back to 'boresight_ra_deg' / 'boresight_dec_deg').
        2. Converts RA/Dec time series to unit vectors on the celestial sphere.
        3. Computes angular motion between consecutive samples.
        4. Converts that angular motion to pixel motion using pixel_scale_rad.
        5. For each window, examines the per-step pixel drift over coarse indices
           [start_index .. end_index] and sets:

               window["tracking_mode"] = "sidereal"
                   if max_step_pix < pix_threshold
               window["tracking_mode"] = "slew"
                   otherwise

           It also stores:
               window["total_drift_pix"] : float
                   Sum of per-step drifts (integrated drift over the window).
               window["max_step_pix"]    : float
                   Maximum per-step drift within the window.

    STRICT behavior:
        - Any missing or malformed data that prevents a reliable classification
          results in a RuntimeError. This includes:
              * missing obs_track for an observer
              * missing or empty RA/Dec arrays
              * RA/Dec length mismatch
              * fewer than 2 coarse samples
              * missing start_index / end_index
              * out-of-bounds indices
              * end_index <= start_index
              * empty drift slice for a window

    Parameters
    ----------
    obs_tracks : dict
        Per-observer track dictionaries containing boresight pointing arrays.
    ranked_target_frames : dict
        Per-observer frames-with-windows structure produced by
        NEBULA_TARGET_PHOTONS / NEBULA_PHOTON_FRAME_BUILDER.
        This function modifies each window entry in-place.
    pixel_scale_rad : float
        Sensor plate scale [radians per pixel]. Must be > 0.
    pix_threshold : float
        Threshold on per-frame drift in pixels. If max_step_pix is less than
        this value, the window is classified as 'sidereal'; otherwise 'slew'.
    logger : logging.Logger, optional
        Logger for informational messages. If None, a default 'sim_test' logger
        is used.

    Raises
    ------
    RuntimeError
        If any observer or window has inconsistent or insufficient data to
        compute tracking mode.
    ValueError
        If pixel_scale_rad is not positive.
    """
    if logger is None:
        logger = logging.getLogger("sim_test")

    if pixel_scale_rad <= 0.0:
        raise ValueError(
            f"annotate_windows_with_tracking_mode: pixel_scale_rad must be "
            f"positive, got {pixel_scale_rad!r}."
        )

    for obs_name, frames_entry in ranked_target_frames.items():
        obs_track = obs_tracks.get(obs_name)
        if obs_track is None:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: no obs_track for "
                f"observer '{obs_name}'."
            )

        # --- Safe selection of RA / Dec arrays (no boolean `or` on numpy arrays) ---
        if "pointing_boresight_ra_deg" in obs_track:
            ra_data = obs_track["pointing_boresight_ra_deg"]
        elif "boresight_ra_deg" in obs_track:
            ra_data = obs_track["boresight_ra_deg"]
        else:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: observer '{obs_name}' "
                f"has no 'pointing_boresight_ra_deg' or 'boresight_ra_deg'."
            )

        if "pointing_boresight_dec_deg" in obs_track:
            dec_data = obs_track["pointing_boresight_dec_deg"]
        elif "boresight_dec_deg" in obs_track:
            dec_data = obs_track["boresight_dec_deg"]
        else:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: observer '{obs_name}' "
                f"has no 'pointing_boresight_dec_deg' or 'boresight_dec_deg'."
            )

        ra_deg = np.asarray(ra_data, dtype=float)
        dec_deg = np.asarray(dec_data, dtype=float)

        if ra_deg.size == 0 or dec_deg.size == 0:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: empty RA/Dec arrays "
                f"for observer '{obs_name}'."
            )

        if ra_deg.shape != dec_deg.shape:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: RA/Dec length mismatch "
                f"for observer '{obs_name}' (ra={ra_deg.size}, dec={dec_deg.size})."
            )

        # --- Convert boresight RA/Dec to unit vectors on the celestial sphere ---
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)

        cos_dec = np.cos(dec_rad)
        bx = np.cos(ra_rad) * cos_dec
        by = np.sin(ra_rad) * cos_dec
        bz = np.sin(dec_rad)
        b = np.stack((bx, by, bz), axis=1)  # shape (N, 3), N coarse samples

        if b.shape[0] < 2:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: fewer than 2 time samples "
                f"for observer '{obs_name}'; cannot compute drift."
            )

        # Angular motion between consecutive coarse samples: Δθ_j = arccos(b_j · b_{j+1})
        dot = np.einsum("ij,ij->i", b[:-1], b[1:])
        dot = np.clip(dot, -1.0, 1.0)  # numerical safety
        delta_theta_rad = np.arccos(dot)  # shape (N-1,)

        # Convert angular motion to approximate pixel motion using the plate scale.
        # Assumes pixel_scale_rad is [radians per pixel].
        delta_pix = delta_theta_rad / pixel_scale_rad  # pixels per step

        windows = frames_entry.get("windows", [])
        for w in windows:
            start_idx = w.get("start_index")
            end_idx = w.get("end_index")
            w_index = w.get("window_index", None)

            if start_idx is None or end_idx is None:
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"for observer '{obs_name}' missing start_index or end_index."
                )

            if not (0 <= start_idx < ra_deg.size) or not (0 <= end_idx < ra_deg.size):
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"indices [{start_idx}..{end_idx}] out of bounds for observer "
                    f"'{obs_name}' (N={ra_deg.size})."
                )

            if end_idx <= start_idx:
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"for observer '{obs_name}' has non-increasing indices "
                    f"[start_index={start_idx}, end_index={end_idx}]."
                )

            # delta_pix[j] is the drift from coarse index j -> j+1.
            # For a window covering coarse indices [start_idx..end_idx], we want
            # steps j = start_idx .. (end_idx-1) inclusive, which is exactly
            # delta_pix[start_idx:end_idx].
            step_slice = slice(start_idx, end_idx)
            window_delta = delta_pix[step_slice]

            if window_delta.size == 0:
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"for observer '{obs_name}' produced an empty drift slice "
                    f"from indices [{start_idx}..{end_idx}]."
                )

            total_drift_pix = float(np.sum(window_delta))
            max_step_pix = float(np.max(window_delta))

            # Use per-step drift to decide sidereal vs slew.
            # Rationale: even if tiny numerical noise accumulates over a long window,
            # what matters physically is whether stars smear significantly between frames.
            mode = "sidereal" if max_step_pix < pix_threshold else "slew"

            w["tracking_mode"] = mode
            w["total_drift_pix"] = total_drift_pix
            w["max_step_pix"] = max_step_pix

        logger.info(
            "annotate_windows_with_tracking_mode: observer '%s' -> annotated %d windows.",
            obs_name,
            len(windows),
        )

def dispatch_star_projection_pipeline(
    ranked_target_frames: Dict[str, Dict[str, Any]],
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Decide which star-projection pipelines to run (sidereal and/or slew)
    based on the per-window 'tracking_mode' annotations.

    This helper inspects the annotated ranked_target_frames structure
    and determines whether any windows are classified as:

        - tracking_mode == "sidereal"
        - tracking_mode == "slew"

    It then:

        * Calls NEBULA_STAR_PROJECTION.main(...) if there is at least
          one sidereal window (for any observer).

        * Calls NEBULA_STAR_SLEW_PROJECTION.main(...) if there is at
          least one slewing window (for any observer).

    If no 'tracking_mode' annotations are present in the loaded
    ranked_target_frames (e.g., older pickles), this function will
    first call annotate_windows_with_tracking_mode(...) to add them
    in-place using the current obs_tracks and sensor_config, and will
    write the updated structure back to
    obs_target_frames_ranked_with_sky.pkl for future runs.

    Parameters
    ----------
    ranked_target_frames : dict
        The culled and ranked TARGET_PHOTON_FRAMES structure, with each
        window ideally annotated with a 'tracking_mode' field by
        annotate_windows_with_tracking_mode().
    obs_tracks : dict
        Observer tracks (with pointing), as returned by
        NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(...).
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, etc.).
    logger : logging.Logger
        Logger for status and summary messages.

    Returns
    -------
    obs_star_projections : dict or None
        Output of NEBULA_STAR_PROJECTION.main(...) if it was run; otherwise None.
    obs_star_slew_projections : dict or None
        Output of NEBULA_STAR_SLEW_PROJECTION.main(...) if it was run; otherwise None.
    """

    def _scan_tracking_modes(
        frames: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, bool, bool]:
        """
        Scan frames and report:

            has_sidereal : whether any window has tracking_mode == "sidereal"
            has_slew     : whether any window has tracking_mode == "slew"
            any_flag     : whether any window has *any* tracking_mode key
        """
        has_sidereal = False
        has_slew = False
        any_flag = False

        for obs_name, obs_entry in frames.items():
            for w in obs_entry.get("windows", []):
                mode = w.get("tracking_mode", None)
                if mode is not None:
                    any_flag = True
                    if mode == "sidereal":
                        has_sidereal = True
                    elif mode == "slew":
                        has_slew = True

        return has_sidereal, has_slew, any_flag

    # ------------------------------------------------------------------
    # 1) First scan: do we already have tracking_mode annotations?
    # ------------------------------------------------------------------
    has_sidereal, has_slew, any_tracking_flag = _scan_tracking_modes(
        ranked_target_frames
    )

    if not any_tracking_flag:
        logger.info(
            "Star-projection dispatcher: no 'tracking_mode' annotations found in "
            "ranked_target_frames; annotating now using obs_tracks + sensor_config."
        )

        # Reconstruct the per-frame drift threshold (same logic as in main).
        pixel_scale_rad = sensor_config.pixel_scale_rad
        psf_fwhm_pix = sensor_config.psf_fwhm_pix or 1.0
        pix_threshold = 0.3 * psf_fwhm_pix

        # Annotate in-place; this is cheap and does NOT recompute any ICRS geometry.
        annotate_windows_with_tracking_mode(
            obs_tracks=obs_tracks,
            ranked_target_frames=ranked_target_frames,
            pixel_scale_rad=pixel_scale_rad,
            pix_threshold=pix_threshold,
            logger=logger,
        )

        # OPTIONAL: persist the annotations to disk for future runs
        frames_dir = NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES"
        out_path = frames_dir / "obs_target_frames_ranked_with_sky.pkl"
        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(ranked_target_frames, f)
            logger.info(
                "Star-projection dispatcher: wrote annotated ranked_target_frames "
                "with tracking_mode back to '%s'.",
                out_path,
            )
        except Exception as exc:
            logger.warning(
                "Star-projection dispatcher: failed to write annotated "
                "ranked_target_frames to '%s': %r",
                out_path,
                exc,
            )

        # Re-scan after annotation.
        has_sidereal, has_slew, any_tracking_flag = _scan_tracking_modes(
            ranked_target_frames
        )

    logger.info(
        "Star-projection dispatcher: has_sidereal=%s, has_slew=%s",
        has_sidereal,
        has_slew,
    )

    # If, even after annotation, we still have no sidereal or slew windows,
    # there is nothing to do.
    if not has_sidereal and not has_slew:
        logger.warning(
            "Star-projection dispatcher: no windows classified as 'sidereal' or "
            "'slew' after annotation; skipping NEBULA_STAR_PROJECTION and "
            "NEBULA_STAR_SLEW_PROJECTION."
        )
        return None, None

    obs_star_projections: Optional[Dict[str, Any]] = None
    obs_star_slew_projections: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 2) Run sidereal star-projection pipeline if needed.
    # ------------------------------------------------------------------
    if has_sidereal:
        logger.info(
            "Star-projection dispatcher: running NEBULA_STAR_PROJECTION "
            "for sidereal-tracking windows."
        )
        obs_star_projections = NEBULA_STAR_PROJECTION.main(
            sensor_config=sensor_config,
            frames_path=None,  # let the module resolve defaults
            gaia_cache_path=None,
            obs_tracks_path=None,
            output_path=None,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # 3) Run slew star-projection pipeline if needed.
    # ------------------------------------------------------------------
    if has_slew:
        logger.info(
            "Star-projection dispatcher: running NEBULA_STAR_SLEW_PROJECTION "
            "for slewing windows."
        )
        obs_star_slew_projections = NEBULA_STAR_SLEW_PROJECTION.main(
            sensor_config=sensor_config,
            frames_path=None,  # let the module resolve defaults
            gaia_cache_path=None,
            obs_tracks_path=None,
            output_path=None,
            logger=logger,
        )

    return obs_star_projections, obs_star_slew_projections

def load_ranked_target_frames_for_dispatch(
    logger: logging.Logger,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Try to load a previously-saved ranked TARGET_PHOTON_FRAMES structure
    (with tracking_mode annotations) from disk for use by the
    star-projection dispatcher.

    This tries, in order:
        1. obs_target_frames_ranked_with_sky.pkl
        2. obs_target_frames_ranked.pkl

    Both are expected to live under:

        NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES"
    """
    # Base TARGET_PHOTON_FRAMES directory
    frames_dir = NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES"

    # Prefer the with-sky version (which is what NEBULA_STAR_PROJECTION uses),
    # but the plain ranked file will also contain tracking_mode if you ran
    # annotate_windows_with_tracking_mode() before saving it.
    candidates = [
        frames_dir / "obs_target_frames_ranked_with_sky.pkl",
        frames_dir / "obs_target_frames_ranked.pkl",
    ]

    for path in candidates:
        if path.exists():
            logger.info(
                "sim_test: loading ranked_target_frames for dispatcher from '%s'.",
                path,
            )
            with open(path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                logger.warning(
                    "sim_test: ranked_target_frames pickle at '%s' is not a dict; "
                    "ignoring.",
                    path,
                )
                return None

            return data

    logger.warning(
        "sim_test: no ranked TARGET_PHOTON_FRAMES pickle found under '%s'; "
        "star-projection dispatcher will be skipped.",
        frames_dir,
    )
    return None

def attach_star_photons_to_target_frames2(
    ranked_target_frames: Dict[str, Any],
    obs_star_photons: Dict[str, Any],
    obs_frame_catalog: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build a *combined* per-frame view of targets + stars for the
    top-ranked window of each observer.

    Philosophy (Avenue 1)
    ----------------------
    This helper does NOT assume any implicit time alignment between
    different products. Instead, it uses:

      - ranked_target_frames:
          * only to choose the "top" window per observer (window with
            the most targets, then most frames).

      - obs_frame_catalog (from NEBULA_PHOTON_FRAME_BUILDER):
          * the canonical per-frame time grid and per-frame target
            sources:
                frames[j] = {
                    "coarse_index": int,
                    "t_utc": datetime,
                    "t_exp_s": float,
                    "sources": [  # per-frame targets
                        {
                            "source_id": str,
                            "source_type": "target",
                            "x_pix": float,
                            "y_pix": float,
                            "phi_ph_m2_s": float,
                            "flux_ph_m2_frame": float,
                            "app_mag_g": float or np.nan,
                            "range_km": float or np.nan,
                        },
                        ...
                    ],
                }

      - obs_star_photons (from NEBULA_STAR_PHOTONS):
          * per-star, per-frame photon time series for each window:
                star_window["stars"][star_id] = {
                    "source_id": str,
                    "source_type": "star",
                    "gaia_source_id": int or str,
                    "t_utc": np.ndarray[object],    # shape (n_frames,)
                    "t_exp_s": np.ndarray[float],   # shape (n_frames,)
                    "x_pix": np.ndarray[float],     # shape (n_frames,)
                    "y_pix": np.ndarray[float],     # shape (n_frames,)
                    "phi_ph_m2_s": np.ndarray[float],
                    "flux_ph_m2_frame": np.ndarray[float],
                    "mag_G": np.ndarray[float],
                    "on_detector": np.ndarray[bool],
                }

    For each observer present in ranked_target_frames, this helper:

      1. Picks the top-ranked window (windows[0]) from ranked_target_frames.
      2. Finds the matching window_index in:
            - obs_frame_catalog[obs_name]["windows"]
            - obs_star_photons[obs_name]["windows"] (if available)
      3. For that window, loops over every frame j and constructs:

            frame_out = {
                "frame_index": j,
                "coarse_index": int,
                "t_utc": datetime,
                "t_exp_s": float,
                "targets": {
                    "<target_id>": {
                        "source_id": str,
                        "source_type": "target",
                        "x_pix": float,
                        "y_pix": float,
                        "phi_ph_m2_s": float,
                        "flux_ph_m2_frame": float,
                        "app_mag_g": float or np.nan,
                        "range_km": float or np.nan,
                    },
                    ...
                },
                "stars": {
                    "<gaia_source_id_str>": {
                        "source_id": str,
                        "source_type": "star",
                        "gaia_source_id": int or str,
                        "x_pix": float,
                        "y_pix": float,
                        "phi_ph_m2_s": float,
                        "flux_ph_m2_frame": float,
                        "mag_G": float,
                        "on_detector": bool,
                    },
                    ...
                },
            }

      4. Wraps those frames in a compact per-observer structure:

            combined[obs_name] = {
                "observer_name": obs_name,
                "sensor_name": frames_entry.get("sensor_name"),
                "rows": int,
                "cols": int,
                "dt_frame_s": float,
                "window_index": int,
                "tracking_mode": str or None,
                "n_frames": int,
                "n_targets": int,
                "n_stars": int,
                "frames": [...],   # list of frame_out dicts
            }

    This gives you, for each observer, a single top window that you can
    inspect in Spyder as "what does the detector see frame-by-frame?",
    with targets and stars clearly separated.

    Parameters
    ----------
    ranked_target_frames : dict
        Output of NEBULA_TARGET_PHOTONS.cull_and_rank_obs_target_frames().
        Used only to select the top-ranked window per observer and to
        read tracking_mode / n_targets metadata.

    obs_star_photons : dict
        Output of NEBULA_STAR_PHOTONS.build_star_photons_for_all_observers().
        Provides per-window, per-star photon time series. If an observer
        or window is missing here, that observer/window will simply have
        an empty "stars" dict in the combined frames.

    obs_frame_catalog : dict
        Output of NEBULA_PHOTON_FRAME_BUILDER.build_frames_by_observer_and_window_photon().
        Provides the canonical per-frame time grid and the per-frame
        target "sources" entries that are copied into "targets".

    logger : logging.Logger, optional
        Logger for informational and warning messages. If None, a
        default 'sim_test' logger is used.

    Returns
    -------
    combined : dict
        Dictionary keyed by observer_name. Each value has the structure
        described above with a single top window expanded into per-frame
        "targets" and "stars" dictionaries.

    Raises
    ------
    RuntimeError
        If a required window cannot be found in obs_frame_catalog or if
        the star time series length does not match the number of frames
        in the chosen window (hard fail; this indicates a structural
        mismatch between products).
    """
    log = logger or logging.getLogger("sim_test")

    combined: Dict[str, Any] = {}

    # Loop over all observers that have ranked target windows.
    for obs_name, ranked_entry in ranked_target_frames.items():
        windows_ranked = ranked_entry.get("windows", [])
        if not windows_ranked:
            log.warning(
                "attach_star_photons_to_target_frames: observer '%s' has no "
                "ranked windows; skipping.",
                obs_name,
            )
            continue

        # Top-ranked window for this observer (already sorted by n_targets, n_frames).
        top_window = windows_ranked[0]
        win_idx = int(top_window.get("window_index", -1))

        # Look up the corresponding entry in the canonical frame catalog.
        frames_entry = obs_frame_catalog.get(obs_name)
        if frames_entry is None:
            log.warning(
                "attach_star_photons_to_target_frames: observer '%s' missing "
                "from obs_frame_catalog; skipping.",
                obs_name,
            )
            continue

        frame_windows = frames_entry.get("windows", [])
        # Find the window with the same window_index in the frame catalog.
        frame_window = None
        for w in frame_windows:
            if int(w.get("window_index", -1)) == win_idx:
                frame_window = w
                break

        if frame_window is None:
            raise RuntimeError(
                f"attach_star_photons_to_target_frames: observer '{obs_name}' "
                f"has no window_index={win_idx} in obs_frame_catalog."
            )

        # Optional: corresponding star window (may be missing).
        star_obs_entry = obs_star_photons.get(obs_name, {})
        star_windows = star_obs_entry.get("windows", []) if star_obs_entry else []
        star_window = None
        for sw in star_windows:
            if int(sw.get("window_index", -1)) == win_idx:
                star_window = sw
                break

        # Extract canonical frames list (from photon frame builder).
        frames_list = frame_window.get("frames", [])
        if not frames_list:
            raise RuntimeError(
                f"attach_star_photons_to_target_frames: observer '{obs_name}', "
                f"window_index={win_idx} has no 'frames' in obs_frame_catalog."
            )

        n_frames = len(frames_list)

        # If we do have a star window, sanity-check frame count.
        if star_window is not None:
            # If there are no stars, this is still OK; we just report zero.
            n_star_frames = None
            for star_entry in star_window.get("stars", {}).values():
                # Use the first star to infer the series length.
                n_star_frames = len(star_entry.get("t_exp_s", []))
                break
            if n_star_frames is not None and n_star_frames != n_frames:
                raise RuntimeError(
                    f"attach_star_photons_to_target_frames: observer '{obs_name}', "
                    f"window_index={win_idx} has n_frames={n_frames} in "
                    f"obs_frame_catalog but star time series length={n_star_frames}."
                )

        # Build per-frame combined structure.
        frames_out: list[Dict[str, Any]] = []

        # Pre-extract star dict for faster access.
        stars_for_window: Dict[str, Any] = (
            star_window.get("stars", {}) if star_window is not None else {}
        )

        for j, base_frame in enumerate(frames_list):
            coarse_index = int(base_frame.get("coarse_index", -1))
            t_utc = base_frame.get("t_utc")
            t_exp_s = float(base_frame.get("t_exp_s", float("nan")))

            # Per-frame targets from photon frame builder
            targets_dict: Dict[str, Any] = {}
            for src in base_frame.get("sources", []):
                # Only targets are expected here, but we guard on type.
                src_type = src.get("source_type", "target")
                if src_type != "target":
                    continue

                target_id = src.get("source_id", None)
                if target_id is None:
                    continue

                targets_dict[str(target_id)] = {
                    "source_id": str(target_id),
                    "source_type": "target",
                    "x_pix": float(src.get("x_pix", float("nan"))),
                    "y_pix": float(src.get("y_pix", float("nan"))),
                    "phi_ph_m2_s": float(src.get("phi_ph_m2_s", float("nan"))),
                    "flux_ph_m2_frame": float(
                        src.get("flux_ph_m2_frame", float("nan"))
                    ),
                    "app_mag_g": float(src.get("app_mag_g", float("nan"))),
                    "range_km": float(src.get("range_km", float("nan"))),
                }

            # Per-frame stars from obs_star_photons
            stars_dict: Dict[str, Any] = {}
            for star_key, star_entry in stars_for_window.items():
                # Defensive: ensure arrays are present; if not, skip this star.
                try:
                    x_arr = star_entry["x_pix"]
                    y_arr = star_entry["y_pix"]
                    phi_arr = star_entry["phi_ph_m2_s"]
                    flux_arr = star_entry["flux_ph_m2_frame"]
                    mag_arr = star_entry["mag_G"]
                    ondet_arr = star_entry["on_detector"]
                except KeyError:
                    continue

                if j >= len(x_arr):
                    # Already checked lengths above; this is just extra safety.
                    continue

                xj = float(x_arr[j])
                yj = float(y_arr[j])
                phij = float(phi_arr[j])
                fluxj = float(flux_arr[j])
                magj = float(mag_arr[j])
                ondet = bool(ondet_arr[j])

                # Optionally skip off-detector samples; for now we keep them,
                # but you can uncomment the “continue” if you prefer.
                # if not ondet:
                #     continue

                stars_dict[str(star_key)] = {
                    "source_id": str(star_entry.get("source_id", star_key)),
                    "source_type": "star",
                    "gaia_source_id": star_entry.get("gaia_source_id", star_key),
                    "x_pix": xj,
                    "y_pix": yj,
                    "phi_ph_m2_s": phij,
                    "flux_ph_m2_frame": fluxj,
                    "mag_G": magj,
                    "on_detector": ondet,
                }

            frame_out: Dict[str, Any] = {
                "frame_index": j,
                "coarse_index": coarse_index,
                "t_utc": t_utc,
                "t_exp_s": t_exp_s,
                "targets": targets_dict,
                "stars": stars_dict,
            }

            frames_out.append(frame_out)

        # Observer-level wrapper.
        rows = int(frames_entry.get("rows", 0))
        cols = int(frames_entry.get("cols", 0))
        dt_frame_s = float(frames_entry.get("dt_frame_s", float("nan")))
        sensor_name = frames_entry.get("sensor_name", None)

        n_targets = int(top_window.get("n_targets", len(top_window.get("targets", {}))))
        n_stars = int(star_window.get("n_stars", 0)) if star_window is not None else 0

        combined[obs_name] = {
            "observer_name": obs_name,
            "sensor_name": sensor_name,
            "rows": rows,
            "cols": cols,
            "dt_frame_s": dt_frame_s,
            "window_index": win_idx,
            "tracking_mode": top_window.get("tracking_mode"),
            "n_frames": n_frames,
            "n_targets": n_targets,
            "n_stars": n_stars,
            "frames": frames_out,
        }

        log.info(
            "attach_star_photons_to_target_frames: observer '%s' -> "
            "window_index=%d, n_frames=%d, n_targets=%d, n_stars=%d.",
            obs_name,
            win_idx,
            n_frames,
            n_targets,
            n_stars,
        )

    return combined

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for sim_test.

    This function:

        1. Configures logging.
        2. Runs NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs() to
           ensure all upstream and pixel pickles are up to date.
        3. Prints a brief summary of the resulting tracks, focusing on
           pixel-level fields for a quick sanity check.
        4. Builds a canonical photon-frame catalog for all observers
           and windows via NEBULA_PHOTON_FRAME_BUILDER. This provides
           the per-frame time grid (t_utc, t_exp_s, ...) that both
           target and star photon time series are defined on.
        5. Optionally (BUILD_TARGET_PHOTON_FRAMES=True), runs the
           photon-domain pipeline via NEBULA_TARGET_PHOTONS to build
           per-target photon time series for all observers and windows,
           and saves both "raw" and "ranked" pickles under
           NEBULA_OUTPUT/TARGET_PHOTON_FRAMES.
        6. Optionally (RUN_GAIA_PIPELINE=True), runs the sky-footprint
           and Gaia catalog pipeline:

           - NEBULA_SKY_SELECTOR.main(logger=...)
           - NEBULA_QUERY_GAIA.main(mag_limit_sensor_G=None, logger=...)

        7. Dispatches the star-projection pipeline based on window
           tracking_mode (sidereal vs slew) and, when possible, runs
           NEBULA_STAR_PHOTONS to convert those star projections into
           per-star photon time series aligned to the same frame grid
           as the target photons.
    """

    # Expose these in the interactive namespace (Spyder variable explorer).
    global obs_tracks, tar_tracks
    global obs_frame_catalog          # NEW: canonical photon-frame catalog
    global obs_target_frames, ranked_target_frames
    global gaia_cache
    global obs_star_projections, obs_star_slew_projections, obs_star_photons



    # Configure a logger for this script.
    logger = configure_logging()

    # Log that the sim_test pixel pipeline is starting.
    logger.info(
        "sim_test: Starting NEBULA pipeline through NEBULA_PIXEL_PICKLER "
        "(force_recompute=%s).",
        FORCE_RECOMPUTE,
    )

    # 1) Pixel pipeline (this cascades through all upstream picklers as needed)
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=FORCE_RECOMPUTE,
        sensor_config=ACTIVE_SENSOR,
        logger=logger,
    )

    # 2) Brief pixel-level summary
    summarize_tracks(
        obs_tracks=obs_tracks,
        tar_tracks=tar_tracks,
        logger=logger,
    )

    logger.info("sim_test: Completed NEBULA pixel pipeline successfully.")

    # ------------------------------------------------------------------
    # 3) Canonical photon frame catalog (per-observer, per-window, per-frame)
    # ------------------------------------------------------------------
    logger.info(
        "sim_test: Building canonical photon-frame catalog via "
        "NEBULA_PHOTON_FRAME_BUILDER."
    )

    obs_frame_catalog = PFB.build_frames_by_observer_and_window_photon(
        max_frames_per_window=None,   # or a small int for quick tests
        logger=logger,
    )

    logger.info(
        "sim_test: Photon frame builder produced catalogs for %d observers.",
        len(obs_frame_catalog),
    )

    # Annotate the canonical photon-frame catalog with tracking_mode so that
    # NEBULA_STAR_PHOTONS can distinguish sidereal vs slew per window.
    #
    # This uses the same physical logic as the target pipeline:
    #   - boresight RA/Dec from obs_tracks,
    #   - ACTIVE_SENSOR.pixel_scale_rad as the plate scale,
    #   - a per-step drift threshold of ~0.3 * PSF_FWHM in pixels.
    pixel_scale_rad_catalog = ACTIVE_SENSOR.pixel_scale_rad
    psf_fwhm_pix_catalog = ACTIVE_SENSOR.psf_fwhm_pix or 1.0
    pix_threshold_catalog = 0.3 * psf_fwhm_pix_catalog

    annotate_windows_with_tracking_mode(
        obs_tracks=obs_tracks,
        ranked_target_frames=obs_frame_catalog,  # here: the photon-frame catalog
        pixel_scale_rad=pixel_scale_rad_catalog,
        pix_threshold=pix_threshold_catalog,
        logger=logger,
    )

    # 4) Optional photon-domain pipeline (all observers, all windows)
    if BUILD_TARGET_PHOTON_FRAMES:

        logger.info(
            "sim_test: Building per-target photon time series for all "
            "observers via NEBULA_TARGET_PHOTONS."
        )


        # Build per-observer, per-window photon catalogs
        obs_target_frames = NTP.build_obs_target_frames_for_all_observers(
            max_frames_per_window=None,
            logger=logger,
        )

        # Save "raw" version (includes windows with zero targets)
        raw_path = NTP.save_obs_target_frames_pickle(
            obs_target_frames,
            filename="obs_target_frames_raw.pkl",
            logger=logger,
        )

        # Cull empty windows and rank by (n_targets, n_frames)
        ranked_target_frames = NTP.cull_and_rank_obs_target_frames(
            obs_target_frames,
            logger=logger,
        )
        
        # Annotate windows with tracking mode.
        #
        # Plate scale from the active sensor (radians per pixel).
        pixel_scale_rad = ACTIVE_SENSOR.pixel_scale_rad

        # Per-frame drift threshold in pixels, expressed as a fraction of the PSF FWHM.
        # Rationale:
        #   - Treat boresight drift during a single frame as a small smear added in quadrature
        #     to the intrinsic PSF.
        #   - Keeping drift ≲ 0.3 * FWHM inflates the effective FWHM by only ~2% for a
        #     Gaussian PSF convolved with a short uniform trail, which is well below typical
        #     tolerances in astronomical imaging and consistent with guiding-error heuristics
        #     that tracking RMS should be ≲ 0.25–0.3 * FWHM.
        psf_fwhm_pix = ACTIVE_SENSOR.psf_fwhm_pix or 1.0
        pix_threshold = 0.3 * psf_fwhm_pix

        annotate_windows_with_tracking_mode(
            obs_tracks=obs_tracks,
            ranked_target_frames=ranked_target_frames,
            pixel_scale_rad=pixel_scale_rad,
            pix_threshold=pix_threshold,
            logger=logger,
        )


        # Save the culled/sorted version
        ranked_path = NTP.save_obs_target_frames_pickle(
            ranked_target_frames,
            filename="obs_target_frames_ranked.pkl",
            logger=logger,
        )

        # Compact summary per observer
        for obs_name, obs_entry in ranked_target_frames.items():
            windows = obs_entry.get("windows", [])
            logger.info(
                "sim_test: Observer '%s' has %d non-empty photon windows "
                "after culling.",
                obs_name,
                len(windows),
            )
            for w in windows[:3]:
                logger.info(
                    "  Window %d: n_frames=%d, n_targets=%d, coarse_index=[%d..%d]",
                    w.get("window_index", -1),
                    w.get("n_frames", -1),
                    w.get("n_targets", 0),
                    w.get("start_index", -1),
                    w.get("end_index", -1),
                )

        logger.info(
            "sim_test: Photon-target frame pipeline complete. "
            "Raw='%s', ranked='%s'.",
            raw_path,
            ranked_path,
        )
    else:
        logger.info(
            "sim_test: BUILD_TARGET_PHOTON_FRAMES=False; skipping photon "
            "time-series and TARGET_PHOTON_FRAMES pickles."
        )

    # 4) Optional sky-footprint + Gaia catalog pipeline
    if RUN_GAIA_PIPELINE:
        logger.info(
            "sim_test: Running NEBULA_SKY_SELECTOR to attach sky footprints "
            "to TARGET_PHOTON_FRAMES windows."
        )
        NSS.main(logger=logger)

        logger.info(
            "sim_test: Running NEBULA_QUERY_GAIA to query Gaia for each "
            "eligible window."
        )
        # mag_limit_sensor_G=None => use ACTIVE_SENSOR.mag_limit internally
        gaia_cache = NQG.main(
            mag_limit_sensor_G=None,
            logger=logger,
        )

        # Brief per-observer summary from the cache
        for obs_name, obs_entry in gaia_cache.items():
            run_meta = obs_entry.get("run_meta", {})
            counts = run_meta.get("window_counts", {})
            logger.info(
                "sim_test: Gaia cache summary for observer '%s' -> "
                "queried=%d, ok=%d, error=%d, skipped_zero_targets=%d, "
                "skipped_bad_sky=%d, skipped_broken=%d, total_stars_ok=%d",
                obs_name,
                counts.get("queried", 0),
                counts.get("ok", 0),
                counts.get("error", 0),
                counts.get("skipped_zero_targets", 0),
                counts.get("skipped_bad_sky", 0),
                counts.get("skipped_broken", 0),
                run_meta.get("total_stars_ok_windows", 0),
            )
    else:
        logger.info(
            "sim_test: RUN_GAIA_PIPELINE=False; skipping sky-footprint and "
            "Gaia catalog queries."
        )


    # 5) Star-projection pipeline dispatcher (sidereal vs slew)
    #
    # At this point:
    #   - ranked_target_frames have been built (if BUILD_TARGET_PHOTON_FRAMES=True)
    #     and annotated with tracking_mode, and/or
    #   - a previous run has written obs_target_frames_ranked[_with_sky].pkl
    #     under NEBULA_OUTPUT/TARGET_PHOTON_FRAMES.
    #
    # We now dispatch to NEBULA_STAR_PROJECTION and/or
    # NEBULA_STAR_SLEW_PROJECTION depending on which tracking modes
    # actually exist in the annotated windows.
    global obs_star_projections, obs_star_slew_projections

    logger.info(
        "sim_test: dispatching star-projection pipeline based on window tracking_mode."
    )

    # Decide where to get ranked_target_frames for the dispatcher:
    if BUILD_TARGET_PHOTON_FRAMES:
        # Use the in-memory structure we just built and annotated.
        ranked_for_dispatch = ranked_target_frames
    else:
        # Reuse previously-saved ranked frames (with tracking_mode) from disk.
        ranked_for_dispatch = load_ranked_target_frames_for_dispatch(logger=logger)

    if ranked_for_dispatch is None:
        logger.info(
            "sim_test: no ranked_target_frames available on disk; skipping "
            "star-projection dispatcher."
        )
    else:
        (
            obs_star_projections,
            obs_star_slew_projections,
        ) = dispatch_star_projection_pipeline(
            ranked_target_frames=ranked_for_dispatch,
            obs_tracks=obs_tracks,
            sensor_config=ACTIVE_SENSOR,
            logger=logger,
        )


        # Brief sanity summary for each projection type, if present.
        if obs_star_projections is not None:
            logger.info(
                "sim_test: NEBULA_STAR_PROJECTION completed for %d observers.",
                len(obs_star_projections),
            )
            for obs_name, obs_entry in list(obs_star_projections.items())[:3]:
                windows = obs_entry.get("windows", [])
                logger.info(
                    "  [sidereal] observer '%s' -> %d windows with star projections.",
                    obs_name,
                    len(windows),
                )
                if windows:
                    w0 = windows[0]
                    logger.info(
                        "    First window %d: n_stars_input=%d, n_stars_on_detector=%d",
                        w0.get("window_index", -1),
                        w0.get("n_stars_input", 0),
                        w0.get("n_stars_on_detector", 0),
                    )

        if obs_star_slew_projections is not None:
            logger.info(
                "sim_test: NEBULA_STAR_SLEW_PROJECTION completed for %d observers.",
                len(obs_star_slew_projections),
            )
            for obs_name, obs_entry in list(obs_star_slew_projections.items())[:3]:
                windows = obs_entry.get("windows", [])
                logger.info(
                    "  [slew] observer '%s' -> %d windows with star projections.",
                    obs_name,
                    len(windows),
                )
                if windows:
                    w0 = windows[0]
                    logger.info(
                        "    First window %d: n_stars_input=%d, n_stars_on_detector=%d",
                        w0.get("window_index", -1),
                        w0.get("n_stars_input", 0),
                        w0.get("n_stars_on_detector", 0),
                    )
                    
    # 6) Star-photon pipeline (Gaia stars -> per-frame photon time series)
    #
    # At this point:
    #   - obs_frame_catalog is the canonical photon-frame catalog built
    #     via NEBULA_PHOTON_FRAME_BUILDER, containing per-window "frames"
    #     entries with t_utc and t_exp_s for each frame,
    #   - ranked_for_dispatch provides tracking_mode annotations and
    #     sky-footprint / Gaia status at the window level (used earlier
    #     by the star-projection dispatcher),
    #   - obs_star_projections contains epoch-based star pixels from
    #     NEBULA_STAR_PROJECTION.main(...),
    #   - obs_star_slew_projections is either None (no slews) or a dict
    #     from NEBULA_STAR_SLEW_PROJECTION.main(...).
    global obs_star_photons, obs_target_star_frames, combined_top_windows

    if obs_frame_catalog is None or obs_star_projections is None:
        logger.info(
            "sim_test: no photon-frame catalog or star projections; "
            "skipping star-photon pipeline."
        )
    else:
        logger.info(
            "sim_test: building per-window star photon time series via "
            "NEBULA_STAR_PHOTONS, using the canonical photon-frame "
            "catalog as the time grid."
        )

        obs_star_photons = NEBULA_STAR_PHOTONS.build_star_photons_for_all_observers(
            # IMPORTANT: use the PHOTON_FRAME_BUILDER output as the
            # frames-with-times input, following the Avenue 1 philosophy.
            obs_target_frames=obs_frame_catalog,
            obs_star_projections=obs_star_projections,
            obs_star_slew_tracks=obs_star_slew_projections,
            logger=logger,
        )

        logger.info(
            "sim_test: NEBULA_STAR_PHOTONS built star photons for %d observer(s).",
            len(obs_star_photons),
        )


    # 7) Attach star photons into the target windows (Option A).
    logger.info(
        "sim_test: attaching star photons to target frames to build "
        "combined target + star windows."
    )
    
    obs_target_star_frames = NEBULA_STAR_PHOTONS.attach_star_photons_to_target_frames(
        obs_target_frames=ranked_for_dispatch,
        obs_star_photons=obs_star_photons,
        logger=logger,
    )
    
    # Optionally: save this collated product for later stages / debugging.
    NTP.save_obs_target_frames_pickle(
        obs_target_star_frames,
        filename="obs_target_star_frames_ranked.pkl",
        logger=logger,
    )

    combined_top_windows = attach_star_photons_to_target_frames2(
        ranked_target_frames=ranked_target_frames,
        obs_star_photons=obs_star_photons,
        obs_frame_catalog=obs_frame_catalog,
        logger=logger,
    )

    anim_dir = NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR / "ANIMATION"
    anim_dir.mkdir(parents=True, exist_ok=True)
    
    combined_path = anim_dir / "combined_top_windows.pkl"
    with open(combined_path, "wb") as f:
        pickle.dump(combined_top_windows, f)
    
    logger.info(
        "sim_test: wrote combined_top_windows for %d observer(s) to '%s'.",
        len(combined_top_windows),
        combined_path,
    )
    combined_pickle_path = (
        NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR / "ANIMATION" / "combined_top_windows.pkl"
    )
    
    combined = NOA.load_combined_top_windows_pickle(
        pickle_path=combined_pickle_path,
        logger=logger,
    )
    
    sbss_entry = combined["SBSS (USA 216)"]
    
    NOA.make_observer_animation_with_target_markers(
        observer_entry=sbss_entry,
        output_path=NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR / "ANIMATION" / "SBSS_window_markers.mp4",
        fps=2.0,
        logger=logger,
    )
    
    NOA.make_observer_animations_from_combined(
        combined_top_windows=combined_top_windows,
        out_dir=anim_dir,      # or some path you like
        fps= 2,                   # or 1.0 / dt_frame_s if you prefer
        logger=logger,
    )

# ---------------------------------------------------------------------------
# Script guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If this file is executed as a script, call the main() function.
    main()
