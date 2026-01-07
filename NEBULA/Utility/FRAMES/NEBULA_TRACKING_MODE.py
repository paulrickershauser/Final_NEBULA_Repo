"""
NEBULA_TRACKING_MODE
====================

Strict / fail-hard window annotation helper to classify per-window
tracking mode ("sidereal" vs "slew") using boresight RA/Dec time series.

This is intentionally placed in Utility/FRAMES because it operates on
"frames-with-windows" style dictionaries (anything with windows containing
start_index/end_index) and should be reusable by multiple post-PIXELS stages.

Design notes
------------
- STRICT behavior: any inconsistency raises RuntimeError.
- Does not import pipeline manager, manifest, or picklers.
- Caller supplies pixel_scale_rad and pix_threshold to avoid coupling to
  ACTIVE_SENSOR and to make unit-testing easier.

Expected inputs
---------------
obs_tracks[obs_name] must contain one of these key pairs:

1) ("pointing_boresight_ra_deg", "pointing_boresight_dec_deg")  [preferred]
2) ("boresight_ra_deg",          "boresight_dec_deg")           [fallback]

ranked_target_frames[obs_name]["windows"] must contain:
- "start_index" : int
- "end_index"   : int
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR



def annotate_windows_with_tracking_mode(
    obs_tracks: Dict[str, Any],
    ranked_target_frames: Dict[str, Any],
    pixel_scale_rad: float,
    pix_threshold: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Annotate each window in ranked_target_frames with a tracking mode.

    For each observer and each window, this function:
        1) Reads boresight pointing from obs_tracks (preferring
           'pointing_boresight_ra_deg' / 'pointing_boresight_dec_deg',
           falling back to 'boresight_ra_deg' / 'boresight_dec_deg').
        2) Converts RA/Dec time series to unit vectors.
        3) Computes angular motion between consecutive samples.
        4) Converts angular motion to pixel motion using pixel_scale_rad.
        5) For each window covering coarse indices [start_index .. end_index],
           examines per-step pixel drift over steps start_index .. end_index-1 and sets:

               window["tracking_mode"] = "sidereal" if max_step_pix < pix_threshold
                                       = "slew"     otherwise

           It also stores:
               window["total_drift_pix"] : float
               window["max_step_pix"]    : float

    STRICT behavior: raises RuntimeError on any missing/malformed prerequisite.

    Parameters
    ----------
    obs_tracks : dict
        Per-observer track dictionaries containing boresight pointing arrays.
    ranked_target_frames : dict
        Per-observer dict containing "windows" list with start_index/end_index.
        Modified in-place.
    pixel_scale_rad : float
        Plate scale [radians per pixel]. Must be > 0.
    pix_threshold : float
        Threshold on per-step drift in pixels.
    logger : logging.Logger, optional
        Logger for informational messages. If None, uses logging.getLogger
        with name "NEBULA_TRACKING_MODE".

    Raises
    ------
    ValueError
        If pixel_scale_rad <= 0.
    RuntimeError
        If any observer/window has inconsistent or insufficient data.
    """
    if logger is None:
        logger = logging.getLogger("NEBULA_TRACKING_MODE")

    if (not np.isfinite(pixel_scale_rad)) or pixel_scale_rad <= 0.0:
        raise ValueError(
            "annotate_windows_with_tracking_mode: pixel_scale_rad must be finite and > 0, "
            f"got {pixel_scale_rad!r}."
        )
    
    if (not np.isfinite(pix_threshold)) or pix_threshold <= 0.0:
        raise ValueError(
            "annotate_windows_with_tracking_mode: pix_threshold must be finite and > 0, "
            f"got {pix_threshold!r}."
        )


    for obs_name, frames_entry in ranked_target_frames.items():
        obs_track = obs_tracks.get(obs_name)
        if obs_track is None:
            raise RuntimeError(
                "annotate_windows_with_tracking_mode: no obs_track for "
                f"observer '{obs_name}'."
            )

        # --- Safe selection of RA / Dec arrays (avoid boolean `or` on numpy arrays) ---
        if "pointing_boresight_ra_deg" in obs_track:
            ra_data = obs_track["pointing_boresight_ra_deg"]
        elif "boresight_ra_deg" in obs_track:
            ra_data = obs_track["boresight_ra_deg"]
        else:
            raise RuntimeError(
                "annotate_windows_with_tracking_mode: observer '{0}' has no "
                "'pointing_boresight_ra_deg' or 'boresight_ra_deg'.".format(obs_name)
            )

        if "pointing_boresight_dec_deg" in obs_track:
            dec_data = obs_track["pointing_boresight_dec_deg"]
        elif "boresight_dec_deg" in obs_track:
            dec_data = obs_track["boresight_dec_deg"]
        else:
            raise RuntimeError(
                "annotate_windows_with_tracking_mode: observer '{0}' has no "
                "'pointing_boresight_dec_deg' or 'boresight_dec_deg'.".format(obs_name)
            )

        ra_deg = np.asarray(ra_data, dtype=float)
        dec_deg = np.asarray(dec_data, dtype=float)

        if ra_deg.size == 0 or dec_deg.size == 0:
            raise RuntimeError(
                "annotate_windows_with_tracking_mode: empty RA/Dec arrays "
                f"for observer '{obs_name}'."
            )

        if ra_deg.shape != dec_deg.shape:
            raise RuntimeError(
                "annotate_windows_with_tracking_mode: RA/Dec shape mismatch "
                f"for observer '{obs_name}' (ra={ra_deg.shape}, dec={dec_deg.shape})."
            )

        # --- Convert boresight RA/Dec to unit vectors on the celestial sphere ---
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)

        cos_dec = np.cos(dec_rad)
        bx = np.cos(ra_rad) * cos_dec
        by = np.sin(ra_rad) * cos_dec
        bz = np.sin(dec_rad)
        b = np.stack((bx, by, bz), axis=1)  # (N, 3)

        if b.shape[0] < 2:
            raise RuntimeError(
                "annotate_windows_with_tracking_mode: fewer than 2 coarse samples "
                f"for observer '{obs_name}'; cannot compute drift."
            )

        # Angular motion per coarse step: Δθ_j = arccos(b_j · b_{j+1})
        dot = np.einsum("ij,ij->i", b[:-1], b[1:])
        dot = np.clip(dot, -1.0, 1.0)
        delta_theta_rad = np.arccos(dot)  # (N-1,)

        # Convert angular motion to approximate pixel motion using plate scale.
        delta_pix = delta_theta_rad / pixel_scale_rad  # pixels per step

        windows = frames_entry.get("windows", [])
        # Skip windows with no targets: do not annotate or write tracking keys.
        for w in windows:
            if int(w.get("n_targets", 0)) <= 0:
                continue
        
            start_idx = w.get("start_index")
            end_idx = w.get("end_index")
            w_index = w.get("window_index", None)


            if start_idx is None or end_idx is None:
                raise RuntimeError(
                    "annotate_windows_with_tracking_mode: window (index={0}) for "
                    "observer '{1}' missing start_index or end_index.".format(
                        w_index, obs_name
                    )
                )

            start_idx = int(start_idx)
            end_idx = int(end_idx)

            # Bounds: indices refer to coarse samples (0..N-1)
            n = ra_deg.size
            if not (0 <= start_idx < n) or not (0 <= end_idx < n):
                raise RuntimeError(
                    "annotate_windows_with_tracking_mode: window (index={0}) indices "
                    "[{1}..{2}] out of bounds for observer '{3}' (N={4}).".format(
                        w_index, start_idx, end_idx, obs_name, n
                    )
                )

            if end_idx <= start_idx:
                raise RuntimeError(
                    "annotate_windows_with_tracking_mode: window (index={0}) for "
                    "observer '{1}' has non-increasing indices "
                    "[start_index={2}, end_index={3}].".format(
                        w_index, obs_name, start_idx, end_idx
                    )
                )

            # delta_pix[j] is drift from coarse j -> j+1.
            # For window covering samples [start_idx..end_idx], steps are
            # j = start_idx .. end_idx-1, i.e. delta_pix[start_idx:end_idx].
            window_delta = delta_pix[start_idx:end_idx]

            if window_delta.size == 0:
                raise RuntimeError(
                    "annotate_windows_with_tracking_mode: window (index={0}) for "
                    "observer '{1}' produced an empty drift slice from "
                    "[{2}..{3}].".format(w_index, obs_name, start_idx, end_idx)
                )

            total_drift_pix = float(np.sum(window_delta))
            max_step_pix = float(np.max(window_delta))

            mode = "sidereal" if max_step_pix < pix_threshold else "slew"

            w["tracking_mode"] = mode
            w["total_drift_pix"] = total_drift_pix
            w["max_step_pix"] = max_step_pix

        n_annotated = sum(1 for w in windows if w.get("tracking_mode") in ("sidereal", "slew"))
        logger.info(
            "annotate_windows_with_tracking_mode: observer '%s' -> annotated %d windows.",
            obs_name,
            n_annotated,
        )

def main(logger: logging.Logger | None = None) -> None:
    """
    Pipeline stage entrypoint for TRACKING_MODE.

    Inputs (must exist)
    -------------------
    - TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl
    - POINT_SatPickles/observer_tracks_with_pointing.pkl

    Output (owned by this stage)
    ----------------------------
    - TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_tracking.pkl
    """
    if logger is None:
        logger = logging.getLogger("NEBULA_TRACKING_MODE")

    # --- Input paths ---
    frames_dir = os.path.join(str(NEBULA_OUTPUT_DIR), "TARGET_PHOTON_FRAMES")
    ranked_in_path = os.path.join(frames_dir, "obs_target_frames_ranked.pkl")

    pointing_dir = os.path.join(str(NEBULA_OUTPUT_DIR), "POINT_SatPickles")
    obs_pointing_path = os.path.join(pointing_dir, "observer_tracks_with_pointing.pkl")

    # --- Output path (owned output) ---
    ranked_out_path = os.path.join(frames_dir, "obs_target_frames_ranked_with_tracking.pkl")

    # Ensure directory exists for output
    os.makedirs(frames_dir, exist_ok=True)

    # Fail-hard on missing prerequisites (pipeline manager should have ensured these exist)
    if not os.path.exists(ranked_in_path):
        raise FileNotFoundError(
            f"NEBULA_TRACKING_MODE.main: missing input ranked frames pickle: '{ranked_in_path}'. "
            "Run TARGET_PHOTONS stage first."
        )

    if not os.path.exists(obs_pointing_path):
        raise FileNotFoundError(
            f"NEBULA_TRACKING_MODE.main: missing prerequisite pointing pickle: '{obs_pointing_path}'. "
            "Run POINTING stage first."
        )

    logger.info("Loading ranked target frames from '%s'.", ranked_in_path)
    with open(ranked_in_path, "rb") as f:
        obs_target_frames_ranked: Dict[str, Any] = pickle.load(f)

    logger.info("Loading observer tracks with pointing from '%s'.", obs_pointing_path)
    with open(obs_pointing_path, "rb") as f:
        obs_tracks: Dict[str, Any] = pickle.load(f)

    # Mirror sim_test policy for thresholding:
    #   - plate scale from ACTIVE_SENSOR (rad/pix)
    #   - threshold = 0.3 * PSF_FWHM (pixels)
    pixel_scale_rad = float(ACTIVE_SENSOR.pixel_scale_rad)

    # Fail-fast: this property raises if psf_fwhm_pix or tracking_pix_threshold_factor
    # are missing/invalid (no silent defaults).
    pix_threshold = float(ACTIVE_SENSOR.tracking_pix_threshold_pix)

    logger.info(
        "NEBULA_TRACKING_MODE: using pix_threshold=%.3f (tracking_pix_threshold_factor=%.3f * psf_fwhm_pix=%.3f) "
        "for slew vs sidereal decision.",
        pix_threshold,
        float(ACTIVE_SENSOR.tracking_pix_threshold_factor),
        float(ACTIVE_SENSOR.psf_fwhm_pix),
    )

    annotate_windows_with_tracking_mode(
        obs_tracks=obs_tracks,
        ranked_target_frames=obs_target_frames_ranked,
        pixel_scale_rad=pixel_scale_rad,
        pix_threshold=pix_threshold,
        logger=logger,
    )

    logger.info("Writing ranked frames with tracking_mode to '%s'.", ranked_out_path)
    with open(ranked_out_path, "wb") as f:
        pickle.dump(obs_target_frames_ranked, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compact summary
    for obs_name, obs_entry in obs_target_frames_ranked.items():
        windows = obs_entry.get("windows", [])
        n_sidereal = sum(1 for w in windows if w.get("tracking_mode") == "sidereal")
        n_slew = sum(1 for w in windows if w.get("tracking_mode") == "slew")
        n_annotated = n_sidereal + n_slew
        logger.info(
            "Observer '%s': %d windows annotated (sidereal=%d, slew=%d).",
            obs_name,
            n_annotated,
            n_sidereal,
            n_slew,
        )


    logger.info("NEBULA_TRACKING_MODE complete. Wrote '%s'.", ranked_out_path)
