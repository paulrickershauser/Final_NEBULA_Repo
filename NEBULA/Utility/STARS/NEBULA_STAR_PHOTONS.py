"""
NEBULA_STAR_PHOTONS
===================

This module converts *star projection* products into per-star photon
time series that mirror the existing **target photon** schema.

Inputs (all per-observer):
--------------------------
1) obs_star_projections
   - Loaded from the pickle produced by NEBULA_STAR_PROJECTION.
   - For each observer and window, provides:
       * mid-window star positions on the detector (x_pix_mid, y_pix_mid)
       * Gaia G-band magnitude (mag_G)
       * flags like on_detector, gaia_status, etc.

2) frames_with_sky
   - Loaded via NEBULA_TARGET_PHOTONS (same pickle used by
     NEBULA_PHOTON_FRAME_BUILDER).
   - For each observer and window, provides:
       * window segmentation (start_index, end_index, n_frames)
       * per-frame timing (t_utc, t_exp_s or equivalent)
       * sky selector metadata (sky_center_ra_deg, sky_radius_deg, etc.)

3) SensorConfig / ACTIVE_SENSOR
   - From Configuration.NEBULA_SENSOR_CONFIG.
   - Describes sensor geometry (rows, cols, pixel pitch, etc.)
     and is embedded as metadata in the star photon products.

Outputs:
--------
obs_star_photons : dict
    Top-level mapping keyed by observer name. For each observer:

        obs_star_photons[obs_name] = {
            "observer_name": str,
            "rows": int,
            "cols": int,
            "catalog_name": str,
            "catalog_band": str,        # e.g., "G"
            "run_meta": {...},
            "windows": [StarPhotonWindow, ...],
        }

    Each StarPhotonWindow contains per-window, per-star photon time
    series, aligned in spirit with the existing target photon pickles.

The resulting dict is typically serialized to:

    NEBULA_OUTPUT/FRAMES/obs_star_photons.pkl

and can be consumed by later sensor / event simulation stages.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library imports
# ----------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging
import os
import pickle
from datetime import datetime, timezone, timedelta  # at top


# ----------------------------------------------------------------------
# Third-party imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
# NEBULA configuration imports (must succeed; failures should be loud)
# ----------------------------------------------------------------------

# Base path configuration for NEBULA input/output directories.
from Configuration import NEBULA_PATH_CONFIG

# Sensor configuration: dataclass + active sensor definition.
from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig, ACTIVE_SENSOR

# Star catalog configuration: name, band metadata, etc.
from Configuration.NEBULA_STAR_CONFIG import NEBULA_STAR_CATALOG

# ----------------------------------------------------------------------
# NEBULA utility imports (must also succeed in a proper NEBULA run)
# ----------------------------------------------------------------------

# Radiometry / magnitude-to-photon-flux routines
# (C:\Users\prick\Desktop\Research\NEBULA\Utility\RADIOMETRY\NEBULA_FLUX.py)
from Utility.RADIOMETRY import NEBULA_FLUX

# ----------------------------------------------------------------------
# Module-wide constants and type aliases
# ----------------------------------------------------------------------

# Version tag for this star-photon stage (embedded in run_meta).
STAR_PHOTONS_RUN_META_VERSION: str = "0.1"

# Type alias for a per-window star photon dict.
StarPhotonWindow = Dict[str, Any]

# Type alias for the top-level mapping: observer_name -> per-observer entry.
ObsStarPhotons = Dict[str, Dict[str, Any]]


# ----------------------------------------------------------------------
# Logger helper
# ----------------------------------------------------------------------
def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger instance suitable for this module.

    Parameters
    ----------
    logger : logging.Logger or None, optional
        Existing logger to reuse. If None, a module-level logger
        named after this module (``__name__``) is returned.

        Logging configuration (handlers, levels, formatting) is assumed
        to be handled by the NEBULA driver script (e.g., sim_test.py)
        or by the top-level application; this helper does not modify
        global logging setup.

    Returns
    -------
    logging.Logger
        Logger to use inside NEBULA_STAR_PHOTONS.
    """
    # If the caller passed a logger, just use it.
    if logger is not None:
        return logger

    # Otherwise, return a standard module-level logger.
    return logging.getLogger(__name__)

def compute_star_photon_flux_from_mag(
    mag_G: np.ndarray,
    eta_eff: float = 1.0,
) -> np.ndarray:
    """
    Convert Gaia G-band magnitude(s) into photon flux at the aperture.

    This helper mirrors the radiometric conventions used in NEBULA_FLUX:
    it uses the Sun's G-band magnitude at 1 AU and an effective G-band
    solar irradiance F_SUN_G_1AU_W_M2 as the reference, then applies the
    standard magnitude–flux relation to recover the absolute energy flux
    and photon flux.

    Parameters
    ----------
    mag_G : array-like
        Gaia G-band apparent magnitudes for one or more stars. This can be
        a scalar, list, or NumPy array; it will be converted to a float
        array internally.

    eta_eff : float, optional
        Overall throughput / quantum-efficiency factor (0–1). This scales
        the photon flux after conversion from energy flux. Default is 1.0,
        i.e., no additional loss beyond what is already implicit in the
        reference irradiance.

    Returns
    -------
    np.ndarray
        Photon flux in units of photons m⁻² s⁻¹ at the aperture, with the
        same shape as ``mag_G`` broadcast to a NumPy array.

    Notes
    -----
    The conversion is:

        Δm = mag_G - GAIA_G_M_SUN_APP_1AU
        p  = 10^(-0.4 * Δm)
        F  = F_SUN_G_1AU_W_M2 * p
        Ṅ = F / GAIA_G_PHOTON_ENERGY_J * eta_eff

    where F_SUN_G_1AU_W_M2, GAIA_G_M_SUN_APP_1AU, GAIA_G_PHOTON_ENERGY_J,
    and NUM_EPS are taken from NEBULA_FLUX to ensure consistency with the
    target radiometry pipeline.
    """
    # Convert input magnitudes to a NumPy float array for vectorized math.
    mag_arr = np.asarray(mag_G, dtype=float)

    # Pull the reference G-band quantities from NEBULA_FLUX.
    m_sun_g = NEBULA_FLUX.GAIA_G_M_SUN_APP_1AU
    F_sun_g = NEBULA_FLUX.F_SUN_G_1AU_W_M2

    # Magnitude difference relative to the Sun in G band.
    delta_m = mag_arr - m_sun_g

    # Flux ratio p = F_star / F_sun,G using the standard magnitude–flux relation.
    p_flux_ratio = 10.0 ** (-0.4 * delta_m)

    # Convert dimensionless flux ratio into an absolute G-band energy flux
    # at the aperture (W m⁻²), using the same normalization as NEBULA_FLUX.
    flux_g_w_m2 = F_sun_g * p_flux_ratio

    # Convert energy flux to photon flux using the effective G-band photon energy.
    # Use NUM_EPS as a floor to avoid division by zero in pathological cases.
    photon_flux_g_m2_s = flux_g_w_m2 / max(
        NEBULA_FLUX.GAIA_G_PHOTON_ENERGY_J,
        NEBULA_FLUX.NUM_EPS,
    )

    # Apply overall instrument throughput / QE.
    photon_flux_g_m2_s *= float(eta_eff)

    return photon_flux_g_m2_s

def build_star_photon_timeseries_for_window_sidereal(
    obs_name: str,
    window_frames_entry: Dict[str, Any],
    window_star_projection: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    mode: str = "sidereal",
) -> StarPhotonWindow:
    """
    Build per-star photon time series for a single observer + window
    in **sidereal** tracking mode.

    This helper takes:
        - one window entry from the frames-with-sky structure
          (produced by NEBULA_PHOTON_FRAME_BUILDER / NEBULA_TARGET_PHOTONS), and
        - the matching star projection entry from obs_star_projections
          (produced by NEBULA_STAR_PROJECTION),

    and returns a StarPhotonWindow dictionary with per-star, per-frame
    photon flux time series.

    Current implementation
    ----------------------
    * Only the "sidereal" tracking mode is implemented here.
    * For sidereal windows, each star's pixel coordinates (x_pix, y_pix)
      are assumed to be fixed across all frames in the window and are
      taken directly from the star projection stage:

          x_pix_epoch, y_pix_epoch  ->  x_pix(t_j), y_pix(t_j) for all frames j.

      i.e., we are not yet modeling intra-window motion of stars on the
      detector; we just use the epoch-based projection as constant over
      the frames in that window.
    * Each star's Gaia G magnitude mag_G is treated as constant over the
      window. The corresponding photon flux at the aperture
      phi_ph_m2_s is computed once via compute_star_photon_flux_from_mag()
      and broadcast across frames.
    * Per-frame photon counts are then:

            flux_ph_m2_frame = phi_ph_m2_s * t_exp_s

      where t_exp_s is the exposure time of each frame.

    Expected input schema (minimal, actual)
    ---------------------------------------
    window_frames_entry (from frames-with-sky) should provide:
        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,
            # Optional summarizing times; photon-frame builder dependent:
            "t_start_utc": datetime (optional),
            "t_end_utc": datetime (optional),

            # Sky selector meta (copied from frames-with-sky):
            "sky_center_ra_deg": float (optional),
            "sky_center_dec_deg": float (optional),
            "sky_radius_deg": float (optional),
            "sky_selector_status": str (optional),

            # Optional tracking label attached later by sim_test:
            "tracking_mode": str (optional, e.g., "sidereal" or "slew"),

            "frames": [
                {
                    "frame_index": int,
                    "t_utc": datetime or str,
                    "t_mjd_utc": float (optional),
                    "t_exp_s": float,
                    ...
                },
                ...
            ],
        }

    window_star_projection (from obs_star_projections) should provide:
        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,
            "sky_center_ra_deg": float,
            "sky_center_dec_deg": float,
            "sky_radius_deg": float,
            "sky_selector_status": str,

            # Gaia query status:
            "gaia_status": str,
            "gaia_error_message": Optional[str],
            "n_stars_input": int,
            "n_stars_on_detector": int,
            "star_density_on_detector_per_deg2": float,

            "stars": {
                "<gaia_source_id_str>": {
                    "gaia_source_id": int or str,
                    "source_id": str,          # often same as key
                    "source_type": "star",
                    "mag_G": float,

                    # Epoch-based RA/Dec at the projection time:
                    "ra_deg_catalog": float,
                    "dec_deg_catalog": float,
                    "ra_deg_epoch": float,
                    "dec_deg_epoch": float,

                    # Proper motion / astrometry (optional for this function):
                    "pm_ra_masyr": float,
                    "pm_dec_masyr": float,
                    # parallax_mas, radial_velocity_km_s may also be present.

                    # Pixel coordinates at the projection epoch:
                    "x_pix_epoch": float,
                    "y_pix_epoch": float,

                    # Whether this star was found to fall on the detector:
                    "on_detector": bool,
                },
                ...
            },
        }

    Returned schema
    ---------------
    The returned StarPhotonWindow has the form:

        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,

            # Window-level time summary copied from frames-with-sky:
            "t_start_utc": datetime or None,
            "t_end_utc": datetime or None,

            # Sky selector meta:
            "sky_center_ra_deg": float or None,
            "sky_center_dec_deg": float or None,
            "sky_radius_deg": float or None,
            "sky_selector_status": str or None,

            # Tracking label for this window ("sidereal" here):
            "tracking_mode": str,

            # Per-star time series:
            "n_stars": int,
            "stars": {
                "<gaia_source_id_str>": {
                    "source_id": str,
                    "source_type": "star",
                    "gaia_source_id": int or str,

                    "t_utc": np.ndarray[object],    # shape (n_frames,)
                    "t_exp_s": np.ndarray[float],   # shape (n_frames,)

                    "x_pix": np.ndarray[float],     # shape (n_frames,)
                    "y_pix": np.ndarray[float],     # shape (n_frames,)

                    "phi_ph_m2_s": np.ndarray[float],      # per-frame flux at aperture
                    "flux_ph_m2_frame": np.ndarray[float], # phi * t_exp_s

                    "mag_G": np.ndarray[float],     # constant per window here
                    "on_detector": np.ndarray[bool],# constant per window here
                },
                ...
            },

            # For future extensibility when combining other source types:
            "n_sources_total": int,  # == n_stars for now
        }

    Parameters
    ----------
    obs_name : str
        Name of the observer (used only for logging / error messages).

    window_frames_entry : dict
        Single-window entry from the frames-with-sky structure for this
        observer.

    window_star_projection : dict
        Matching single-window star projection entry from
        obs_star_projections[obs_name]["windows"].

    logger : logging.Logger, optional
        Logger for debug/info messages. If None, a local module logger
        obtained via _get_logger() is used.

    mode : {"sidereal", ...}, optional
        Tracking mode for this window. Currently only "sidereal" is
        supported; any other value results in a ValueError.

    Returns
    -------
    StarPhotonWindow
        Dictionary containing per-star photon time series for this
        observer + window.

    Raises
    ------
    ValueError
        If mode is not "sidereal".
    RuntimeError
        If the window has no frames, or required keys are missing.
    """
    # Resolve a logger to use inside this function.
    log = _get_logger(logger)

    # For now we only support sidereal handling here.
    if mode != "sidereal":
        raise ValueError(
            f"build_star_photon_timeseries_for_window_sidereal: only 'sidereal' "
            f"mode is implemented, got mode={mode!r} for observer '{obs_name}'."
        )

    # # ------------------------------------------------------------------
    # # Extract and validate frame-level information for this window.
    # # ------------------------------------------------------------------
    # frames = window_frames_entry.get("frames", None)
    # if not frames:
    #     raise RuntimeError(
    #         f"build_star_photon_timeseries_for_window_sidereal: window "
    #         f"{window_frames_entry.get('window_index')} for observer "
    #         f"'{obs_name}' has no 'frames' entry."
    #     )

    # # Number of frames in this window.
    # n_frames = len(frames)

    # # Collect per-frame times and exposure durations.
    # t_utc_list: list[Any] = []
    # t_exp_s_list: list[float] = []

    # for f in frames:
    #     # Each frame must provide a time stamp and exposure duration.
    #     if "t_utc" not in f:
    #         raise RuntimeError(
    #             f"build_star_photon_timeseries_for_window_sidereal: frame in "
    #             f"window {window_frames_entry.get('window_index')} for observer "
    #             f"'{obs_name}' is missing 't_utc'."
    #         )
    #     if "t_exp_s" not in f:
    #         raise RuntimeError(
    #             f"build_star_photon_timeseries_for_window_sidereal: frame in "
    #             f"window {window_frames_entry.get('window_index')} for observer "
    #             f"'{obs_name}' is missing 't_exp_s'."
    #         )

    #     t_utc_list.append(f["t_utc"])
    #     t_exp_s_list.append(float(f["t_exp_s"]))

    # # Convert per-frame exposure durations to a NumPy array for vectorized math.
    # t_exp_s = np.asarray(t_exp_s_list, dtype=float)
    
    
    # # ------------------------------------------------------------------
    # # Extract and validate frame-level information for this window.
    # #
    # # We support two layouts:
    # #   (a) Newer "frames-with-sky" style where the window explicitly
    # #       contains a "frames" list with per-frame metadata, or
    # #   (b) The current NEBULA_TARGET_PHOTONS layout where the window
    # #       has only summary fields (n_frames, start/end_time, etc.) and
    # #       per-frame times/exposures live inside each target's time
    # #       series under window_frames_entry["targets"].
    # #
    # # For (b), we treat the first target's t_utc / t_exp_s as canonical
    # # for this window and assume all targets share the same frame grid.
    # # ------------------------------------------------------------------
    # frames = window_frames_entry.get("frames", None)

    # # Collect per-frame times and exposure durations (in seconds).
    # t_utc_list: list[Any] = []
    # t_exp_s: np.ndarray

    # if frames:
    #     # ------------------------------------------------------------------
    #     # Case (a): explicit "frames" list is present.
    #     # ------------------------------------------------------------------
    #     n_frames = len(frames)

    #     t_exp_s_list: list[float] = []

    #     for f in frames:
    #         # Each frame must provide a time stamp and exposure duration.
    #         if "t_utc" not in f:
    #             raise RuntimeError(
    #                 "build_star_photon_timeseries_for_window_sidereal: frame "
    #                 f"in window {window_frames_entry.get('window_index')} for "
    #                 f"observer '{obs_name}' is missing 't_utc'."
    #             )
    #         if "t_exp_s" not in f:
    #             raise RuntimeError(
    #                 "build_star_photon_timeseries_for_window_sidereal: frame "
    #                 f"in window {window_frames_entry.get('window_index')} for "
    #                 f"observer '{obs_name}' is missing 't_exp_s'."
    #             )

    #         t_utc_list.append(f["t_utc"])
    #         t_exp_s_list.append(float(f["t_exp_s"]))

    #     # Convert per-frame exposure durations to a NumPy array for vectorized math.
    #     t_exp_s = np.asarray(t_exp_s_list, dtype=float)

    # else:
    #     # ------------------------------------------------------------------
    #     # Case (b): no explicit "frames" list; reconstruct the frame grid
    #     # from one of the per-target time series in this window.
    #     # ------------------------------------------------------------------
    #     targets = window_frames_entry.get("targets", {}) or {}
    #     if not targets:
    #         raise RuntimeError(
    #             "build_star_photon_timeseries_for_window_sidereal: window "
    #             f"{window_frames_entry.get('window_index')} for observer "
    #             f"'{obs_name}' has neither 'frames' nor 'targets'; cannot "
    #             "reconstruct frame grid."
    #         )

    #     # Use the first target as the canonical frame grid.
    #     first_target_key, first_target = next(iter(targets.items()))

    #     try:
    #         t_utc_arr = np.asarray(first_target["t_utc"], dtype=object)
    #         t_exp_s_arr = np.asarray(first_target["t_exp_s"], dtype=float)
    #     except KeyError as exc:
    #         raise RuntimeError(
    #             "build_star_photon_timeseries_for_window_sidereal: target "
    #             f"{first_target_key!r} in window "
    #             f"{window_frames_entry.get('window_index')} for observer "
    #             f"'{obs_name}' is missing 't_utc' or 't_exp_s'; cannot "
    #             "reconstruct frame grid."
    #         ) from exc

    #     if t_utc_arr.shape != t_exp_s_arr.shape:
    #         raise RuntimeError(
    #             "build_star_photon_timeseries_for_window_sidereal: target "
    #             f"{first_target_key!r} in window "
    #             f"{window_frames_entry.get('window_index')} for observer "
    #             f"'{obs_name}' has mismatched t_utc / t_exp_s shapes "
    #             f"{t_utc_arr.shape} vs {t_exp_s_arr.shape}."
    #         )

    #     # Use the window's n_frames if present; otherwise, infer from the
    #     # target time series length.
    #     n_frames_window = window_frames_entry.get("n_frames", None)
    #     if n_frames_window is None:
    #         n_frames = int(t_utc_arr.shape[0])
    #     else:
    #         n_frames = int(n_frames_window)
    #         if t_utc_arr.shape[0] < n_frames:
    #             raise RuntimeError(
    #                 "build_star_photon_timeseries_for_window_sidereal: target "
    #                 f"{first_target_key!r} in window "
    #                 f"{window_frames_entry.get('window_index')} for observer "
    #                 f"'{obs_name}' has only {t_utc_arr.shape[0]} frames but "
    #                 f"window n_frames={n_frames}."
    #             )

    #     # Truncate arrays to the window's declared frame count if needed.
    #     t_utc_arr = t_utc_arr[:n_frames]
    #     t_exp_s_arr = t_exp_s_arr[:n_frames]

    #     # Populate the lists/arrays used downstream.
    #     t_utc_list = list(t_utc_arr)
    #     t_exp_s = t_exp_s_arr
    # ------------------------------------------------------------------
    # Extract and validate frame-level information for this window.
    #
    # SIDEREAL star photons must use the *full-window* per-frame timebase
    # from window_frames_entry["frames"]. We do NOT fall back to per-target
    # series (those can be sparse and shorter than n_frames).
    # ------------------------------------------------------------------
    frames = window_frames_entry.get("frames", None) or []
    if len(frames) == 0:
        raise RuntimeError(
            "build_star_photon_timeseries_for_window_sidereal: missing full-frame timebase "
            f"(window_frames_entry['frames']) for observer '{obs_name}', "
            f"window_index={window_frames_entry.get('window_index')}. "
            "STAR_PHOTONS requires a canonical per-window frame catalog."
        )

    # Window frame count: prefer declared n_frames; otherwise accept frames length.
    n_frames = int(window_frames_entry.get("n_frames") or len(frames))
    if len(frames) < n_frames:
        raise RuntimeError(
            "build_star_photon_timeseries_for_window_sidereal: frame timebase has only "
            f"{len(frames)} frames but window declares n_frames={n_frames} "
            f"for observer '{obs_name}', window_index={window_frames_entry.get('window_index')}."
        )

    # Truncate to declared n_frames (ignore any extras safely).
    frames = frames[:n_frames]

    # Collect per-frame times and exposure durations (in seconds).
    t_utc_list: list[Any] = []
    t_exp_s_list: list[float] = []

    for f in frames:
        if "t_utc" not in f:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_sidereal: frame "
                f"in window {window_frames_entry.get('window_index')} for "
                f"observer '{obs_name}' is missing 't_utc'."
            )
        if "t_exp_s" not in f:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_sidereal: frame "
                f"in window {window_frames_entry.get('window_index')} for "
                f"observer '{obs_name}' is missing 't_exp_s'."
            )

        t_utc_list.append(f["t_utc"])
        t_exp_s_list.append(float(f["t_exp_s"]))

    t_exp_s = np.asarray(t_exp_s_list, dtype=float)


    # ------------------------------------------------------------------
    # Extract and validate per-star projection information.
    # ------------------------------------------------------------------
    stars_proj: Dict[str, Any] = window_star_projection.get("stars", {}) or {}

    # Dictionary to accumulate per-star time series.
    stars_timeseries: Dict[str, Dict[str, Any]] = {}

    # If there are no stars at all, we still return a valid (but empty)
    # StarPhotonWindow structure.
    if len(stars_proj) == 0:
        log.debug(
            "Observer '%s', window %s: no stars in projection; returning "
            "empty StarPhotonWindow.",
            obs_name,
            window_frames_entry.get("window_index"),
        )

    for star_key, star_entry in stars_proj.items():
        # Respect the on_detector flag. If False, we skip the star to keep the
        # photon cube focused on actual sources on the detector.
        on_det_flag = bool(star_entry.get("on_detector", True))
        if not on_det_flag:
            continue

        # Gaia source identifier. Prefer an explicit gaia_source_id field
        # from NEBULA_STAR_PROJECTION; fall back to the dict key if absent.
        gaia_source_id = star_entry.get("gaia_source_id", star_key)

        # Human-readable/short source identifier; fall back to the key.
        source_id = star_entry.get("source_id", str(star_key))

        # Epoch-based pixel coordinates from the star projection stage.
        try:
            x_epoch = float(star_entry["x_pix_epoch"])
            y_epoch = float(star_entry["y_pix_epoch"])
        except KeyError as exc:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_sidereal: missing "
                f"x_pix_epoch/y_pix_epoch for star {star_key!r} in observer "
                f"'{obs_name}', window "
                f"{window_frames_entry.get('window_index')}."
            ) from exc

        # Gaia G magnitude; treat as constant over this window.
        if "mag_G" not in star_entry:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_sidereal: missing "
                f"mag_G for star {star_key!r} in observer '{obs_name}', window "
                f"{window_frames_entry.get('window_index')}."
            )

        mag_val = float(np.asarray(star_entry["mag_G"], dtype=float).ravel()[0])

        # Convert magnitude to photon flux [ph m^-2 s^-1] at the aperture.
        phi_ph_m2_s_scalar = float(compute_star_photon_flux_from_mag(mag_val))

        # Broadcast scalar quantities over all frames in this window.
        x_pix = np.full(n_frames, x_epoch, dtype=float)
        y_pix = np.full(n_frames, y_epoch, dtype=float)
        phi_ph_m2_s = np.full(n_frames, phi_ph_m2_s_scalar, dtype=float)
        mag_G_series = np.full(n_frames, mag_val, dtype=float)
        on_detector_series = np.full(n_frames, on_det_flag, dtype=bool)

        # Per-frame photon counts: phi * exposure time.
        flux_ph_m2_frame = phi_ph_m2_s * t_exp_s

        # Assemble per-star time series record.
        star_rec: Dict[str, Any] = {
            "source_id": source_id,
            "source_type": "star",
            "gaia_source_id": gaia_source_id,
            "t_utc": np.asarray(t_utc_list, dtype=object),
            "t_exp_s": t_exp_s,
            "x_pix": x_pix,
            "y_pix": y_pix,
            "phi_ph_m2_s": phi_ph_m2_s,
            "flux_ph_m2_frame": flux_ph_m2_frame,
            "mag_G": mag_G_series,
            "on_detector": on_detector_series,
        }

        stars_timeseries[str(star_key)] = star_rec

    # ------------------------------------------------------------------
    # Build the window-level StarPhotonWindow wrapper.
    # ------------------------------------------------------------------
    window_index = window_frames_entry.get(
        "window_index", window_star_projection.get("window_index")
    )

    star_window: StarPhotonWindow = {
        "window_index": window_index,
        "start_index": window_frames_entry.get(
            "start_index", window_star_projection.get("start_index")
        ),
        "end_index": window_frames_entry.get(
            "end_index", window_star_projection.get("end_index")
        ),
        "n_frames": n_frames,

        # Prefer the photon-frame builder's naming, but fall back
        # gracefully if older field names are present instead.
        "t_start_utc": window_frames_entry.get(
            "t_start_utc", window_frames_entry.get("start_time")
        ),
        "t_end_utc": window_frames_entry.get(
            "t_end_utc", window_frames_entry.get("end_time")
        ),

        # Sky meta: prefer the star projection (which is copied from frames),
        # but allow falling back to the frames-with-sky entry if needed.
        "sky_center_ra_deg": window_star_projection.get(
            "sky_center_ra_deg", window_frames_entry.get("sky_center_ra_deg")
        ),
        "sky_center_dec_deg": window_star_projection.get(
            "sky_center_dec_deg", window_frames_entry.get("sky_center_dec_deg")
        ),
        "sky_radius_deg": window_star_projection.get(
            "sky_radius_deg", window_frames_entry.get("sky_radius_deg")
        ),
        "sky_selector_status": window_star_projection.get(
            "sky_selector_status", window_frames_entry.get("sky_selector_status")
        ),

        # Tracking label: whatever sim_test decided, falling back to the
        # provided mode for safety.
        "tracking_mode": window_frames_entry.get("tracking_mode", mode),

        "n_stars": len(stars_timeseries),
        "stars": stars_timeseries,
        "n_sources_total": len(stars_timeseries),
    }

    log.debug(
        "Observer '%s', window %s: built star photon time series for %d stars.",
        obs_name,
        window_index,
        star_window["n_stars"],
    )

    return star_window

def build_star_photon_timeseries_for_window_slew(
    obs_name: str,
    window_frames_entry: Dict[str, Any],
    window_star_slew_tracks: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    mode: str = "slew",
) -> StarPhotonWindow:
    """
    Build per-star photon time series for a single observer + window in **slew** mode,
    using ONLY the independent output of NEBULA_STAR_SLEW_PROJECTION.

    Key design constraints (per your request)
    -----------------------------------------
    1) No "mid-time" fields are read, required, documented, or returned.
       In particular: we do NOT reference or emit t_mid_utc / t_mid_mjd_utc.
    2) This helper is "slew-tracks-only":
       - It does not rely on NEBULA_STAR_PROJECTION outputs or concepts.
       - The only star-geometry input is `window_star_slew_tracks`, produced by
         NEBULA_STAR_SLEW_PROJECTION (per-star tracks over coarse indices).

    Conceptual behavior
    -------------------
    - The photon-frame builder provides the canonical per-frame grid for the window
      (each frame has a `coarse_index`, `t_utc`, and `t_exp_s`).
    - NEBULA_STAR_SLEW_PROJECTION provides, for each star, arrays sampled on the
      *coarse* grid: `coarse_indices[j]`, `x_pix[j]`, `y_pix[j]`, `on_detector[j]`.
    - This function aligns star samples to frames by matching `coarse_index` and
      produces per-frame arrays for each star.

    Expected input schema (minimal)
    -------------------------------
    window_frames_entry must provide (from frames-with-sky / frame catalog):
        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "t_start_utc": ...,
            "t_end_utc": ...,
            "tracking_mode": "slew" (recommended),
            "frames": [
                {
                    "frame_index": int,
                    "coarse_index": int,
                    "t_utc": datetime or str,
                    "t_exp_s": float,
                    ...
                },
                ...
            ],
            # Optional sky selector metadata is simply propagated if present:
            "sky_center_ra_deg": float (optional),
            "sky_center_dec_deg": float (optional),
            "sky_radius_deg": float (optional),
            "sky_selector_status": str (optional),
        }

    window_star_slew_tracks must provide (from NEBULA_STAR_SLEW_PROJECTION):
        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "coarse_indices": np.ndarray[int],
            "t_mjd_utc": np.ndarray[float],      # may exist; not required here
            "t_ref_mjd_utc": float,              # may exist; not required here
            "stars": {
                "<gaia_source_id_str>": {
                    "gaia_source_id": int or str,
                    "source_id": str (optional),
                    "mag_G": float,
                    "coarse_indices": np.ndarray[int],
                    "x_pix": np.ndarray[float],
                    "y_pix": np.ndarray[float],
                    "on_detector": np.ndarray[bool],
                    ...
                },
                ...
            },
        }

    Returned schema
    ---------------
    Matches the sidereal-window star schema you already use (start/end only; no mid-times):

        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,
            "t_start_utc": ...,
            "t_end_utc": ...,
            "sky_center_ra_deg": float or None,
            "sky_center_dec_deg": float or None,
            "sky_radius_deg": float or None,
            "sky_selector_status": str or None,
            "tracking_mode": "slew" (or from frames entry),
            "n_stars": int,
            "stars": {
                "<gaia_source_id_str>": {
                    "source_id": str,
                    "source_type": "star",
                    "gaia_source_id": int or str,
                    "t_utc": np.ndarray[object],
                    "t_exp_s": np.ndarray[float],
                    "x_pix": np.ndarray[float],
                    "y_pix": np.ndarray[float],
                    "phi_ph_m2_s": np.ndarray[float],
                    "flux_ph_m2_frame": np.ndarray[float],
                    "mag_G": np.ndarray[float],
                    "on_detector": np.ndarray[bool],
                },
                ...
            },
            "n_sources_total": int,  # == n_stars for now
        }

    Parameters / Returns / Raises
    -----------------------------
    Same intent as your previous version:
    - ValueError if mode != "slew"
    - RuntimeError for missing/inconsistent required inputs
    """
    log = _get_logger(logger)

    # Enforce that this helper is only used for slew windows.
    if mode != "slew":
        raise ValueError(
            "build_star_photon_timeseries_for_window_slew: only 'slew' mode is "
            f"implemented here, got mode={mode!r} for observer '{obs_name}'."
        )

    # # ------------------------------------------------------------------
    # # Validate window_frames_entry and extract canonical per-frame fields.
    # # ------------------------------------------------------------------
    # frames = window_frames_entry.get("frames", None)
    # if not frames:
    #     raise RuntimeError(
    #         "build_star_photon_timeseries_for_window_slew: window "
    #         f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
    #         "has no 'frames' entry."
    #     )

    # n_frames = len(frames)

    # t_utc_list: List[Any] = []
    # t_exp_s_list: List[float] = []
    # coarse_idx_list: List[int] = []

    # for f in frames:
    #     if "t_utc" not in f:
    #         raise RuntimeError(
    #             "build_star_photon_timeseries_for_window_slew: frame in window "
    #             f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
    #             "is missing 't_utc'."
    #         )
    #     if "t_exp_s" not in f:
    #         raise RuntimeError(
    #             "build_star_photon_timeseries_for_window_slew: frame in window "
    #             f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
    #             "is missing 't_exp_s'."
    #         )
    #     if "coarse_index" not in f:
    #         raise RuntimeError(
    #             "build_star_photon_timeseries_for_window_slew: frame in window "
    #             f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
    #             "is missing 'coarse_index'."
    #         )

    #     t_utc_list.append(f["t_utc"])
    #     t_exp_s_list.append(float(f["t_exp_s"]))
    #     coarse_idx_list.append(int(f["coarse_index"]))

    # t_utc_arr = np.asarray(t_utc_list, dtype=object)
    # t_exp_s = np.asarray(t_exp_s_list, dtype=float)
    # coarse_idx_frames = np.asarray(coarse_idx_list, dtype=int)

    # # Precompute: coarse_index -> list of frame indices (frames may share a coarse index).
    # frames_by_coarse: Dict[int, List[int]] = {}
    # for i_frame, ci in enumerate(coarse_idx_frames):
    #     frames_by_coarse.setdefault(int(ci), []).append(i_frame)

    # ------------------------------------------------------------------
    # Validate window_frames_entry and extract canonical per-frame fields.
    # ------------------------------------------------------------------
    frames = window_frames_entry.get("frames", None)
    if not frames:
        raise RuntimeError(
            "build_star_photon_timeseries_for_window_slew: window "
            f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
            "has no 'frames' entry."
        )

    # Respect the window's declared length when present (do not blindly trust len(frames)).
    n_frames_window = window_frames_entry.get("n_frames", None)
    n_frames = int(n_frames_window) if n_frames_window is not None else len(frames)

    if len(frames) < n_frames:
        raise RuntimeError(
            "build_star_photon_timeseries_for_window_slew: frame catalog has only "
            f"{len(frames)} frames but window declares n_frames={n_frames} for "
            f"observer '{obs_name}', window_index={window_frames_entry.get('window_index')}."
        )

    # Truncate if upstream provided a longer list than the window declares.
    frames = frames[:n_frames]

    t_utc_list: List[Any] = []
    t_exp_s_list: List[float] = []
    coarse_idx_list: List[int] = []

    for f in frames:
        if "t_utc" not in f:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_slew: frame in window "
                f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
                "is missing 't_utc'."
            )
        if "t_exp_s" not in f:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_slew: frame in window "
                f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
                "is missing 't_exp_s'."
            )
        if "coarse_index" not in f:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_slew: frame in window "
                f"{window_frames_entry.get('window_index')} for observer '{obs_name}' "
                "is missing 'coarse_index'."
            )

        t_utc_list.append(f["t_utc"])
        t_exp_s_list.append(float(f["t_exp_s"]))
        coarse_idx_list.append(int(f["coarse_index"]))

    t_utc_arr = np.asarray(t_utc_list, dtype=object)
    t_exp_s = np.asarray(t_exp_s_list, dtype=float)
    coarse_idx_frames = np.asarray(coarse_idx_list, dtype=int)

    # Precompute: coarse_index -> list of frame indices (frames may share a coarse index).
    frames_by_coarse: Dict[int, List[int]] = {}
    for i_frame, ci in enumerate(coarse_idx_frames):
        frames_by_coarse.setdefault(int(ci), []).append(i_frame)




    # ------------------------------------------------------------------
    # Validate window_star_slew_tracks and pull star-track dictionary.
    # ------------------------------------------------------------------
    stars_tracks: Dict[str, Any] = window_star_slew_tracks.get("stars", {})
    if stars_tracks is None:
        stars_tracks = {}

    # Optional but strongly recommended: ensure we are pairing the same window.
    # We fail loudly on clear mismatches (start/end/index) because silent pairing
    # errors are extremely expensive to debug downstream.
    wf_wi = window_frames_entry.get("window_index")
    wt_wi = window_star_slew_tracks.get("window_index")
    if wf_wi is not None and wt_wi is not None and int(wf_wi) != int(wt_wi):
        raise RuntimeError(
            "build_star_photon_timeseries_for_window_slew: window_index mismatch "
            f"for observer '{obs_name}': frames={wf_wi} vs slew_tracks={wt_wi}."
        )

    wf_si = window_frames_entry.get("start_index")
    wf_ei = window_frames_entry.get("end_index")
    wt_si = window_star_slew_tracks.get("start_index")
    wt_ei = window_star_slew_tracks.get("end_index")
    if (wf_si is not None and wt_si is not None and int(wf_si) != int(wt_si)) or (
        wf_ei is not None and wt_ei is not None and int(wf_ei) != int(wt_ei)
    ):
        raise RuntimeError(
            "build_star_photon_timeseries_for_window_slew: start/end index mismatch "
            f"for observer '{obs_name}', window {wf_wi}: "
            f"frames=({wf_si},{wf_ei}) vs slew_tracks=({wt_si},{wt_ei})."
        )

    stars_timeseries: Dict[str, Dict[str, Any]] = {}

    if len(stars_tracks) == 0:
        log.debug(
            "Observer '%s', window %s (slew): no stars in slew tracks; returning empty StarPhotonWindow.",
            obs_name,
            wf_wi if wf_wi is not None else wt_wi,
        )

    # ------------------------------------------------------------------
    # Build per-star per-frame records by coarse_index alignment.
    # ------------------------------------------------------------------
    for star_key, star_entry in stars_tracks.items():
        gaia_source_id = star_entry.get("gaia_source_id", star_key)
        source_id = star_entry.get("source_id", str(star_key))

        # Required geometry arrays for this star track.
        try:
            star_coarse_idx = np.asarray(star_entry["coarse_indices"], dtype=int)
            x_track = np.asarray(star_entry["x_pix"], dtype=float)
            y_track = np.asarray(star_entry["y_pix"], dtype=float)
        except KeyError as exc:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_slew: missing required "
                f"'coarse_indices'/'x_pix'/'y_pix' for star {star_key!r} in "
                f"observer '{obs_name}', window {wf_wi if wf_wi is not None else wt_wi}."
            ) from exc

        on_det_track = np.asarray(
            star_entry.get("on_detector", np.ones_like(x_track, dtype=bool)),
            dtype=bool,
        )

        if not (star_coarse_idx.shape == x_track.shape == y_track.shape == on_det_track.shape):
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_slew: inconsistent array lengths "
                f"for star {star_key!r} in observer '{obs_name}', window {wf_wi if wf_wi is not None else wt_wi}."
            )

        # If the star is never on-detector at the coarse samples, it cannot contribute.
        if not np.any(on_det_track):
            continue

        if "mag_G" not in star_entry:
            raise RuntimeError(
                "build_star_photon_timeseries_for_window_slew: missing mag_G "
                f"for star {star_key!r} in observer '{obs_name}', window {wf_wi if wf_wi is not None else wt_wi}."
            )
        mag_val = float(np.asarray(star_entry["mag_G"], dtype=float).ravel()[0])

        # Scalar photon flux at the aperture from magnitude (your existing helper).
        phi_ph_m2_s_scalar = float(compute_star_photon_flux_from_mag(mag_val))

        # Allocate per-frame arrays (defaults mean "no sample / off-detector").
        x_pix = np.full(n_frames, np.nan, dtype=float)
        y_pix = np.full(n_frames, np.nan, dtype=float)
        on_detector_series = np.zeros(n_frames, dtype=bool)

        # Build mapping: coarse_index -> track index (duplicate coarse indices indicate a projection bug).
        track_by_coarse: Dict[int, int] = {}
        for idx_ci, ci in enumerate(star_coarse_idx):
            ici = int(ci)
            if ici in track_by_coarse:
                raise RuntimeError(
                    "build_star_photon_timeseries_for_window_slew: duplicate coarse_index "
                    f"{ici} for star {star_key!r} in observer '{obs_name}', window {wf_wi if wf_wi is not None else wt_wi}."
                )
            track_by_coarse[ici] = idx_ci

        # Fill per-frame values by matching coarse_index.
        # If multiple frames share the same coarse_index, they all receive the same star sample.
        for ci_frame, frame_indices in frames_by_coarse.items():
            idx_track = track_by_coarse.get(ci_frame)
            if idx_track is None:
                continue

            # Apply the star sample to all frames with this coarse index.
            x_val = float(x_track[idx_track])
            y_val = float(y_track[idx_track])
            on_val = bool(on_det_track[idx_track])

            x_pix[frame_indices] = x_val
            y_pix[frame_indices] = y_val
            on_detector_series[frame_indices] = on_val

        # Drop stars that end up never on-detector on the frame grid.
        if not np.any(on_detector_series):
            continue

        # Broadcast magnitude across frames.
        mag_G_series = np.full(n_frames, mag_val, dtype=float)

        # Photon flux time series:
        # We zero out flux where the star is off-detector so downstream code can safely sum fluxes
        # without having to remember to apply on_detector masks.
        phi_ph_m2_s = np.where(on_detector_series, phi_ph_m2_s_scalar, 0.0).astype(float)
        flux_ph_m2_frame = phi_ph_m2_s * t_exp_s

        stars_timeseries[str(star_key)] = {
            "source_id": source_id,
            "source_type": "star",
            "gaia_source_id": gaia_source_id,
            "t_utc": t_utc_arr,
            "t_exp_s": t_exp_s,
            "x_pix": x_pix,
            "y_pix": y_pix,
            "phi_ph_m2_s": phi_ph_m2_s,
            "flux_ph_m2_frame": flux_ph_m2_frame,
            "mag_G": mag_G_series,
            "on_detector": on_detector_series,
        }

    # ------------------------------------------------------------------
    # Window-level wrapper (NO mid-time fields).
    # ------------------------------------------------------------------
    window_index = window_frames_entry.get("window_index", window_star_slew_tracks.get("window_index"))

    star_window: StarPhotonWindow = {
        "window_index": window_index,
        "start_index": window_frames_entry.get("start_index"),
        "end_index": window_frames_entry.get("end_index"),
        "n_frames": n_frames,
        "t_start_utc": window_frames_entry.get("t_start_utc", window_frames_entry.get("start_time")),
        "t_end_utc": window_frames_entry.get("t_end_utc", window_frames_entry.get("end_time")),
        "sky_center_ra_deg": window_frames_entry.get("sky_center_ra_deg"),
        "sky_center_dec_deg": window_frames_entry.get("sky_center_dec_deg"),
        "sky_radius_deg": window_frames_entry.get("sky_radius_deg"),
        "sky_selector_status": window_frames_entry.get("sky_selector_status"),
        "tracking_mode": window_frames_entry.get("tracking_mode", mode),
        "n_stars": len(stars_timeseries),
        "stars": stars_timeseries,
        "n_sources_total": len(stars_timeseries),
    }

    log.debug(
        "Observer '%s', window %s (slew): built star photon time series for %d stars.",
        obs_name,
        window_index,
        star_window["n_stars"],
    )

    return star_window

def build_star_photon_timeseries_for_window(
    obs_name: str,
    window_frames_entry: Dict[str, Any],
    window_star_projection: Optional[Dict[str, Any]] = None,
    window_star_slew_entry: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> StarPhotonWindow:
    """
    Dispatch helper: build per-star photon time series for one window,
    choosing the appropriate implementation based on tracking_mode.

    This is the *single* entry point that higher-level code (e.g.,
    "build_star_photons_for_all_observers") should call.

    It inspects the window's tracking_mode (as annotated earlier by
    annotate_windows_with_tracking_mode in sim_test) and then:

        - If tracking_mode == "sidereal":
              -> calls build_star_photon_timeseries_for_window_sidereal(...)
                 using window_star_projection.

        - If tracking_mode == "slew":
              -> calls build_star_photon_timeseries_for_window_slew(...)
                 using window_star_slew_entry.

    Any missing inputs or unknown tracking_mode values are treated as
    *fatal* errors (fail-hard), since they indicate a mismatch between
    the windows, projection products, or earlier classification.

    Parameters
    ----------
    obs_name : str
        Name of the observer (for logging / error messages).

    window_frames_entry : dict
        Single-window entry from the frames-with-sky structure for this
        observer, i.e. one element of:
            frames_with_sky[obs_name]["windows"].

        Must contain a "tracking_mode" field set to "sidereal" or "slew"
        by annotate_windows_with_tracking_mode.

    window_star_projection : dict or None, optional
        Matching single-window entry from obs_star_projections for this
        observer, used when tracking_mode == "sidereal". If None in that
        case, a RuntimeError is raised.

    window_star_slew_entry : dict or None, optional
        Matching single-window entry from obs_star_slew_tracks for this
        observer, used when tracking_mode == "slew". If None in that
        case, a RuntimeError is raised.

    logger : logging.Logger, optional
        Logger for informational / debug messages. If None, a module-
        level logger obtained via _get_logger() is used.

    Returns
    -------
    StarPhotonWindow
        Dictionary containing per-star photon time series for this
        observer + window, with either sidereal or slew kinematics
        applied as appropriate.

    Raises
    ------
    RuntimeError
        If tracking_mode is missing, or the required star projection
        entry for the selected mode is None.

    ValueError
        If tracking_mode is not one of the recognized values
        ("sidereal", "slew").
    """
    log = _get_logger(logger)

    # ------------------------------------------------------------------
    # Determine tracking mode for this window.
    # ------------------------------------------------------------------
    mode_raw = window_frames_entry.get("tracking_mode", None)
    if mode_raw is None:
        raise RuntimeError(
            f"build_star_photon_timeseries_for_window: window "
            f"{window_frames_entry.get('window_index')} for observer "
            f"'{obs_name}' has no 'tracking_mode' field. "
            f"Did you run annotate_windows_with_tracking_mode first?"
        )

    mode = str(mode_raw).lower()

    # ------------------------------------------------------------------
    # Dispatch based on mode.
    # ------------------------------------------------------------------
    if mode == "sidereal":
        if window_star_projection is None:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: sidereal window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' has no matching window_star_projection."
            )

        return build_star_photon_timeseries_for_window_sidereal(
            obs_name=obs_name,
            window_frames_entry=window_frames_entry,
            window_star_projection=window_star_projection,
            logger=log,
            mode="sidereal",
        )

    elif mode == "slew":
        if window_star_slew_entry is None:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: slew window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' has no matching window_star_slew_entry."
            )

        return build_star_photon_timeseries_for_window_slew(
            obs_name=obs_name,
            window_frames_entry=window_frames_entry,
            window_star_slew_tracks=window_star_slew_entry,  
            logger=log,
            mode="slew",
        )

    else:
        raise ValueError(
            f"build_star_photon_timeseries_for_window: unknown tracking_mode "
            f"{mode_raw!r} for observer '{obs_name}', window "
            f"{window_frames_entry.get('window_index')}."
        )
'''
def build_star_photons_for_observer(
    obs_name: str,
    frames_for_obs: Dict[str, Any],
    star_projections_for_obs: Dict[str, Any],
    star_slew_tracks_for_obs: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build star photon time series for all windows of a single observer.

    This is the main per-observer helper for NEBULA_STAR_PHOTONS. It takes:

      * the frames-with-sky view for one observer (``frames_for_obs``),
      * the corresponding sidereal star projections for that observer
        (``star_projections_for_obs`` from NEBULA_STAR_PROJECTION), and
      * optionally, the per-frame star tracks for slewing windows
        (``star_slew_tracks_for_obs`` from NEBULA_STAR_SLEW_PROJECTION),

    and returns a per-observer star-photon catalog:

        {
            "observer_name": str,
            "rows": int,
            "cols": int,
            "catalog_name": str,
            "catalog_band": str,
            "run_meta": {...},
            "windows": [StarPhotonWindow, ...],
        }

    For each window in ``frames_for_obs["windows"]``:

      1. Determine the tracking mode from
             window["tracking_mode"]  (annotated earlier by sim_test).

      2. Use ``window_index`` to look up the matching star information:

             * For ``tracking_mode == "sidereal"``:
                   star_projections_for_obs["windows"] is used, and
                   build_star_photon_timeseries_for_window_sidereal(...)
                   is called.

             * For ``tracking_mode == "slew"``:
                   star_slew_tracks_for_obs["windows"] is used, and
                   build_star_photon_timeseries_for_window_slew(...)
                   is called.

      3. Collect the resulting StarPhotonWindow dict into the output
         list ``windows`` for this observer.

    STRICT / fail-loud behaviour
    ----------------------------
    This function is intentionally strict to catch pipeline mismatches:

      * If a window has an unrecognized tracking_mode, a ValueError is raised.
      * If a sidereal window is missing a matching entry in
        ``star_projections_for_obs["windows"]``, a RuntimeError is raised.
      * If a slew window is missing either ``star_slew_tracks_for_obs`` or
        the matching entry in ``star_slew_tracks_for_obs["windows"]``,
        a RuntimeError is raised.

    Parameters
    ----------
    obs_name : str
        Name of the observer whose windows are being processed.

    frames_for_obs : dict
        Frames-with-sky structure for this observer, as produced by
        NEBULA_PHOTON_FRAME_BUILDER / NEBULA_TARGET_PHOTONS. Expected
        minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "windows": [
                    {
                        "window_index": int,
                        "tracking_mode": str,  # "sidereal" or "slew"
                        "frames": [...],
                        ...
                    },
                    ...
                ],
                ...
            }

    star_projections_for_obs : dict
        Per-observer star projection product from NEBULA_STAR_PROJECTION.
        Expected minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "run_meta": {...},
                "windows": [
                    {
                        "window_index": int,
                        "n_stars_input": int,
                        "n_stars_on_detector": int,
                        "stars": { ... },
                        ...
                    },
                    ...
                ],
            }

        There should be one StarWindowProjection entry per window_index,
        even if it contains zero stars.

    star_slew_tracks_for_obs : dict or None, optional
        Per-observer star tracks for slewing windows from
        NEBULA_STAR_SLEW_PROJECTION. Expected minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "run_meta": {...},
                "windows": [
                    {
                        "window_index": int,
                        "coarse_indices": [...],
                        "stars": {
                            "<gaia_source_id_str>": {
                                "gaia_source_id": int or str,
                                "source_id": str,
                                "x_pix": np.ndarray[float],
                                "y_pix": np.ndarray[float],
                                "on_detector": np.ndarray[bool],
                                "mag_G": float,
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    ...
                ],
            }

        If None, any window in slewing mode will cause a RuntimeError.

    logger : logging.Logger or None, optional
        Logger for informational / debug messages. If None, a module-level
        logger from _get_logger() is used.

    Returns
    -------
    dict
        Per-observer star photon catalog with the schema:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "catalog_name": str,
                "catalog_band": str,
                "run_meta": {...},
                "windows": [StarPhotonWindow, ...],
            }

        where each element of "windows" is the result of either
        build_star_photon_timeseries_for_window_sidereal(...) or
        build_star_photon_timeseries_for_window_slew(...).

    Raises
    ------
    RuntimeError
        If a required star window (sidereal or slew) is missing, or if
        frames_for_obs does not have the expected "windows" structure.

    ValueError
        If a window has an unsupported tracking_mode.
    """
    # Resolve a logger for this function.
    log = _get_logger(logger)

    # ------------------------------------------------------------------
    # Basic validation of input structures.
    # ------------------------------------------------------------------
    windows_frames = frames_for_obs.get("windows", None)
    if windows_frames is None:
        raise RuntimeError(
            f"build_star_photons_for_observer: frames_for_obs for observer "
            f"'{obs_name}' is missing a 'windows' entry."
        )

    # Determine whether this observer actually needs sidereal projections.
    has_sidereal = any(
        str(w.get("tracking_mode", "")).lower() == "sidereal"
        for w in windows_frames
    )

    # Build lookup: window_index -> StarWindowProjection (sidereal), only if needed.
    sidereal_by_index: Dict[int, Dict[str, Any]] = {}
    if has_sidereal:
        if star_projections_for_obs is None:
            raise RuntimeError(
                f"build_star_photons_for_observer: observer '{obs_name}' has "
                f"sidereal windows but star_projections_for_obs is None."
            )

        sidereal_windows = star_projections_for_obs.get("windows", None)
        if sidereal_windows is None:
            raise RuntimeError(
                f"build_star_photons_for_observer: star_projections_for_obs for "
                f"observer '{obs_name}' is missing a 'windows' entry."
            )

        for w in sidereal_windows:
            idx = int(w.get("window_index"))
            if idx in sidereal_by_index:
                raise RuntimeError(
                    f"build_star_photons_for_observer: duplicate sidereal "
                    f"window_index={idx} for observer '{obs_name}'."
                )
            sidereal_by_index[idx] = w

    # If slew tracks are provided, build a similar lookup.
    slew_by_index: Dict[int, Dict[str, Any]] = {}
    if star_slew_tracks_for_obs is not None:
        slew_windows = star_slew_tracks_for_obs.get("windows", [])
        for w in slew_windows:
            idx = int(w.get("window_index"))
            if idx in slew_by_index:
                raise RuntimeError(
                    f"build_star_photons_for_observer: duplicate slew "
                    f"window_index={idx} for observer '{obs_name}'."
                )
            slew_by_index[idx] = w

    # ------------------------------------------------------------------
    # Loop over all windows for this observer and build star photons.
    # ------------------------------------------------------------------
    star_windows: List[StarPhotonWindow] = []

    for window_frames_entry in windows_frames:
        if "window_index" not in window_frames_entry:
            raise RuntimeError(
                f"build_star_photons_for_observer: a window for observer "
                f"'{obs_name}' is missing 'window_index'."
            )

        window_index = int(window_frames_entry["window_index"])

        # tracking_mode MUST have been annotated earlier.
        mode_raw = window_frames_entry.get("tracking_mode", None)
        if mode_raw is None:
            raise RuntimeError(
                f"build_star_photons_for_observer: window_index={window_index} "
                f"for observer '{obs_name}' has no 'tracking_mode'. "
                f"Did you run annotate_windows_with_tracking_mode?"
            )
        mode = str(mode_raw).lower()

        # Look up matching star products; we do NOT assume anything about
        # mid-window fields here, only that the per-window dicts exist.
        window_star_proj: Optional[Dict[str, Any]] = None
        window_star_slew_entry: Optional[Dict[str, Any]] = None

        if mode == "sidereal":
            window_star_proj = sidereal_by_index.get(window_index)
            if window_star_proj is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: no sidereal star "
                    f"projection found for observer '{obs_name}', "
                    f"window_index={window_index}."
                )
        elif mode == "slew":
            if star_slew_tracks_for_obs is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: encountered a 'slew' "
                    f"window (index={window_index}) for observer '{obs_name}', "
                    f"but star_slew_tracks_for_obs is None."
                )
            window_star_slew_entry = slew_by_index.get(window_index)
            if window_star_slew_entry is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: no slew star tracks "
                    f"found for observer '{obs_name}', window_index={window_index}."
                )
        else:
            raise ValueError(
                f"build_star_photons_for_observer: unsupported tracking_mode="
                f"{mode_raw!r} for observer '{obs_name}', "
                f"window_index={window_index}."
            )

        # Delegate to the unified per-window dispatcher, which itself
        # chooses the correct sidereal/slew implementation and assumes
        # no mid-window pixel/time fields.
        star_window = build_star_photon_timeseries_for_window(
            obs_name=obs_name,
            window_frames_entry=window_frames_entry,
            window_star_projection=window_star_proj,
            window_star_slew_entry=window_star_slew_entry,
            logger=log,
        )

        star_windows.append(star_window)

    # ------------------------------------------------------------------
    # Build the per-observer wrapper structure.
    # ------------------------------------------------------------------
    observer_name = frames_for_obs.get("observer_name", obs_name)
    rows = int(frames_for_obs.get("rows", getattr(ACTIVE_SENSOR, "rows", 0)))
    cols = int(frames_for_obs.get("cols", getattr(ACTIVE_SENSOR, "cols", 0)))

    catalog_name = getattr(
        NEBULA_STAR_CATALOG, "name",
        getattr(NEBULA_STAR_CATALOG, "catalog_name", "Gaia"),
    )
    catalog_band = getattr(NEBULA_STAR_CATALOG, "band", "G")

    run_meta: Dict[str, Any] = {
        "star_photons_version": STAR_PHOTONS_RUN_META_VERSION,
        "builder": "NEBULA_STAR_PHOTONS.build_star_photons_for_observer",
        "observer_name": observer_name,
        "n_windows_input": len(windows_frames),
        "n_windows_output": len(star_windows),
        "created_utc": datetime.now(timezone.utc).isoformat() + "Z",
    }

    obs_star_photons: Dict[str, Any] = {
        "observer_name": observer_name,
        "rows": rows,
        "cols": cols,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "run_meta": run_meta,
        "windows": star_windows,
    }

    log.info(
        "build_star_photons_for_observer: built star photon catalog for "
        "observer '%s' with %d windows.",
        observer_name,
        len(star_windows),
    )

    return obs_star_photons
'''
# AFTER (corrected)
def build_star_photons_for_observer( 
    obs_name: str,
    frames_for_obs: Dict[str, Any],
    star_projections_for_obs: Dict[str, Any],
    star_slew_tracks_for_obs: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build star photon time series for all windows of a single observer.

    This is the main per-observer helper for NEBULA_STAR_PHOTONS. It takes:

      * the frames-with-sky view for one observer (``frames_for_obs``),
      * the corresponding sidereal star projections for that observer
        (``star_projections_for_obs`` from NEBULA_STAR_PROJECTION), and
      * optionally, the per-frame star tracks for slewing windows
        (``star_slew_tracks_for_obs`` from NEBULA_STAR_SLEW_PROJECTION),

    and returns a per-observer star-photon catalog:

        {
            "observer_name": str,
            "rows": int,
            "cols": int,
            "catalog_name": str,
            "catalog_band": str,
            "run_meta": {...},
            "windows": [StarPhotonWindow, ...],
        }

    For each window in ``frames_for_obs["windows"]``:

      1. Determine the tracking mode from
             window["tracking_mode"]  (annotated earlier by sim_test).

      2. Use ``window_index`` to look up the matching star information:

             * For ``tracking_mode == "sidereal"``:
                   star_projections_for_obs["windows"] is used, and
                   build_star_photon_timeseries_for_window_sidereal(...)
                   is called.

             * For ``tracking_mode == "slew"``:
                   star_slew_tracks_for_obs["windows"] is used, and
                   build_star_photon_timeseries_for_window_slew(...)
                   is called.

      3. Collect the resulting StarPhotonWindow dict into the output
         list ``windows`` for this observer.

    Timebase construction / target-agnostic behavior
    -----------------------------------------------
    In some pipeline products (including your obs_target_frames_ranked_with_sky.pkl),
    windows may **not** carry a full per-frame "frames" list. When that happens,
    downstream star-photon builders can fall back to using per-target time series
    (e.g., the first target's t_utc array) to infer the window timebase.

    This is invalid when target series are *sparse* (only present while a target is
    "active/on-detector") because those arrays are shorter than the true window
    n_frames. The result is a hard failure such as:

        "... has only 1023 frames but window n_frames=7272"

    To make STAR_PHOTONS target-agnostic, this function ensures that each window
    passed downstream has a canonical per-frame timebase by synthesizing
    window["frames"] from window metadata:

      * window["start_time"] (preferred) or window["end_time"] (fallback)
      * frames_for_obs["dt_frame_s"] (cadence / exposure)
      * window["start_index"] and window["n_frames"]

    This is applied for BOTH sidereal and slew windows.

    STRICT / fail-loud behaviour
    ----------------------------
    This function is intentionally strict to catch pipeline mismatches:

      * If a window has an unrecognized tracking_mode, a ValueError is raised.
      * If a sidereal window is missing a matching entry in
        ``star_projections_for_obs["windows"]``, a RuntimeError is raised.
      * If a slew window is missing either ``star_slew_tracks_for_obs`` or
        the matching entry in ``star_slew_tracks_for_obs["windows"]``,
        a RuntimeError is raised.

    Parameters
    ----------
    obs_name : str
        Name of the observer whose windows are being processed.

    frames_for_obs : dict
        Frames-with-sky structure for this observer, as produced by
        NEBULA_PHOTON_FRAME_BUILDER / NEBULA_TARGET_PHOTONS. Expected
        minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "windows": [
                    {
                        "window_index": int,
                        "tracking_mode": str,  # "sidereal" or "slew"
                        "frames": [...],
                        ...
                    },
                    ...
                ],
                ...
            }

    star_projections_for_obs : dict
        Per-observer star projection product from NEBULA_STAR_PROJECTION.
        Expected minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "run_meta": {...},
                "windows": [
                    {
                        "window_index": int,
                        "n_stars_input": int,
                        "n_stars_on_detector": int,
                        "stars": { ... },
                        ...
                    },
                    ...
                ],
            }

        There should be one StarWindowProjection entry per window_index,
        even if it contains zero stars.

    star_slew_tracks_for_obs : dict or None, optional
        Per-observer star tracks for slewing windows from
        NEBULA_STAR_SLEW_PROJECTION. Expected minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "run_meta": {...},
                "windows": [
                    {
                        "window_index": int,
                        "coarse_indices": [...],
                        "stars": {
                            "<gaia_source_id_str>": {
                                "gaia_source_id": int or str,
                                "source_id": str,
                                "x_pix": np.ndarray[float],
                                "y_pix": np.ndarray[float],
                                "on_detector": np.ndarray[bool],
                                "mag_G": float,
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    ...
                ],
            }

        If None, any window in slewing mode will cause a RuntimeError.

    logger : logging.Logger or None, optional
        Logger for informational / debug messages. If None, a module-level
        logger from _get_logger() is used.

    Returns
    -------
    dict
        Per-observer star photon catalog with the schema:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "catalog_name": str,
                "catalog_band": str,
                "run_meta": {...},
                "windows": [StarPhotonWindow, ...],
            }

        where each element of "windows" is the result of either
        build_star_photon_timeseries_for_window_sidereal(...) or
        build_star_photon_timeseries_for_window_slew(...).

    Raises
    ------
    RuntimeError
        If a required star window (sidereal or slew) is missing, or if
        frames_for_obs does not have the expected "windows" structure.

    ValueError
        If a window has an unsupported tracking_mode.
    """
    # Resolve a logger for this function.
    log = _get_logger(logger)

    # ------------------------------------------------------------------
    # Helper: coerce datetime-like values (datetime or ISO8601 str) to datetime.
    # ------------------------------------------------------------------
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            s = value.strip()
            # Handle common UTC "Z" suffix in a stdlib-friendly way.
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Helper: ensure window_frames_entry contains a canonical per-frame timebase.
    # ------------------------------------------------------------------
    def _ensure_window_frames_timebase(
        window_entry: Dict[str, Any],
        dt_frame_s: Optional[float],
    ) -> Dict[str, Any]:
        """
        Ensure window_entry has a full-length window_entry["frames"] list.

        The STAR_PHOTONS builders require a per-frame time array of length n_frames.
        If "frames" is missing, or present but length-mismatched, we synthesize it
        using window metadata (start_time/end_time, start_index, n_frames, dt_frame_s).

        IMPORTANT: We return a shallow copy so we do not mutate the original
        pickle-backed window dict stored inside frames_for_obs.
        """
        window_copy: Dict[str, Any] = dict(window_entry)

        # Determine cadence (dt_frame_s) from the observer-level product if possible.
        dt = dt_frame_s
        if dt is None:
            dt = window_copy.get("dt_frame_s", None)
        if dt is not None:
            dt = float(dt)

        # Determine n_frames for this window.
        n_frames_raw = window_copy.get("n_frames", None)
        frames_existing = window_copy.get("frames", None)

        if n_frames_raw is None:
            # If n_frames is missing, attempt a best-effort inference.
            if isinstance(frames_existing, list):
                n_frames = len(frames_existing)
            else:
                raise RuntimeError(
                    f"_ensure_window_frames_timebase: window_index={window_copy.get('window_index')} "
                    f"for observer '{obs_name}' is missing 'n_frames' and has no usable 'frames' list."
                )
        else:
            n_frames = int(n_frames_raw)

        # If frames exist and are the right length, accept them as-is.
        if isinstance(frames_existing, list) and len(frames_existing) == n_frames:
            return window_copy

        # If frames exist but are length-mismatched, we will rebuild them from metadata.
        # This avoids any downstream fallback to sparse per-target series.
        if isinstance(frames_existing, list) and len(frames_existing) != n_frames:
            log.info(
                "_ensure_window_frames_timebase: observer '%s' window_index=%s has frames len=%d != n_frames=%d; "
                "rebuilding canonical frames list from window metadata.",
                obs_name,
                window_copy.get("window_index"),
                len(frames_existing),
                n_frames,
            )

        # We require dt to synthesize times meaningfully.
        if dt is None:
            raise RuntimeError(
                f"_ensure_window_frames_timebase: cannot synthesize frames for observer '{obs_name}', "
                f"window_index={window_copy.get('window_index')}, because dt_frame_s is missing."
            )

        # Determine start_index for coarse_index bookkeeping.
        start_index_raw = window_copy.get("start_index", None)
        start_index = int(start_index_raw) if start_index_raw is not None else 0

        # Determine start_time (preferred) or compute from end_time (fallback).
        start_time = _coerce_datetime(window_copy.get("start_time", None))
        if start_time is None:
            end_time = _coerce_datetime(window_copy.get("end_time", None))
            if end_time is not None:
                start_time = end_time - timedelta(seconds=dt * float(max(n_frames - 1, 0)))
            else:
                # Last resort: infer start_time from any available target series by back-propagating
                # from its first (t_utc, coarse_index) to window start_index. This does NOT use the
                # *length* of target series (which is the original bug), only a single anchor sample.
                targets = window_copy.get("targets", {}) or {}
                inferred_start: Optional[datetime] = None
                for t_entry in targets.values():
                    try:
                        t_utc_arr = t_entry.get("t_utc", None)
                        c_idx_arr = t_entry.get("coarse_indices", None)
                        if t_utc_arr is None or c_idx_arr is None:
                            continue
                        if len(t_utc_arr) < 1 or len(c_idx_arr) < 1:
                            continue
                        t0 = _coerce_datetime(t_utc_arr[0])
                        c0 = int(c_idx_arr[0])
                        if t0 is None:
                            continue
                        inferred_start = t0 - timedelta(seconds=dt * float(c0 - start_index))
                        break
                    except Exception:
                        continue

                if inferred_start is None:
                    raise RuntimeError(
                        f"_ensure_window_frames_timebase: cannot synthesize frames for observer '{obs_name}', "
                        f"window_index={window_copy.get('window_index')}, because start_time/end_time are missing "
                        "and no usable target anchor sample could be inferred."
                    )
                start_time = inferred_start

        # Synthesize canonical frames list matching the PHOTON_FRAMES schema style.
        frames_list: List[Dict[str, Any]] = []
        for i in range(n_frames):
            frames_list.append(
                {
                    "coarse_index": start_index + i,
                    "sources": [],
                    "t_exp_s": float(dt),
                    "t_utc": start_time + timedelta(seconds=float(dt) * float(i)),
                }
            )

        window_copy["frames"] = frames_list
        return window_copy

    # ------------------------------------------------------------------
    # Basic validation of input structures.
    # ------------------------------------------------------------------
    windows_frames = frames_for_obs.get("windows", None)
    if windows_frames is None:
        raise RuntimeError(
            f"build_star_photons_for_observer: frames_for_obs for observer "
            f"'{obs_name}' is missing a 'windows' entry."
        )

    # Determine whether this observer actually needs sidereal projections.
    has_sidereal = any(
        str(w.get("tracking_mode", "")).lower() == "sidereal"
        for w in windows_frames
    )

    # Build lookup: window_index -> StarWindowProjection (sidereal), only if needed.
    sidereal_by_index: Dict[int, Dict[str, Any]] = {}
    if has_sidereal:
        if star_projections_for_obs is None:
            raise RuntimeError(
                f"build_star_photons_for_observer: observer '{obs_name}' has "
                f"sidereal windows but star_projections_for_obs is None."
            )

        sidereal_windows = star_projections_for_obs.get("windows", None)
        if sidereal_windows is None:
            raise RuntimeError(
                f"build_star_photons_for_observer: star_projections_for_obs for "
                f"observer '{obs_name}' is missing a 'windows' entry."
            )

        for w in sidereal_windows:
            idx = int(w.get("window_index"))
            if idx in sidereal_by_index:
                raise RuntimeError(
                    f"build_star_photons_for_observer: duplicate sidereal "
                    f"window_index={idx} for observer '{obs_name}'."
                )
            sidereal_by_index[idx] = w

    # If slew tracks are provided, build a similar lookup.
    slew_by_index: Dict[int, Dict[str, Any]] = {}
    if star_slew_tracks_for_obs is not None:
        slew_windows = star_slew_tracks_for_obs.get("windows", [])
        for w in slew_windows:
            idx = int(w.get("window_index"))
            if idx in slew_by_index:
                raise RuntimeError(
                    f"build_star_photons_for_observer: duplicate slew "
                    f"window_index={idx} for observer '{obs_name}'."
                )
            slew_by_index[idx] = w

    # ------------------------------------------------------------------
    # Loop over all windows for this observer and build star photons.
    # ------------------------------------------------------------------
    star_windows: List[StarPhotonWindow] = []

    # Observer-level cadence (used to synthesize per-frame timebase when needed).
    dt_frame_s = frames_for_obs.get("dt_frame_s", None)
    if dt_frame_s is not None:
        dt_frame_s = float(dt_frame_s)

    for window_frames_entry in windows_frames:
        if "window_index" not in window_frames_entry:
            raise RuntimeError(
                f"build_star_photons_for_observer: a window for observer "
                f"'{obs_name}' is missing 'window_index'."
            )

        window_index = int(window_frames_entry["window_index"])

        # tracking_mode MUST have been annotated earlier.
        mode_raw = window_frames_entry.get("tracking_mode", None)
        if mode_raw is None:
            raise RuntimeError(
                f"build_star_photons_for_observer: window_index={window_index} "
                f"for observer '{obs_name}' has no 'tracking_mode'. "
                f"Did you run annotate_windows_with_tracking_mode?"
            )
        mode = str(mode_raw).lower()

        # ------------------------------------------------------------------
        # Ensure a target-agnostic per-frame timebase exists for this window.
        # This prevents downstream star-photon logic from falling back to
        # sparse per-target arrays (which can be shorter than n_frames).
        # Applied for BOTH sidereal and slew windows.
        # ------------------------------------------------------------------
        window_frames_entry_canonical = _ensure_window_frames_timebase(
            window_entry=window_frames_entry,
            dt_frame_s=dt_frame_s,
        )

        # Look up matching star products; we do NOT assume anything about
        # mid-window fields here, only that the per-window dicts exist.
        window_star_proj: Optional[Dict[str, Any]] = None
        window_star_slew_entry: Optional[Dict[str, Any]] = None

        if mode == "sidereal":
            window_star_proj = sidereal_by_index.get(window_index)
            if window_star_proj is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: no sidereal star "
                    f"projection found for observer '{obs_name}', "
                    f"window_index={window_index}."
                )
        elif mode == "slew":
            if star_slew_tracks_for_obs is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: encountered a 'slew' "
                    f"window (index={window_index}) for observer '{obs_name}', "
                    f"but star_slew_tracks_for_obs is None."
                )
            window_star_slew_entry = slew_by_index.get(window_index)
            if window_star_slew_entry is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: no slew star tracks "
                    f"found for observer '{obs_name}', window_index={window_index}."
                )
        else:
            raise ValueError(
                f"build_star_photons_for_observer: unsupported tracking_mode="
                f"{mode_raw!r} for observer '{obs_name}', "
                f"window_index={window_index}."
            )

        # Delegate to the unified per-window dispatcher, which itself
        # chooses the correct sidereal/slew implementation and assumes
        # no mid-window pixel/time fields.
        star_window = build_star_photon_timeseries_for_window(
            obs_name=obs_name,
            window_frames_entry=window_frames_entry_canonical,
            window_star_projection=window_star_proj,
            window_star_slew_entry=window_star_slew_entry,
            logger=log,
        )

        star_windows.append(star_window)

    # ------------------------------------------------------------------
    # Build the per-observer wrapper structure.
    # ------------------------------------------------------------------
    observer_name = frames_for_obs.get("observer_name", obs_name)
    rows = int(frames_for_obs.get("rows", getattr(ACTIVE_SENSOR, "rows", 0)))
    cols = int(frames_for_obs.get("cols", getattr(ACTIVE_SENSOR, "cols", 0)))

    catalog_name = getattr(
        NEBULA_STAR_CATALOG, "name",
        getattr(NEBULA_STAR_CATALOG, "catalog_name", "Gaia"),
    )
    catalog_band = getattr(NEBULA_STAR_CATALOG, "band", "G")

    run_meta: Dict[str, Any] = {
        "star_photons_version": STAR_PHOTONS_RUN_META_VERSION,
        "builder": "NEBULA_STAR_PHOTONS.build_star_photons_for_observer",
        "observer_name": observer_name,
        "n_windows_input": len(windows_frames),
        "n_windows_output": len(star_windows),
        "created_utc": datetime.now(timezone.utc).isoformat() + "Z",
    }

    obs_star_photons: Dict[str, Any] = {
        "observer_name": observer_name,
        "rows": rows,
        "cols": cols,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "run_meta": run_meta,
        "windows": star_windows,
    }

    log.info(
        "build_star_photons_for_observer: built star photon catalog for "
        "observer '%s' with %d windows.",
        observer_name,
        len(star_windows),
    )

    return obs_star_photons


'''
def build_star_photons_for_all_observers(
    obs_target_frames: Dict[str, Any],
    obs_star_projections: Dict[str, Any],
    obs_star_slew_tracks: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> ObsStarPhotons:
    """
    Build star photon time series for *all* observers present in the
    target-photon frames dictionary.

    This is the main in-memory dispatcher for NEBULA_STAR_PHOTONS:
    given per-observer windowed photon frames (from NEBULA_TARGET_PHOTONS /
    NEBULA_PHOTON_FRAME_BUILDER) and per-observer star projections
    (from NEBULA_STAR_PROJECTION and, optionally, NEBULA_STAR_SLEW_PROJECTION),
    it calls :func:`build_star_photons_for_observer` for each observer and
    aggregates the results.

    Parameters
    ----------
    obs_target_frames : dict
        Per-observer target photon frames, typically the
        ``obs_target_frames_ranked`` structure produced by
        :func:`NEBULA_TARGET_PHOTONS.build_obs_target_frames_for_all_observers`.

        Expected minimal structure per observer:

        .. code-block:: python

            obs_target_frames = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "windows": [
                        {
                            "window_index": int,
                            "start_index": int,
                            "end_index": int,
                            "n_frames": int,
                            "tracking_mode": str,  # "sidereal" or "slew"
                            "frames": [
                                {
                                    "frame_index": int,
                                    "t_utc": datetime or str,
                                    "t_mjd_utc": float,      # optional but typical
                                    "t_exp_s": float,
                                    ...
                                },
                                ...
                            ],
                            ...
                        },
                        ...
                    ],
                    ...
                },
                ...
            }

    obs_star_projections : dict
        Per-observer star projection products, typically the output of
        :func:`NEBULA_STAR_PROJECTION.build_star_projections_for_all_observers`.

        This reflects the **epoch-based** projection strategy: each window
        stores star positions and pixel coordinates evaluated at a chosen
        epoch for that window (e.g. mid-window), but this function does
        not rely on any explicit "mid" fields — only on the epoch-level
        fields written by NEBULA_STAR_PROJECTION.

        Expected minimal structure per observer (aligned with your actual
        ``obs_star_projections.pkl``):

        .. code-block:: python

            obs_star_projections = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "catalog_name": str,
                    "catalog_band": str,
                    "run_meta": dict,
                    "windows": [
                        {
                            "window_index": int,
                            "start_index": int,
                            "end_index": int,
                            "n_frames": int,
                            "gaia_status": str,           # e.g., "ok"
                            "gaia_error_message": str or None,
                            "n_stars_input": int,
                            "n_stars_on_detector": int,
                            "sky_center_ra_deg": float,
                            "sky_center_dec_deg": float,
                            "sky_radius_deg": float,
                            "sky_selector_status": str,
                            # No t_mid_utc / t_mid_mjd_utc / x_pix_mid required.
                            "stars": {
                                "<gaia_source_id_str>": {
                                    "gaia_source_id": int or str,
                                    "source_id": str,
                                    "source_type": "star",
                                    "mag_G": float,
                                    "ra_deg_catalog": float,
                                    "dec_deg_catalog": float,
                                    "ra_deg_epoch": float,
                                    "dec_deg_epoch": float,
                                    "pm_ra_masyr": float,
                                    "pm_dec_masyr": float,
                                    "x_pix_epoch": float,
                                    "y_pix_epoch": float,
                                    "on_detector": bool,
                                    ...
                                },
                                ...
                            },
                            ...
                        },
                        ...
                    ],
                },
                ...
            }

        The observer keys (``"<observer_name>"``) must match those in
        ``obs_target_frames`` for star photons to be constructed.

    obs_star_slew_tracks : dict or None, optional
        Per-observer star tracks for **slewing** windows, typically the
        output of :mod:`NEBULA_STAR_SLEW_PROJECTION`.

        Expected minimal structure per observer:

        .. code-block:: python

            obs_star_slew_tracks = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "run_meta": dict,
                    "windows": [
                        {
                            "window_index": int,
                            # any additional bookkeeping fields as needed
                            "stars": {
                                "<gaia_source_id_str>": {
                                    "gaia_source_id": int or str,
                                    "source_id": str,
                                    "source_type": "star",
                                    "mag_G": float,
                                    # Per-frame kinematics for this window:
                                    "x_pix": np.ndarray,      # shape ~ (n_frames,)
                                    "y_pix": np.ndarray,      # shape ~ (n_frames,)
                                    "on_detector": np.ndarray,  # bool, shape ~ (n_frames,)
                                    ...
                                },
                                ...
                            },
                            ...
                        },
                        ...
                    ],
                },
                ...
            }

        If this argument is ``None`` and any window is annotated with
        ``tracking_mode == "slew"``, a :class:`RuntimeError` will be raised
        inside :func:`build_star_photons_for_observer`, since slew mode
        requires per-frame star tracks.

    logger : logging.Logger, optional
        Logger for informational / diagnostic messages. If ``None``,
        a module-level logger obtained via :func:`_get_logger` is used.

    Returns
    -------
    ObsStarPhotons
        Dictionary mapping each observer name to its star photon product:

        .. code-block:: python

            obs_star_photons = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "catalog_name": str,
                    "catalog_band": str,
                    "run_meta": dict,
                    "windows": [StarPhotonWindow, ...],
                },
                ...
            }

        Each ``StarPhotonWindow`` is produced by
        :func:`build_star_photons_for_observer`, which in turn uses
        :func:`build_star_photon_timeseries_for_window` to choose the
        appropriate sidereal or slew implementation based on
        ``tracking_mode``. No mid-window time or pixel fields are
        required at this stage.

    Raises
    ------
    RuntimeError
        If an observer present in ``obs_target_frames`` has no matching
        entry in ``obs_star_projections``. This is a fail-hard design:
        star photons are only considered valid if both target frames and
        star projections exist for the same observer.
    """
    # Resolve a logger for this function.
    log = _get_logger(logger)

    # Container for the per-observer star photon products.
    obs_star_photons: ObsStarPhotons = {}

    # Loop over every observer that has target photon frames.
    for obs_name, frames_for_obs in obs_target_frames.items():
        windows = frames_for_obs.get("windows", []) or []
        needs_sidereal = any(
            str(w.get("tracking_mode", "")).lower() == "sidereal"
            for w in windows
        )

        # Only require sidereal projections if this observer has sidereal windows.
        star_proj_for_obs: Optional[Dict[str, Any]] = None
        if needs_sidereal:
            if obs_star_projections is None:
                raise RuntimeError(
                    "build_star_photons_for_all_observers: encountered sidereal windows "
                    f"for observer '{obs_name}' but obs_star_projections is None."
                )

            star_proj_for_obs = obs_star_projections.get(obs_name)
            if star_proj_for_obs is None:
                raise RuntimeError(
                    "build_star_photons_for_all_observers: no star projection "
                    f"entry found for observer '{obs_name}', but sidereal windows exist. "
                    "Ensure NEBULA_STAR_PROJECTION has been run for the same set of observers."
                )

        # Look up the optional slew tracks for this observer, if provided.
        star_slew_for_obs: Optional[Dict[str, Any]] = None
        if obs_star_slew_tracks is not None:
            star_slew_for_obs = obs_star_slew_tracks.get(obs_name)

        # Delegate the actual per-observer construction to the helper.
        obs_entry = build_star_photons_for_observer(
            obs_name=obs_name,
            frames_for_obs=frames_for_obs,
            star_projections_for_obs=star_proj_for_obs,
            star_slew_tracks_for_obs=star_slew_for_obs,
            logger=log,
        )

        # Store the result in the top-level mapping.
        obs_star_photons[obs_name] = obs_entry

        # Log a compact summary for this observer.
        n_windows = len(obs_entry.get("windows", []))
        log.info(
            "NEBULA_STAR_PHOTONS: observer '%s' -> built star photons for %d window(s).",
            obs_name,
            n_windows,
        )

    # Optionally, log a one-line summary across all observers.
    log.info(
        "NEBULA_STAR_PHOTONS: completed star photon construction for %d observer(s).",
        len(obs_star_photons),
    )

    return obs_star_photons
'''
def build_star_photons_for_all_observers(
    obs_target_frames: Dict[str, Any],
    obs_star_projections: Dict[str, Any],
    obs_star_slew_tracks: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> ObsStarPhotons:
    """
    Build star photon time series for *all* observers present in the
    target-photon frames dictionary.

    This is the main in-memory dispatcher for NEBULA_STAR_PHOTONS:
    given per-observer windowed photon frames (from NEBULA_TARGET_PHOTONS /
    NEBULA_PHOTON_FRAME_BUILDER) and per-observer star projections
    (from NEBULA_STAR_PROJECTION and, optionally, NEBULA_STAR_SLEW_PROJECTION),
    it calls :func:`build_star_photons_for_observer` for each observer and
    aggregates the results.

    Parameters
    ----------
    obs_target_frames : dict
        Per-observer target photon frames, typically the
        ``obs_target_frames_ranked`` structure produced by
        :func:`NEBULA_TARGET_PHOTONS.build_obs_target_frames_for_all_observers`.

        Expected minimal structure per observer:

        .. code-block:: python

            obs_target_frames = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "windows": [
                        {
                            "window_index": int,
                            "start_index": int,
                            "end_index": int,
                            "n_frames": int,
                            "tracking_mode": str,  # "sidereal" or "slew"
                            "frames": [
                                {
                                    "frame_index": int,
                                    "t_utc": datetime or str,
                                    "t_mjd_utc": float,      # optional but typical
                                    "t_exp_s": float,
                                    ...
                                },
                                ...
                            ],
                            ...
                        },
                        ...
                    ],
                    ...
                },
                ...
            }

    obs_star_projections : dict
        Per-observer star projection products, typically the output of
        :func:`NEBULA_STAR_PROJECTION.build_star_projections_for_all_observers`.

        This reflects the **epoch-based** projection strategy: each window
        stores star positions and pixel coordinates evaluated at a chosen
        epoch for that window (e.g. mid-window), but this function does
        not rely on any explicit "mid" fields — only on the epoch-level
        fields written by NEBULA_STAR_PROJECTION.

        Expected minimal structure per observer (aligned with your actual
        ``obs_star_projections.pkl``):

        .. code-block:: python

            obs_star_projections = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "catalog_name": str,
                    "catalog_band": str,
                    "run_meta": dict,
                    "windows": [
                        {
                            "window_index": int,
                            "start_index": int,
                            "end_index": int,
                            "n_frames": int,
                            "gaia_status": str,           # e.g., "ok"
                            "gaia_error_message": str or None,
                            "n_stars_input": int,
                            "n_stars_on_detector": int,
                            "sky_center_ra_deg": float,
                            "sky_center_dec_deg": float,
                            "sky_radius_deg": float,
                            "sky_selector_status": str,
                            # No t_mid_utc / t_mid_mjd_utc / x_pix_mid required.
                            "stars": {
                                "<gaia_source_id_str>": {
                                    "gaia_source_id": int or str,
                                    "source_id": str,
                                    "source_type": "star",
                                    "mag_G": float,
                                    "ra_deg_catalog": float,
                                    "dec_deg_catalog": float,
                                    "ra_deg_epoch": float,
                                    "dec_deg_epoch": float,
                                    "pm_ra_masyr": float,
                                    "pm_dec_masyr": float,
                                    "x_pix_epoch": float,
                                    "y_pix_epoch": float,
                                    "on_detector": bool,
                                    ...
                                },
                                ...
                            },
                            ...
                        },
                        ...
                    ],
                },
                ...
            }

        The observer keys (``"<observer_name>"``) must match those in
        ``obs_target_frames`` for star photons to be constructed.

    obs_star_slew_tracks : dict or None, optional
        Per-observer star tracks for **slewing** windows, typically the
        output of :mod:`NEBULA_STAR_SLEW_PROJECTION`.

        Expected minimal structure per observer:

        .. code-block:: python

            obs_star_slew_tracks = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "run_meta": dict,
                    "windows": [
                        {
                            "window_index": int,
                            # any additional bookkeeping fields as needed
                            "stars": {
                                "<gaia_source_id_str>": {
                                    "gaia_source_id": int or str,
                                    "source_id": str,
                                    "source_type": "star",
                                    "mag_G": float,
                                    # Per-frame kinematics for this window:
                                    "x_pix": np.ndarray,      # shape ~ (n_frames,)
                                    "y_pix": np.ndarray,      # shape ~ (n_frames,)
                                    "on_detector": np.ndarray,  # bool, shape ~ (n_frames,)
                                    ...
                                },
                                ...
                            },
                            ...
                        },
                        ...
                    ],
                },
                ...
            }

        If this argument is ``None`` and any window is annotated with
        ``tracking_mode == "slew"``, a :class:`RuntimeError` will be raised
        inside :func:`build_star_photons_for_observer`, since slew mode
        requires per-frame star tracks.

    logger : logging.Logger, optional
        Logger for informational / diagnostic messages. If ``None``,
        a module-level logger obtained via :func:`_get_logger` is used.

    Returns
    -------
    ObsStarPhotons
        Dictionary mapping each observer name to its star photon product:

        .. code-block:: python

            obs_star_photons = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "catalog_name": str,
                    "catalog_band": str,
                    "run_meta": dict,
                    "windows": [StarPhotonWindow, ...],
                },
                ...
            }

        Each ``StarPhotonWindow`` is produced by
        :func:`build_star_photons_for_observer`, which in turn uses
        :func:`build_star_photon_timeseries_for_window` to choose the
        appropriate sidereal or slew implementation based on
        ``tracking_mode``. No mid-window time or pixel fields are
        required at this stage.

    Raises
    ------
    RuntimeError
        If an observer present in ``obs_target_frames`` has no matching
        entry in ``obs_star_projections``. This is a fail-hard design:
        star photons are only considered valid if both target frames and
        star projections exist for the same observer.
    """
    # Resolve a logger for this function.
    log = _get_logger(logger)

    # Container for the per-observer star photon products.
    obs_star_photons: ObsStarPhotons = {}

    # Loop over every observer that has target photon frames.
    for obs_name, frames_for_obs in obs_target_frames.items():
        windows = frames_for_obs.get("windows", []) or []

        # ------------------------------------------------------------------
        # Ensure each window has a full per-frame timebase ("frames" list).
        #
        # Some upstream products (e.g., obs_target_frames_ranked_with_sky.pkl)
        # store only per-target series (shorter than the window) and omit the
        # full window-length "frames" list. Star photons require a window-length
        # time array, so we synthesize window["frames"] from:
        #   start_time + i * dt_frame_s, for i=0..n_frames-1
        #
        # This applies to BOTH sidereal and slew windows.
        # ------------------------------------------------------------------
        from datetime import datetime, timedelta

        dt_frame_s = frames_for_obs.get("dt_frame_s", None)

        for w in windows:
            if not isinstance(w, dict):
                continue

            n_frames_window = w.get("n_frames", None)
            if n_frames_window is None:
                continue

            try:
                n_frames_int = int(n_frames_window)
            except Exception:
                continue

            existing_frames = w.get("frames", None)
            if isinstance(existing_frames, list) and len(existing_frames) == n_frames_int:
                continue

            # Parse / recover start_time (preferred) or infer it from end_time.
            start_time = w.get("start_time", None)
            end_time = w.get("end_time", None)

            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                except Exception:
                    start_time = None

            if isinstance(end_time, str):
                try:
                    end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                except Exception:
                    end_time = None

            if dt_frame_s is None:
                dt_frame_s = w.get("dt_frame_s", None)

            if start_time is None and end_time is not None and dt_frame_s is not None:
                try:
                    start_time = end_time - timedelta(seconds=float(dt_frame_s) * float(n_frames_int - 1))
                except Exception:
                    start_time = None

            if start_time is None or dt_frame_s is None:
                log.warning(
                    "NEBULA_STAR_PHOTONS: cannot synthesize window 'frames' timebase for observer '%s' "
                    "(window_index=%s): start_time=%r, dt_frame_s=%r, n_frames=%r. "
                    "Downstream may fall back to per-target time arrays.",
                    obs_name,
                    w.get("window_index", None),
                    start_time,
                    dt_frame_s,
                    n_frames_window,
                )
                continue

            start_index = w.get("start_index", None)
            frames_list: List[Dict[str, Any]] = []
            for i in range(n_frames_int):
                frame_entry: Dict[str, Any] = {
                    "t_utc": start_time + timedelta(seconds=float(dt_frame_s) * float(i)),
                    "t_exp_s": float(dt_frame_s),
                    "sources": [],
                }
                if start_index is not None:
                    try:
                        frame_entry["coarse_index"] = int(start_index) + int(i)
                    except Exception:
                        pass
                frames_list.append(frame_entry)

            w["frames"] = frames_list

            log.info(
                "NEBULA_STAR_PHOTONS: synthesized window 'frames' timebase for observer '%s' "
                "(window_index=%s) with %d frames.",
                obs_name,
                w.get("window_index", None),
                len(frames_list),
            )

        needs_sidereal = any(
            str(w.get("tracking_mode", "")).lower() == "sidereal"
            for w in windows
        )

        # Only require sidereal projections if this observer has sidereal windows.
        star_proj_for_obs: Optional[Dict[str, Any]] = None
        if needs_sidereal:
            if obs_star_projections is None:
                raise RuntimeError(
                    "build_star_photons_for_all_observers: encountered sidereal windows "
                    f"for observer '{obs_name}' but obs_star_projections is None."
                )

            star_proj_for_obs = obs_star_projections.get(obs_name)
            if star_proj_for_obs is None:
                raise RuntimeError(
                    "build_star_photons_for_all_observers: no star projection "
                    f"entry found for observer '{obs_name}', but sidereal windows exist. "
                    "Ensure NEBULA_STAR_PROJECTION has been run for the same set of observers."
                )

        # Look up the optional slew tracks for this observer, if provided.
        star_slew_for_obs: Optional[Dict[str, Any]] = None
        if obs_star_slew_tracks is not None:
            star_slew_for_obs = obs_star_slew_tracks.get(obs_name)

        # Delegate the actual per-observer construction to the helper.
        obs_entry = build_star_photons_for_observer(
            obs_name=obs_name,
            frames_for_obs=frames_for_obs,
            star_projections_for_obs=star_proj_for_obs,
            star_slew_tracks_for_obs=star_slew_for_obs,
            logger=log,
        )

        # Store the result in the top-level mapping.
        obs_star_photons[obs_name] = obs_entry

        # Log a compact summary for this observer.
        n_windows = len(obs_entry.get("windows", []))
        log.info(
            "NEBULA_STAR_PHOTONS: observer '%s' -> built star photons for %d window(s).",
            obs_name,
            n_windows,
        )

    # Optionally, log a one-line summary across all observers.
    log.info(
        "NEBULA_STAR_PHOTONS: completed star photon construction for %d observer(s).",
        len(obs_star_photons),
    )

    return obs_star_photons

'''
# def run_star_photons_pipeline_from_pickles(
#     frames_with_sky_path: Optional[str] = None,
#     star_projection_sidereal_path: Optional[str] = None,
#     star_projection_slew_path: Optional[str] = None,
#     output_path: Optional[str] = None,
#     logger: Optional[logging.Logger] = None,
# ) -> ObsStarPhotons:
#     """
#     High-level driver: load existing pickles, build star photons,
#     and write obs_star_photons.pkl.

#     This function is intentionally *read-only* with respect to upstream
#     stages: it assumes that:

#         1) NEBULA_TARGET_PHOTONS / NEBULA_PHOTON_FRAME_BUILDER have
#            already produced a ranked frames-with-sky product:

#                NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl

#            with per-observer entries like:

#                {
#                    "observer_name": str,
#                    "rows": int,
#                    "cols": int,
#                    "windows": [
#                        {
#                            "window_index": int,
#                            "start_index": int,
#                            "end_index": int,
#                            "n_frames": int,
#                            "tracking_mode": str,   # "sidereal" or "slew"
#                            "frames": [
#                                {
#                                    "frame_index": int,
#                                    "t_utc": datetime or str,
#                                    "t_mjd_utc": float,   # optional but typical
#                                    "t_exp_s": float,
#                                    ...
#                                },
#                                ...
#                            ],
#                            ...
#                        },
#                        ...
#                    ],
#                    ...
#                }

#         2) NEBULA_STAR_PROJECTION has produced sidereal star projections
#            for the same observers:

#                NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

#            with per-observer entries like your actual
#            obs_star_projections.pkl:

#                {
#                    "observer_name": str,
#                    "rows": int,
#                    "cols": int,
#                    "catalog_name": str,
#                    "catalog_band": str,
#                    "run_meta": dict,
#                    "windows": [
#                        {
#                            "window_index": int,
#                            "start_index": int,
#                            "end_index": int,
#                            "n_frames": int,
#                            "gaia_status": str,
#                            "gaia_error_message": str or None,
#                            "n_stars_input": int,
#                            "n_stars_on_detector": int,
#                            "sky_center_ra_deg": float,
#                            "sky_center_dec_deg": float,
#                            "sky_radius_deg": float,
#                            "sky_selector_status": str,
#                            "stars": {
#                                "<gaia_source_id_str>": {
#                                    "gaia_source_id": int or str,
#                                    "source_id": str,
#                                    "source_type": "star",
#                                    "mag_G": float,
#                                    "ra_deg_catalog": float,
#                                    "dec_deg_catalog": float,
#                                    "ra_deg_epoch": float,
#                                    "dec_deg_epoch": float,
#                                    "pm_ra_masyr": float,
#                                    "pm_dec_masyr": float,
#                                    "x_pix_epoch": float,
#                                    "y_pix_epoch": float,
#                                    "on_detector": bool,
#                                    ...
#                                },
#                                ...
#                            },
#                            ...
#                        },
#                        ...
#                    ],
#                }

#            Note: the star photons pipeline does **not** assume any
#            explicit “mid-window” fields; it simply consumes the
#            epoch-level projections as given (e.g., x_pix_epoch).

#         3) NEBULA_STAR_SLEW_PROJECTION has (optionally) produced
#            non-sidereal (slew) star tracks:

#                NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_slew_tracks.pkl

#            with per-observer entries like:

#                {
#                    "observer_name": str,
#                    "rows": int,
#                    "cols": int,
#                    "run_meta": dict,
#                    "windows": [
#                        {
#                            "window_index": int,
#                            "stars": {
#                                "<gaia_source_id_str>": {
#                                    "gaia_source_id": int or str,
#                                    "source_id": str,
#                                    "source_type": "star",
#                                    "mag_G": float,
#                                    "x_pix": np.ndarray,       # per-frame
#                                    "y_pix": np.ndarray,       # per-frame
#                                    "on_detector": np.ndarray, # per-frame bool
#                                    ...
#                                },
#                                ...
#                            },
#                            ...
#                        },
#                        ...
#                    ],
#                }

#     It then:

#         * loads these pickles,
#         * calls build_star_photons_for_all_observers(...) to construct
#           per-observer, per-window star photon time series, and
#         * writes:

#                NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_photons.pkl

#     Parameters
#     ----------
#     frames_with_sky_path : str or None, optional
#         Path to the ranked target-frames pickle (frames-with-sky structure),
#         typically:

#             NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl

#         If None, this default location is used.

#     star_projection_sidereal_path : str or None, optional
#         Path to the sidereal star projection pickle produced by
#         NEBULA_STAR_PROJECTION, typically:

#             NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

#         If None, this default location is used.

#     star_projection_slew_path : str or None, optional
#         Path to the non-sidereal (slew) star tracks pickle produced by
#         NEBULA_STAR_SLEW_PROJECTION, typically:

#             NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_slew_tracks.pkl

#         If None, the function will look for this file at the default
#         location. If it is missing, the pipeline proceeds with
#         ``obs_star_slew_tracks=None`` (i.e., sidereal-only stars).
#         If any windows are annotated with ``tracking_mode == "slew"``,
#         downstream helpers will raise a RuntimeError because slew
#         windows require per-frame star tracks.

#     output_path : str or None, optional
#         Path where the resulting obs_star_photons pickle will be written.
#         If None, the default is:

#             NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_photons.pkl

#     logger : logging.Logger or None, optional
#         Logger for status / debug messages. If None, a module-level
#         logger obtained via _get_logger() is used.

#     Returns
#     -------
#     ObsStarPhotons
#         The in-memory obs_star_photons mapping keyed by observer name.

#     Raises
#     ------
#     FileNotFoundError
#         If the ranked target-frames pickle or the sidereal star
#         projection pickle cannot be found at the resolved paths.
#     RuntimeError
#         If downstream helper functions detect inconsistent data
#         (e.g., slew windows with no slew tracks).
#     """
#     # Resolve a logger to use internally.
#     log = _get_logger(logger)

#     # ------------------------------------------------------------------
#     # Resolve default paths if caller did not supply them explicitly.
#     # ------------------------------------------------------------------
#     nebula_output_dir = NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR
#     catalog_name = getattr(
#         NEBULA_STAR_CATALOG,
#         "name",
#         getattr(NEBULA_STAR_CATALOG, "catalog_name", "UNKNOWN_CATALOG"),
#     )

#     # Ranked target frames (frames-with-sky) from NEBULA_TARGET_PHOTONS.
#     if frames_with_sky_path is None:
#         frames_with_sky_dir = os.path.join(
#             nebula_output_dir,
#             "TARGET_PHOTON_FRAMES",
#         )
#         frames_with_sky_path = os.path.join(
#             frames_with_sky_dir,
#             "obs_target_frames_ranked_with_sky.pkl",
#         )

#     # Sidereal star projections from NEBULA_STAR_PROJECTION.
#     if star_projection_sidereal_path is None:
#         stars_dir = os.path.join(
#             nebula_output_dir,
#             "STARS",
#             f"{catalog_name}",
#         )
#         star_projection_sidereal_path = os.path.join(
#             stars_dir,
#             "obs_star_projections.pkl",
#         )

#     # Slew star tracks from NEBULA_STAR_SLEW_PROJECTION (optional).
#     if star_projection_slew_path is None:
#         stars_dir = os.path.join(
#             nebula_output_dir,
#             "STARS",
#             f"{catalog_name}",
#         )
#         star_projection_slew_path = os.path.join(
#             stars_dir,
#             "obs_star_slew_tracks.pkl",
#         )

#     # Output path for obs_star_photons.
#     if output_path is None:
#         stars_dir = os.path.join(
#             nebula_output_dir,
#             "STARS",
#             f"{catalog_name}",
#         )
#         output_path = os.path.join(
#             stars_dir,
#             "obs_star_photons.pkl",
#         )

#     log.info(
#         "NEBULA_STAR_PHOTONS: using frames_with_sky_path=%s",
#         frames_with_sky_path,
#     )
#     log.info(
#         "NEBULA_STAR_PHOTONS: using star_projection_sidereal_path=%s",
#         star_projection_sidereal_path,
#     )
#     log.info(
#         "NEBULA_STAR_PHOTONS: using star_projection_slew_path=%s",
#         star_projection_slew_path,
#     )
#     log.info(
#         "NEBULA_STAR_PHOTONS: output will be written to %s",
#         output_path,
#     )

#     # ------------------------------------------------------------------
#     # Load input pickles from disk.
#     # ------------------------------------------------------------------
#     # ------------------------------------------------------------------
#     # Load input pickles from disk.
#     # ------------------------------------------------------------------
#     if not os.path.exists(frames_with_sky_path):
#         raise FileNotFoundError(
#             f"run_star_photons_pipeline_from_pickles: frames_with_sky_path "
#             f"does not exist: {frames_with_sky_path!r}"
#         )

#     # Load ranked target frames (frames-with-sky) first so we can decide
#     # whether sidereal projections are actually required.
#     with open(frames_with_sky_path, "rb") as f:
#         frames_with_sky = pickle.load(f)

#     # Determine whether ANY sidereal windows exist across all observers.
#     needs_sidereal = any(
#         str(w.get("tracking_mode", "")).lower() == "sidereal"
#         for obs_entry in (frames_with_sky.values() if isinstance(frames_with_sky, dict) else [])
#         for w in (obs_entry.get("windows", []) or [])
#         if isinstance(obs_entry, dict)
#     )

#     # Load sidereal star projections only if required by the data.
#     if needs_sidereal:
#         if not os.path.exists(star_projection_sidereal_path):
#             raise FileNotFoundError(
#                 f"run_star_photons_pipeline_from_pickles: sidereal star "
#                 f"projection pickle not found at: "
#                 f"{star_projection_sidereal_path!r}.\n"
#                 f"Did you run NEBULA_STAR_PROJECTION?"
#             )

#         with open(star_projection_sidereal_path, "rb") as f:
#             obs_star_projections_sidereal = pickle.load(f)
#     else:
#         obs_star_projections_sidereal = None

#     # Load slew star tracks if present; otherwise, treat as None
#     # (pure-sidereal pipeline). Downstream logic will fail-loud if
#     # any window is in "slew" mode but no tracks exist.
#     if os.path.exists(star_projection_slew_path):
#         with open(star_projection_slew_path, "rb") as f:
#             obs_star_projections_slew = pickle.load(f)
#         log.info(
#             "NEBULA_STAR_PHOTONS: loaded slew star tracks from '%s'.",
#             star_projection_slew_path,
#         )
#     else:
#         obs_star_projections_slew = None
#         log.info(
#             "NEBULA_STAR_PHOTONS: no slew star tracks found at '%s'; "
#             "proceeding with sidereal-only star projections.",
#             star_projection_slew_path,
#         )

#     # ------------------------------------------------------------------
#     # Build star photon time series for all observers.
#     # ------------------------------------------------------------------
#     obs_star_photons = build_star_photons_for_all_observers(
#         obs_target_frames=frames_with_sky,
#         obs_star_projections=obs_star_projections_sidereal,
#         obs_star_slew_tracks=obs_star_projections_slew,
#         logger=log,
#     )

#     # ------------------------------------------------------------------
#     # Write the resulting obs_star_photons to disk.
#     # ------------------------------------------------------------------
#     output_dir = os.path.dirname(output_path)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     with open(output_path, "wb") as f:
#         pickle.dump(obs_star_photons, f)

#     log.info(
#         "NEBULA_STAR_PHOTONS: wrote obs_star_photons for %d observers to '%s'.",
#         len(obs_star_photons),
#         output_path,
#     )

#     return obs_star_photons
'''
def run_star_photons_pipeline_from_pickles(
    frames_with_sky_path: Optional[str] = None,
    star_projection_sidereal_path: Optional[str] = None,
    star_projection_slew_path: Optional[str] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    frame_catalog_path: Optional[str] = None,
) -> ObsStarPhotons:
    """
    High-level driver: load existing pickles, build star photons,
    and write obs_star_photons.pkl.

    This function is intentionally *read-only* with respect to upstream
    stages: it assumes that:

        1) NEBULA_TARGET_PHOTONS / NEBULA_PHOTON_FRAME_BUILDER have
           already produced a ranked frames-with-sky product:

               NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl

           with per-observer entries like:

               {
                   "observer_name": str,
                   "rows": int,
                   "cols": int,
                   "windows": [
                       {
                           "window_index": int,
                           "start_index": int,
                           "end_index": int,
                           "n_frames": int,
                           "tracking_mode": str,   # "sidereal" or "slew"
                           "frames": [
                               {
                                   "frame_index": int,
                                   "t_utc": datetime or str,
                                   "t_mjd_utc": float,   # optional but typical
                                   "t_exp_s": float,
                                   ...
                               },
                               ...
                           ],
                           ...
                       },
                       ...
                   ],
                   ...
               }

        1a) NEBULA_PHOTON_FRAME_BUILDER has also produced a full per-window
            frame catalog (canonical timebase) for each observer:

               NEBULA_OUTPUT/PHOTON_FRAMES/obs_photon_frame_catalog.pkl

            This is used as the canonical per-frame time array for star
            photons, so the star pipeline is agnostic to target “active”
            spans (targets can be sparse within a window).

        2) NEBULA_STAR_PROJECTION has produced sidereal star projections
           for the same observers:

               NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

           with per-observer entries like your actual
           obs_star_projections.pkl:

               {
                   "observer_name": str,
                   "rows": int,
                   "cols": int,
                   "catalog_name": str,
                   "catalog_band": str,
                   "run_meta": dict,
                   "windows": [
                       {
                           "window_index": int,
                           "start_index": int,
                           "end_index": int,
                           "n_frames": int,
                           "gaia_status": str,
                           "gaia_error_message": str or None,
                           "n_stars_input": int,
                           "n_stars_on_detector": int,
                           "sky_center_ra_deg": float,
                           "sky_center_dec_deg": float,
                           "sky_radius_deg": float,
                           "sky_selector_status": str,
                           "stars": {
                               "<gaia_source_id_str>": {
                                   "gaia_source_id": int or str,
                                   "source_id": str,
                                   "source_type": "star",
                                   "mag_G": float,
                                   "ra_deg_catalog": float,
                                   "dec_deg_catalog": float,
                                   "ra_deg_epoch": float,
                                   "dec_deg_epoch": float,
                                   "pm_ra_masyr": float,
                                   "pm_dec_masyr": float,
                                   "x_pix_epoch": float,
                                   "y_pix_epoch": float,
                                   "on_detector": bool,
                                   ...
                               },
                               ...
                           },
                           ...
                       },
                       ...
                   ],
               }

           Note: the star photons pipeline does **not** assume any
           explicit “mid-window” fields; it simply consumes the
           epoch-level projections as given (e.g., x_pix_epoch).

        3) NEBULA_STAR_SLEW_PROJECTION has (optionally) produced
           non-sidereal (slew) star tracks:

               NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_slew_tracks.pkl

           with per-observer entries like:

               {
                   "observer_name": str,
                   "rows": int,
                   "cols": int,
                   "run_meta": dict,
                   "windows": [
                       {
                           "window_index": int,
                           "stars": {
                               "<gaia_source_id_str>": {
                                   "gaia_source_id": int or str,
                                   "source_id": str,
                                   "source_type": "star",
                                   "mag_G": float,
                                   "x_pix": np.ndarray,       # per-frame
                                   "y_pix": np.ndarray,       # per-frame
                                   "on_detector": np.ndarray, # per-frame bool
                                   ...
                               },
                               ...
                           },
                           ...
                       },
                       ...
                   ],
               }

    It then:

        * loads these pickles,
        * calls build_star_photons_for_all_observers(...) to construct
          per-observer, per-window star photon time series, and
        * writes:

               NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_photons.pkl

    Parameters
    ----------
    frames_with_sky_path : str or None, optional
        Path to the ranked target-frames pickle (frames-with-sky structure),
        typically:

            NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl

        If None, this default location is used.

    star_projection_sidereal_path : str or None, optional
        Path to the sidereal star projection pickle produced by
        NEBULA_STAR_PROJECTION, typically:

            NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

        If None, this default location is used.

    star_projection_slew_path : str or None, optional
        Path to the non-sidereal (slew) star tracks pickle produced by
        NEBULA_STAR_SLEW_PROJECTION, typically:

            NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_slew_tracks.pkl

        If None, the function will look for this file at the default
        location. If it is missing, the pipeline proceeds with
        ``obs_star_slew_tracks=None`` (i.e., sidereal-only stars).
        If any windows are annotated with ``tracking_mode == "slew"``,
        downstream helpers will raise a RuntimeError because slew
        windows require per-frame star tracks.

    output_path : str or None, optional
        Path where the resulting obs_star_photons pickle will be written.
        If None, the default is:

            NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_photons.pkl

    logger : logging.Logger or None, optional
        Logger for status / debug messages. If None, a module-level
        logger obtained via _get_logger() is used.

    frame_catalog_path : str or None, optional
        Path to the full per-window frame catalog pickle produced by
        NEBULA_PHOTON_FRAME_BUILDER, typically:

            NEBULA_OUTPUT/PHOTON_FRAMES/obs_photon_frame_catalog.pkl

        If None, this default location is used.

    Returns
    -------
    ObsStarPhotons
        The in-memory obs_star_photons mapping keyed by observer name.

    Raises
    ------
    FileNotFoundError
        If the ranked target-frames pickle or the sidereal star
        projection pickle cannot be found at the resolved paths.
    RuntimeError
        If downstream helper functions detect inconsistent data
        (e.g., slew windows with no slew tracks).
    """
    # Resolve a logger to use internally.
    log = _get_logger(logger)

    # ------------------------------------------------------------------
    # Resolve default paths if caller did not supply them explicitly.
    # ------------------------------------------------------------------
    nebula_output_dir = NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR
    catalog_name = getattr(
        NEBULA_STAR_CATALOG,
        "name",
        getattr(NEBULA_STAR_CATALOG, "catalog_name", "UNKNOWN_CATALOG"),
    )

    # Ranked target frames (frames-with-sky) from NEBULA_TARGET_PHOTONS.
    if frames_with_sky_path is None:
        frames_with_sky_dir = os.path.join(
            nebula_output_dir,
            "TARGET_PHOTON_FRAMES",
        )
        frames_with_sky_path = os.path.join(
            frames_with_sky_dir,
            "obs_target_frames_ranked_with_sky.pkl",
        )

    # Full per-window frame catalog (canonical timebase) from NEBULA_PHOTON_FRAME_BUILDER.
    if frame_catalog_path is None:
        frame_catalog_dir = os.path.join(
            nebula_output_dir,
            "PHOTON_FRAMES",
        )
        frame_catalog_path = os.path.join(
            frame_catalog_dir,
            "obs_photon_frame_catalog.pkl",
        )

    # Sidereal star projections from NEBULA_STAR_PROJECTION.
    if star_projection_sidereal_path is None:
        stars_dir = os.path.join(
            nebula_output_dir,
            "STARS",
            f"{catalog_name}",
        )
        star_projection_sidereal_path = os.path.join(
            stars_dir,
            "obs_star_projections.pkl",
        )

    # Slew star tracks from NEBULA_STAR_SLEW_PROJECTION (optional).
    if star_projection_slew_path is None:
        stars_dir = os.path.join(
            nebula_output_dir,
            "STARS",
            f"{catalog_name}",
        )
        star_projection_slew_path = os.path.join(
            stars_dir,
            "obs_star_slew_tracks.pkl",
        )

    # Output path for obs_star_photons.
    if output_path is None:
        stars_dir = os.path.join(
            nebula_output_dir,
            "STARS",
            f"{catalog_name}",
        )
        output_path = os.path.join(
            stars_dir,
            "obs_star_photons.pkl",
        )

    log.info(
        "NEBULA_STAR_PHOTONS: using frames_with_sky_path=%s",
        frames_with_sky_path,
    )
    log.info(
        "NEBULA_STAR_PHOTONS: using frame_catalog_path=%s",
        frame_catalog_path,
    )
    log.info(
        "NEBULA_STAR_PHOTONS: using star_projection_sidereal_path=%s",
        star_projection_sidereal_path,
    )
    log.info(
        "NEBULA_STAR_PHOTONS: using star_projection_slew_path=%s",
        star_projection_slew_path,
    )
    log.info(
        "NEBULA_STAR_PHOTONS: output will be written to %s",
        output_path,
    )

    # ------------------------------------------------------------------
    # Load input pickles from disk.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Load input pickles from disk.
    # ------------------------------------------------------------------
    if not os.path.exists(frames_with_sky_path):
        raise FileNotFoundError(
            f"run_star_photons_pipeline_from_pickles: frames_with_sky_path "
            f"does not exist: {frames_with_sky_path!r}"
        )

    if not os.path.exists(frame_catalog_path):
        raise FileNotFoundError(
            f"run_star_photons_pipeline_from_pickles: frame_catalog_path "
            f"does not exist: {frame_catalog_path!r}"
        )

    # Load ranked target frames (frames-with-sky) first so we can decide
    # whether sidereal projections are actually required.
    with open(frames_with_sky_path, "rb") as f:
        frames_with_sky = pickle.load(f)

    # Load the full per-window frame catalog (canonical timebase).
    with open(frame_catalog_path, "rb") as f:
        obs_frame_catalog = pickle.load(f)

    # Determine whether ANY sidereal windows exist across all observers.
    needs_sidereal = any(
        str(w.get("tracking_mode", "")).lower() == "sidereal"
        for obs_entry in (frames_with_sky.values() if isinstance(frames_with_sky, dict) else [])
        for w in (obs_entry.get("windows", []) or [])
        if isinstance(obs_entry, dict)
    )

    # Load sidereal star projections only if required by the data.
    if needs_sidereal:
        if not os.path.exists(star_projection_sidereal_path):
            raise FileNotFoundError(
                f"run_star_photons_pipeline_from_pickles: sidereal star "
                f"projection pickle not found at: "
                f"{star_projection_sidereal_path!r}.\n"
                f"Did you run NEBULA_STAR_PROJECTION?"
            )

        with open(star_projection_sidereal_path, "rb") as f:
            obs_star_projections_sidereal = pickle.load(f)
    else:
        obs_star_projections_sidereal = None

    # Load slew star tracks if present; otherwise, treat as None
    # (pure-sidereal pipeline). Downstream logic will fail-loud if
    # any window is in "slew" mode but no tracks exist.
    if os.path.exists(star_projection_slew_path):
        with open(star_projection_slew_path, "rb") as f:
            obs_star_projections_slew = pickle.load(f)
        log.info(
            "NEBULA_STAR_PHOTONS: loaded slew star tracks from '%s'.",
            star_projection_slew_path,
        )
    else:
        obs_star_projections_slew = None
        log.info(
            "NEBULA_STAR_PHOTONS: no slew star tracks found at '%s'; "
            "proceeding with sidereal-only star projections.",
            star_projection_slew_path,
        )

    # ------------------------------------------------------------------
    # Build star photon time series for all observers.
    # ------------------------------------------------------------------
    build_kwargs = dict(
        obs_target_frames=frames_with_sky,
        obs_star_projections=obs_star_projections_sidereal,
        obs_star_slew_tracks=obs_star_projections_slew,
        logger=log,
    )

    # Pass the canonical per-window frame catalog (timebase) if the builder supports it.
    try:
        import inspect
        if "obs_frame_catalog" in inspect.signature(build_star_photons_for_all_observers).parameters:
            build_kwargs["obs_frame_catalog"] = obs_frame_catalog
    except (TypeError, ValueError):
        pass

    obs_star_photons = build_star_photons_for_all_observers(
        **build_kwargs
    )

    # ------------------------------------------------------------------
    # Write the resulting obs_star_photons to disk.
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(obs_star_photons, f)

    log.info(
        "NEBULA_STAR_PHOTONS: wrote obs_star_photons for %d observers to '%s'.",
        len(obs_star_photons),
        output_path,
    )

    return obs_star_photons


def attach_star_photons_to_target_frames(
    obs_target_frames: Dict[str, Any],
    obs_star_photons: ObsStarPhotons,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Attach per-window star photon time series into the existing
    obs_target_frames structure, so each window contains both targets
    and stars.

    This implements **Option A**:

        - Treat the target windows (from NEBULA_TARGET_PHOTONS) as the
          canonical owners of:
              * window_index
              * start_index / end_index
              * n_frames
              * window-level time / sky meta
              * per-target photon time series

        - Treat the star photon product (from NEBULA_STAR_PHOTONS) as a
          **parallel** per-window catalog of star time series.

        - Join them by window_index for each observer, and attach:

              window["stars"]        <- star_window["stars"]
              window["n_stars"]      <- star_window["n_stars"]
              window["n_sources_total"] = n_targets + n_stars

    The function mutates obs_target_frames **in place** and also returns
    the same dictionary for convenience.

    Expected input (minimal, actual)
    --------------------------------

    obs_target_frames
        This is the ranked (and optionally sky-selected) target-frames
        structure you already have on disk, e.g. the contents of:

            NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl
            or
            NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_sky.pkl

        Per observer:

            obs_target_frames[obs_name] = {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "windows": [
                    {
                        "window_index": int,
                        "start_index": int,
                        "end_index": int,
                        "start_time": datetime,
                        "end_time": datetime,
                        "n_frames": int,
                        "n_targets": int,
                        "targets": {
                            "<target_id>": {
                                "t_utc": np.ndarray(n_frames),
                                "t_exp_s": np.ndarray(n_frames),
                                "x_pix": np.ndarray(n_frames),
                                "y_pix": np.ndarray(n_frames),
                                "flux_ph_m2_frame": np.ndarray(n_frames),
                                ...
                            },
                            ...
                        },
                        # plus sky_* and tracking_mode, etc.
                    },
                    ...
                ],
                ...
            }

    obs_star_photons
        Output from NEBULA_STAR_PHOTONS.build_star_photons_for_all_observers.
        Per observer:

            obs_star_photons[obs_name] = {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "catalog_name": str,
                "catalog_band": str,
                "run_meta": dict,
                "windows": [
                    {
                        "window_index": int,
                        "start_index": int,
                        "end_index": int,
                        "n_frames": int,
                        "tracking_mode": str,   # "sidereal" or "slew"
                        "n_stars": int,
                        "stars": {
                            "<gaia_source_id_str>": {
                                "source_id": str,
                                "source_type": "star",
                                "gaia_source_id": int or str,
                                "t_utc": np.ndarray(n_frames),
                                "t_exp_s": np.ndarray(n_frames),
                                "x_pix": np.ndarray(n_frames),
                                "y_pix": np.ndarray(n_frames),
                                "phi_ph_m2_s": np.ndarray(n_frames),
                                "flux_ph_m2_frame": np.ndarray(n_frames),
                                "mag_G": np.ndarray(n_frames),
                                "on_detector": np.ndarray(n_frames),
                            },
                            ...
                        },
                    },
                    ...
                ],
            }

    Parameters
    ----------
    obs_target_frames : dict
        Per-observer target photon frames (ranked / with-sky).

    obs_star_photons : dict
        Per-observer star photon products from NEBULA_STAR_PHOTONS.

    logger : logging.Logger, optional
        Logger for informational / diagnostic messages. If None, a
        module-level logger from _get_logger() is used.

    Returns
    -------
    dict
        The same obs_target_frames dictionary, with each window augmented
        by:

            - "n_stars"
            - "stars"
            - "n_sources_total" = n_targets + n_stars

    Raises
    ------
    RuntimeError
        If the star photon windows and target windows disagree on
        n_frames for a given window_index, or if multiple star windows
        share the same window_index for a given observer.
    """
    log = _get_logger(logger)

    # Iterate over all observers that have target frames.
    for obs_name, tgt_obs_entry in obs_target_frames.items():
        star_obs_entry = obs_star_photons.get(obs_name)

        if star_obs_entry is None:
            # Fail hard or warn? Here we warn and skip so that you can
            # still inspect partial results if desired.
            log.warning(
                "attach_star_photons_to_target_frames: no star photon entry "
                "found for observer '%s'; leaving its windows unchanged.",
                obs_name,
            )
            continue

        tgt_windows = tgt_obs_entry.get("windows", [])
        star_windows = star_obs_entry.get("windows", [])

        # Build a simple index: window_index -> star_window
        star_by_index: Dict[int, Dict[str, Any]] = {}
        for w in star_windows:
            idx = int(w.get("window_index"))
            if idx in star_by_index:
                raise RuntimeError(
                    "attach_star_photons_to_target_frames: duplicate star "
                    f"window_index={idx} for observer '{obs_name}'."
                )
            star_by_index[idx] = w

        attached_count = 0

        # Loop over target windows and bolt on matching star windows.
        for tgt_w in tgt_windows:
            if "window_index" not in tgt_w:
                raise RuntimeError(
                    "attach_star_photons_to_target_frames: a target window "
                    f"for observer '{obs_name}' is missing 'window_index'."
                )

            idx = int(tgt_w["window_index"])
            star_w = star_by_index.get(idx)

            if star_w is None:
                # This window simply has no stars in the catalog / on the
                # detector. Make that explicit but do not error.
                tgt_w.setdefault("n_stars", 0)
                tgt_w.setdefault("stars", {})
                tgt_w["n_sources_total"] = tgt_w.get("n_targets", 0)
                continue

            # Safety check: number of frames must agree if both non-zero.
            n_frames_tgt = int(tgt_w.get("n_frames", 0))
            n_frames_star = int(star_w.get("n_frames", 0))
            if n_frames_tgt and n_frames_star and (n_frames_tgt != n_frames_star):
                raise RuntimeError(
                    "attach_star_photons_to_target_frames: mismatch in n_frames "
                    f"for observer '{obs_name}', window_index={idx}: "
                    f"targets n_frames={n_frames_tgt}, stars n_frames={n_frames_star}."
                )

            # Attach star time series wholesale.
            stars_dict: Dict[str, Any] = star_w.get("stars", {}) or {}
            n_stars = int(star_w.get("n_stars", len(stars_dict)))

            tgt_w["stars"] = stars_dict
            tgt_w["n_stars"] = n_stars
            tgt_w["n_sources_total"] = tgt_w.get("n_targets", 0) + n_stars

            attached_count += 1

        log.info(
            "attach_star_photons_to_target_frames: observer '%s' -> "
            "attached stars to %d window(s).",
            obs_name,
            attached_count,
        )

    return obs_target_frames

# ----------------------------------------------------------------------
# Script guard
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # This module is intended to be used as a library, called from a
    # driver (e.g., sim_test) once all upstream pickles exist.
    #
    # Typical usage pattern:
    #
    #   from Utility.STARS import NEBULA_STAR_PHOTONS as NSP
    #
    #   obs_star_photons = NSP.run_star_photons_pipeline_from_pickles(
    #       frames_path=...,
    #       star_projection_path=...,
    #       star_slew_tracks_path=...,   # optional / for slewing mode
    #       output_path=...,             # optional, to save a pickle
    #       logger=logger,
    #   )
    #
    # For now, running this file directly without arguments will just
    # exit with a message. If you want an ad-hoc test harness, you can
    # edit this block locally to hard-code paths and call your pipeline
    # function.
    raise SystemExit(
        "NEBULA_STAR_PHOTONS is a library module. Import it and call your "
        "pipeline function (e.g., run_star_photons_pipeline_from_pickles(...)) "
        "from a driver such as sim_test."
    )
