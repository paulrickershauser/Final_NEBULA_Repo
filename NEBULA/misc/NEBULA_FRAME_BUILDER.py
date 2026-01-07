# NEBULA_FRAME_BUILDER.py
# ---------------------------------------------------------------------------
# Build per-frame source catalogs (no images yet) from NEBULA pixel pickles
# ---------------------------------------------------------------------------
"""
NEBULA_FRAME_BUILDER
====================

Purpose
-------
Organize NEBULA's coarse-time satellite products into *frame catalogs* that
can later be fed to an image / event generator (e.g., with photutils).

This module does **not** render images. It just answers:

    "At each frame time for observer X, which targets are on the detector,
     sunlit, and how bright are they at which pixel coordinates?"

Key ideas
---------
- We rely entirely on existing NEBULA picklers:

    NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs()

  which guarantees that upstream LOS, illumination, flux, and pointing
  products have already been attached.

- Frame existence is decided *only* by the observer-level flag:

    obs["pointing_valid_for_projection"][i] == True

- Source inclusion inside a frame is decided *only* by the per-target,
  per-observer flag:

    entry["on_detector_visible_sunlit"][i] == True

- For each included source we convert the LOS-gated photon flux
  (rad_photon_flux_g_m2_s_los_only) into total detected electrons per
  exposure using the EVK4 sensor configuration.

Outputs (SBSS-only prototype)
-----------------------------
For a single observer, build_frames_for_observer(...) returns:

    {
      "observer_name": str,
      "sensor_name":   str,
      "rows":          int,
      "cols":          int,
      "dt_frame_s":    float,
      "frames": [
        {
          "coarse_index": int,
          "t_utc":        datetime,
          "t_exp_s":      float,
          "sources": [
            {
              "source_id":    str,
              "source_type":  "target",
              "x_pix":        float,
              "y_pix":        float,
              "phi_ph_m2_s":  float,
              "flux_e_frame": float,
              "app_mag_g":    float,
              "range_km":     float,
            },
            ...
          ],
        },
        ...
      ],
    }

Later we can generalize this to multiple observers and/or add stars and
background models, but this file is deliberately focused on the simplest
"targets only" case.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# NEBULA sensor config (for EVK4 geometry + radiometry)
from Configuration.NEBULA_SENSOR_CONFIG import EVK4_SENSOR, SensorConfig  # type: ignore

# Pixel pickler (sits on top of all upstream picklers)
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_logger(name: str = "NEBULA_FRAME_BUILDER") -> logging.Logger:
    """
    Create or return a simple console logger for this module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_frame_time_info_for_observer(
    obs_track: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Determine which coarse timesteps will be treated as frames.

    Parameters
    ----------
    obs_track : dict
        One observer track dictionary from obs_tracks[obs_name], as returned
        by NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(). Must contain:
            - "times" (list/array of datetime objects)
            - "pointing_valid_for_projection" (array of bool)
    logger : logging.Logger, optional
        For status messages. If None, a default logger is created.

    Returns
    -------
    info : dict
        Dictionary with keys:
            - "frame_indices" : np.ndarray[int]
            - "frame_times"   : np.ndarray[datetime]
            - "dt_frame_s"    : float
    """
    if logger is None:
        logger = _get_logger()

    times = np.asarray(obs_track["times"])
    mask_valid = np.asarray(
        obs_track["pointing_valid_for_projection"], dtype=bool
    )

    if times.shape[0] != mask_valid.shape[0]:
        raise ValueError(
            "get_frame_time_info_for_observer: length mismatch between "
            f"times ({times.shape[0]}) and pointing_valid_for_projection "
            f"({mask_valid.shape[0]})."
        )

    # Indices where the observer is in a valid pointing configuration.
    frame_indices = np.where(mask_valid)[0]
    frame_times = times[frame_indices]

    # Estimate dt_frame_s from the coarse time grid. For now, use the
    # median difference in seconds between consecutive coarse samples.
    if len(times) > 1:
        dt_seconds = np.median(
            [
                (times[i + 1] - times[i]).total_seconds()
                for i in range(len(times) - 1)
            ]
        )
    else:
        dt_seconds = 0.0

    logger.info(
        "Frame builder: observer '%s' has %d coarse timesteps, "
        "%d with pointing_valid_for_projection=True. Estimated dt=%.3f s.",
        obs_track.get("name", "UNKNOWN"),
        len(times),
        len(frame_indices),
        dt_seconds,
    )

    return {
        "frame_indices": frame_indices,
        "frame_times": frame_times,
        "dt_frame_s": float(dt_seconds),
    }

def split_frame_indices_into_windows(
    frame_indices: np.ndarray,
    frame_times: np.ndarray,
    *,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Group contiguous frame indices into pointing *windows*.

    A new window starts whenever there is a gap in the coarse index sequence,
    i.e. frame_indices[i] != frame_indices[i-1] + 1. This effectively
    groups together runs of `pointing_valid_for_projection == True`.

    Parameters
    ----------
    frame_indices : np.ndarray[int]
        Indices where the observer is valid for projection.
    frame_times : np.ndarray[datetime]
        Times corresponding to `frame_indices`, same length.
    logger : logging.Logger, optional
        For status messages.

    Returns
    -------
    windows : list of dict
        Each entry has:
            - "window_index"  : int (0, 1, 2, ...)
            - "frame_indices" : np.ndarray[int]
            - "frame_times"   : np.ndarray[datetime]
            - "start_index"   : int
            - "end_index"     : int
            - "start_time"    : datetime
            - "end_time"      : datetime
    """
    if logger is None:
        logger = _get_logger()

    windows: List[Dict[str, Any]] = []

    if frame_indices.size == 0:
        logger.info(
            "Frame builder: no valid pointing timesteps; 0 windows."
        )
        return windows

    start = 0
    win_idx = 0

    # Walk through the indices and start a new window on any gap.
    for i in range(1, len(frame_indices)):
        if frame_indices[i] != frame_indices[i - 1] + 1:
            # Close out previous window [start, i)
            idx_slice = frame_indices[start:i]
            time_slice = frame_times[start:i]
            windows.append(
                {
                    "window_index": win_idx,
                    "frame_indices": idx_slice,
                    "frame_times": time_slice,
                    "start_index": int(idx_slice[0]),
                    "end_index": int(idx_slice[-1]),
                    "start_time": time_slice[0],
                    "end_time": time_slice[-1],
                }
            )
            win_idx += 1
            start = i

    # Final window [start, end]
    idx_slice = frame_indices[start:]
    time_slice = frame_times[start:]
    windows.append(
        {
            "window_index": win_idx,
            "frame_indices": idx_slice,
            "frame_times": time_slice,
            "start_index": int(idx_slice[0]),
            "end_index": int(idx_slice[-1]),
            "start_time": time_slice[0],
            "end_time": time_slice[-1],
        }
    )

    logger.info(
        "Frame builder: split %d valid timesteps into %d windows.",
        len(frame_indices),
        len(windows),
    )

    return windows

def build_frame_sources_for_index(
    idx: int,
    observer_name: str,
    tar_tracks: Dict[str, Any],
    *,
    t_exp_s: float,
    sensor_config: SensorConfig = EVK4_SENSOR,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Build a list of source entries for a single frame time index.

    This applies the *per-target* inclusion mask:

        entry["on_detector_visible_sunlit"][idx] == True

    and for each included target converts the LOS-gated photon flux into
    total detected electrons per frame.

    Parameters
    ----------
    idx : int
        Coarse time index (must already satisfy pointing_valid_for_projection).
    observer_name : str
        Name of the observer, used to select the correct by_observer block
        inside each target track.
    tar_tracks : dict
        Target track dictionary returned by NEBULA_PIXEL_PICKLER.
    t_exp_s : float
        Exposure time for this frame (seconds). For now, typically equal
        to the coarse dt_frame_s.
    sensor_config : SensorConfig, optional
        Sensor configuration to use (geometry + radiometry). Default is
        EVK4_SENSOR.
    logger : logging.Logger, optional
        Logger for debug messages.

    Returns
    -------
    sources : list of dict
        One dictionary per included target, with keys:
            - "source_id"
            - "source_type"
            - "x_pix", "y_pix"
            - "phi_ph_m2_s"
            - "flux_e_frame"
            - "app_mag_g"
            - "range_km"
    """
    if logger is None:
        logger = _get_logger()

    sources: List[Dict[str, Any]] = []

    # Radiometric factors from the sensor configuration
    A_collect = sensor_config.collecting_area_m2
    if A_collect is None:
        # For EVK4_SENSOR this should not happen, but guard just in case.
        raise RuntimeError(
            "SensorConfig.collecting_area_m2 is None; cannot convert photon "
            "flux to electrons. Please set aperture_diameter_m or "
            "aperture_area_m2 in NEBULA_SENSOR_CONFIG."
        )

    T_opt = sensor_config.optical_throughput or 1.0
    QE = sensor_config.quantum_efficiency or 1.0

    # Loop over all targets and build per-source entries.
    for tar_name, tar_track in tar_tracks.items():
        by_obs = tar_track.get("by_observer", {})
        if observer_name not in by_obs:
            # This target was never paired with this observer; skip.
            continue

        entry = by_obs[observer_name]

        # Safety: ensure the LOS-gated + on-detector arrays exist and are long enough.
        if "on_detector_visible_sunlit" not in entry:
            continue
        if idx >= len(entry["on_detector_visible_sunlit"]):
            continue

        if not bool(entry["on_detector_visible_sunlit"][idx]):
            # Target is not on the detector + sunlit at this frame time.
            continue

        # Extract pixel coordinates (float) for this timestep.
        x_pix = float(entry["pix_x"][idx])
        y_pix = float(entry["pix_y"][idx])

        # LOS-gated photon flux at the aperture [photons m^-2 s^-1].
        # If the key is missing, skip the target.
        if "rad_photon_flux_g_m2_s_los_only" not in entry:
            continue
        phi_ph_m2_s = float(entry["rad_photon_flux_g_m2_s_los_only"][idx])

        # Apparent magnitude in G band when LOS + visible; may be +inf when
        # there is no meaningful brightness. If missing, default to NaN.
        app_mag_g = float(
            entry.get("rad_app_mag_g_los_only", [np.nan])[idx]
        )

        # Range [km] (optional but useful for diagnostics).
        range_km = float(
            entry.get("los_icrs_range_km", [np.nan])[idx]
        )

        # Convert photon flux to electrons per frame:
        #   N_e = phi * A_collect * T_opt * QE * t_exp
        flux_e_frame = phi_ph_m2_s * A_collect * T_opt * QE * t_exp_s

        source_entry: Dict[str, Any] = {
            "source_id": tar_name,
            "source_type": "target",
            "x_pix": x_pix,
            "y_pix": y_pix,
            "phi_ph_m2_s": phi_ph_m2_s,
            "flux_e_frame": flux_e_frame,
            "app_mag_g": app_mag_g,
            "range_km": range_km,
        }

        sources.append(source_entry)

    return sources


# ---------------------------------------------------------------------------
# Public API: build frames for a single observer
# ---------------------------------------------------------------------------

def build_frames_for_observer(
    observer_name: str,
    *,
    max_frames: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build a per-frame source catalog for one observer.

    This is the main entry point for the SBSS-only prototype. It:

        1. Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(...) to load
           (or build) the pixel-augmented observer and target tracks.

        2. Uses get_frame_time_info_for_observer(...) to find all coarse
           timesteps where 'pointing_valid_for_projection' is True and uses
           those timesteps as frame times.

        3. For each frame index, calls build_frame_sources_for_index(...) to
           build a list of contributing targets.

        4. Packages the results into an 'ObserverFrames' dictionary that
           can later be pickled or passed to an image generator.

    Parameters
    ----------
    observer_name : str
        Name of the observer to process, e.g. "SBSS (USA 216)".
    max_frames : int or None, optional
        Optional cap on the number of frames to build (useful for quick
        experiments). If None, all valid frames are built.
    logger : logging.Logger, optional
        Logger to use. If None, a default console logger is created.

    Returns
    -------
    observer_frames : dict
        Dictionary with keys:
            - "observer_name"
            - "sensor_name"
            - "rows"
            - "cols"
            - "dt_frame_s"
            - "frames" : list of per-frame dicts
    """
    if logger is None:
        logger = _get_logger()

    # 1) Load pixel-augmented tracks for all observers and targets.
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=False,
        sensor_config=EVK4_SENSOR,
        logger=logger,
    )

    if observer_name not in obs_tracks:
        raise KeyError(
            f"Observer '{observer_name}' not found in obs_tracks. "
            f"Available observers: {list(obs_tracks.keys())}"
        )

    obs_track = obs_tracks[observer_name]

    # 2) Determine which coarse indices become frames.
    time_info = get_frame_time_info_for_observer(
        obs_track,
        logger=logger,
    )
    frame_indices: np.ndarray = time_info["frame_indices"]
    frame_times: np.ndarray = time_info["frame_times"]
    dt_frame_s: float = time_info["dt_frame_s"]

    if max_frames is not None and max_frames < len(frame_indices):
        frame_indices = frame_indices[:max_frames]
        frame_times = frame_times[:max_frames]
        logger.info(
            "Frame builder: restricting to first %d frames for observer '%s'.",
            len(frame_indices),
            observer_name,
        )

    # 3) Build per-frame source lists.
    frames: List[Dict[str, Any]] = []

    for idx, t_utc in zip(frame_indices, frame_times):
        sources = build_frame_sources_for_index(
            idx=idx,
            observer_name=observer_name,
            tar_tracks=tar_tracks,
            t_exp_s=dt_frame_s,
            sensor_config=EVK4_SENSOR,
            logger=logger,
        )

        frame_entry: Dict[str, Any] = {
            "coarse_index": int(idx),
            "t_utc": t_utc,
            "t_exp_s": dt_frame_s,
            "sources": sources,
        }

        frames.append(frame_entry)

    logger.info(
        "Frame builder: constructed %d frames for observer '%s'.",
        len(frames),
        observer_name,
    )

    # 4) Package everything into an ObserverFrames dictionary.
    observer_frames: Dict[str, Any] = {
        "observer_name": observer_name,
        "sensor_name": "EVK4",
        "rows": EVK4_SENSOR.rows,
        "cols": EVK4_SENSOR.cols,
        "dt_frame_s": dt_frame_s,
        "frames": frames,
    }

    return observer_frames

def build_frames_by_observer_and_window(
    *,
    max_frames_per_window: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build frame catalogs for *all* observers, grouped by pointing window.

    This is the "by observer → by window → by frame" version of
    NEBULA_FRAME_BUILDER. It:

        1. Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(...) once
           to load (or build) the pixel-augmented observer + target tracks.

        2. For each observer, uses get_frame_time_info_for_observer(...) to
           find the coarse indices where 'pointing_valid_for_projection' is
           True.

        3. Splits those indices into contiguous windows using
           split_frame_indices_into_windows(...).

        4. For each window and each frame index inside that window, calls
           build_frame_sources_for_index(...) to build the per-frame
           source list.

    Parameters
    ----------
    max_frames_per_window : int or None, optional
        Optional cap on the number of frames *per window* (useful for quick
        experiments). If None, all valid frames in each window are built.
    logger : logging.Logger, optional
        Logger to use. If None, a default console logger is created.

    Returns
    -------
    frames_by_observer : dict
        Dictionary keyed by observer_name. Each value has:

            {
              "observer_name": str,
              "sensor_name":   "EVK4",
              "rows":          int,
              "cols":          int,
              "dt_frame_s":    float,
              "windows": [
                {
                  "window_index":        int,
                  "start_index":         int,
                  "end_index":           int,
                  "start_time":          datetime,
                  "end_time":            datetime,
                  "n_frames":            int,
                  "n_unique_targets":    int,
                  "unique_target_ids":   list[str],
                  "frames": [
                    {
                      "coarse_index": int,
                      "t_utc":        datetime,
                      "t_exp_s":      float,
                      "sources":      [... as in build_frames_for_observer ...]
                    },
                    ...
                  ],
                },
                ...
              ],
            }
    """
    if logger is None:
        logger = _get_logger()

    # 1) Load pixel-augmented tracks once for all observers and targets.
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=False,
        sensor_config=EVK4_SENSOR,
        logger=logger,
    )

    frames_by_observer: Dict[str, Any] = {}

    logger.info(
        "Frame builder: constructing frames for %d observers.",
        len(obs_tracks),
    )

    for obs_name, obs_track in obs_tracks.items():
        logger.info("Frame builder: processing observer '%s'.", obs_name)

        # 2) Find all valid frame indices for this observer.
        time_info = get_frame_time_info_for_observer(
            obs_track,
            logger=logger,
        )
        frame_indices: np.ndarray = time_info["frame_indices"]
        frame_times: np.ndarray = time_info["frame_times"]
        dt_frame_s: float = time_info["dt_frame_s"]

        # 3) Split those indices into contiguous pointing windows.
        windows_meta = split_frame_indices_into_windows(
            frame_indices,
            frame_times,
            logger=logger,
        )

        windows_out: List[Dict[str, Any]] = []

        for win in windows_meta:
            win_indices = win["frame_indices"]
            win_times = win["frame_times"]

            # Optional cap per window.
            if (
                max_frames_per_window is not None
                and len(win_indices) > max_frames_per_window
            ):
                logger.info(
                    "Frame builder: window %d for observer '%s' has %d frames; "
                    "restricting to first %d for inspection.",
                    win["window_index"],
                    obs_name,
                    len(win_indices),
                    max_frames_per_window,
                )
                win_indices = win_indices[:max_frames_per_window]
                win_times = win_times[:max_frames_per_window]

            frames: List[Dict[str, Any]] = []

            # 4) Build per-frame source lists inside this window.
            for idx, t_utc in zip(win_indices, win_times):
                sources = build_frame_sources_for_index(
                    idx=int(idx),
                    observer_name=obs_name,
                    tar_tracks=tar_tracks,
                    t_exp_s=dt_frame_s,
                    sensor_config=EVK4_SENSOR,
                    logger=logger,
                )

                frame_entry: Dict[str, Any] = {
                    "coarse_index": int(idx),
                    "t_utc": t_utc,
                    "t_exp_s": dt_frame_s,
                    "sources": sources,
                }
                frames.append(frame_entry)

            # --- NEW: window-level summary of targets --------------------
            # Collect all distinct source_ids seen in any frame of this window.
            unique_target_ids = sorted(
                {
                    src["source_id"]
                    for frame in frames
                    for src in frame.get("sources", [])
                }
            )
            n_unique_targets = len(unique_target_ids)
            n_frames = len(frames)
            # -------------------------------------------------------------

            window_out: Dict[str, Any] = {
                "window_index": int(win["window_index"]),
                "start_index": int(win["start_index"]),
                "end_index": int(win["end_index"]),
                "start_time": win["start_time"],
                "end_time": win["end_time"],
                "n_frames": n_frames,
                "n_unique_targets": n_unique_targets,
                "unique_target_ids": unique_target_ids,
                "frames": frames,
            }
            windows_out.append(window_out)

        total_frames = sum(len(w["frames"]) for w in windows_out)
        logger.info(
            "Frame builder: observer '%s' has %d windows with %d total frames.",
            obs_name,
            len(windows_out),
            total_frames,
        )

        frames_by_observer[obs_name] = {
            "observer_name": obs_name,
            "sensor_name": "EVK4",
            "rows": EVK4_SENSOR.rows,
            "cols": EVK4_SENSOR.cols,
            "dt_frame_s": dt_frame_s,
            "windows": windows_out,
        }

    logger.info(
        "Frame builder: built windowed frame catalogs for %d observers.",
        len(frames_by_observer),
    )

    return frames_by_observer

if __name__ == "__main__":
    # Example manual invocation from Spyder:
    #
    #   %runfile '.../Utility/FRAMES/NEBULA_FRAME_BUILDER.py' --wdir
    #
    # Then inspect the 'all_frames' object in the variable explorer.
    #
    logger = _get_logger()
    logger.info(
        "NEBULA_FRAME_BUILDER: building frames for ALL observers, "
        "grouped by pointing windows."
    )

    all_frames = build_frames_by_observer_and_window(
        max_frames_per_window=None,  # use all frames; change if you want a quick subset
        logger=logger,
    )

    # Simple summary log:
    for obs_name, obs_data in all_frames.items():
        n_windows = len(obs_data.get("windows", []))
        n_frames = sum(len(w.get("frames", [])) for w in obs_data.get("windows", []))
        logger.info(
            "NEBULA_FRAME_BUILDER: observer '%s' -> %d windows, %d frames.",
            obs_name,
            n_windows,
            n_frames,
        )
