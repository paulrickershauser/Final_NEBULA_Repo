# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:57:13 2025

@author: prick
"""

import logging
from Utility.FRAMES import NEBULA_FRAME_BUILDER as FB

def pick_best_window_for_first_observer(
    max_frames_per_window=None,
    logger: logging.Logger | None = None,
):
    """
    Build frames for all observers, then for the *first* observer in the dict
    return the pointing window with the largest number of unique targets.

    Returns
    -------
    obs_name : str
        Name of the chosen observer.
    best_window : dict
        Window dict from frames_by_observer[obs_name]["windows"].
    frames_by_observer : dict
        Full output of build_frames_by_observer_and_window, in case you
        also want other observers/windows.
    """
    if logger is None:
        logger = FB._get_logger()  # reuse the frame-builder logger

    frames_by_observer = FB.build_frames_by_observer_and_window(
        max_frames_per_window=max_frames_per_window,
        logger=logger,
    )

    if not frames_by_observer:
        raise RuntimeError("No observers found in frames_by_observer.")

    # “First observer” without hard-coding SBSS/etc.
    obs_names = sorted(frames_by_observer.keys())
    obs_name = obs_names[0]
    obs_data = frames_by_observer[obs_name]

    windows = obs_data.get("windows", [])
    if not windows:
        raise RuntimeError(f"Observer '{obs_name}' has no pointing windows.")

    # Sort by most unique targets, then by number of frames as a tiebreaker.
    best_window = max(
        windows,
        key=lambda w: (w.get("n_unique_targets", 0), w.get("n_frames", 0)),
    )

    logger.info(
        "Selected best window %d for observer '%s': %d unique targets, %d frames.",
        best_window["window_index"],
        obs_name,
        best_window.get("n_unique_targets", 0),
        best_window.get("n_frames", 0),
    )

    return obs_name, best_window, frames_by_observer

import numpy as np
from astropy.nddata import CCDData
import astropy.units as u

def frame_to_ccd(obs_data: dict, frame: dict) -> CCDData:
    """
    Convert a NEBULA frame dict into a simple CCDData image where each
    target deposits its electrons into the nearest pixel.
    """
    nrows = obs_data["rows"]
    ncols = obs_data["cols"]

    img = np.zeros((nrows, ncols), dtype=float)

    for src in frame.get("sources", []):
        x = src["x_pix"]
        y = src["y_pix"]
        flux_e = src["flux_e_frame"]  # electrons per frame

        # Nearest-neighbour deposit; you can later upgrade this to a PSF.
        j = int(round(x))   # column index
        i = int(round(y))   # row index

        if 0 <= i < nrows and 0 <= j < ncols:
            img[i, j] += flux_e

    # Wrap as CCDData. Using 'electron' as the natural unit for your detector.
    ccd = CCDData(img, unit=u.electron)

    # Attach some metadata (add WCS later from NEBULA_WCS)
    ccd.meta["OBSERVER"] = obs_data["observer_name"]
    ccd.meta["T_EXP"] = frame["t_exp_s"]
    ccd.meta["T_UTC"] = frame["t_utc"].isoformat()

    return ccd

# Pick observer + window
obs_name, best_window, frames_by_observer = pick_best_window_for_first_observer()
obs_data = frames_by_observer[obs_name]

# For now, just take the first frame from that window
frame0 = best_window["frames"][0]

ccd = frame_to_ccd(obs_data, frame0)

# Example photutils import + aperture photometry:
from photutils.aperture import CircularAperture, aperture_photometry

positions = [(src["x_pix"], src["y_pix"]) for src in frame0["sources"]]
apertures = CircularAperture(positions, r=2.0)  # tweak r as you like

tbl = aperture_photometry(ccd, apertures)
print(tbl)
