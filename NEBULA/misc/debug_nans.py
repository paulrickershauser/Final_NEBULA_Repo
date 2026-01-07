# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:06:50 2025

@author: prick
"""

import numpy as np

def debug_nan_pixels_for_pair(
    obs_tracks,
    tar_tracks,
    observer_name="SBSS (USA 216)",
    target_name="TDRS 3",
    start_idx=563,
    end_idx=1276,
):
    """
    Investigate NaN pix_x / pix_y values for a specific observer–target pair.

    This function assumes that obs_tracks and tar_tracks are the dictionaries
    produced by NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(), i.e., that
    they contain per-observer pixel and LOS information for each target.

    For the requested observer and target, it:

        1. Extracts pix_x, pix_y, LOS RA/Dec, and boresight RA/Dec.
        2. Looks at the index range [start_idx, end_idx] (inclusive).
        3. Finds the timesteps in that range where pix_x or pix_y is NaN.
        4. For those timesteps, prints:
            - Index i
            - pix_x[i], pix_y[i]
            - LOS RA/Dec (deg)
            - Boresight RA/Dec (deg)
            - Angular separation between LOS and boresight (deg)
            - LOS visibility and illumination flags if available.

    Parameters
    ----------
    obs_tracks : dict
        Observer track dictionary returned by NEBULA_PIXEL_PICKLER.
    tar_tracks : dict
        Target track dictionary returned by NEBULA_PIXEL_PICKLER.
    observer_name : str, optional
        Name of the observer to inspect (default: "SBSS (USA 216)").
    target_name : str, optional
        Name of the target to inspect (default: "TDRS 3").
    start_idx : int, optional
        Starting index (inclusive) of the timestep range to inspect.
    end_idx : int, optional
        Ending index (inclusive) of the timestep range to inspect.

    Returns
    -------
    None
        Prints diagnostic information to the console.
    """

    # ----------------------------------------------------------------------
    # Helper to compute angular separation between two RA/Dec directions.
    # ----------------------------------------------------------------------
    def ang_sep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
        # Convert all angles from degrees to radians.
        ra1 = np.radians(ra1_deg)
        dec1 = np.radians(dec1_deg)
        ra2 = np.radians(ra2_deg)
        dec2 = np.radians(dec2_deg)

        # Compute cosine of angular separation using spherical law of cosines.
        cosang = (
            np.sin(dec1) * np.sin(dec2)
            + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        )

        # Clip to valid domain [-1, 1] to avoid numerical issues.
        cosang = np.clip(cosang, -1.0, 1.0)

        # Return separation in degrees.
        return np.degrees(np.arccos(cosang))

    # ----------------------------------------------------------------------
    # Basic existence checks for the requested observer and target.
    # ----------------------------------------------------------------------

    # Ensure the observer exists in obs_tracks.
    if observer_name not in obs_tracks:
        print(f"[ERROR] Observer '{observer_name}' not found in obs_tracks.")
        return

    # Ensure the target exists in tar_tracks.
    if target_name not in tar_tracks:
        print(f"[ERROR] Target '{target_name}' not found in tar_tracks.")
        return

    # Get the observer and target track dictionaries.
    obs_track = obs_tracks[observer_name]
    tar_track = tar_tracks[target_name]

    # ----------------------------------------------------------------------
    # Extract per-observer dictionary for this target.
    # ----------------------------------------------------------------------

    # Get the by_observer dictionary for this target.
    by_observer = tar_track.get("by_observer", None)
    if by_observer is None or observer_name not in by_observer:
        print(
            f"[ERROR] Target '{target_name}' has no by_observer entry for "
            f"observer '{observer_name}'."
        )
        return

    # Extract the sub-dict for this observer–target pair.
    by_obs = by_observer[observer_name]

    # ----------------------------------------------------------------------
    # Extract pixel coordinates and LOS / mask arrays.
    # ----------------------------------------------------------------------

    # Convert pix_x and pix_y to numpy arrays for easier masking.
    pix_x = np.asarray(by_obs.get("pix_x", []), dtype=float)
    pix_y = np.asarray(by_obs.get("pix_y", []), dtype=float)

    # Extract LOS RA/Dec arrays (in degrees) for this pair.
    los_ra = np.asarray(by_obs.get("los_icrs_ra_deg", []), dtype=float)
    los_dec = np.asarray(by_obs.get("los_icrs_dec_deg", []), dtype=float)

    # Extract observer boresight RA/Dec arrays from the observer track.
    bore_ra = np.asarray(obs_track.get("pointing_boresight_ra_deg", []), dtype=float)
    bore_dec = np.asarray(obs_track.get("pointing_boresight_dec_deg", []), dtype=float)

    # Extract LOS and illumination flags if present.
    los_visible = np.asarray(by_obs.get("los_visible", []), dtype=bool)
    illum_is_sunlit = np.asarray(by_obs.get("illum_is_sunlit", []), dtype=bool)
    on_detector = np.asarray(by_obs.get("on_detector", []), dtype=bool)
    on_det_vis_sun = np.asarray(
        by_obs.get("on_detector_visible_sunlit", []), dtype=bool
    )

    # ----------------------------------------------------------------------
    # Sanity check: all key arrays should have consistent length.
    # ----------------------------------------------------------------------

    n = len(pix_x)
    print(f"Total timesteps for this pair: {n}")

    # Define a helper to check a single array length.
    def check_len(name, arr):
        if len(arr) not in (0, n):
            print(
                f"[WARN] Length mismatch for '{name}': len={len(arr)}, expected {n}"
            )

    # Check each relevant array.
    check_len("pix_y", pix_y)
    check_len("los_ra", los_ra)
    check_len("los_dec", los_dec)
    check_len("bore_ra", bore_ra)
    check_len("bore_dec", bore_dec)
    check_len("los_visible", los_visible)
    check_len("illum_is_sunlit", illum_is_sunlit)
    check_len("on_detector", on_detector)
    check_len("on_detector_visible_sunlit", on_det_vis_sun)

    # ----------------------------------------------------------------------
    # Define the index range to inspect; clamp to [0, n-1].
    # ----------------------------------------------------------------------

    i0 = max(0, start_idx)
    i1 = min(n - 1, end_idx)

    print(f"Inspecting indices from {i0} to {i1} (inclusive).")

    # Build a boolean mask for NaN pixels in this range.
    idx = np.arange(n)
    in_range = (idx >= i0) & (idx <= i1)
    nan_mask = in_range & (np.isnan(pix_x) | np.isnan(pix_y))

    # Get the actual indices where pixels are NaN in the requested range.
    nan_indices = idx[nan_mask]

    print(f"Number of NaN timesteps in this range: {nan_indices.size}")

    if nan_indices.size == 0:
        print("No NaNs in pix_x/pix_y in the requested index range.")
        return

    # ----------------------------------------------------------------------
    # Print detailed info for each NaN index (or a subset, if many).
    # ----------------------------------------------------------------------

    # If there are many NaNs, you might want to limit how many you print.
    max_print = 714
    if nan_indices.size > max_print:
        print(
            f"[INFO] Showing details for the first {max_print} NaN indices "
            f"out of {nan_indices.size}."
        )
        nan_indices_to_print = nan_indices[:max_print]
    else:
        nan_indices_to_print = nan_indices

    for i in nan_indices_to_print:
        # Safely get LOS RA/Dec; if array is too short, use NaN.
        ra_i = los_ra[i] if len(los_ra) == n else np.nan
        dec_i = los_dec[i] if len(los_dec) == n else np.nan

        # Safely get boresight RA/Dec.
        bore_ra_i = bore_ra[i] if len(bore_ra) == n else np.nan
        bore_dec_i = bore_dec[i] if len(bore_dec) == n else np.nan

        # Compute angular separation if all values are finite.
        if np.isfinite(ra_i) and np.isfinite(dec_i) and np.isfinite(bore_ra_i) and np.isfinite(bore_dec_i):
            sep = ang_sep_deg(ra_i, dec_i, bore_ra_i, bore_dec_i)
        else:
            sep = np.nan

        # Get flags for this timestep (if arrays have correct length).
        los_i = bool(los_visible[i]) if len(los_visible) == n else False
        illum_i = bool(illum_is_sunlit[i]) if len(illum_is_sunlit) == n else False
        on_det_i = bool(on_detector[i]) if len(on_detector) == n else False
        on_det_vis_sun_i = bool(on_det_vis_sun[i]) if len(on_det_vis_sun) == n else False

        # Print a summary line for this index.
        print(
            f"i={i}: pix_x=NaN, pix_y=NaN | "
            f"LOS(ra,dec)=({ra_i:.3f}, {dec_i:.3f}) deg | "
            f"Bore(ra,dec)=({bore_ra_i:.3f}, {bore_dec_i:.3f}) deg | "
            f"sep={sep:.3f} deg | "
            f"los_visible={los_i}, illum_sun={illum_i}, "
            f"on_detector={on_det_i}, on_det_vis_sun={on_det_vis_sun_i}"
        )

    print("Done inspecting NaN pixels for this observer–target pair.")
