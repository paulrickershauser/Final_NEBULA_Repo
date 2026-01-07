"""
NEBULA_OBS_TAR_STAR_ANIMATION
=============================

Simple "delta-PSF" animation utilities for combined target + star products.

This module is designed to work primarily with the `combined_top_windows`
product produced in `sim_test`, with structure:

    combined_top_windows[obs_name] = {
        "observer_name": str,
        "sensor_name": Optional[str],
        "rows": int,
        "cols": int,
        "dt_frame_s": float,
        "window_index": int,
        "tracking_mode": Optional[str],
        "n_frames": int,
        "n_targets": int,
        "n_stars": int,
        "frames": [frame_out_0, frame_out_1, ..., frame_out_{N-1}],
    }

Each `frame_out_j` is expected to be a dict with:

    {
        "frame_index": int,
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

This module does NOT perform any physics or WCS. It simply:

  1) Reads this structure,
  2) Builds a per-frame 2D intensity image (delta-PSF: one pixel per source),
  3) Writes an MP4 (or similar) animation.

Brightness scaling
------------------

Pixel intensities are taken directly from `flux_ph_m2_frame` for both
targets and stars:

    I_pixel = sum_over_sources( flux_ph_m2_frame(source) )

This ensures that targets and stars live on the SAME physical photon-flux
scale. No normalization is applied at the cube-building stage; scaling to
display range (vmin / vmax) happens only at the image-rendering level and
remains linear in flux.

Dependencies
------------

Relies only on standard library + numpy + matplotlib.

You must have `ffmpeg` available on your PATH for the FFMpegWriter backend
to work.

"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm  # for logarithmic display stretch


# Type alias for clarity
CombinedTopWindows = Dict[str, Dict[str, Any]]

def stamp_gaussian_psf(
    frame2d: np.ndarray,
    x_center: float,
    y_center: float,
    intensity: float,
    size: int = 7,
    fwhm_pix: float = 1.5,
) -> None:
    """
    Stamp a simple 2D Gaussian PSF into a single 2D frame at a
    (sub-pixel) center location.

    Parameters
    ----------
    frame2d : np.ndarray
        2D array with shape (rows, cols) into which the PSF will be added.
    x_center : float
        PSF center in pixel *column* coordinates (0 .. cols).
    y_center : float
        PSF center in pixel *row* coordinates (0 .. rows).
    intensity : float
        Total intensity to distribute across the PSF. The function
        normalizes weights so that the sum of all contributions is
        approximately equal to 'intensity' (exact if the PSF patch is
        fully on-chip).
    size : int, optional
        Linear size of the PSF patch in pixels (must be odd). Default: 7.
    fwhm_pix : float, optional
        Full width at half maximum of the Gaussian in pixels. Default: 1.5.
    """
    rows, cols = frame2d.shape

    # Ensure 'size' is odd so we have a symmetric patch around the center.
    if size % 2 == 0:
        size += 1
    radius = size // 2

    # Convert FWHM to Gaussian sigma in pixels.
    # FWHM = 2 * sqrt(2*ln(2)) * sigma
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if sigma <= 0.0 or not np.isfinite(sigma):
        # Fallback: if something is weird, just drop back to a delta-PSF.
        col_idx = int(np.floor(x_center))
        row_idx = int(np.floor(y_center))
        if 0 <= row_idx < rows and 0 <= col_idx < cols:
            frame2d[row_idx, col_idx] += intensity
        return

    # Precompute all weights in the patch and accumulate for normalization.
    weights = []
    weight_sum = 0.0

    for dy in range(-radius, radius + 1):
        # Candidate row index for this patch pixel
        row = int(np.floor(y_center)) + dy
        if (row < 0) or (row >= rows):
            continue

        for dx in range(-radius, radius + 1):
            # Candidate column index
            col = int(np.floor(x_center)) + dx
            if (col < 0) or (col >= cols):
                continue

            # Compute offset from PSF center to pixel center (sub-pixel aware).
            # Here we assume pixel centers at (col + 0.5, row + 0.5).
            dcol = (col + 0.5) - x_center
            drow = (row + 0.5) - y_center

            r2 = dcol * dcol + drow * drow
            w = np.exp(-0.5 * r2 / (sigma * sigma))

            weights.append((row, col, w))
            weight_sum += w

    # Guard against empty or degenerate patches
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        # Again, fallback to delta-PSF if something goes wrong
        col_idx = int(np.floor(x_center))
        row_idx = int(np.floor(y_center))
        if 0 <= row_idx < rows and 0 <= col_idx < cols:
            frame2d[row_idx, col_idx] += intensity
        return

    # Distribute intensity across the patch so that the total ~ intensity.
    scale = intensity / weight_sum

    for row, col, w in weights:
        frame2d[row, col] += scale * w

def load_combined_top_windows_pickle(
    pickle_path: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> CombinedTopWindows:
    """
    Load a combined_top_windows pickle from disk.

    Parameters
    ----------
    pickle_path : str or Path
        Path to the pickle file produced by sim_test that contains
        the combined_top_windows dict.
    logger : logging.Logger, optional
        Optional logger for status messages.

    Returns
    -------
    combined : dict
        The combined_top_windows dictionary as loaded from pickle.
    """
    # Normalize the path to a Path object
    pickle_path = Path(pickle_path)

    # Log the load attempt if a logger is provided
    if logger is not None:
        logger.info(
            "NEBULA_OBS_TAR_STAR_ANIMATION: Loading combined_top_windows from %s",
            pickle_path,
        )

    # Use binary read mode for pickle
    with pickle_path.open("rb") as f:
        combined = pickle.load(f)

    # Basic sanity check: combined should be a dict
    if not isinstance(combined, dict):
        raise TypeError(
            f"Expected a dict from {pickle_path}, got {type(combined)} instead."
        )

    return combined


def build_frame_cube_for_observer(
    observer_entry: Dict[str, Any],
    intensity_key: str = "flux_ph_m2_frame",
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Build a simple delta-PSF frame cube for a single observer.

    Parameters
    ----------
    observer_entry : dict
        One entry from combined_top_windows, e.g.:
            combined_top_windows["SBSS (USA 216)"].
        Must contain:
            - "rows", "cols": detector dimensions
            - "frames": list of per-frame dicts
            - Each frame dict must contain "targets" and "stars" dicts.
    intensity_key : str, optional
        Which per-source field to use as the pixel intensity. By default,
        "flux_ph_m2_frame", i.e. photon flux at the aperture integrated
        over the exposure time for that frame.
    logger : logging.Logger, optional
        Optional logger for progress messages.

    Returns
    -------
    cube : np.ndarray
        3D array of shape (n_frames, rows, cols) containing per-pixel
        intensities in the SAME units and linear scale as the chosen
        intensity_key (default: flux_ph_m2_frame).

        cube[j, r, c] = sum of intensity_key for all sources that land
        in pixel (r, c) in frame j, with delta-PSF (one pixel per source).

    Notes
    -----
    - x_pix / y_pix are assumed to be pixel coordinates in [0, cols),
      [0, rows), where x=0, y=0 correspond to the lower-left pixel.
    - We convert to integer indices via floor() and clip to valid range.
    - Sources with NaN coordinates or NaN intensities are ignored.
    - Stars with on_detector == False are ignored (if the flag exists).
    """

    # Extract detector dimensions
    rows = int(observer_entry.get("rows"))
    cols = int(observer_entry.get("cols"))

    # Extract the list of frame dicts
    frames = observer_entry.get("frames", [])
    n_frames = len(frames)

    if n_frames == 0:
        raise ValueError("Observer entry has zero frames; nothing to animate.")

    if logger is not None:
        logger.info(
            "Building frame cube for observer '%s': %d frames, %d x %d pixels.",
            observer_entry.get("observer_name", "<unknown>"),
            n_frames,
            rows,
            cols,
        )

    # Allocate the cube: (n_frames, rows, cols).
    # Use float64 to retain dynamic range.
    cube = np.zeros((n_frames, rows, cols), dtype=np.float64)

    # Iterate over frames
    for j, frame in enumerate(frames):
        # Optional: log every N frames if many frames exist
        # (only if logger is provided and N is chosen reasonably)
        # if logger is not None and j % 100 == 0:
        #     logger.debug("  Processing frame %d / %d", j, n_frames)

        # Handle targets first
        # target_dict = frame.get("targets", {})
        # for target_id, target in target_dict.items():
        #     # Get pixel coordinates as floats
        #     x = float(target.get("x_pix", np.nan))
        #     y = float(target.get("y_pix", np.nan))

        #     # Skip if coordinates are NaN
        #     if not np.isfinite(x) or not np.isfinite(y):
        #         continue

        #     # Convert to integer pixel indices using floor
        #     # (delta-PSF: source contributes to exactly one pixel)
        #     col_idx = int(np.floor(x))
        #     row_idx = int(np.floor(y))

        #     # Bounds check: skip sources that fall off the detector
        #     if (col_idx < 0) or (col_idx >= cols):
        #         continue
        #     if (row_idx < 0) or (row_idx >= rows):
        #         continue

        #     # Retrieve intensity in the requested units
        #     val = target.get(intensity_key, np.nan)

        #     # Convert to float and skip if NaN
        #     try:
        #         intensity = float(val)
        #     except (TypeError, ValueError):
        #         continue

        #     if not np.isfinite(intensity):
        #         continue

        #     # Add intensity to the corresponding pixel
        #     cube[j, row_idx, col_idx] += intensity
        # Handle targets first
        target_dict = frame.get("targets", {})
        for target_id, target in target_dict.items():
            # Get pixel coordinates as floats
            x = float(target.get("x_pix", np.nan))
            y = float(target.get("y_pix", np.nan))
        
            # Skip if coordinates are NaN
            if not np.isfinite(x) or not np.isfinite(y):
                continue
        
            # Retrieve intensity in the requested units
            val = target.get(intensity_key, np.nan)
            try:
                intensity = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(intensity) or intensity <= 0.0:
                continue
        
            # Stamp a small Gaussian PSF centered at (x, y)
            stamp_gaussian_psf(
                frame2d=cube[j],
                x_center=x,
                y_center=y,
                intensity=intensity,
                size=7,        # can tweak; 5–9 are typical for mock PSF
                fwhm_pix=1.5,  # match your ACTIVE_SENSOR.psf_fwhm_pix if you like
            )


        # # Handle stars next
        # star_dict = frame.get("stars", {})
        # for star_id, star in star_dict.items():
        #     # Respect on_detector flag if present
        #     on_detector = star.get("on_detector", True)
        #     if isinstance(on_detector, (list, np.ndarray)):
        #         # Defensive: if somehow an array sneaks in, treat False if any False
        #         if not bool(np.all(on_detector)):
        #             continue
        #     else:
        #         if not bool(on_detector):
        #             continue

        #     # Get pixel coordinates
        #     x = float(star.get("x_pix", np.nan))
        #     y = float(star.get("y_pix", np.nan))

        #     # Skip if coordinates are NaN
        #     if not np.isfinite(x) or not np.isfinite(y):
        #         continue

        #     # Convert to integer indices
        #     col_idx = int(np.floor(x))
        #     row_idx = int(np.floor(y))

        #     # Bounds check
        #     if (col_idx < 0) or (col_idx >= cols):
        #         continue
        #     if (row_idx < 0) or (row_idx >= rows):
        #         continue

        #     # Retrieve intensity
        #     val = star.get(intensity_key, np.nan)

        #     try:
        #         intensity = float(val)
        #     except (TypeError, ValueError):
        #         continue

        #     if not np.isfinite(intensity):
        #         continue

        #     # Add to cube
        #     cube[j, row_idx, col_idx] += intensity
        # Handle stars next
        star_dict = frame.get("stars", {})
        for star_id, star in star_dict.items():
            # Respect on_detector flag if present
            on_detector = star.get("on_detector", True)
            if not bool(on_detector):
                continue
        
            x = float(star.get("x_pix", np.nan))
            y = float(star.get("y_pix", np.nan))
            if not np.isfinite(x) or not np.isfinite(y):
                continue
        
            val = star.get(intensity_key, np.nan)
            try:
                intensity = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(intensity) or intensity <= 0.0:
                continue
        
            stamp_gaussian_psf(
                frame2d=cube[j],
                x_center=x,
                y_center=y,
                intensity=intensity,
                size=7,
                fwhm_pix=1.5,
            )

    return cube


def compute_linear_vmin_vmax(
    cube: np.ndarray,
    clip_percentile: float = 95,
) -> tuple[float, float]:
    """
    Compute a global linear display range (vmin, vmax) for a frame cube.

    Parameters
    ----------
    cube : np.ndarray
        3D array of shape (n_frames, rows, cols) with pixel intensities
        in physical units (e.g., photon flux per frame).
    clip_percentile : float, optional
        Upper percentile used to define vmax. For example, 99.5 means
        we take the 99.5th percentile of all positive finite intensities
        across the entire cube as vmax. This is a simple way to avoid
        a single extremely bright source saturating the colormap.

    Returns
    -------
    vmin : float
        Lower bound for imshow; always 0.0 for linear photon flux.
    vmax : float
        Upper bound for imshow; percentile of the positive finite pixels.

    Notes
    -----
    This function does NOT rescale or warp the underlying data. It only
    provides a sensible global range for visualization while maintaining
    linearity in flux across targets and stars.
    """
    # Flatten cube to 1D
    data = cube.reshape(-1)

    # Keep only positive finite values
    mask = np.isfinite(data) & (data > 0.0)
    if not np.any(mask):
        # Fallback: if everything is zero / non-finite, just return (0, 1)
        return 0.0, 1.0

    finite_vals = data[mask]

    # Compute the desired percentile as vmax
    vmax = float(np.percentile(finite_vals, clip_percentile))

    # Ensure vmax is positive and non-zero
    if vmax <= 0.0 or not np.isfinite(vmax):
        vmax = 1.0

    # For photon flux, a natural vmin is 0.0
    vmin = 0.0

    return vmin, vmax


def write_movie_from_cube(
    frame_cube: np.ndarray,
    output_path: Union[str, Path],
    fps: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "gray",
    dpi: int = 150,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Write an MP4 movie from a (n_frames, rows, cols) frame cube.

    Parameters
    ----------
    frame_cube : np.ndarray
        3D array of shape (n_frames, rows, cols) containing pixel
        intensities in physical units (e.g. photon flux per frame).
    output_path : str or Path
        Path to the output movie file. Typically should end with ".mp4".
    fps : float, optional
        Frames per second for the movie. This does not affect the
        underlying physics; it just controls how fast the animation
        plays back.
    vmin : float, optional
        Lower bound for imshow. If None, computed from the cube via
        compute_linear_vmin_vmax.
    vmax : float, optional
        Upper bound for imshow. If None, computed from the cube via
        compute_linear_vmin_vmax.
    cmap : str, optional
        Matplotlib colormap name.
    dpi : int, optional
        Dots-per-inch setting for the output figure.
    title : str, optional
        Optional figure title.
    logger : logging.Logger, optional
        Optional logger for progress messages.

    Notes
    -----
    - Uses matplotlib.animation.FFMpegWriter; requires ffmpeg in PATH.
    - This function treats the cube values as linear intensities and
      does NOT apply any non-linear stretching (e.g. log scale).
    """

    # Normalize output path
    output_path = Path(output_path)

    if logger is not None:
        logger.info(
            "Writing movie to %s (fps=%.3f, cmap=%s).",
            output_path,
            fps,
            cmap,
        )

    # Determine number of frames and image dimensions
    n_frames, rows, cols = frame_cube.shape

    # Determine vmin / vmax if not provided
    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = compute_linear_vmin_vmax(frame_cube)
        if vmin is None:
            vmin = vmin_auto
        if vmax is None:
            vmax = vmax_auto

    # Create a figure and axes
    fig, ax = plt.subplots()
    fig.set_dpi(dpi)

    # Initial image for frame 0
    im = ax.imshow(
        frame_cube[0],
        origin="lower",  # y increasing upwards
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        animated=True,
    )



    # Add colorbar for reference
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Flux [ph m$^{-2}$ frame$^{-1}$]")


    # Axis labels
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")

    # Title
    if title is not None:
        ax.set_title(title)

    # Define the update function for animation
    def _update(frame_index: int):
        """
        Update function called by FuncAnimation for each frame.
        """
        # Update image data in-place
        im.set_array(frame_cube[frame_index])

        # Optionally, update a dynamic title if desired
        # (Disabled by default to avoid jitter)
        # ax.set_title(f"{title} - frame {frame_index}")

        return (im,)

    # Set up the animation object
    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000.0 / max(fps, 1e-6),  # ms per frame
        blit=True,
    )

    # Use FFMpegWriter to save as MP4
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(str(output_path), writer=writer)

    # Close figure to free memory
    plt.close(fig)

    if logger is not None:
        logger.info("Finished writing movie to %s.", output_path)


def make_observer_animation(
    observer_entry: Dict[str, Any],
    output_path: Union[str, Path],
    fps: float = 1.0,
    intensity_key: str = "flux_ph_m2_frame",
    clip_percentile: float = 99.5,
    cmap: str = "gray",
    dpi: int = 150,
    logger: Optional[logging.Logger] = None,
) -> None:


    """
    High-level helper: build a frame cube for one observer and write a movie.

    Parameters
    ----------
    observer_entry : dict
        One entry from combined_top_windows, e.g.:
            combined_top_windows["SBSS (USA 216)"].
    output_path : str or Path
        Output movie path (e.g. 'SBSS_top_window.mp4').
    fps : float, optional
        Frames per second for the movie.
    intensity_key : str, optional
        Field used as per-source intensity (default: 'flux_ph_m2_frame').
    clip_percentile : float, optional
        Percentile used for global linear vmax (see compute_linear_vmin_vmax).
    cmap : str, optional
        Matplotlib colormap.
    dpi : int, optional
        Figure DPI.
    logger : logging.Logger, optional
        Optional logger for status messages.

    Notes
    -----
    The resulting movie will show the top-ranked window for this observer
    (as stored in observer_entry) with per-pixel brightness linearly
    proportional to the sum of `intensity_key` for all targets and stars.
    """
    # Build the frame cube from the observer entry
    cube = build_frame_cube_for_observer(
        observer_entry=observer_entry,
        intensity_key=intensity_key,
        logger=logger,
    )

    # Compute global visualization range
    vmin, vmax = compute_linear_vmin_vmax(cube, clip_percentile=clip_percentile)

    # Construct a default title from observer metadata
    obs_name = observer_entry.get("observer_name", "<observer>")
    tracking_mode = observer_entry.get("tracking_mode", None)
    win_idx = observer_entry.get("window_index", None)

    if tracking_mode is not None and win_idx is not None:
        title = f"{obs_name} - window {win_idx} ({tracking_mode})"
    elif win_idx is not None:
        title = f"{obs_name} - window {win_idx}"
    else:
        title = obs_name

    # Write the movie using the frame cube
    write_movie_from_cube(
        frame_cube=cube,
        output_path=output_path,
        fps=fps,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        dpi=dpi,
        title=title,
        logger=logger,
    )

def make_observer_animation_with_target_markers(
    observer_entry: Dict[str, Any],
    output_path: Union[str, Path],
    fps: float = 1.0,
    intensity_key: str = "flux_ph_m2_frame",
    clip_percentile: float = 95.0,
    cmap: str = "gray",
    dpi: int = 150,
    marker_size: float = 40.0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Build an animation for one observer with red circles over targets.

    This is identical in spirit to make_observer_animation, but in
    addition to rendering the flux image (targets + stars), it overlays
    red, hollow circle markers at the per-frame target positions.

    Parameters
    ----------
    observer_entry : dict
        One entry from combined_top_windows, e.g.:
            combined_top_windows["SBSS (USA 216)"].
        Must contain a "frames" list, each with a "targets" dict
        whose entries have "x_pix", "y_pix".
    output_path : str or Path
        Output movie path (e.g. 'SBSS_top_window_markers.mp4').
    fps : float, optional
        Frames per second for the movie.
    intensity_key : str, optional
        Field used as per-source intensity (default: 'flux_ph_m2_frame').
    clip_percentile : float, optional
        Percentile used for global linear vmax (see compute_linear_vmin_vmax).
    cmap : str, optional
        Matplotlib colormap.
    dpi : int, optional
        Figure DPI.
    marker_size : float, optional
        Size of the target markers passed to ax.scatter(..., s=marker_size).
    logger : logging.Logger, optional
        Optional logger for status messages.
    """

    # Build the frame cube (targets + stars flux) as usual
    cube = build_frame_cube_for_observer(
        observer_entry=observer_entry,
        intensity_key=intensity_key,
        logger=logger,
    )

    # Get the original per-frame metadata (for target positions)
    frames = observer_entry.get("frames", [])
    n_frames = cube.shape[0]
    if len(frames) != n_frames:
        raise ValueError(
            f"Observer entry has {len(frames)} frame dicts but cube has "
            f"{n_frames} frames; cannot attach target markers."
        )

    # Compute global visualization range
    vmin, vmax = compute_linear_vmin_vmax(cube, clip_percentile=clip_percentile)

    # Construct a default title from observer metadata
    obs_name = observer_entry.get("observer_name", "<observer>")
    tracking_mode = observer_entry.get("tracking_mode", None)
    win_idx = observer_entry.get("window_index", None)

    if tracking_mode is not None and win_idx is not None:
        title = f"{obs_name} - window {win_idx} ({tracking_mode})"
    elif win_idx is not None:
        title = f"{obs_name} - window {win_idx}"
    else:
        title = obs_name

    # Normalize output path
    output_path = Path(output_path)

    if logger is not None:
        logger.info(
            "Writing movie with target markers to %s (fps=%.3f, cmap=%s).",
            output_path,
            fps,
            cmap,
        )

    # Create figure and axes
    fig, ax = plt.subplots()
    fig.set_dpi(dpi)

    # Base image for frame 0
    im = ax.imshow(
        cube[0],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        animated=True,
    )

    # Initial target marker artist (empty; we will update offsets per frame)
    target_scat = ax.scatter(
        [], [],
        facecolors="none",
        edgecolors="red",
        s=marker_size,
        linewidths=1.0,
        animated=True,
    )

    # Colorbar and labels
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Flux [ph m$^{-2}$ frame$^{-1}$]")

    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    ax.set_title(title)

    def _update(frame_index: int):
        """
        Update function for FuncAnimation: update image + target markers.
        """
        # Update image
        im.set_array(cube[frame_index])

        # Extract target positions for this frame
        frame = frames[frame_index]
        target_dict = frame.get("targets", {})

        xs = []
        ys = []
        for target in target_dict.values():
            x = float(target.get("x_pix", np.nan))
            y = float(target.get("y_pix", np.nan))
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xs.append(x)
            ys.append(y)

        if xs:
            # Set marker positions (N x 2 array)
            coords = np.column_stack([xs, ys])
            target_scat.set_offsets(coords)
        else:
            # No targets this frame: clear the offsets
            target_scat.set_offsets(np.empty((0, 2)))

        return (im, target_scat)

    # Build animation object
    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000.0 / max(fps, 1e-6),
        blit=True,
    )

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(str(output_path), writer=writer)

    plt.close(fig)

    if logger is not None:
        logger.info("Finished writing movie with target markers to %s.", output_path)


def make_observer_animations_from_combined(
    combined_top_windows: CombinedTopWindows,
    out_dir: Union[str, Path],
    fps: float = 1.0,
    intensity_key: str = "flux_ph_m2_frame",
    clip_percentile: float = 99.5,
    cmap: str = "gray",
    dpi: int = 150,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:

    """
    Build and write movies for ALL observers present in combined_top_windows.

    Parameters
    ----------
    combined_top_windows : dict
        Combined top window product keyed by observer name, as produced
        by sim_test via attach_star_photons_to_target_frames2.
    out_dir : str or Path
        Directory in which to place the output movies. One movie per
        observer, named '<observer_name_safe>.mp4'.
    fps : float, optional
        Frames per second for all movies.
    intensity_key : str, optional
        Per-source intensity field (default: 'flux_ph_m2_frame').
    clip_percentile : float, optional
        Percentile used for global vmin/vmax per observer.
    cmap : str, optional
        Matplotlib colormap.
    dpi : int, optional
        Figure DPI.
    logger : logging.Logger, optional
        Optional logger for progress messages.

    Returns
    -------
    outputs : dict
        Dict mapping observer_name -> output_movie_path (as a string).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}

    for obs_name, obs_entry in combined_top_windows.items():
        # Construct a filesystem-safe filename by replacing problematic chars
        safe_name = "".join(
            c if c.isalnum() or c in ("_", "-", ".") else "_" for c in obs_name
        )
        movie_path = out_dir / f"{safe_name}_top_window.mp4"

        if logger is not None:
            logger.info(
                "Creating animation for observer '%s' -> %s",
                obs_name,
                movie_path,
            )

        make_observer_animation(
            observer_entry=obs_entry,
            output_path=movie_path,
            fps=fps,
            intensity_key=intensity_key,
            clip_percentile=clip_percentile,
            cmap=cmap,
            dpi=dpi,
            logger=logger,
        )


        outputs[obs_name] = str(movie_path)

    return outputs


def make_observer_animations_from_pickle(
    pickle_path: Union[str, Path],
    out_dir: Union[str, Path],
    fps: float = 1.0,
    intensity_key: str = "flux_ph_m2_frame",
    clip_percentile: float = 99.5,
    cmap: str = "gray",
    dpi: int = 150,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:

    """
    Convenience wrapper: load combined_top_windows from pickle and animate.

    Parameters
    ----------
    pickle_path : str or Path
        Path to combined_top_windows.pkl produced by sim_test.
    out_dir : str or Path
        Directory in which to place the output movies.
    fps : float, optional
        Frames per second for all movies.
    intensity_key : str, optional
        Per-source intensity field (default: 'flux_ph_m2_frame').
    clip_percentile : float, optional
        Percentile used for vmin/vmax per observer.
    cmap : str, optional
        Matplotlib colormap.
    dpi : int, optional
        Figure DPI.
    logger : logging.Logger, optional
        Optional logger for status messages.

    Returns
    -------
    outputs : dict
        Dict mapping observer_name -> output_movie_path (as a string).
    """
    combined = load_combined_top_windows_pickle(
        pickle_path=pickle_path,
        logger=logger,
    )

    return make_observer_animations_from_combined(
        combined_top_windows=combined,
        out_dir=out_dir,
        fps=fps,
        intensity_key=intensity_key,
        clip_percentile=clip_percentile,
        cmap=cmap,
        dpi=dpi,
        logger=logger,
    )

