"""
NEBULA_PHOTUTILS_WORKFLOW.py

Glue between NEBULA frame builder and astropy/photutils:

- Build frames for all observers using
  NEBULA_FRAME_BUILDER.build_frames_by_observer_and_window().
- For a preferred observer (currently "SBSS (USA 216)" if present),
  select the pointing window with the largest number of unique targets
  (ties broken by number of frames).
- Convert each frame in that window into a CCDData image, rendering each
  source via an analytic Gaussian PSF with sub-pixel centering.
- Treat flux physically: if the PSF extends beyond the detector edges,
  the corresponding flux is *lost* (no artificial conservation on-chip).
- Run a basic photutils pipeline on each frame:
    * 2D background estimate (Background2D)
    * detection threshold (detect_threshold)
    * segmentation (detect_sources)
    * aperture photometry at the true NEBULA source positions.
"""

from __future__ import annotations  # Ensure forward-reference type hints work in all Python versions

import logging  # Standard logging library for diagnostics
from typing import Any, Dict, List  # Typing helpers for dicts and lists

import numpy as np  # Numerical array computations
import astropy.units as u  # Physical unit handling for CCDData
from astropy.nddata import CCDData  # Container for image + metadata
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma  # Sigma clipping & FWHM->sigma conversion

from astropy.convolution import convolve  # Convolution for PSF-matched filtering
from astropy.modeling.models import Gaussian2D  # Analytic 2D Gaussian PSF model

from photutils.background import Background2D, MedianBackground  # 2D background estimation
from photutils.segmentation import detect_threshold, detect_sources, make_2dgaussian_kernel  # Detection tools
from photutils.aperture import CircularAperture, aperture_photometry  # Aperture photometry tools

from Configuration.NEBULA_SENSOR_CONFIG import EVK4_SENSOR  # Sensor configuration (rows, cols, PSF FWHM, etc.)
from Utility.FRAMES import NEBULA_FRAME_BUILDER as FB  # NEBULA frame builder (geometry -> frames)

import matplotlib.pyplot as plt  # Quick-look plotting for debug

# Import tqdm for progress bars over frames
from tqdm.auto import tqdm

import warnings
from photutils.utils.exceptions import NoDetectionsWarning
warnings.filterwarnings("ignore", category=NoDetectionsWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _get_logger() -> logging.Logger:
    """
    Get the NEBULA frame builder logger for consistent logging behavior.

    Returns
    -------
    logger : logging.Logger
        The shared NEBULA logger instance used across the frame builder and
        this workflow module.
    """
    # Delegate logger creation/configuration to NEBULA_FRAME_BUILDER
    logger = FB._get_logger()
    # Return the logger so callers can emit messages
    return logger


# ---------------------------------------------------------------------------
# Step 1: Build frames and pick "best" window for the preferred observer
# ---------------------------------------------------------------------------

def build_frames_and_pick_best_window(
    max_frames_per_window: int | None = None,
    logger: logging.Logger | None = None,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Build frames for all observers, then select the "best" window for a
    single observer (currently SBSS, or first alphabetically if SBSS is
    not present).

    The "best" window is defined as the one with:
      - the largest number of unique targets (n_unique_targets), and
      - in case of ties, the largest number of frames (n_frames).

    Parameters
    ----------
    max_frames_per_window : int or None, optional
        If not None, cap the number of frames per window when building
        frames via NEBULA_FRAME_BUILDER. This is passed directly through
        to FB.build_frames_by_observer_and_window().
    logger : logging.Logger or None, optional
        Optional logger instance. If None, the shared NEBULA logger is
        obtained via _get_logger().

    Returns
    -------
    obs_name : str
        Name of the observer whose window was selected.
    obs_data : dict
        Observer data dictionary from NEBULA_FRAME_BUILDER containing
        global metadata (rows, cols, etc.) and the list of windows.
    best_window : dict
        The chosen window dictionary. It contains fields such as
        "window_index", "n_unique_targets", "n_frames", and "frames"
        (list of per-frame dicts).
    """
    # Use provided logger or reuse NEBULA's shared logger
    if logger is None:
        logger = _get_logger()

    # Log that we are about to build frames for all observers
    logger.info("Building frames for all observers via NEBULA_FRAME_BUILDER.")

    # Call into the NEBULA frame builder to produce frames grouped by observer and window
    frames_by_observer = FB.build_frames_by_observer_and_window(
        max_frames_per_window=max_frames_per_window,
        logger=logger,
    )

    # If the result is empty, raise an error because there is nothing to process
    if not frames_by_observer:
        raise RuntimeError("No observers found in frames_by_observer.")

    # Define the preferred observer name for this workflow (SBSS if available)
    preferred_name = "SBSS (USA 216)"

    # If the preferred observer is present, use it; otherwise, fall back to first sorted observer
    if preferred_name in frames_by_observer:
        obs_name = preferred_name
    else:
        # Get observer names sorted alphabetically for deterministic behavior
        obs_names = sorted(frames_by_observer.keys())
        # Choose the first one when SBSS is absent
        obs_name = obs_names[0]
        # Log a warning so the user knows SBSS wasn't available
        logger.warning(
            "Preferred SBSS observer not found; falling back to '%s'.",
            obs_name,
        )

    # Retrieve the data dict for the selected observer
    obs_data = frames_by_observer[obs_name]
    # Get the list of windows (each window is a dict with metadata + frame list)
    windows = obs_data.get("windows", [])

    # If there are no windows for this observer, raise an error
    if not windows:
        raise RuntimeError(f"Observer '{obs_name}' has no pointing windows.")

    # Use Python's max() with a key function that prefers more unique targets, then more frames
    best_window = max(
        windows,
        key=lambda w: (w.get("n_unique_targets", 0), w.get("n_frames", 0)),
    )

    # Log which window was selected and its basic stats
    logger.info(
        "Selected best window %d for observer '%s': %d unique targets, %d frames.",
        best_window.get("window_index", -1),
        obs_name,
        best_window.get("n_unique_targets", 0),
        best_window.get("n_frames", 0),
    )

    # Return observer name, observer data dict, and the chosen window dict
    return obs_name, obs_data, best_window


# ---------------------------------------------------------------------------
# Step 2: Convert NEBULA frame -> CCDData image with physical PSF clipping
# ---------------------------------------------------------------------------

def frame_to_ccd(
    obs_data: Dict[str, Any],
    frame: Dict[str, Any],
    background_e: float = 0.0,
) -> CCDData:
    """
    Convert a single NEBULA frame dictionary into an Astropy CCDData image.

    Each source in the frame is rendered as a 2D analytic Gaussian PSF
    (Gaussian2D) centered at its *true* floating-point pixel coordinates
    (x_pix, y_pix). The PSF:

      - Uses a FWHM (in pixels) read from EVK4_SENSOR.psf_fwhm_pix.
      - Is converted to a 1-sigma width via astropy.stats.gaussian_fwhm_to_sigma.
      - Is evaluated on a finite stamp grid (~4 sigma radius) around the
        source position.
      - Is physically clipped at the detector edges: if part of the PSF
        lies off-chip, the corresponding flux is effectively lost and is
        NOT renormalized onto the detector.

    The rendered image is in electrons, then shot noise (Poisson) is applied
    to simulate realistic sensor statistics.

    Parameters
    ----------
    obs_data : dict
        Observer metadata dictionary from NEBULA_FRAME_BUILDER for the
        relevant observer. Must contain:
          - "rows": number of detector rows.
          - "cols": number of detector columns.
          - Optionally "observer_name" and "sensor_name" for metadata.
    frame : dict
        Frame dictionary from the chosen window. It must contain:
          - "sources": list of per-source dicts, each with:
              * "x_pix": x coordinate in pixel space (float).
              * "y_pix": y coordinate in pixel space (float).
              * "flux_e_frame": total electrons from that source in this frame.
          - "t_exp_s": exposure time in seconds (optional, for metadata).
          - "t_utc": UTC timestamp (optional, for metadata).
    background_e : float, optional
        Uniform background level in electrons per pixel to add before
        source rendering. Default is 0.0.

    Returns
    -------
    ccd : astropy.nddata.CCDData
        CCDData object containing:
          - data: 2D ndarray of electron counts (after Poisson noise).
          - unit: u.electron
          - meta: OBSERVER, SENSOR, T_EXP, T_UTC fields when available.
    """
    # Get a logger instance for debug / info messages
    logger = _get_logger()

    # Extract number of rows from observer metadata and ensure it is an integer
    nrows = int(obs_data["rows"])
    # Extract number of columns from observer metadata and ensure it is an integer
    ncols = int(obs_data["cols"])

    # Create the initial image as a 2D array filled with the background level
    img = np.full((nrows, ncols), float(background_e), dtype=float)

    # Retrieve the PSF FWHM (in pixels) from the sensor configuration
    psf_fwhm_pix = getattr(EVK4_SENSOR, "psf_fwhm_pix", None)

    # Check that psf_fwhm_pix is present and strictly positive; if not, raise
    if psf_fwhm_pix is None or psf_fwhm_pix <= 0.0:
        raise ValueError(
            "EVK4_SENSOR.psf_fwhm_pix must be set to a positive value "
            "in NEBULA_SENSOR_CONFIG for PSF rendering."
        )

    # Convert the FWHM to a 1-sigma Gaussian width using Astropy's standard factor
    # (gaussian_fwhm_to_sigma is defined so sigma = FWHM * gaussian_fwhm_to_sigma)
    sigma = float(psf_fwhm_pix * gaussian_fwhm_to_sigma)

    # Define a stamp radius in pixels (~4 sigma captures > 99.9% of the Gaussian flux)
    stamp_radius_pix = int(np.ceil(4.0 * sigma))

    # Emit a debug log describing the PSF parameters for sanity checking
    logger.debug(
        "frame_to_ccd: using Gaussian PSF with FWHM=%.3f pix (sigma=%.3f pix, stamp_radius=%d pix).",
        psf_fwhm_pix,
        sigma,
        stamp_radius_pix,
    )

    # Loop over each source in this frame and deposit its PSF into the image
    for src in frame.get("sources", []):
        # Convert the floating-point x coordinate of the source pixel position
        x_true = float(src["x_pix"])
        # Convert the floating-point y coordinate of the source pixel position
        y_true = float(src["y_pix"])
        # Extract the total electrons from this source in this frame
        flux_e = float(src["flux_e_frame"])

        # If the source has zero or negative flux, skip rendering
        if flux_e <= 0.0:
            continue

        # Compute an integer "central" pixel index in x (used only to define the PSF grid)
        ix_center = int(round(x_true))
        # Compute an integer "central" pixel index in y (used only to define the PSF grid)
        iy_center = int(round(y_true))

        # Define the PSF-grid bounding box in x (may extend outside detector)
        x_min_psf = ix_center - stamp_radius_pix
        x_max_psf = ix_center + stamp_radius_pix + 1  # +1 because Python slices are half-open
        # Define the PSF-grid bounding box in y (may extend outside detector)
        y_min_psf = iy_center - stamp_radius_pix
        y_max_psf = iy_center + stamp_radius_pix + 1  # +1 because Python slices are half-open

        # Create a 2D grid of y, x coordinates in the PSF frame (not clipped to detector)
        y_psf, x_psf = np.mgrid[y_min_psf:y_max_psf, x_min_psf:x_max_psf]

        # Instantiate a Gaussian2D model with amplitude=1.0 and sub-pixel center at (x_true, y_true)
        g_model = Gaussian2D(
            amplitude=1.0,
            x_mean=x_true,
            y_mean=y_true,
            x_stddev=sigma,
            y_stddev=sigma,
        )

        # Evaluate the Gaussian at each PSF-grid coordinate to obtain an unscaled PSF stamp
        stamp_psf = g_model(x_psf, y_psf)

        # Compute the total PSF sum on this finite grid (approximating the continuous integral)
        total_psf = float(np.sum(stamp_psf))

        # If the PSF stamp has zero or negative total (should not happen), skip this source
        if total_psf <= 0.0:
            continue

        # Compute the overlapping x-range between the PSF grid and the detector [0, ncols)
        x_min_det = max(0, x_min_psf)
        x_max_det = min(ncols, x_max_psf)
        # Compute the overlapping y-range between the PSF grid and the detector [0, nrows)
        y_min_det = max(0, y_min_psf)
        y_max_det = min(nrows, y_max_psf)

        # If there is no overlap (source fully off-chip), skip this source
        if x_min_det >= x_max_det or y_min_det >= y_max_det:
            continue

        # Compute the slice indices that map detector coordinates into the PSF stamp
        x_slice = slice(x_min_det - x_min_psf, x_max_det - x_min_psf)
        y_slice = slice(y_min_det - y_min_psf, y_max_det - y_min_psf)

        # Extract the portion of the PSF stamp that actually lies on the detector
        stamp_on = stamp_psf[y_slice, x_slice]

        # Convert the PSF *shape* into actual electrons: fractional PSF * total source flux
        stamp_electrons = flux_e * (stamp_on / total_psf)

        # Add the electron stamp into the corresponding region of the main image
        img[y_min_det:y_max_det, x_min_det:x_max_det] += stamp_electrons

    # Ensure no pixel has negative electrons (can occur from numerical noise)
    img = np.maximum(img, 0.0)

    # Draw Poisson-distributed electron counts per pixel to simulate shot noise
    img = np.random.poisson(lam=img).astype(float)

    # Wrap the image into a CCDData object with electron units
    ccd = CCDData(img, unit=u.electron)

    # Attach the observer name to the CCD metadata if provided
    ccd.meta["OBSERVER"] = obs_data.get("observer_name", "")
    # Attach the sensor name to the CCD metadata if provided
    ccd.meta["SENSOR"] = obs_data.get("sensor_name", "")
    # Attach the exposure time (seconds) if present in the frame
    ccd.meta["T_EXP"] = frame.get("t_exp_s", np.nan)

    # Extract the UTC timestamp from the frame metadata if present
    t_utc = frame.get("t_utc", None)
    # If a timestamp exists, try to serialize it in ISO 8601 format
    if t_utc is not None:
        try:
            ccd.meta["T_UTC"] = t_utc.isoformat()
        except Exception:
            # If ISO formatting fails (e.g., non-datetime object), fall back to string conversion
            ccd.meta["T_UTC"] = str(t_utc)

    # Return the fully populated CCDData object
    return ccd


# ---------------------------------------------------------------------------
# Step 3: Run photutils pipeline on all frames in the chosen window
# ---------------------------------------------------------------------------

def run_photutils_pipeline_on_window(
    obs_name: str,
    obs_data: Dict[str, Any],
    window: Dict[str, Any],
    logger: logging.Logger | None = None,
    bkg_box_size: int = 64,
    bkg_filter_size: int = 3,
    nsigma_det: float = 3.0,
    npixels: int = 5,
    aperture_radius: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Run a basic photutils pipeline on each frame in a given pointing window.

    For each frame in the selected window, this function:

      1. Builds a CCDData image using frame_to_ccd() with an analytic PSF.
      2. Estimates a 2D background using photutils.background.Background2D
         with median statistics and sigma clipping. :contentReference[oaicite:1]{index=1}
      3. Computes a detection threshold via photutils.segmentation.detect_threshold
         using the background model and its RMS. :contentReference[oaicite:2]{index=2}
      4. Optionally performs PSF-matched filtering using a Gaussian kernel
         constructed by photutils.segmentation.make_2dgaussian_kernel
         with the same PSF FWHM as the sensor. :contentReference[oaicite:3]{index=3}
      5. Generates a segmentation map (labeled sources) via detect_sources.
      6. Performs circular aperture photometry at the *true* NEBULA source
         positions (x_pix, y_pix) in each frame.

    Parameters
    ----------
    obs_name : str
        Name of the observer whose window is being processed. Used only
        for logging and diagnostics.
    obs_data : dict
        Observer data dictionary from NEBULA_FRAME_BUILDER. Used primarily
        to extract detector dimensions and metadata for frame_to_ccd().
    window : dict
        Window dictionary selected by build_frames_and_pick_best_window().
        It must contain a "frames" list of per-frame dicts.
    logger : logging.Logger or None, optional
        Optional logger instance. If None, the shared NEBULA logger is
        obtained via _get_logger().
    bkg_box_size : int, optional
        Approximate box size (in pixels) for Background2D. The actual
        box size used is clamped to the image dimensions and a minimum
        value to avoid degenerate boxes.
    bkg_filter_size : int, optional
        Filter kernel size (in pixels) used to smooth the low-resolution
        background map produced by Background2D.
    nsigma_det : float, optional
        Detection threshold in units of sigma above the background level
        for detect_threshold().
    npixels : int, optional
        Minimum number of connected pixels above threshold required for
        a detection in detect_sources().
    aperture_radius : float, optional
        Radius (in pixels) of the circular apertures used for forced
        aperture photometry at the true NEBULA positions.

    Returns
    -------
    results : list of dict
        One dictionary per frame with the following keys:
          - "frame": original NEBULA frame dict.
          - "ccd": CCDData image constructed by frame_to_ccd().
          - "background": Background2D result object.
          - "segmentation": SegmentationImage or None (if no detections).
          - "aperture_phot": QTable from aperture_photometry or None if
            no sources were present in the frame.
    """
    # Use provided logger or retrieve the shared NEBULA logger
    if logger is None:
        logger = _get_logger()

    # Retrieve the list of frames contained in this pointing window
    frames = window.get("frames", [])

    # If there are no frames, log a warning and return an empty result list
    if not frames:
        logger.warning(
            "Observer '%s' window %s has no frames.",
            obs_name,
            window.get("window_index", -1),
        )
        return []

    # Retrieve PSF FWHM from the sensor config and enforce validity for detection as well
    psf_fwhm_pix = getattr(EVK4_SENSOR, "psf_fwhm_pix", None)

    # If psf_fwhm_pix is invalid, raise an error to avoid inconsistent kernels
    if psf_fwhm_pix is None or psf_fwhm_pix <= 0.0:
        raise ValueError(
            "EVK4_SENSOR.psf_fwhm_pix must be set to a positive value "
            "in NEBULA_SENSOR_CONFIG for photutils detection kernels."
        )

    # Prepare an empty list that will accumulate per-frame photutils results
    results: List[Dict[str, Any]] = []

    # Log the overall plan for this window (how many frames will be processed)
    logger.info(
        "Running photutils pipeline for observer '%s', window %s, %d frames.",
        obs_name,
        window.get("window_index", -1),
        len(frames),
    )

    # Loop over each frame and perform the photutils background/detection/photometry pipeline
    # Create a description string for the tqdm progress bar
    tqdm_desc = f"Photutils pipeline: {obs_name}, window {window.get('window_index', -1)}"

    # Wrap the frames list in a tqdm iterator to show a progress bar over frames
    for idx, frame in enumerate(
        tqdm(frames, total=len(frames), desc=tqdm_desc, unit="frame"), start=1
    ):
        # ------------------------------------------------------------------
        # Build CCDData image for this frame via frame_to_ccd()
        # ------------------------------------------------------------------

        # Convert the NEBULA frame (truth) into a CCD-like image with PSF rendering and shot noise
        ccd = frame_to_ccd(obs_data, frame, background_e=0.0)
        # Extract the underlying numpy array of electron counts from the CCDData
        data = ccd.data

        # Get the image dimensions (rows, cols) from the data array
        nrows, ncols = data.shape

        # ------------------------------------------------------------------
        # 2D background estimate (Background2D)
        # ------------------------------------------------------------------

        # Compute a reasonable box size in y: no larger than bkg_box_size, no smaller than 10,
        # and not more than one quarter of the image dimension (to retain multiple boxes).
        box_y = min(bkg_box_size, max(10, nrows // 4))
        # Compute a reasonable box size in x with the same logic as y
        box_x = min(bkg_box_size, max(10, ncols // 4))
        # Package the box sizes into a tuple for Background2D
        box_size = (box_y, box_x)

        # Construct a SigmaClip object for iterative sigma-clipped statistics in the background estimation
        sigma_clip = SigmaClip(sigma=3.0)
        # Use the median as the background estimator, robust to outliers
        bkg_estimator = MedianBackground()

        # Run Background2D to estimate the 2D background and its RMS across the image
        bkg = Background2D(
            data,
            box_size=box_size,
            filter_size=(bkg_filter_size, bkg_filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )

        # Subtract the estimated background from the data to get a background-subtracted image
        data_sub = data - bkg.background

        # ------------------------------------------------------------------
        # Detection threshold and segmentation map
        # ------------------------------------------------------------------

        # Compute the size of the Gaussian kernel in pixels: ~6 * FWHM, rounded up,
        # and forced to be odd-sized so the kernel is centered on a pixel.
        kernel_size = int(np.ceil(6.0 * psf_fwhm_pix))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Build a normalized 2D Gaussian kernel for PSF-matched filtering of the background-subtracted data
        det_kernel = make_2dgaussian_kernel(psf_fwhm_pix, size=kernel_size)

        # Convolve the image with the Gaussian kernel to enhance point-source detectability
        data_filt = convolve(data_sub, det_kernel)

        # Compute a pixel-wise detection threshold image using detect_threshold()
        threshold = detect_threshold(
            data_filt,
            nsigma=nsigma_det,
            background=bkg.background,
            error=bkg.background_rms,
            sigma_clip=sigma_clip,
        )

        # Use detect_sources() to generate a segmentation image of connected pixels above threshold
        segm = detect_sources(
            data_filt,
            threshold,
            npixels=npixels,
        )

        # Determine the number of labeled sources in the segmentation map (0 if segm is None)
        n_labels = int(segm.nlabels) if segm is not None else 0

        # ------------------------------------------------------------------
        # Aperture photometry at *true* NEBULA positions
        # ------------------------------------------------------------------

        # Retrieve the list of truth sources present in this frame
        sources = frame.get("sources", [])
        # Count how many sources NEBULA says are on detector in this frame
        n_sources = len(sources)

        # If no sources are present, log and store results with no aperture photometry
        if n_sources == 0:
            logger.debug(
                "Frame %d/%d (coarse_index=%s): no sources in this frame; "
                "skipping aperture photometry. Segmentation labels=%d.",
                idx + 1,
                len(frames),
                frame.get("coarse_index", -1),
                n_labels,
            )

            # Append a result dict with segmentation and background, but no photometry table
            results.append(
                {
                    "frame": frame,
                    "ccd": ccd,
                    "background": bkg,
                    "segmentation": segm,
                    "aperture_phot": None,
                }
            )

            # Move on to the next frame in the loop
            continue

        # Build a list of (x, y) positions for each source using its true pixel coordinates
        positions = [(float(s["x_pix"]), float(s["y_pix"])) for s in sources]
        # Define circular apertures centered at these positions with the requested radius
        apertures = CircularAperture(positions, r=aperture_radius)

        # Perform aperture photometry on the background-subtracted image at these apertures
        phot_table = aperture_photometry(data_sub, apertures)

        # Log a summary of the frame processing: how many truth sources and detected segments
        logger.debug(
            "Frame %d/%d (coarse_index=%s): %d true sources, %d segm labels.",
            idx + 1,
            len(frames),
            frame.get("coarse_index", -1),
            n_sources,
            n_labels,
        )

        # Append a result dict with the frame, CCDData, background, segmentation, and photometry table
        results.append(
            {
                "frame": frame,
                "ccd": ccd,
                "background": bkg,
                "segmentation": segm,
                "aperture_phot": phot_table,
            }
        )

    # Once all frames are processed, log that the pipeline has finished for this window
    logger.info(
        "Completed photutils pipeline for observer '%s', window %s.",
        obs_name,
        window.get("window_index", -1),
    )

    # Return the full list of per-frame photutils results
    return results


# ---------------------------------------------------------------------------
# Script-style entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Obtain a logger instance for this script-style execution
    logger = _get_logger()

    # ----------------------------------------------------------------------
    # 1. Build frames and pick the "best" window for the preferred observer
    # ----------------------------------------------------------------------

    # Call helper function to build frames for all observers and select a window
    obs_name, obs_data, best_window = build_frames_and_pick_best_window(
        max_frames_per_window=None,
        logger=logger,
    )

    # ----------------------------------------------------------------------
    # 2. Run the photutils pipeline on the best window
    # ----------------------------------------------------------------------

    # Execute the background/detection/photometry pipeline for the chosen window
    results = run_photutils_pipeline_on_window(
        obs_name,
        obs_data,
        best_window,
        logger=logger,
    )

    # Log a high-level completion message including number of frames processed
    logger.info(
        "NEBULA_PHOTUTILS_WORKFLOW: finished. %d frames processed for '%s' window %s.",
        len(results),
        obs_name,
        best_window.get("window_index", -1),
    )

    # ----------------------------------------------------------------------
    # 3. Debug / validation block: compare true vs measured fluxes
    #    across *all* frames with aperture photometry
    # ----------------------------------------------------------------------

    # Lists to accumulate fluxes from all frames
    all_true_flux = []
    all_meas_flux = []

    # Optional: count how many frames actually contributed
    n_frames_with_phot = 0

    # Separate accumulators for central vs edge sources
    central_true = []
    central_meas = []
    edge_true = []
    edge_meas = []

    # Detector geometry
    nrows = int(obs_data["rows"])
    ncols = int(obs_data["cols"])

    # PSF sigma and an edge threshold in pixels (~4 sigma)
    psf_fwhm_pix = getattr(EVK4_SENSOR, "psf_fwhm_pix", None)
    if psf_fwhm_pix is None or psf_fwhm_pix <= 0.0:
        raise ValueError(
            "EVK4_SENSOR.psf_fwhm_pix must be set to a positive value "
            "for edge/central classification."
        )

    sigma = float(psf_fwhm_pix * gaussian_fwhm_to_sigma)
    edge_thresh_pix = 4.0 * sigma  # "edge" = within ~4σ of a detector border

    for res in results:
        phot = res["aperture_phot"]
        if phot is None:
            # This frame had no sources on detector
            continue

        frame = res["frame"]
        sources = frame.get("sources", [])

        if not sources:
            # Shouldn't happen if phot is not None, but be safe
            continue

        # True electrons per source in this frame
        true_flux_frame = np.array([s["flux_e_frame"] for s in sources], dtype=float)

        # Measured electrons from photutils (aperture-summed, background-subtracted)
        meas_flux_frame = np.array(phot["aperture_sum"], dtype=float)

        # Sanity: enforce length match
        if true_flux_frame.shape != meas_flux_frame.shape:
            logger.warning(
                "Flux length mismatch in frame with coarse_index=%s: true=%d, meas=%d",
                frame.get("coarse_index", -1),
                true_flux_frame.size,
                meas_flux_frame.size,
            )
            continue

        # Global accumulators
        all_true_flux.append(true_flux_frame)
        all_meas_flux.append(meas_flux_frame)
        n_frames_with_phot += 1

        # ------------------------------------------------------------------
        # Central vs edge classification for this frame
        # ------------------------------------------------------------------
        xs = np.array([s["x_pix"] for s in sources], dtype=float)
        ys = np.array([s["y_pix"] for s in sources], dtype=float)

        # Distance to nearest detector edge in pixels
        d_edge = np.minimum.reduce(
            [
                xs,                # distance to left edge (x = 0)
                ncols - 1 - xs,    # distance to right edge (x = ncols-1)
                ys,                # distance to bottom edge (y = 0)
                nrows - 1 - ys,    # distance to top edge (y = nrows-1)
            ]
        )

        is_edge = d_edge < edge_thresh_pix

        # Append to central / edge accumulators
        if np.any(~is_edge):
            central_true.append(true_flux_frame[~is_edge])
            central_meas.append(meas_flux_frame[~is_edge])

        if np.any(is_edge):
            edge_true.append(true_flux_frame[is_edge])
            edge_meas.append(meas_flux_frame[is_edge])


    if not all_true_flux:
        raise RuntimeError("No frames with valid aperture_phot and sources were found.")

    # Concatenate all per-frame arrays into global 1D arrays
    all_true_flux = np.concatenate(all_true_flux)
    all_meas_flux = np.concatenate(all_meas_flux)

    # Compute global ratios and deltas
    ratio = all_meas_flux / all_true_flux
    delta = all_meas_flux - all_true_flux

    # ----------------------------------------------------------------------
    # Central vs edge stats
    # ----------------------------------------------------------------------
    if central_true:
        central_true_all = np.concatenate(central_true)
        central_meas_all = np.concatenate(central_meas)
        central_ratio = central_meas_all / central_true_all
        central_delta = central_meas_all - central_true_all

        print("\nCentral sources (far from edges):")
        print(f"  count = {central_true_all.size}")
        print("  flux ratio (meas / true):")
        print("    min  =", np.min(central_ratio))
        print("    max  =", np.max(central_ratio))
        print("    mean =", np.mean(central_ratio))
        print("    std  =", np.std(central_ratio))

        print("  flux delta (meas - true) [electrons]:")
        print("    min  =", np.min(central_delta))
        print("    max  =", np.max(central_delta))
        print("    mean =", np.mean(central_delta))
        print("    std  =", np.std(central_delta))
    else:
        print("\nNo central sources found (edge_thresh_pix may be too small).")

    if edge_true:
        edge_true_all = np.concatenate(edge_true)
        edge_meas_all = np.concatenate(edge_meas)
        edge_ratio = edge_meas_all / edge_true_all
        edge_delta = edge_meas_all - edge_true_all

        print("\nEdge sources (within ~4σ of a border):")
        print(f"  count = {edge_true_all.size}")
        print("  flux ratio (meas / true):")
        print("    min  =", np.min(edge_ratio))
        print("    max  =", np.max(edge_ratio))
        print("    mean =", np.mean(edge_ratio))
        print("    std  =", np.std(edge_ratio))

        print("  flux delta (meas - true) [electrons]:")
        print("    min  =", np.min(edge_delta))
        print("    max  =", np.max(edge_delta))
        print("    mean =", np.mean(edge_delta))
        print("    std  =", np.std(edge_delta))
    else:
        print("\nNo edge sources found (field may be small or all sources central).")

    print(f"Total frames with photometry: {n_frames_with_phot}")
    print(f"Total sources across those frames: {all_true_flux.size}")

    print("Global flux ratio stats (meas / true):")
    print("  min  =", np.min(ratio))
    print("  max  =", np.max(ratio))
    print("  mean =", np.mean(ratio))
    print("  std  =", np.std(ratio))

    print("Global flux delta stats (meas - true) [electrons]:")
    print("  min  =", np.min(delta))
    print("  max  =", np.max(delta))
    print("  mean =", np.mean(delta))
    print("  std  =", np.std(delta))

    # ----------------------------------------------------------------------
    # 4. Quick-look plot: frame with the most sources
    # ----------------------------------------------------------------------
    max_sources = 0
    k_example = None

    # Find the frame with the largest number of truth sources that
    # also has an aperture_phot table (i.e., photometry was done)
    for k, res in enumerate(results):
        phot = res.get("aperture_phot", None)
        frame = res.get("frame", {})
        sources = frame.get("sources", [])
        n_sources = len(sources)

        if phot is not None and n_sources > max_sources:
            max_sources = n_sources
            k_example = k

    if k_example is not None and max_sources > 0:
        res = results[k_example]
        frame = res["frame"]
        ccd = res["ccd"]
        data = ccd.data

        # Plot the image
        plt.figure()
        plt.imshow(data, origin="lower")
        plt.colorbar(label="electrons")

        # Overlay all source positions in that frame
        xs = [float(s["x_pix"]) for s in frame["sources"]]
        ys = [float(s["y_pix"]) for s in frame["sources"]]
        plt.scatter(xs, ys, s=50, edgecolor="cyan", facecolor="none")

        plt.title(
            f"Frame {k_example} with {max_sources} sources "
            f"(window {best_window.get('window_index', -1)})"
        )
        plt.show()
    else:
        print("No frame with sources + photometry found to plot.")
