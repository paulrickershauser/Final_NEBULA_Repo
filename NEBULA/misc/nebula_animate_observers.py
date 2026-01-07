# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 13:56:03 2025

@author: prick
"""

# nebula_animate_observers.py
# ---------------------------------------------------------------------------
# Load NEBULA pixel pickles, animate each observer's FOV, and report how many
# targets are ever visible (on_detector_visible_sunlit) per observer.
# ---------------------------------------------------------------------------

"""
nebula_animate_observers
========================

This script:

    1. Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs() to ensure
       pixel-augmented pickles exist and loads:

           obs_tracks, tar_tracks

       into the workspace.

    2. For each observer in obs_tracks, builds a simple animation that
       shows all targets moving through that observer's field of view,
       using only the:

           by_observer[obs_name]["on_detector_visible_sunlit"]

       mask as the visibility gate.

       At each animation frame (timestep), all targets that satisfy:

           on_detector_visible_sunlit[t] == True

       for that observer are plotted at their corresponding:

           pix_x[t], pix_y[t]

       on the sensor pixel grid.

    3. At the end, prints a summary of how many targets are ever visible
       (on_detector_visible_sunlit == True at any timestep) for each
       observer, so you know which animations are worth watching.

Notes
-----

- This script **does not** recompute pixel pickles by default; it reuses
  existing ones unless you change FORCE_RECOMPUTE to True.

- Currently, all targets are considered; if you want to restrict to GEO
  only, you can add a simple filter in count_visible_targets_per_observer()
  and animate_targets_for_observer_from_tracks() based on whatever GEO
  metadata you have stored in tar_tracks.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
from pathlib import Path
from Configuration.NEBULA_PATH_CONFIG import (
    NEBULA_OUTPUT_DIR,
    RUN_SUBDIR_ANIMATION,
)

# Import typing helpers for type hints.
from typing import Dict, Any, Tuple

# Import logging to report progress and diagnostics.
import logging
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# Import numpy for array operations.
import numpy as np

# Import matplotlib for plotting.
import matplotlib.pyplot as plt

# Import the animation module from matplotlib.
from matplotlib import animation

# Keep global references to animations so they are not garbage-collected
# when created inside helper functions in an interactive environment.
GLOBAL_ANIMATIONS = []
# ---------------------------------------------------------------------------
# NEBULA imports
# ---------------------------------------------------------------------------

# Import sensor configuration to get detector dimensions.
from Configuration.NEBULA_SENSOR_CONFIG import EVK4_SENSOR

# Import the pixel pickler that provides obs_tracks and tar_tracks.
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER


# ---------------------------------------------------------------------------
# Configuration flag
# ---------------------------------------------------------------------------

# Flag controlling whether to force recomputation of upstream and pixel
# products. If False, existing pixel pickles will be reused when present.
FORCE_RECOMPUTE: bool = False


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def build_logger() -> logging.Logger:
    """
    Build and return a simple console logger for this script.

    Returns
    -------
    logger : logging.Logger
        Logger configured with INFO level and a basic stream handler.
    """
    # Get or create a logger using this module's name.
    logger = logging.getLogger("NEBULA_OBSERVER_FOV_ANIM")

    # If the logger has no handlers yet, configure a stream handler.
    if not logger.handlers:
        # Create a stream handler that writes to stderr.
        handler = logging.StreamHandler()
        # Create a simple format with time, name, level, and message.
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Attach the formatter to the handler.
        handler.setFormatter(fmt)
        # Add the handler to the logger.
        logger.addHandler(handler)
        # Set the logging level to INFO.
        logger.setLevel(logging.INFO)

    # Return the configured logger.
    return logger


# ---------------------------------------------------------------------------
# Visibility counting helper
# ---------------------------------------------------------------------------

def count_visible_targets_per_observer(
    obs_tracks: Dict[str, Any],
    tar_tracks: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Tuple[int, list]]:
    """
    Count how many targets are ever visible for each observer, using only
    the on_detector_visible_sunlit mask.

    For each observer:

        * Loop over all targets.
        * If target["by_observer"][obs_name]["on_detector_visible_sunlit"]
          exists and is True for at least one timestep, count that target
          as "visible" for that observer.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary mapping observer names to observer track dictionaries.
    tar_tracks : dict
        Dictionary mapping target names to target track dictionaries.
    logger : logging.Logger
        Logger for reporting summary information.

    Returns
    -------
    visibility_summary : dict
        Dictionary mapping observer name to (count, target_list), where:

            count : int
                Number of targets ever visible for that observer.
            target_list : list of str
                Names of targets that are ever visible.
    """
    # Create a dictionary to hold the results per observer.
    visibility_summary: Dict[str, Tuple[int, list]] = {}

    # Loop over each observer in the observer tracks.
    for obs_name in obs_tracks.keys():
        # Initialize a counter for how many targets are ever visible.
        num_visible = 0
        # Initialize a list to store the names of visible targets.
        visible_targets: list = []

        # Loop over all targets in the target tracks.
        for tar_name, tar_track in tar_tracks.items():
            # Get the by_observer dict for this target, if present.
            by_observer = tar_track.get("by_observer", None)
            # If by_observer is missing or does not include this observer, skip.
            if not by_observer or obs_name not in by_observer:
                continue

            # Get the per-observer fields for this observer–target pair.
            by_obs = by_observer[obs_name]

            # Get the on_detector_visible_sunlit mask; if missing, treat as not visible.
            mask = by_obs.get("on_detector_visible_sunlit", None)
            if mask is None:
                continue

            # Convert the mask to a boolean numpy array.
            mask_bool = np.asarray(mask, dtype=bool)

            # If the mask has any True values, the target is ever visible.
            if mask_bool.any():
                num_visible += 1
                visible_targets.append(tar_name)

        # Store the results for this observer in the summary dict.
        visibility_summary[obs_name] = (num_visible, visible_targets)

        # Log a one-line summary for this observer.
        logger.info(
            "Observer '%s': %d targets ever visible (on_detector_visible_sunlit).",
            obs_name,
            num_visible,
        )

    # After processing all observers, return the summary dict.
    return visibility_summary


# ---------------------------------------------------------------------------
# Animation helper (from already-loaded tracks)
# ---------------------------------------------------------------------------

def animate_targets_for_observer_from_tracks(
    obs_tracks: Dict[str, Any],
    tar_tracks: Dict[str, Any],
    observer_name: str,
    logger: logging.Logger,
    frame_stride: int = 1,
) -> None:
    """
    Animate targets moving through a single observer's field of view, using
    already-loaded obs_tracks and tar_tracks.

    This function assumes obs_tracks and tar_tracks have already been
    produced by NEBULA_PIXEL_PICKLER and that each target's per-observer
    dictionary contains:

        by_observer[obs_name]["pix_x"]
        by_observer[obs_name]["pix_y"]
        by_observer[obs_name]["on_detector_visible_sunlit"]

    For the specified observer:

        1) Infer the number of timesteps from one suitable target.
        2) For each timestep (optionally subsampled by frame_stride),
           gather all targets with on_detector_visible_sunlit[t] == True.
        3) Plot those visible targets as points on the sensor grid and
           animate their motion over time.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary mapping observer names to observer track dictionaries.
    tar_tracks : dict
        Dictionary mapping target names to target track dictionaries.
    observer_name : str
        Name/key of the observer to visualize.
    logger : logging.Logger
        Logger for diagnostic messages.
    frame_stride : int, optional
        Factor by which to subsample timesteps. frame_stride=1 uses every
        timestep; frame_stride=10 uses every 10th timestep, etc.

    Returns
    -------
    None
        Displays the animation with plt.show().
    """
    # If the observer name is not in obs_tracks, log and bail out.
    if observer_name not in obs_tracks:
        logger.warning(
            "animate_targets_for_observer_from_tracks: Observer '%s' not found.",
            observer_name,
        )
        return

    # If there are no targets, there is nothing to animate.
    if not tar_tracks:
        logger.warning(
            "animate_targets_for_observer_from_tracks: tar_tracks is empty; "
            "nothing to animate."
        )
        return

    # ------------------------------------------------------------------
    # Infer the number of timesteps (n_times) from one suitable target
    # ------------------------------------------------------------------
    n_times = None

    # Loop over targets until we find one with pix_x for this observer.
    for tar_name, tar_track in tar_tracks.items():
        # Get the by_observer dict for this target.
        by_observer = tar_track.get("by_observer", None)
        # Skip if this target has no per-observer data for this observer.
        if not by_observer or observer_name not in by_observer:
            continue

        # Extract the per-observer dict.
        by_obs = by_observer[observer_name]
        # If pix_x is missing, skip this target.
        if "pix_x" not in by_obs:
            continue

        # Use the length of pix_x to define n_times.
        n_times = len(np.asarray(by_obs["pix_x"]))
        logger.info(
            "animate_targets_for_observer_from_tracks: Using target '%s' "
            "to infer n_times=%d for observer '%s'.",
            tar_name,
            n_times,
            observer_name,
        )
        # Break after finding the first suitable target.
        break

    # If we never found a suitable target, we cannot animate anything.
    if n_times is None:
        logger.warning(
            "animate_targets_for_observer_from_tracks: Could not infer n_times "
            "for observer '%s'; no target has pix_x for this observer.",
            observer_name,
        )
        return

    # ------------------------------------------------------------------
    # Get sensor dimensions from EVK4_SENSOR
    # ------------------------------------------------------------------
    rows = EVK4_SENSOR.rows
    cols = EVK4_SENSOR.cols

    logger.info(
        "animate_targets_for_observer_from_tracks: Observer '%s' sensor size = "
        "(%d rows, %d cols).",
        observer_name,
        rows,
        cols,
    )

    # ------------------------------------------------------------------
    # Build list of frame indices to animate (subsample by frame_stride)
    # ------------------------------------------------------------------
    all_indices = np.arange(n_times, dtype=int)
    frame_indices = all_indices[::frame_stride]
    n_frames = frame_indices.size

    logger.info(
        "animate_targets_for_observer_from_tracks: Animating %d frames (stride=%d).",
        n_frames,
        frame_stride,
    )

    # ------------------------------------------------------------------
    # Set up the matplotlib figure and initial scatter plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create an initially empty scatter; we'll update its offsets each frame.
    scatter = ax.scatter([], [], s=20)

    # Set axis limits to match sensor pixel coordinates.
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # invert y so (0,0) is top-left

    # Label axes and set title.
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title(f"Observer '{observer_name}' FOV (frame 0)")

    # Ensure pixels are square on the plot.
    ax.set_aspect("equal", adjustable="box")

    # Turn on a light grid for reference.
    ax.grid(True, linestyle="--", alpha=0.3)

    # ------------------------------------------------------------------
    # Define update function for a single simulation timestep
    # ------------------------------------------------------------------
    def update(sim_idx: int):
        """
        Update the scatter plot for a single simulation timestep.

        Parameters
        ----------
        sim_idx : int
            Simulation timestep index.

        Returns
        -------
        scatter : matplotlib.collections.PathCollection
            Updated scatter object.
        """
        xs = []
        ys = []

        # Loop over all targets to see which are visible for this observer.
        for tar_name, tar_track in tar_tracks.items():
            by_observer = tar_track.get("by_observer", None)
            if not by_observer or observer_name not in by_observer:
                continue

            by_obs = by_observer[observer_name]

            mask = by_obs.get("on_detector_visible_sunlit", None)
            if mask is None:
                continue

            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.shape[0] != n_times:
                continue

            if not mask_bool[sim_idx]:
                continue

            pix_x = np.asarray(by_obs["pix_x"], dtype=float)
            pix_y = np.asarray(by_obs["pix_y"], dtype=float)

            if pix_x.shape[0] != n_times or pix_y.shape[0] != n_times:
                continue

            x = pix_x[sim_idx]
            y = pix_y[sim_idx]

            xs.append(x)
            ys.append(y)

        if xs:
            offsets = np.column_stack([xs, ys])
        else:
            offsets = np.empty((0, 2))

        scatter.set_offsets(offsets)
        ax.set_title(
            f"Observer '{observer_name}' FOV (frame {sim_idx}/{n_times - 1})"
        )

        return scatter,

    # ------------------------------------------------------------------
    # Build output path under NEBULA_OUTPUT/animation
    # ------------------------------------------------------------------
    # Ensure the base NEBULA output directory exists.
    base_out = NEBULA_OUTPUT_DIR
    base_out.mkdir(parents=True, exist_ok=True)

    # Create the animation subdirectory.
    anim_dir = base_out / RUN_SUBDIR_ANIMATION
    anim_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize observer name for use in a filename.
    safe_obs_name = (
        observer_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    out_path = anim_dir / f"{safe_obs_name}_fov.mp4"

    logger.info(
        "Saving FOV animation for observer '%s' to '%s'.",
        observer_name,
        out_path,
    )

    # ------------------------------------------------------------------
    # Save animation as MP4 with tqdm over frames
    # ------------------------------------------------------------------
    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=20, metadata={"artist": "NEBULA"}, bitrate=1800)
    except Exception as exc:
        logger.error(
            "Could not create ffmpeg writer for observer '%s': %s",
            observer_name,
            exc,
        )
        plt.close(fig)
        return

    try:
        # Use the low-level saving API so we can hook in tqdm.
        with writer.saving(fig, str(out_path), dpi=100):
            for sim_idx in tqdm(
                frame_indices,
                desc=f"{observer_name} frames",
                leave=True,
            ):
                # Ensure int index
                sim_idx = int(sim_idx)
                # Update the figure for this timestep.
                update(sim_idx)
                # Grab the current frame.
                writer.grab_frame()

        logger.info("Successfully wrote animation to '%s'.", out_path)
    except Exception as exc:
        logger.error(
            "Failed to save MP4 animation for observer '%s' at '%s': %s",
            observer_name,
            out_path,
            exc,
        )
    finally:
        # Close the figure to free memory, especially when looping over observers.
        plt.close(fig)





# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a logger for this script.
    logger = build_logger()

    # Log that we are starting the pixel-loading step.
    logger.info(
        "Starting NEBULA pixel pipeline via NEBULA_PIXEL_PICKLER "
        "(force_recompute=%s).",
        FORCE_RECOMPUTE,
    )

    # Call NEBULA_PIXEL_PICKLER to load or compute obs_tracks and tar_tracks.
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=FORCE_RECOMPUTE,
        sensor_config=EVK4_SENSOR,
        logger=logger,
    )

    # At this point, obs_tracks and tar_tracks are in the workspace.

    # Count how many targets are ever visible per observer.
    visibility_summary = count_visible_targets_per_observer(
        obs_tracks=obs_tracks,
        tar_tracks=tar_tracks,
        logger=logger,
    )

    # Print a clean summary to the console.
    print("\n=== Visibility summary (on_detector_visible_sunlit) ===")
    for obs_name, (count, targets) in visibility_summary.items():
        print(f"Observer '{obs_name}': {count} targets ever visible.")
        if count > 0:
            # You can comment this out if the list is too long.
            print("  Targets:", ", ".join(sorted(targets)))
    print("======================================================\n")

    # Animate each observer that has at least one visible target.
    for obs_name, (count, _) in visibility_summary.items():
        if count == 0:
            logger.info(
                "Skipping animation for observer '%s' (no visible targets).",
                obs_name,
            )
            continue

        logger.info(
            "Animating observer '%s' with %d visible targets.",
            obs_name,
            count,
        )

        # Use a modest frame_stride to keep things responsive if n_times is large.
        animate_targets_for_observer_from_tracks(
            obs_tracks=obs_tracks,
            tar_tracks=tar_tracks,
            observer_name=obs_name,
            logger=logger,
            frame_stride=1,
        )
