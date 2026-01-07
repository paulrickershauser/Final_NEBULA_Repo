"""
NEBULA_SIMPLE_FRAME_VIZ.py

Make a simple MP4 movie of all frames in the "best" pointing window
for SBSS (USA 216), using the PSF-aware frame_to_ccd() from
NEBULA_PHOTUTILS_WORKFLOW.

Usage (from NEBULA repo root in Spyder):

    %runfile 'Utility/FRAMES/NEBULA_SIMPLE_FRAME_VIZ.py' --wdir

The movie is saved under:

    NEBULA_OUTPUT/FRAME_MOVIES/<observer>_window<idx>_frames.mp4
"""

from __future__ import annotations

import os
import sys

# Ensure NEBULA root is on sys.path
THIS_DIR = os.path.dirname(__file__)
NEBULA_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if NEBULA_ROOT not in sys.path:
    sys.path.insert(0, NEBULA_ROOT)

from Utility.FRAMES import NEBULA_PHOTUTILS_WORKFLOW as PWF

import logging
from typing import Any, Dict, List

import numpy as np

# Use non-interactive backend so this works in batch / Spyder
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_logger() -> logging.Logger:
    """Reuse the photutils-workflow logger for consistency."""
    return PWF._get_logger()


def _compute_global_intensity_scale(
    obs_data: Dict[str, Any],
    frames: List[Dict[str, Any]],
    *,
    n_samples: int = 100,
    background_e: float = 0.0,
    logger: logging.Logger | None = None,
) -> tuple[float, float]:
    """
    Estimate a global (vmin, vmax) intensity scale over the window by
    sampling up to `n_samples` frames and taking the 99th percentile
    of the non-zero pixels.

    This avoids frame-to-frame flicker in the movie.
    """
    if logger is None:
        logger = _get_logger()

    if not frames:
        raise ValueError("No frames provided to _compute_global_intensity_scale.")

    n_frames = len(frames)

    # Choose up to n_samples roughly evenly spaced indices
    if n_frames <= n_samples:
        sample_indices = range(n_frames)
    else:
        sample_indices = np.linspace(0, n_frames - 1, num=n_samples, dtype=int)

    vmax_samples: List[float] = []

    logger.info(
        "Computing global intensity scale from %d sampled frames (of %d total).",
        len(list(sample_indices)),
        n_frames,
    )

    for idx in sample_indices:
        frame = frames[idx]
        ccd = PWF.frame_to_ccd(obs_data, frame, background_e=background_e)
        data = np.asarray(ccd.data, dtype=float)

        # Ignore zeros for percentile so pure-background frames don't dominate
        nonzero = data[data > 0.0]
        if nonzero.size == 0:
            continue

        vmax_samples.append(np.percentile(nonzero, 99.0))

    if not vmax_samples:
        logger.warning(
            "All sampled frames were empty or zero-valued; using vmax = 1.0."
        )
        vmin, vmax = 0.0, 1.0
    else:
        vmax = max(vmax_samples)
        vmin = 0.0

    logger.info("Global intensity scale: vmin=%.3g, vmax=%.3g", vmin, vmax)
    return vmin, vmax


# ---------------------------------------------------------------------------
# Main visualization entry point
# ---------------------------------------------------------------------------

def make_frame_movie(
    *,
    background_e: float = 0.0,
    fps: int = 10,
    sample_for_scale: int = 100,
    frame_stride: int = 1,
    logger: logging.Logger | None = None,
) -> str:
    """
    Build an MP4 movie of all frames in the "best" window for SBSS.

    Parameters
    ----------
    background_e : float, optional
        Constant background level in electrons per pixel passed to frame_to_ccd.
    fps : int, optional
        Frames per second for the output movie.
    sample_for_scale : int, optional
        Number of frames to sample when estimating the global intensity
        scale (vmin, vmax). Higher is more accurate but slower.
    frame_stride : int, optional
        Use every `frame_stride`-th frame when building the movie. For
        example, 2 will use every other frame. 1 uses all frames.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    output_path : str
        Full path to the saved MP4 file.
    """
    if logger is None:
        logger = _get_logger()

    # 1) Build frames and pick the best window for SBSS (via PWF helper)
    obs_name, obs_data, best_window = PWF.build_frames_and_pick_best_window(
        max_frames_per_window=None,
        logger=logger,
    )

    frames_all: List[Dict[str, Any]] = best_window.get("frames", [])
    if not frames_all:
        raise RuntimeError(
            f"Observer '{obs_name}' best window has no frames; nothing to visualize."
        )

    # Apply frame_stride if > 1
    if frame_stride > 1:
        frames = frames_all[::frame_stride]
        logger.info(
            "Using frame_stride=%d: %d of %d frames will be rendered.",
            frame_stride,
            len(frames),
            len(frames_all),
        )
    else:
        frames = frames_all

    n_frames = len(frames)
    nrows = int(obs_data["rows"])
    ncols = int(obs_data["cols"])

    # 2) Compute global intensity scale
    vmin, vmax = _compute_global_intensity_scale(
        obs_data,
        frames,
        n_samples=sample_for_scale,
        background_e=background_e,
        logger=logger,
    )

    # 3) Set up Matplotlib figure
    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        constrained_layout=True,
    )
    # Initial blank image
    img0 = np.zeros((nrows, ncols), dtype=float)
    im = ax.imshow(
        img0,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Electrons per pixel")

    window_idx = best_window.get("window_index", -1)
    title = ax.set_title(f"{obs_name} window {window_idx} (initializing...)")
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")

    # 4) Animation functions
    def init():
        im.set_data(img0)
        title.set_text(f"{obs_name} window {window_idx} (init)")
        return im, title

    def update(i: int):
        frame = frames[i]
        ccd = PWF.frame_to_ccd(obs_data, frame, background_e=background_e)
        data = np.asarray(ccd.data, dtype=float)

        im.set_data(data)

        t_utc = frame.get("t_utc", None)
        if t_utc is not None:
            t_str = t_utc.isoformat()
        else:
            t_str = "unknown time"

        title.set_text(
            f"{obs_name} window {window_idx} | "
            f"frame {i + 1}/{n_frames} | {t_str}"
        )
        return im, title

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=False,
        repeat=False,
    )

    # 5) Prepare output path
    output_dir = os.path.join(NEBULA_ROOT, "NEBULA_OUTPUT", "FRAME_MOVIES")
    os.makedirs(output_dir, exist_ok=True)

    safe_obs_name = (
        obs_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    output_path = os.path.join(
        output_dir,
        f"{safe_obs_name}_window{window_idx}_frames.mp4",
    )

    # 6) Save as MP4 (requires ffmpeg)
    logger.info(
        "Saving frame movie to '%s' at %d fps (this may take a while)...",
        output_path,
        fps,
    )

    writer = FFMpegWriter(fps=fps, bitrate=2000)

    try:
        anim.save(output_path, writer=writer)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to save MP4 movie; ensure ffmpeg is installed and on PATH. "
            "Error: %s",
            exc,
        )
        raise

    plt.close(fig)

    logger.info("Frame movie saved to '%s'.", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger = _get_logger()
    try:
        path = make_frame_movie(
            background_e=0.0,  # tweak if you add a sky/dark background
            fps=10,
            sample_for_scale=100,
            frame_stride=1,    # increase to >1 to thin frames for speed
            logger=logger,
        )
        logger.info("NEBULA_SIMPLE_FRAME_VIZ: done. Movie at: %s", path)
    except Exception as e:  # noqa: BLE001
        logger.exception("NEBULA_SIMPLE_FRAME_VIZ: failed: %s", e)
