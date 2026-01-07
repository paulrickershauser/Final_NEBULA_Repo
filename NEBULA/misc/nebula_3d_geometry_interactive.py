# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 11:13:36 2025

@author: prick
"""

"""
nebula_3d_geometry_interactive.py

Interactive 3D geometry viewer for NEBULA.

This script:
    1) Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(force_recompute=False)
       to get:
           - observer tracks with positions + pointing (RA/Dec),
           - target tracks with positions + per-observer visibility flags,
           - per-target, per-observer:
                 by_observer[obs_name]["on_detector_visible_sunlit"].
    2) Computes Sun positions over the NEBULA time grid using the same
       DE440s ephemeris as NEBULA_SKYFIELD_ILLUMINATION.
    3) Builds an interactive 3D figure showing:
           - Earth as a sphere at the origin,
           - observer orbits + current positions,
           - the Sun direction,
           - each observer's FOV frustum,
           - all targets as 3D points, colored:
               * green if ANY observer has on_detector_visible_sunlit=True
                 at that timestep,
               * red otherwise.
    4) Adds:
           - a time slider to scrub through frames,
           - a FuncAnimation that auto-plays frames in a loop.

Run from Spyder:

    %matplotlib qt
    %runfile C:/Users/prick/Desktop/Research/NEBULA/nebula_3d_geometry_interactive.py --wdir

You can:
    - left-drag in the 3D window to rotate the camera,
    - scroll to zoom,
    - use the slider to jump to a specific timestep.

This is for qualitative, interactive debugging of NEBULA geometry.
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed to activate 3D
from matplotlib import animation
from matplotlib.widgets import Slider

from skyfield.api import load

# NEBULA configuration imports
from Configuration.NEBULA_ENV_CONFIG import R_EARTH
from Configuration.NEBULA_SENSOR_CONFIG import EVK4_SENSOR

# NEBULA pipeline imports
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER
from Utility.RADIOMETRY import NEBULA_SKYFIELD_ILLUMINATION


# ---------------------------------------------------------------------------
# Logger + small vector helpers
# ---------------------------------------------------------------------------

def _build_logger(name: str = "NEBULA_GEOM3D_INTERACTIVE") -> logging.Logger:
    """
    Build a simple console logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D vector; if near-zero, returns the input unchanged.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


def _build_fov_basis(ra_deg: float, dec_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis (u_bore, u_right, u_up) in ECI from a boresight
    given by RA/Dec.

    u_bore  : pointing direction (unit) from observer to boresight.
    u_right : local "right" direction in the tangent plane.
    u_up    : local "up" direction in the tangent plane.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # Boresight unit vector
    u_bore = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])
    u_bore = _unit_vector(u_bore)

    # Choose a reference axis not parallel to u_bore
    z_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u_bore, z_axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = z_axis

    # Local right = ref × u_bore
    u_right = np.cross(ref, u_bore)
    u_right = _unit_vector(u_right)

    # Local up = u_bore × u_right
    u_up = np.cross(u_bore, u_right)
    u_up = _unit_vector(u_up)

    return u_bore, u_right, u_up


def compute_fov_corners(
    r_obs_km: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    fov_deg_horiz: float,
    rows: int,
    cols: int,
    extent_km: float,
) -> np.ndarray:
    """
    Compute 3D positions of the four FOV far-plane corners from an observer.

    Parameters
    ----------
    r_obs_km : (3,) array
        Observer position in ECI [km].
    ra_deg, dec_deg : float
        Boresight RA/Dec in degrees.
    fov_deg_horiz : float
        Horizontal FOV in degrees (e.g. EVK4_SENSOR.fov_deg).
    rows, cols : int
        Sensor pixel dimensions, used to approximate vertical FOV.
    extent_km : float
        Distance along boresight at which to place the FOV rectangle.

    Returns
    -------
    corners_km : (4, 3) array
        Corner positions in ECI [km], ordered around the rectangle.
    """
    r_obs_km = np.asarray(r_obs_km, dtype=float).reshape(3)

    # Build local basis: boresight, right, up
    u_bore, u_right, u_up = _build_fov_basis(ra_deg, dec_deg)

    # Horizontal and vertical half-FOV angles
    hfov_rad = np.deg2rad(fov_deg_horiz) / 2.0
    vfov_deg = fov_deg_horiz * (rows / cols)  # approximate vertical FOV
    vfov_rad = np.deg2rad(vfov_deg) / 2.0

    # Tangent factors
    tan_h = np.tan(hfov_rad)
    tan_v = np.tan(vfov_rad)

    # Corner sign combinations: (-x,-y), (+x,-y), (+x,+y), (-x,+y)
    signs = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
    corners = []
    for sx, sy in signs:
        dir_vec = (
            u_bore
            + sx * tan_h * u_right
            + sy * tan_v * u_up
        )
        dir_vec = _unit_vector(dir_vec)
        corner = r_obs_km + extent_km * dir_vec
        corners.append(corner)

    return np.vstack(corners)


def build_sun_positions_eci(
    times: np.ndarray,
    ephemeris_path: str,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Compute Sun position in ECI coordinates [km] for each datetime in `times`.

    Uses the same ephemeris as NEBULA_SKYFIELD_ILLUMINATION.
    """
    times = np.asarray(times)
    logger.info("Loading DE440s ephemeris from: %s", ephemeris_path)
    eph = load(ephemeris_path)
    ts = load.timescale()
    t_sf = ts.from_datetimes(times)

    earth = eph["earth"]
    sun = eph["sun"]

    astrometric_sun = earth.at(t_sf).observe(sun)
    r_sun_eci_km = astrometric_sun.position.km.T  # shape (N, 3)

    logger.info("Computed Sun positions for %d timesteps.", r_sun_eci_km.shape[0])
    return r_sun_eci_km


# ---------------------------------------------------------------------------
# Main interactive viewer
# ---------------------------------------------------------------------------

def run_interactive_geometry(
    force_recompute_pixels: bool = False,
    frame_stride: int = 10,
) -> None:
    """
    Launch an interactive 3D viewer for NEBULA geometry.

    Parameters
    ----------
    force_recompute_pixels : bool, optional
        If True, NEBULA_PIXEL_PICKLER will recompute upstream products.
        If False (default), existing PIXEL pickles are reused.
    frame_stride : int, optional
        Subsampling factor for timesteps. 1 uses every timestep; 10 uses
        every 10th timestep, etc.
    """
    logger = _build_logger()

    # ----------------------------------------------------------------------
    # 1) Load or compute pixel-augmented tracks
    # ----------------------------------------------------------------------
    logger.info(
        "Starting NEBULA pixel pipeline via NEBULA_PIXEL_PICKLER "
        "(force_recompute=%s).",
        force_recompute_pixels,
    )

    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=force_recompute_pixels,
        sensor_config=EVK4_SENSOR,
        logger=logger,
    )

    if not obs_tracks:
        logger.error("No observer tracks returned from NEBULA_PIXEL_PICKLER.")
        return
    if not tar_tracks:
        logger.error("No target tracks returned from NEBULA_PIXEL_PICKLER.")
        return

    observer_names = list(obs_tracks.keys())
    target_names = list(tar_tracks.keys())
    logger.info(
        "Loaded %d observers and %d targets from pixel tracks.",
        len(observer_names),
        len(target_names),
    )

    # ----------------------------------------------------------------------
    # 2) Build a common time grid
    # ----------------------------------------------------------------------
    first_obs_name = observer_names[0]
    times = np.asarray(obs_tracks[first_obs_name]["times"])
    n_times = times.shape[0]
    logger.info("Common time grid has %d timesteps.", n_times)

    # Frame indices (subsampled)
    all_indices = np.arange(n_times, dtype=int)
    frame_indices = all_indices[::frame_stride]
    n_frames = frame_indices.size
    logger.info("Interactive viewer will use %d frames (stride=%d).", n_frames, frame_stride)

    # ----------------------------------------------------------------------
    # 3) Compute Sun positions
    # ----------------------------------------------------------------------
    eph_path = NEBULA_SKYFIELD_ILLUMINATION.EPHEMERIS_PATH_DEFAULT
    r_sun_eci_km = build_sun_positions_eci(times, eph_path, logger)

    # ----------------------------------------------------------------------
    # 4) Gather observer positions and pointing arrays
    # ----------------------------------------------------------------------
    obs_pos: Dict[str, np.ndarray] = {}
    obs_bore_ra: Dict[str, np.ndarray] = {}
    obs_bore_dec: Dict[str, np.ndarray] = {}

    for obs_name, obs_track in obs_tracks.items():
        r_eci = np.asarray(obs_track["r_eci_km"], dtype=float)
        obs_pos[obs_name] = r_eci

        ra_deg = np.asarray(obs_track["pointing_boresight_ra_deg"], dtype=float)
        dec_deg = np.asarray(obs_track["pointing_boresight_dec_deg"], dtype=float)
        obs_bore_ra[obs_name] = ra_deg
        obs_bore_dec[obs_name] = dec_deg

        logger.info(
            "Observer '%s': positions shape=%s, boresight shape=%s.",
            obs_name,
            r_eci.shape,
            ra_deg.shape,
        )

    # ----------------------------------------------------------------------
    # 5) Gather target positions and visibility (any observer)
    # ----------------------------------------------------------------------
    tar_pos: Dict[str, np.ndarray] = {}
    tar_masks_good: Dict[str, np.ndarray] = {}

    for tar_name, tar_track in tar_tracks.items():
        r_eci = np.asarray(tar_track["r_eci_km"], dtype=float)
        tar_pos[tar_name] = r_eci

        by_observer = tar_track.get("by_observer", {})
        good_any = np.zeros(n_times, dtype=bool)

        for obs_name in observer_names:
            by_obs = by_observer.get(obs_name, None)
            if by_obs is None:
                continue
            mask = by_obs.get("on_detector_visible_sunlit", None)
            if mask is None:
                continue
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.shape[0] != n_times:
                continue
            good_any |= mask_bool

        tar_masks_good[tar_name] = good_any

    logger.info("Prepared target positions and 'good' visibility masks.")

    # ----------------------------------------------------------------------
    # 6) Build the interactive 3D figure (Earth, orbits, etc.)
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- Earth ---
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x_earth,
        y_earth,
        z_earth,
        alpha=0.2,
        edgecolor="none",
    )

    # Set a symmetric volume that covers GEO
    max_range = 7.0 * R_EARTH
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")

    # Initial camera view (user can change this interactively)
    ax.view_init(elev=25.0, azim=45.0)

    # --- Observer orbits and moving markers ---
    obs_lines: Dict[str, Any] = {}
    obs_markers: Dict[str, Any] = {}

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, obs_name in enumerate(observer_names):
        r_eci = obs_pos[obs_name]
        c = color_cycle[i % len(color_cycle)]

        # Static orbit path
        line, = ax.plot(
            r_eci[:, 0],
            r_eci[:, 1],
            r_eci[:, 2],
            linestyle="-",
            linewidth=1.0,
            color=c,
            alpha=0.7,
            label=obs_name,
        )
        obs_lines[obs_name] = line

        # Moving marker
        marker = ax.scatter(
            [], [], [],
            s=40,
            marker="^",
            color=c,
            edgecolor="k",
            zorder=10,
        )
        obs_markers[obs_name] = marker

    ax.legend(loc="upper left")

    # --- Sun direction line ---
    sun_line, = ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        linestyle="--",
        linewidth=2.0,
        color="yellow",
        label="Sun dir",
    )

    # --- Target scatter (positions + colors) ---
    scatter_targets = ax.scatter([], [], [], s=5)

    # --- FOV lines for each observer (8 lines: 4 apex->corner, 4 rectangle edges) ---
    fov_lines: Dict[str, List[Any]] = {}
    for obs_name in observer_names:
        fov_lines[obs_name] = []
        for _ in range(8):
            fl, = ax.plot(
                [], [], [],
                linestyle="-",
                linewidth=0.7,
                color="grey",
                alpha=0.6,
            )
            fov_lines[obs_name].append(fl)

    # ----------------------------------------------------------------------
    # 7) Define per-frame geometry update
    # ----------------------------------------------------------------------
    rows = EVK4_SENSOR.rows
    cols = EVK4_SENSOR.cols
    fov_deg_horiz = EVK4_SENSOR.fov_deg
    fov_extent_km = 6.0 * R_EARTH

    def update_for_sim_index(sim_idx: int) -> None:
        """
        Update all artists to represent a given simulation timestep.
        """
        # Sun direction (normalized, then scaled for display)
        r_sun = r_sun_eci_km[sim_idx]
        sun_dir = _unit_vector(r_sun)
        sun_point = 8.0 * R_EARTH * sun_dir
        sun_line.set_data_3d(
            [0.0, sun_point[0]],
            [0.0, sun_point[1]],
            [0.0, sun_point[2]],
        )

        # Observers + their FOVs
        for obs_name in observer_names:
            r_obs = obs_pos[obs_name][sim_idx, :]

            # Update moving marker
            obs_markers[obs_name]._offsets3d = (
                np.array([r_obs[0]]),
                np.array([r_obs[1]]),
                np.array([r_obs[2]]),
            )

            ra_deg = obs_bore_ra[obs_name][sim_idx]
            dec_deg = obs_bore_dec[obs_name][sim_idx]

            corners = compute_fov_corners(
                r_obs_km=r_obs,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                fov_deg_horiz=fov_deg_horiz,
                rows=rows,
                cols=cols,
                extent_km=fov_extent_km,
            )

            # 4 apex->corner lines
            for i_corner in range(4):
                line = fov_lines[obs_name][i_corner]
                x_vals = [r_obs[0], corners[i_corner, 0]]
                y_vals = [r_obs[1], corners[i_corner, 1]]
                z_vals = [r_obs[2], corners[i_corner, 2]]
                line.set_data_3d(x_vals, y_vals, z_vals)

            # 4 rectangle edges
            rect_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for j, (i0, i1) in enumerate(rect_edges):
                line = fov_lines[obs_name][4 + j]
                x_vals = [corners[i0, 0], corners[i1, 0]]
                y_vals = [corners[i0, 1], corners[i1, 1]]
                z_vals = [corners[i0, 2], corners[i1, 2]]
                line.set_data_3d(x_vals, y_vals, z_vals)

        # Targets: positions + colors (green if "good", red otherwise)
        xs = []
        ys = []
        zs = []
        colors = []
        for tar_name in target_names:
            r_tar = tar_pos[tar_name][sim_idx, :]
            xs.append(r_tar[0])
            ys.append(r_tar[1])
            zs.append(r_tar[2])

            good_mask = tar_masks_good[tar_name]
            is_good = bool(good_mask[sim_idx])
            colors.append("g" if is_good else "r")

        scatter_targets._offsets3d = (
            np.array(xs),
            np.array(ys),
            np.array(zs),
        )
        scatter_targets.set_color(colors)

        # Update title with timestep info
        ax.set_title(
            f"NEBULA 3D Geometry (timestep {sim_idx}/{n_times - 1})"
        )

    # ----------------------------------------------------------------------
    # 8) Slider + animation hookup
    # ----------------------------------------------------------------------
    # Reserve some space at the bottom for the slider
    plt.subplots_adjust(bottom=0.15)

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax_slider,
        "Frame",
        0,
        n_frames - 1,
        valinit=0,
        valstep=1,
    )

    def update_for_frame_index(frame_i: int) -> None:
        """
        Convert a 0..n_frames-1 frame index into a simulation index, and update.
        """
        frame_i = int(frame_i)
        sim_idx = int(frame_indices[frame_i])
        update_for_sim_index(sim_idx)

        # Keep slider in sync when animation drives frame changes
        # (avoid recursive callback via eventson flag)
        slider.eventson = False
        slider.set_val(frame_i)
        slider.eventson = True

    def on_slider_change(val: float) -> None:
        """
        Slider callback: user moved the slider to select a frame.
        """
        frame_i = int(val)
        update_for_frame_index(frame_i)

    slider.on_changed(on_slider_change)

    # Initialize to frame 0
    update_for_frame_index(0)

    # Create a simple looping animation over frames
    def anim_func(frame_i: int):
        update_for_frame_index(frame_i)
        # Return a tuple of artists; mpl doesn't require everything here
        return ()

    anim = animation.FuncAnimation(
        fig,
        anim_func,
        frames=n_frames,
        interval=50,   # ms between frames (~20 fps)
        repeat=True,
        blit=False,
    )

    # Keep reference so it's not garbage-collected
    global _NEBULA_GEOM3D_INTERACTIVE_ANIM
    _NEBULA_GEOM3D_INTERACTIVE_ANIM = anim

    logger.info(
        "Interactive viewer is ready. You can rotate the camera with the mouse "
        "and use the slider to scrub through time."
    )

    # Show the interactive window (Qt backend recommended: %matplotlib qt)
    plt.show()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_interactive_geometry(
        force_recompute_pixels=False,
        frame_stride=10,  # you can drop to 1 if you want every timestep
    )
