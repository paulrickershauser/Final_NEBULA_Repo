"""
nebula_3d_geometry_anim.py

High-level 3D geometry visualization for NEBULA.

This script:
    1) Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(force_recompute=False)
       to ensure we have:
          - observer tracks with positions + pointing (RA/Dec)
          - target tracks with positions + per-observer visibility flags
          - per-target, per-observer:
                by_observer[obs_name]["on_detector_visible_sunlit"]
    2) Uses Skyfield + the same DE440s ephemeris as NEBULA_SKYFIELD_ILLUMINATION
       to compute the Sun direction over the common time grid.
    3) Builds a 3D matplotlib figure showing:
          - Earth as a sphere at the origin,
          - observer orbits + current positions,
          - the Sun direction as a vector,
          - a field-of-view pyramid for each observer (approximate cone),
          - all targets as 3D points.
       Targets are colored:
          - green if ANY observer has on_detector_visible_sunlit == True
            at that timestep,
          - red otherwise.
    4) Writes a single MP4 movie to:

           NEBULA_OUTPUT/GEOM3D_Animations/nebula_3d_geometry.mp4

Usage (from Spyder):

    %runfile C:/Users/prick/Desktop/Research/NEBULA/nebula_3d_geometry_anim.py --wdir

Dependencies:
    - matplotlib (with ffmpeg available in PATH)
    - tqdm
    - skyfield
    - Existing NEBULA pipeline and PIXEL pickles

This is meant as a qualitative, global sanity-check visualization of the
NEBULA geometry, not a publication-quality plotting tool.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
from matplotlib import animation
from tqdm.auto import tqdm

from skyfield.api import load

# NEBULA configuration imports
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Configuration.NEBULA_ENV_CONFIG import R_EARTH
from Configuration.NEBULA_SENSOR_CONFIG import EVK4_SENSOR

# NEBULA pipeline imports
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER
from Utility.RADIOMETRY import NEBULA_SKYFIELD_ILLUMINATION


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _build_logger(name: str = "NEBULA_GEOM3D") -> logging.Logger:
    """
    Build a simple console logger for this script.
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
    Build a simple orthonormal basis (u_bore, u_right, u_up) in ECI from
    a boresight given by RA/Dec.

    u_bore  : pointing direction (unit) from observer to boresight.
    u_right : local "right" direction in the tangent plane.
    u_up    : local "up" direction in the tangent plane.
    """
    # Convert RA/Dec from degrees to radians.
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # Boresight unit vector in ECI-like coordinates.
    u_bore = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

    u_bore = _unit_vector(u_bore)

    # Choose a reference "up" vector that is not parallel to u_bore.
    z_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u_bore, z_axis)) > 0.9:
        # If boresight is too close to +Z/-Z, choose Y as reference.
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
    Compute the 3D positions of the four FOV corners at some distance
    along the boresight.

    Parameters
    ----------
    r_obs_km : (3,) array
        Observer position in ECI coordinates [km].
    ra_deg, dec_deg : float
        Boresight right ascension and declination in degrees.
    fov_deg_horiz : float
        Horizontal FOV in degrees (e.g. EVK4_SENSOR.fov_deg).
    rows, cols : int
        Sensor pixel dimensions, used to estimate vertical FOV.
    extent_km : float
        Distance from the observer to place the FOV "far plane" [km].

    Returns
    -------
    corners_km : (4, 3) array
        ECI positions of the four corner points on the far FOV rectangle.
        Order: [(-x,-y), (+x,-y), (+x,+y), (-x,+y)] in the local (right, up) basis.
    """
    r_obs_km = np.asarray(r_obs_km, dtype=float).reshape(3)

    # Build local basis.
    u_bore, u_right, u_up = _build_fov_basis(ra_deg, dec_deg)

    # Horizontal and vertical half-angles in radians.
    hfov_rad = np.deg2rad(fov_deg_horiz) / 2.0
    vfov_deg = fov_deg_horiz * (rows / cols)   # approximate vertical FOV
    vfov_rad = np.deg2rad(vfov_deg) / 2.0

    # Tangent factors: how far we go along right/up compared to boresight.
    tan_h = np.tan(hfov_rad)
    tan_v = np.tan(vfov_rad)

    # Local corner directions in the boresight frame.
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
    Compute Sun position in ECI-like coordinates [km] for each datetime
    in the given time array, using Skyfield and the same DE440s ephemeris
    path as NEBULA_SKYFIELD_ILLUMINATION.

    Parameters
    ----------
    times : array-like of datetime
        Length-N array of Python datetime objects.
    ephemeris_path : str
        Path to the DE440s BSP file.
    logger : logging.Logger
        Logger for informational messages.

    Returns
    -------
    r_sun_eci_km : ndarray, shape (N, 3)
        Sun position vectors in km, in the same ECI frame used by NEBULA.
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
# Main movie builder
# ---------------------------------------------------------------------------

def create_geometry_movie(
    force_recompute_pixels: bool = False,
    frame_stride: int = 10,
    mp4_name: str = "nebula_3d_geometry.mp4",
) -> None:
    """
    Create a 3D geometry MP4 showing Earth, observers, targets, the Sun,
    and each observer's FOV cone, with targets colored by visibility.

    Parameters
    ----------
    force_recompute_pixels : bool, optional
        If True, calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs with
        force_recompute=True. If False, reuses existing PIXEL pickles.
    frame_stride : int, optional
        Use every `frame_stride` timestep when rendering to keep the movie
        at a manageable length. For example, 10 renders about 144 frames
        for a 1441-step simulation.
    mp4_name : str, optional
        Filename for the output MP4 (placed under NEBULA_OUTPUT/GEOM3D_Animations).

    Returns
    -------
    None
        Writes the MP4 to disk and logs the location.
    """
    log = _build_logger()

    # ----------------------------------------------------------------------
    # 1) Run (or reuse) the pixel pipeline to get obs/target tracks.
    # ----------------------------------------------------------------------
    log.info(
        "Starting NEBULA pixel pipeline via NEBULA_PIXEL_PICKLER "
        "(force_recompute=%s).",
        force_recompute_pixels,
    )

    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=force_recompute_pixels,
        sensor_config=EVK4_SENSOR,
        logger=log,
    )

    if not obs_tracks:
        log.error("No observer tracks returned from NEBULA_PIXEL_PICKLER.")
        return
    if not tar_tracks:
        log.error("No target tracks returned from NEBULA_PIXEL_PICKLER.")
        return

    observer_names = list(obs_tracks.keys())
    target_names = list(tar_tracks.keys())
    log.info(
        "Loaded %d observers and %d targets from pixel tracks.",
        len(observer_names),
        len(target_names),
    )

    # ----------------------------------------------------------------------
    # 2) Build a common time grid (take from the first observer).
    # ----------------------------------------------------------------------
    first_obs_name = observer_names[0]
    first_obs_track = obs_tracks[first_obs_name]
    times = np.asarray(first_obs_track["times"])
    n_times = times.shape[0]
    log.info("Common time grid: %d timesteps.", n_times)

    # Frame indices to render.
    all_indices = np.arange(n_times, dtype=int)
    frame_indices = all_indices[::frame_stride]
    log.info("Rendering %d frames (stride=%d).", frame_indices.size, frame_stride)

    # ----------------------------------------------------------------------
    # 3) Compute Sun positions over the time grid.
    # ----------------------------------------------------------------------
    eph_path = NEBULA_SKYFIELD_ILLUMINATION.EPHEMERIS_PATH_DEFAULT
    r_sun_eci_km = build_sun_positions_eci(times, eph_path, log)

    # ----------------------------------------------------------------------
    # 4) Gather observer positions and pointing arrays.
    # ----------------------------------------------------------------------
    obs_pos: Dict[str, np.ndarray] = {}
    obs_bore_ra: Dict[str, np.ndarray] = {}
    obs_bore_dec: Dict[str, np.ndarray] = {}

    for obs_name, obs_track in obs_tracks.items():
        r_eci = np.asarray(obs_track["r_eci_km"], dtype=float)
        obs_pos[obs_name] = r_eci

        # These must exist; they are also used by NEBULA_WCS.
        ra_deg = np.asarray(obs_track["pointing_boresight_ra_deg"], dtype=float)
        dec_deg = np.asarray(obs_track["pointing_boresight_dec_deg"], dtype=float)

        obs_bore_ra[obs_name] = ra_deg
        obs_bore_dec[obs_name] = dec_deg

        log.info(
            "Observer '%s': positions shape=%s, boresight arrays shape=%s.",
            obs_name,
            r_eci.shape,
            ra_deg.shape,
        )

    # ----------------------------------------------------------------------
    # 5) Gather target positions and per-target, per-observer visibility.
    # ----------------------------------------------------------------------
    tar_pos: Dict[str, np.ndarray] = {}
    tar_masks_good: Dict[str, np.ndarray] = {}

    for tar_name, tar_track in tar_tracks.items():
        r_eci = np.asarray(tar_track["r_eci_km"], dtype=float)
        tar_pos[tar_name] = r_eci

        by_observer = tar_track.get("by_observer", {})
        # Build "any observer sees this target on_detector_visible_sunlit".
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

    log.info("Prepared target positions and visibility masks.")

    # ----------------------------------------------------------------------
    # 6) Prepare output directory for the MP4.
    # ----------------------------------------------------------------------
    geom_dir = NEBULA_OUTPUT_DIR / "GEOM3D_Animations"
    geom_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = geom_dir / mp4_name
    log.info("Output MP4 will be written to: %s", mp4_path)

    # ----------------------------------------------------------------------
    # 7) Build the 3D figure and static artists (Earth + orbits).
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("NEBULA 3D Geometry")

    # Earth sphere
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

    # Set a symmetric view volume that comfortably fits GEO.
    max_range = 7.0 * R_EARTH
    for axis in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        axis(-max_range, max_range)

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.view_init(elev=25.0, azim=45.0)

    # Observer orbits (static lines) and their moving markers.
    obs_lines: Dict[str, Any] = {}
    obs_markers: Dict[str, Any] = {}

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, obs_name in enumerate(observer_names):
        r_eci = obs_pos[obs_name]
        c = color_cycle[i % len(color_cycle)]

        # Static orbit line
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

        # Moving marker (current satellite position)
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

    # Sun direction line (from origin).
    sun_line, = ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        linestyle="--",
        linewidth=2.0,
        color="yellow",
        label="Sun dir",
    )

    # Target scatter; we will update positions and colors each frame.
    scatter_targets = ax.scatter([], [], [], s=5)

    # FOV lines per observer: 8 lines each (4 from apex to corners, 4 rectangle edges).
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
    # 8) Update function for a single frame index.
    # ----------------------------------------------------------------------
    rows = EVK4_SENSOR.rows
    cols = EVK4_SENSOR.cols
    fov_deg_horiz = EVK4_SENSOR.fov_deg
    fov_extent_km = 6.0 * R_EARTH  # how far to draw the FOV frustum

    def update_frame(sim_idx: int) -> None:
        """
        Update all artists for a given simulation timestep index.
        """
        # ---- Sun direction ----
        r_sun = r_sun_eci_km[sim_idx]
        sun_dir = _unit_vector(r_sun)
        sun_point = 8.0 * R_EARTH * sun_dir
        sun_line.set_data_3d(
            [0.0, sun_point[0]],
            [0.0, sun_point[1]],
            [0.0, sun_point[2]],
        )

        # ---- Observers + their FOV frustums ----
        for obs_name in observer_names:
            r_obs = obs_pos[obs_name][sim_idx, :]
            # Update observer marker.
            obs_markers[obs_name]._offsets3d = (
                np.array([r_obs[0]]),
                np.array([r_obs[1]]),
                np.array([r_obs[2]]),
            )

            # FOV corners for this timestep.
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

            # Edges: 4 lines from apex -> each corner.
            for i_corner in range(4):
                line = fov_lines[obs_name][i_corner]
                x_vals = [r_obs[0], corners[i_corner, 0]]
                y_vals = [r_obs[1], corners[i_corner, 1]]
                z_vals = [r_obs[2], corners[i_corner, 2]]
                line.set_data_3d(x_vals, y_vals, z_vals)

            # Rectangle edges (4 lines).
            rect_edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            ]
            for j, (i0, i1) in enumerate(rect_edges):
                line = fov_lines[obs_name][4 + j]
                x_vals = [corners[i0, 0], corners[i1, 0]]
                y_vals = [corners[i0, 1], corners[i1, 1]]
                z_vals = [corners[i0, 2], corners[i1, 2]]
                line.set_data_3d(x_vals, y_vals, z_vals)

        # ---- Targets ----
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

        # Update title with time index.
        ax.set_title(
            f"NEBULA 3D Geometry (frame {sim_idx}/{n_times - 1})"
        )

    # ----------------------------------------------------------------------
    # 9) Render to MP4 with FFmpeg writer and tqdm progress bar.
    # ----------------------------------------------------------------------
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, metadata={}, bitrate=2000)

    with writer.saving(fig, str(mp4_path), dpi=150):
        for sim_idx in tqdm(frame_indices, desc="Rendering 3D geometry"):
            update_frame(int(sim_idx))
            writer.grab_frame()

    plt.close(fig)
    log.info("Finished writing 3D geometry movie to: %s", mp4_path)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # You can tweak these defaults if you want shorter/longer or denser frames.
    create_geometry_movie(
        force_recompute_pixels=False,
        frame_stride=10,
        mp4_name="nebula_3d_geometry.mp4",
    )
