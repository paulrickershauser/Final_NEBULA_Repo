"""
nebula_pyvista_viewer.py

Interactive 3D visualization for NEBULA using PyVista + Qt.

Features
--------
- Loads pixel-augmented tracks from NEBULA_PIXEL_PICKLER
- Displays:
    * Earth sphere (R_EARTH)
    * Observer orbits as 3D lines
    * Targets as colored points in 3D (ECI)
    * Observer FOV frustum (pyramid)
- Animation over time:
    * Qt QTimer drives sim_idx
    * Play/Pause toolbar button
    * Time slider (scrub through frames)
- Observer selection:
    * Dropdown to choose observer_for_visibility
    * Targets are colored by that observer's on_detector_visible_sunlit
- Observer camera:
    * Toolbar toggle "Observer Camera"
    * When enabled, camera is placed at observer position
      and oriented along the boresight pointing.
- Point picking:
    * Click a target point to print detailed info:
        - Target name
        - Observer
        - Time index
        - Range
        - LOS visible / Sunlit / On detector / Combined mask
        - Pixel (x, y)
        - Phase angle (deg) + simple classification (front-lit, etc.)
        - Flux / photon flux / apparent mag (if available).

Usage
-----
In Spyder (NEBULA env):

    %runfile C:/Users/prick/Desktop/Research/NEBULA/nebula_pyvista_viewer.py --wdir

Make sure:
- PyVista, pyvistaqt, qtpy, tqdm are installed in your NEBULA env.
"""

# -----------------------------
# Standard library imports
# -----------------------------
# Import logging for status messages.
import logging
# Import typing utilities for type hints.
from typing import Dict, Any, Tuple, List, Optional

# -----------------------------
# Third-party imports
# -----------------------------
# Import NumPy for numerical arrays.
import numpy as np
# Import tqdm for progress bars in the console.
from tqdm.auto import tqdm
# Import PyVista for 3D visualization.
import pyvista as pv
# Import BackgroundPlotter for Qt-based interactive window.
from pyvistaqt import BackgroundPlotter
# Import Qt widgets and core from qtpy for cross-backend Qt.
from qtpy import QtWidgets, QtCore

# -----------------------------
# NEBULA configuration imports
# -----------------------------
# Import Earth's radius from NEBULA environment configuration.
from Configuration.NEBULA_ENV_CONFIG import R_EARTH
# Import sensor definition (rows, cols, FOV) from NEBULA sensor config.
from Configuration.NEBULA_SENSOR_CONFIG import EVK4_SENSOR

# -----------------------------
# NEBULA pipeline imports
# -----------------------------
# Import the NEBULA pixel pickler that attaches pixel geometry.
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER


# ============================================================================
# Logging helper
# ============================================================================

def _build_logger(name: str = "NEBULA_PYVISTA_VIEWER") -> logging.Logger:
    """
    Build and return a simple console logger with INFO level.
    """

    # Get (or create) a logger with the given name.
    logger = logging.getLogger(name)

    # Only add a handler if none exist (avoid duplicate handlers).
    if not logger.handlers:
        # Create a StreamHandler to write logs to stdout.
        handler = logging.StreamHandler()
        # Define a log message format with timestamps and levels.
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Attach the formatter to the handler.
        handler.setFormatter(fmt)
        # Add the handler to the logger.
        logger.addHandler(handler)
        # Set the default logging level to INFO.
        logger.setLevel(logging.INFO)

    # Return the configured logger.
    return logger


# ============================================================================
# Small math utilities (vector normalization, FOV basis)
# ============================================================================

def _unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D vector.

    If the norm is ~0, return the original vector to avoid division by zero.
    """

    # Convert input to NumPy float array for robustness.
    v = np.asarray(v, dtype=float)

    # Compute the Euclidean norm (length) of the vector.
    n = np.linalg.norm(v)

    # If the norm is extremely small, return the original vector unchanged.
    if n == 0.0:
        return v

    # Divide the vector by its norm to get a unit vector.
    return v / n


def _build_fov_basis(ra_deg: float, dec_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis (u_bore, u_right, u_up) in ECI from a boresight
    specified by RA/Dec in degrees.

    u_bore  : unit vector pointing along the boresight direction
    u_right : unit vector roughly increasing RA at constant Dec (local "x")
    u_up    : unit vector completing the right-handed triad (local "y")
    """

    # Convert RA from degrees to radians for trigonometric functions.
    ra = np.deg2rad(ra_deg)

    # Convert Dec from degrees to radians.
    dec = np.deg2rad(dec_deg)

    # Build the boresight unit vector in ECI using standard RA/Dec convention.
    u_bore = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

    # Normalize the boresight vector for safety.
    u_bore = _unit_vector(u_bore)

    # Define the global +Z axis as a reference.
    z_axis = np.array([0.0, 0.0, 1.0])

    # If boresight is almost aligned with +Z, choose a different reference
    # (e.g. +Y) to avoid singularity when taking cross products.
    if abs(np.dot(u_bore, z_axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = z_axis

    # Local right vector is ref × boresight (ensures orthogonality).
    u_right = np.cross(ref, u_bore)

    # Normalize local right vector.
    u_right = _unit_vector(u_right)

    # Local up vector is boresight × right to complete the triad.
    u_up = np.cross(u_bore, u_right)

    # Normalize local up vector.
    u_up = _unit_vector(u_up)

    # Return tuple of (boresight, right, up) unit vectors.
    return u_bore, u_right, u_up


# ============================================================================
# FOV geometry helpers
# ============================================================================

def compute_fov_corners(
    r_obs_km: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    fov_deg_horiz: float,
    rows: int,
    cols: int,
    extent_km: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D positions of four FOV far-plane corners from an observer.

    Returns
    -------
    corners_km : (4,3) array of far-plane corner positions
    u_bore     : (3,) boresight unit vector
    u_right    : (3,) local right unit vector
    u_up       : (3,) local up unit vector
    """

    # Ensure observer position is a flat 3-vector of floats.
    r_obs_km = np.asarray(r_obs_km, dtype=float).reshape(3)

    # Build FOV basis vectors from boresight RA/Dec.
    u_bore, u_right, u_up = _build_fov_basis(ra_deg, dec_deg)

    # Compute horizontal half-FOV in radians (sensor FOV_deg / 2).
    hfov_rad = np.deg2rad(fov_deg_horiz) / 2.0

    # Compute vertical FOV by scaling with pixel aspect ratio rows/cols.
    vfov_deg = fov_deg_horiz * (rows / cols)

    # Convert vertical half-FOV to radians.
    vfov_rad = np.deg2rad(vfov_deg) / 2.0

    # Precompute tangent values for half-FOV angles.
    tan_h = np.tan(hfov_rad)
    tan_v = np.tan(vfov_rad)

    # Define sign combinations for the four corners (-x,-y), (+x,-y), (+x,+y), (-x,+y).
    signs = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]

    # Initialize list to accumulate corner positions.
    corners = []

    # Loop over each (sx, sy) combination.
    for sx, sy in signs:
        # Form the direction vector in the boresight basis by adding
        # horizontal and vertical offsets multiplied by tangent of half-FOV.
        dir_vec = (
            u_bore
            + sx * tan_h * u_right
            + sy * tan_v * u_up
        )

        # Normalize the direction vector to get a unit vector.
        dir_vec = _unit_vector(dir_vec)

        # Compute the actual 3D corner by going extent_km along this direction.
        corner = r_obs_km + extent_km * dir_vec

        # Append the corner to our list.
        corners.append(corner)

    # Stack the list of corners into a (4,3) array.
    corners_km = np.vstack(corners)

    # Return corners and the basis vectors used.
    return corners_km, u_bore, u_right, u_up


def build_fov_frustum_mesh(
    r_obs_km: np.ndarray,
    corners_km: np.ndarray,
) -> pv.PolyData:
    """
    Build a triangular FOV frustum (pyramid) mesh for PyVista.

    - 1 apex vertex at the observer position
    - 4 vertices at the FOV far-plane corners
    - 4 triangular faces forming a pyramid
    """

    # Ensure observer position is a (1,3) array for stacking.
    apex = np.asarray(r_obs_km, dtype=float).reshape(1, 3)

    # Ensure corners are a (4,3) array.
    corners = np.asarray(corners_km, dtype=float).reshape(4, 3)

    # Stack apex and corner vertices into a single (5,3) vertex array.
    vertices = np.vstack([apex, corners])

    # Define faces for the pyramid using VTK-style cell encoding.
    # Each face: [3, v0, v1, v2] -> triangle with 3 vertices.
    faces = np.hstack([
        np.array([3, 0, 1, 2], dtype=np.int64),
        np.array([3, 0, 2, 3], dtype=np.int64),
        np.array([3, 0, 3, 4], dtype=np.int64),
        np.array([3, 0, 4, 1], dtype=np.int64),
    ])

    # Construct a PolyData from these vertices and faces.
    frustum = pv.PolyData(vertices, faces)

    # Return the frustum mesh.
    return frustum


# ============================================================================
# Main viewer class
# ============================================================================

class NebulaPyVistaViewer:
    """
    Encapsulates the PyVista/Qt viewer for NEBULA geometry.
    """

    def __init__(
        self,
        force_recompute_pixels: bool = False,
        observer_for_visibility: Optional[str] = None,
        sim_idx_start: int = 0,
        frame_interval_ms: int = 50,
    ):
        """
        Initialize the viewer: load tracks, precompute data, build scene, start timer.
        """

        # Build a logger for this viewer instance.
        self.log = _build_logger()

        # Store frame interval in milliseconds for the Qt timer.
        self.frame_interval_ms = frame_interval_ms

        # Log that we are calling the pixel pickler (which cascades upstream).
        self.log.info(
            "Calling NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs "
            "(force_recompute=%s).",
            force_recompute_pixels,
        )

        # Call the NEBULA pixel pickler to get observer and target tracks
        # with pixel geometry attached.
        obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
            force_recompute=force_recompute_pixels,
            sensor_config=EVK4_SENSOR,
            logger=self.log,
        )

        # If no observers were returned, raise an error (pipeline problem).
        if not obs_tracks:
            raise RuntimeError("No observer tracks returned from NEBULA_PIXEL_PICKLER.")

        # If no targets were returned, raise an error (pipeline problem).
        if not tar_tracks:
            raise RuntimeError("No target tracks returned from NEBULA_PIXEL_PICKLER.")

        # Store the observer tracks dictionary on the instance.
        self.obs_tracks = obs_tracks

        # Store the target tracks dictionary on the instance.
        self.tar_tracks = tar_tracks

        # Extract a list of observer names for convenience.
        self.observer_names = list(obs_tracks.keys())

        # Extract a list of target names for convenience.
        self.target_names = list(tar_tracks.keys())

        # Log how many observers and targets we have.
        self.log.info(
            "Loaded %d observers and %d targets from pixel tracks.",
            len(self.observer_names),
            len(self.target_names),
        )

        # Pick the first observer as a reference to get number of timesteps.
        first_obs_name = self.observer_names[0]

        # Extract that observer's ECI position array.
        first_obs = obs_tracks[first_obs_name]
        r_eci_first = np.asarray(first_obs["r_eci_km"], dtype=float)

        # Determine the number of timesteps from the length of r_eci_km.
        self.n_times = r_eci_first.shape[0]

        # If requested starting index is out of range, clamp it to [0, n_times-1].
        if sim_idx_start < 0 or sim_idx_start >= self.n_times:
            self.log.warning(
                "Requested sim_idx_start=%d out of range [0, %d). Clamping.",
                sim_idx_start,
                self.n_times,
            )
            sim_idx_start = max(0, min(sim_idx_start, self.n_times - 1))

        # Store the current simulation time index.
        self.sim_idx = sim_idx_start

        # If no observer_for_visibility is provided, or if the provided name
        # is not found, fall back to the first observer.
        if (observer_for_visibility is None or
                observer_for_visibility not in self.observer_names):
            if observer_for_visibility is not None:
                self.log.warning(
                    "observer_for_visibility='%s' not found; "
                    "falling back to '%s'.",
                    observer_for_visibility,
                    first_obs_name,
                )
            observer_for_visibility = first_obs_name

        # Store the name of the observer whose perspective we use for coloring.
        self.observer_for_visibility = observer_for_visibility

        # Precompute observer position and boresight arrays for all observers.
        self._precompute_observer_geometry()

        # Build the list of valid targets (with masks) for the current observer.
        self._build_valid_targets_for_observer()

        # Initialize flags for camera behavior (global vs observer).
        self.use_observer_camera = False

        # Store a placeholder for global camera state (filled later).
        self._global_cam_state = None

        # Placeholders for boresight basis at current timestep (for camera).
        self._last_u_bore = None
        self._last_u_up = None

        # IMPORTANT: Initialize playing flag BEFORE building the scene,
        # because _build_scene() calls _status_text(), which references
        # self.playing.
        self.playing = True

        # Build the 3D scene (plotter, static geometry, dynamic actors, GUI).
        self._build_scene()

        # Create a QTimer attached to the plotter's Qt window.
        self.timer = QtCore.QTimer(self.plotter.app_window)

        # Connect the timer's timeout signal to our _on_timer callback.
        self.timer.timeout.connect(self._on_timer)

        # Start the timer with the requested frame interval.
        self.timer.start(self.frame_interval_ms)

        # Log that the viewer is fully initialized.
        self.log.info(
            "Viewer initialized. Close the PyVista window to end the session."
        )

    # =====================================================================
    # Precomputation
    # =====================================================================

    def _precompute_observer_geometry(self) -> None:
        """
        Precompute ECI position and camera basis vectors for each observer.

        For each observer we fill:
            self.obs_r_eci_km[obs_name]  : (n_times, 3) ECI positions [km]
            self.obs_u_bore[obs_name]    : (n_times, 3) boresight unit vectors
            self.obs_u_up[obs_name]      : (n_times, 3) "up" unit vectors

        The function supports two possible track layouts:

        1) Preferred (what your pickles currently have):
           - 'pointing_boresight_ra_deg' : (n_times,) RA of boresight [deg]
           - 'pointing_boresight_dec_deg': (n_times,) Dec of boresight [deg]

           In this case we build boresight/up in ECI via _build_fov_basis
           for each timestep.

        2) Legacy:
           - 'boresight_eci_unit' : (n_times, 3)
           - 'up_eci_unit'        : (n_times, 3)

           If present, we just use them directly.
        """

        # Create dictionaries to hold position and boresight arrays per observer.
        self.obs_r_eci_km: Dict[str, np.ndarray] = {}
        self.obs_u_bore: Dict[str, np.ndarray] = {}
        self.obs_u_up: Dict[str, np.ndarray] = {}

        # Loop over each observer in the tracks dictionary.
        for obs_name, obs_track in self.obs_tracks.items():
            # ECI positions [km].
            r_eci = np.asarray(obs_track["r_eci_km"], dtype=float)
            if r_eci.ndim != 2 or r_eci.shape[1] != 3:
                raise ValueError(
                    f"Observer '{obs_name}': r_eci_km has shape {r_eci.shape}, "
                    "expected (n_times, 3)."
                )
            n_times = r_eci.shape[0]
            self.obs_r_eci_km[obs_name] = r_eci

            # ------------------------------------------------------------------
            # Path 1: direct ECI basis vectors already stored on the track.
            # ------------------------------------------------------------------
            if "boresight_eci_unit" in obs_track and "up_eci_unit" in obs_track:
                u_bore = np.asarray(obs_track["boresight_eci_unit"], dtype=float)
                u_up   = np.asarray(obs_track["up_eci_unit"], dtype=float)

                # Ensure correct shape.
                u_bore = u_bore.reshape(n_times, 3)
                u_up   = u_up.reshape(n_times, 3)

            else:
                # ------------------------------------------------------------------
                # Path 2: build basis from stored boresight RA/Dec (current case).
                # ------------------------------------------------------------------
                if (
                    "pointing_boresight_ra_deg" not in obs_track
                    or "pointing_boresight_dec_deg" not in obs_track
                ):
                    raise KeyError(
                        f"Observer '{obs_name}' has neither "
                        "('boresight_eci_unit', 'up_eci_unit') nor "
                        "('pointing_boresight_ra_deg', 'pointing_boresight_dec_deg')."
                    )

                ra_deg = np.asarray(obs_track["pointing_boresight_ra_deg"], dtype=float)
                dec_deg = np.asarray(obs_track["pointing_boresight_dec_deg"], dtype=float)

                # Allow scalar RA/Dec to be broadcast over time.
                if ra_deg.shape == ():
                    ra_deg = np.full(n_times, float(ra_deg))
                if dec_deg.shape == ():
                    dec_deg = np.full(n_times, float(dec_deg))

                if ra_deg.shape[0] != n_times or dec_deg.shape[0] != n_times:
                    raise ValueError(
                        f"Observer '{obs_name}': RA/Dec arrays have shapes "
                        f"{ra_deg.shape}, {dec_deg.shape} but expected (n_times,)."
                    )

                # Build boresight & up vectors for each timestep.
                u_bore_list = []
                u_up_list = []
                for ra, dec in zip(ra_deg, dec_deg):
                    u_bore_i, _, u_up_i = _build_fov_basis(float(ra), float(dec))
                    u_bore_list.append(u_bore_i)
                    u_up_list.append(u_up_i)

                u_bore = np.vstack(u_bore_list)
                u_up = np.vstack(u_up_list)

            # Final sanity checks: shapes must match r_eci_km.
            if u_bore.shape != r_eci.shape:
                raise ValueError(
                    f"Observer '{obs_name}': boresight basis has shape {u_bore.shape} "
                    f"but r_eci_km has shape {r_eci.shape}."
                )
            if u_up.shape != r_eci.shape:
                raise ValueError(
                    f"Observer '{obs_name}': up basis has shape {u_up.shape} "
                    f"but r_eci_km has shape {r_eci.shape}."
                )

            # Store unit vectors.
            self.obs_u_bore[obs_name] = u_bore
            self.obs_u_up[obs_name] = u_up

        # Log that precomputation completed successfully.
        self.log.info(
            "Precomputed observer geometry for %d observers.",
            len(self.obs_tracks),
        )

    # =====================================================================
    # Target selection / masks
    # =====================================================================

    def _build_valid_targets_for_observer(self) -> None:
        """
        Build arrays of valid targets (and per-target visibility masks) for
        the current observer_for_visibility.

        A target is considered 'valid' if it has a by_observer entry for this
        observer, and at least one timestep where:
          - the LOS flag is true, and
          - the target is illuminated,
          - and the target is within the sensor FOV (pixel indices not NaN).
        """

        # Name of the observer whose perspective we use.
        obs_name = self.observer_for_visibility

        # Lists to accumulate valid target names and their masks.
        valid_target_names: list[str] = []
        per_target_mask: list[np.ndarray] = []

        # Iterate over all targets.
        for tar_name, tar_track in self.tar_tracks.items():
            by_obs = tar_track.get("by_observer", {})
            if obs_name not in by_obs:
                # This target was never processed for this observer.
                continue

            sub = by_obs[obs_name]

            # Extract LOS, illumination, and pixel coordinates (as stored in the pickles).
            los   = np.asarray(sub.get("los_visible", []), dtype=bool)
            illum = np.asarray(sub.get("illum_is_sunlit", []), dtype=bool)
            pix_x = np.asarray(sub.get("pix_x", []), dtype=float)
            pix_y = np.asarray(sub.get("pix_y", []), dtype=float)
            
            # Basic shape checks.
            if (
                los.shape != illum.shape
                or los.shape != pix_x.shape
                or los.shape != pix_y.shape
            ):
                self.log.warning(
                    "Target '%s' for observer '%s' has inconsistent array shapes. Skipping.",
                    tar_name,
                    obs_name,
                )
                continue
            
            # Build a mask: visible, illuminated, and valid (non-NaN) pixels.
            valid_mask = (
                los
                & illum
                & np.isfinite(pix_x)
                & np.isfinite(pix_y)
            )


            # If there is no timestep where the target is valid, skip it.
            if not np.any(valid_mask):
                continue

            # Otherwise keep this target and its mask.
            valid_target_names.append(tar_name)
            per_target_mask.append(valid_mask)

        # If no targets were valid for this observer, raise an error.
        if not valid_target_names:
            raise RuntimeError(
                f"No valid targets found for observer '{obs_name}'. "
                "Check LOS/illumination/pixel fields."
            )

        # Convert to NumPy arrays for convenience.
        self.valid_target_names = np.array(valid_target_names, dtype=object)
        self.valid_target_masks = per_target_mask

        # Log how many valid targets we ended up with.
        self.log.info(
            "Observer '%s': using %d valid targets for animation.",
            obs_name,
            len(self.valid_target_names),
        )

    # =====================================================================
    # Scene construction
    # =====================================================================

    def _build_scene(self) -> None:
        """
        Create the PyVista BackgroundPlotter, add Earth, orbits, boresights,
        and GUI widgets.
        """

        # Create the PyVista background plotter.
        self.plotter = BackgroundPlotter(
            title="NEBULA PyVista Viewer",
            window_size=(1280, 720),
            auto_update=True,
            off_screen=False,
        )

        # Add a dark background for contrast.
        self.plotter.set_background("black")

        # Add Earth sphere (simple representation).
        earth_radius_km = 6378.137
        earth = pv.Sphere(radius=earth_radius_km, theta_resolution=64, phi_resolution=64)
        self.plotter.add_mesh(
            earth,
            color="darkblue",
            specular=0.3,
            smooth_shading=True,
            name="Earth",
        )

        # Determine a global scale for the camera distance.
        all_obs_positions = np.concatenate(list(self.obs_r_eci_km.values()), axis=0)
        max_range = float(np.linalg.norm(all_obs_positions, axis=1).max())

        # Place the camera at (max_range, max_range, max_range) looking toward the origin.
        self.plotter.set_focus((0.0, 0.0, 0.0))
        self.plotter.set_position(point=(max_range, max_range, max_range))
        self.plotter.set_viewup((0.0, 0.0, 1.0))

        # Store default (global) camera state for later toggling.
        self._global_cam_state = self.plotter.camera_position

        # Add orbits and satellite markers.
        self._build_orbits_and_markers()

        # Add initial dynamic actors (lines from observer to targets, etc.).
        self._build_dynamic_actors()

        # Add status text overlay (uses self.playing, so playing must be set).
        self.text_actor = self.plotter.add_text(
            self._status_text(),
            position="upper_left",
            color="white",
            font_size=10,
        )

        # Build Qt-based UI controls (slider, buttons, etc.).
        self._build_qt_controls()

    # =====================================================================
    # Dynamic actors and status text
    # =====================================================================

    def _status_text(self) -> str:
        """
        Build the status text string for the overlay in the upper-left corner.
        Uses current observer, timestep, and playing state.
        """
        return (
            f"Observer: {self.observer_for_visibility}\n"
            f"Time index: {self.sim_idx+1} / {self.n_times}\n"
            f"{'Playing' if self.playing else 'Paused'}"
        )

    def _update_text(self) -> None:
        """
        Update the on-screen status text with current state.
        """
        if self.text_actor is not None:
            self.plotter.remove_actor(self.text_actor, reset_camera=False)
        self.text_actor = self.plotter.add_text(
            self._status_text(),
            position="upper_left",
            color="white",
            font_size=10,
        )


    # ------------------------------------------------------------------ #
    # Camera helpers (store/restore global, set observer camera)
    # ------------------------------------------------------------------ #

    def _store_global_camera_state(self):
        """
        Store the current camera state as the global view state.
        """

        # Access the PyVista camera object.
        cam = self.plotter.camera

        # Save position, focal point, view up, and view angle.
        self._global_cam_state = {
            "position": cam.position,
            "focal_point": cam.focal_point,
            "view_up": cam.up,
            "view_angle": cam.view_angle,
        }

    def _restore_global_camera_state(self):
        """
        Restore the previously stored global camera state.
        """

        # If we never stored a global state, just reset camera and store it.
        if self._global_cam_state is None:
            self.plotter.reset_camera()
            self._store_global_camera_state()
            return

        # Access the camera object.
        cam = self.plotter.camera

        # Restore camera position.
        cam.position = self._global_cam_state["position"]

        # Restore camera focal point.
        cam.focal_point = self._global_cam_state["focal_point"]

        # Restore camera up vector.
        cam.up = self._global_cam_state["view_up"]

        # Restore camera view angle.
        cam.view_angle = self._global_cam_state["view_angle"]

        # Render the updated camera state.
        self.plotter.render()

    def _apply_observer_camera(self):
        """
        Configure the camera to sit at the current observer position
        and look along the current boresight.
        """

        # Extract the name of the observer whose POV we want.
        obs_name = self.observer_for_visibility

        # Get the current observer ECI position at sim_idx.
        r_obs_now = self.obs_r_eci_km[obs_name][self.sim_idx, :]

        # If we do not have boresight basis cached yet, just return.
        if self._last_u_bore is None or self._last_u_up is None:
            return

        # Access the camera object from the plotter.
        cam = self.plotter.camera

        # Set camera position to the observer's current position.
        cam.position = tuple(r_obs_now.tolist())

        # Choose a focal point some distance along the boresight direction.
        cam.focal_point = tuple((r_obs_now + 10.0 * R_EARTH * self._last_u_bore).tolist())

        # Set camera up vector to the local "up" direction of the FOV basis.
        cam.up = tuple(self._last_u_up.tolist())

        # Set camera view angle approximately to sensor horizontal FOV.
        cam.view_angle = EVK4_SENSOR.fov_deg

        # Render the new camera configuration.
        self.plotter.render()

    # ------------------------------------------------------------------ #
    # Point picking setup and callback
    # ------------------------------------------------------------------ #

    def _enable_point_picking(self):
        """
        Enable point picking so user can click on targets and see their info.
        """

        # Define internal callback that forwards picked point info to method.
        def _callback(picked):
            # Call the instance method to handle the picked point.
            self._on_point_picked(picked)

        # Enable point picking on the plotter with our callback.
        self.plotter.enable_point_picking(
            callback=_callback,
            use_mesh=True,
            show_message=True,
            show_point=True,
        )

    def _on_point_picked(self, picked_point):
        """
        Handle a point pick event from PyVista.

        picked_point : (x,y,z) coordinates of the clicked location.
        """

        # If there is no target_cloud yet, do nothing.
        if not hasattr(self, "target_cloud"):
            return

        # Convert picked_point to a NumPy array for numeric operations.
        picked_point = np.asarray(picked_point, dtype=float)

        # Find the index of the closest point in the target_cloud to this coordinate.
        point_idx = self.target_cloud.find_closest_point(picked_point)

        # If we have no target_index field, we cannot map this to a target.
        if "target_index" not in self.target_cloud.array_names:
            self.log.warning("Picked a point but target_index scalar is missing.")
            return

        # Get the integer target index from the target_index scalar array.
        target_idx = int(self.target_cloud["target_index"][point_idx])

        # If the index is out of range for valid_targets, return.
        if target_idx < 0 or target_idx >= len(self.valid_targets):
            self.log.warning("Picked target_idx=%d out of valid_targets range.", target_idx)
            return

        # Extract the target dictionary for this index.
        t = self.valid_targets[target_idx]

        # Get the target name.
        tar_name = t["name"]

        # Extract the target's ECI position array.
        r_tar_all = t["r_eci"]

        # Get the target ECI position at the current timestep.
        r_tar_now = r_tar_all[self.sim_idx, :]

        # Get current observer name used for visibility.
        obs_name = self.observer_for_visibility

        # Get the observer ECI position at current timestep.
        r_obs_now = self.obs_r_eci_km[obs_name][self.sim_idx, :]

        # Compute the range vector from observer to target.
        dr = r_tar_now - r_obs_now

        # Compute the Euclidean distance (range) in km.
        range_km = float(np.linalg.norm(dr))

        # Look up the full track dict for this target from tar_tracks.
        tar_track = self.tar_tracks[tar_name]

        # Get per-observer dictionary for this target and observer.
        by_obs = tar_track.get("by_observer", {}).get(obs_name, {})

        # Helper function to extract a scalar at sim_idx from an array field.
        def _scalar_from_array(field: Optional[Any]) -> Optional[Any]:
            # If field is missing, return None.
            if field is None:
                return None
            # Convert to NumPy array.
            arr = np.asarray(field)
            # If length does not match n_times, return None.
            if arr.shape[0] != self.n_times:
                return None
            # Return the entry at current sim_idx.
            return arr[self.sim_idx]

        # Extract LOS visible flag at current timestep.
        los_visible = _scalar_from_array(by_obs.get("los_visible"))

        # Extract sunlit flag at current timestep.
        illum_sun = _scalar_from_array(by_obs.get("illum_is_sunlit"))

        # Extract on_detector flag at current timestep.
        on_det = _scalar_from_array(by_obs.get("on_detector"))

        # Extract combined on_detector_visible_sunlit mask at current timestep.
        on_det_vis_sun = _scalar_from_array(by_obs.get("on_detector_visible_sunlit"))

        # Extract pixel x coordinate at current timestep.
        pix_x = _scalar_from_array(by_obs.get("pix_x"))

        # Extract pixel y coordinate at current timestep.
        pix_y = _scalar_from_array(by_obs.get("pix_y"))

        # Extract phase angle (rad) from by_obs if available, else from target track.
        phase_arr_by_obs = by_obs.get("illum_phase_angle_rad")
        phase_arr_global = tar_track.get("illum_phase_angle_rad", None)
        # Prefer per-observer array if it has correct shape.
        if phase_arr_by_obs is not None and np.asarray(phase_arr_by_obs).shape[0] == self.n_times:
            phase_rad = float(np.asarray(phase_arr_by_obs)[self.sim_idx])
        elif phase_arr_global is not None and np.asarray(phase_arr_global).shape[0] == self.n_times:
            phase_rad = float(np.asarray(phase_arr_global)[self.sim_idx])
        else:
            phase_rad = None

        # If we got a valid phase, convert to degrees; else keep None.
        if phase_rad is not None:
            phase_deg = float(np.rad2deg(phase_rad))
        else:
            phase_deg = None

        # Classify the phase angle into simple descriptive categories.
        if phase_deg is None:
            phase_desc = "N/A"
        elif phase_deg < 45.0:
            phase_desc = "front-lit"
        elif phase_deg < 90.0:
            phase_desc = "gibbous"
        elif phase_deg < 135.0:
            phase_desc = "quarter"
        elif phase_deg < 170.0:
            phase_desc = "crescent"
        else:
            phase_desc = "back-lit"

        # Extract radiometric fields from by_obs or, if needed, from tar_track.
        flux_w_m2 = None
        photon_flux = None
        app_mag_g = None

        # Try to get flux (W m^-2) from by_obs first.
        if "rad_flux_g_w_m2" in by_obs:
            val = _scalar_from_array(by_obs["rad_flux_g_w_m2"])
            if val is not None:
                flux_w_m2 = float(val)
        # If still None, try global tar_track.
        if flux_w_m2 is None and "rad_flux_g_w_m2" in tar_track:
            val = _scalar_from_array(tar_track["rad_flux_g_w_m2"])
            if val is not None:
                flux_w_m2 = float(val)

        # Try to get photon flux (photons m^-2 s^-1).
        if "rad_photon_flux_g_m2_s" in by_obs:
            val = _scalar_from_array(by_obs["rad_photon_flux_g_m2_s"])
            if val is not None:
                photon_flux = float(val)
        if photon_flux is None and "rad_photon_flux_g_m2_s" in tar_track:
            val = _scalar_from_array(tar_track["rad_photon_flux_g_m2_s"])
            if val is not None:
                photon_flux = float(val)

        # Try to get apparent magnitude in Gaia G band.
        if "rad_app_mag_g" in by_obs:
            val = _scalar_from_array(by_obs["rad_app_mag_g"])
            if val is not None:
                app_mag_g = float(val)
        if app_mag_g is None and "rad_app_mag_g" in tar_track:
            val = _scalar_from_array(tar_track["rad_app_mag_g"])
            if val is not None:
                app_mag_g = float(val)

        # Build a multi-line info string summarizing the picked target state.
        lines = []
        lines.append(f"Target: {tar_name}")
        lines.append(f"Observer: {obs_name}")
        lines.append(f"Time index: {self.sim_idx} / {self.n_times - 1}")
        lines.append(f"Range: {range_km:,.1f} km")
        lines.append(f"LOS visible: {los_visible}")
        lines.append(f"Sunlit: {illum_sun}")
        lines.append(f"On detector: {on_det}")
        lines.append(f"On detector & visible & sunlit: {on_det_vis_sun}")
        lines.append(f"Pixel (x, y): ({pix_x}, {pix_y})")
        lines.append(f"Phase angle (deg): {phase_deg} ({phase_desc})")
        lines.append(f"Flux (W m^-2): {flux_w_m2}")
        lines.append(f"Photon flux (photons m^-2 s^-1): {photon_flux}")
        lines.append(f"Apparent mag (G): {app_mag_g}")

        # Join all lines into one string separated by newlines.
        info_str = "\n".join(lines)

        # Print the information string to the console.
        print("\n=== Picked Target Info ===")
        print(info_str)
        print("==========================\n")

        # Also log the first line at INFO level for quick reference.
        self.log.info("Picked target '%s' at sim_idx=%d.", tar_name, self.sim_idx)

    # ------------------------------------------------------------------ #
    # Core scene update (used by timer and slider)
    # ------------------------------------------------------------------ #

    def update_scene(self, sim_idx: int, from_slider: bool = False):
        """
        Update dynamic actors to a specific timestep (sim_idx).

        If from_slider=True, do not update the slider again (avoid feedback loop).
        """

        # Clamp sim_idx to the valid range [0, n_times-1].
        if sim_idx < 0 or sim_idx >= self.n_times:
            sim_idx = max(0, min(sim_idx, self.n_times - 1))

        # Store the clamped index on the instance.
        self.sim_idx = sim_idx

        # Rebuild dynamic actors (targets, observer, FOV) for this timestep.
        self._build_dynamic_actors_for_sim_idx(self.sim_idx)

        # If we are not responding to the slider, push the new value to the slider.
        if not from_slider:
            # Temporarily block slider signals to avoid recursive calls.
            self.slider.blockSignals(True)
            # Set slider to the new sim_idx.
            self.slider.setValue(self.sim_idx)
            # Re-enable slider signals.
            self.slider.blockSignals(False)

        # Refresh text overlay to show updated frame and state.
        self._update_text()

        # Render the updated scene.
        self.plotter.render()

    # ------------------------------------------------------------------ #
    # Qt / animation callbacks
    # ------------------------------------------------------------------ #

    def _on_timer(self):
        """
        Called by QTimer at regular intervals; advances the animation if playing.
        """

        # If playing flag is False, do nothing on this tick.
        if not self.playing:
            return

        # Compute the next frame index (wrap around at n_times).
        next_idx = (self.sim_idx + 1) % self.n_times

        # Update the scene (not from slider).
        self.update_scene(next_idx, from_slider=False)

    def _on_slider_changed(self, value: int):
        """
        Called when the time slider is moved by the user.
        """

        # When user scrubs with the slider, pause the animation.
        self.playing = False

        # Block signals so changing the action's checked state doesn't re-trigger.
        self.play_action.blockSignals(True)
        # Uncheck the Play action to indicate we are paused.
        self.play_action.setChecked(False)
        # Re-enable signals for the Play action.
        self.play_action.blockSignals(False)

        # Update the scene for the new slider value (from_slider=True).
        self.update_scene(int(value), from_slider=True)

    def _on_play_toggled(self, checked: bool):
        """
        Called when the Play toolbar action is toggled.
        """

        # Update the playing flag based on the checkbox state.
        self.playing = bool(checked)

        # Refresh the overlay text to show "Playing" or "Paused".
        self._update_text()

    def _on_fov_toggled(self, checked: bool):
        """
        Called when the FOV visibility action is toggled.
        """

        # If we have a frustum actor, set its visibility based on checked state.
        if hasattr(self, "frustum_actor"):
            self.frustum_actor.SetVisibility(1 if checked else 0)

            # Render the scene to show/hide the frustum.
            self.plotter.render()

    def _on_orbit_toggled(self, checked: bool):
        """
        Called when the Orbits visibility action is toggled.
        """

        # Map checked flag to 1 (visible) or 0 (hidden).
        vis = 1 if checked else 0

        # Loop over orbit actors and set their visibility.
        for actor in self.orbit_actors:
            actor.SetVisibility(vis)

        # Render the scene to apply visibility changes.
        self.plotter.render()

    def _on_cam_toggled(self, checked: bool):
        """
        Called when the Observer Camera action is toggled.
        """

        # If user turns on observer camera mode:
        if checked:
            # Set the internal flag.
            self.use_observer_camera = True
            # Apply observer camera settings at current timestep.
            self._apply_observer_camera()
        else:
            # If user turns it off, switch back to global camera.
            self.use_observer_camera = False
            # Restore the saved global camera state.
            self._restore_global_camera_state()

        # Refresh the scene text.
        self._update_text()

    def _on_observer_changed(self, new_name: str):
        """
        Called when the observer dropdown selection changes.
        """

        # If the new observer name is not known, do nothing.
        if new_name not in self.observer_names:
            return

        # Update the observer_for_visibility to the new selection.
        self.observer_for_visibility = new_name

        # Log that we switched observers for visibility perspective.
        self.log.info(
            "Switched observer_for_visibility to '%s'. Rebuilding target masks.",
            new_name,
        )

        # Rebuild valid_targets for the new observer (with tqdm).
        self._build_valid_targets_for_observer()

        # Rebuild dynamic actors at current timestep to reflect new masks.
        self._build_dynamic_actors_for_sim_idx(self.sim_idx)

        # If observer camera is currently enabled, re-apply camera transform.
        if self.use_observer_camera:
            self._apply_observer_camera()

        # Refresh overlay text and render.
        self._update_text()
        self.plotter.render()


# ============================================================================
# Entry-point convenience function
# ============================================================================

def run_pyvista_viewer(
    force_recompute_pixels: bool = False,
    observer_for_visibility: str = "SBSS (USA 216)",
    sim_idx_start: int = 0,
    frame_interval_ms: int = 50,
) -> NebulaPyVistaViewer:
    """
    Convenience function to construct and return a NebulaPyVistaViewer instance.
    """

    # Create and return the viewer instance with given options.
    viewer = NebulaPyVistaViewer(
        force_recompute_pixels=force_recompute_pixels,
        observer_for_visibility=observer_for_visibility,
        sim_idx_start=sim_idx_start,
        frame_interval_ms=frame_interval_ms,
    )
    return viewer


# ============================================================================
# Script guard
# ============================================================================

if __name__ == "__main__":
    # If run as a script, construct the viewer with typical defaults.
    run_pyvista_viewer(
        force_recompute_pixels=False,
        observer_for_visibility="SBSS (USA 216)",  # or "SAPPHIRE", "NEOSSAT"
        sim_idx_start=0,
        frame_interval_ms=50,  # ~20 fps
    )
