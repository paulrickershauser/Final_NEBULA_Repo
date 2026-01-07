"""
NEBULA_PYVISTA_SCENE
====================

PyVista scene construction helpers for the NEBULA simulation viewer.

This module is responsible for taking a *viewer dataset* (as produced by
:mod:`NEBULA_PYVISTA_DATA`) and turning it into a PyVista 3-D scene with:

    * An Earth sphere at the origin.
    * Orbit polylines for all observers and targets.
    * Point markers for each observer and target that can be animated
      over time.
    * Optional line-of-sight (LOS) line segments for each
      (observer, target) pair with visibility information.

The module does **not** load pickles or compute LOS/illumination/flux
itself; it only consumes the already-prepared dataset.

Key entry points
----------------

1. :func:`create_scene`

   Build the initial PyVista scene and return a :class:`SceneState`
   object that holds all actor handles and the underlying plotter.

2. :func:`update_scene_time_index`

   Given a :class:`SceneState`, a time index, and the name of the
   active observer, update all marker positions and visual
   highlighting to reflect current visibility.

Typical usage (from a higher-level script)
-----------------------------------------

>>> from Simulation_Viewer import NEBULA_PYVISTA_DATA, NEBULA_PYVISTA_SCENE
>>> from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER
>>>
>>> obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
...     force_recompute=False
... )
>>> dataset = NEBULA_PYVISTA_DATA.build_viewer_dataset(
...     observer_tracks=obs_tracks,
...     target_tracks=tar_tracks,
...     visibility_field="los_visible",
... )
>>>
>>> scene = NEBULA_PYVISTA_SCENE.create_scene(dataset)
>>> plotter = scene.plotter
>>>
>>> # Example: update to a given timestep and active observer
>>> NEBULA_PYVISTA_SCENE.update_scene_time_index(
...     scene_state=scene,
...     index=0,
...     active_observer=dataset["observer_names"][0],
... )
>>>
>>> # Add slider / key callbacks in a separate viewer module, then:
>>> plotter.show()

The goal is to keep this module focused solely on *scene management*,
leaving widget logic and UI controls to a higher-level script.
"""

from __future__ import annotations

# Standard-library imports
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Logger helper
# ---------------------------------------------------------------------------

def _build_default_logger() -> logging.Logger:
    """
    Build a simple console logger for this module.

    This is used whenever the caller does not provide a logger explicitly.
    """
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Visual style configuration
# ---------------------------------------------------------------------------

# Earth
DEFAULT_EARTH_RADIUS_KM: float = 6378.0

# Colors (RGB tuples in 0–1 range or color names understood by PyVista)
COLOR_EARTH: Any = "lightblue"
COLOR_OBSERVER_ORBIT: Any = "white"
COLOR_TARGET_ORBIT: Any = "gray"
COLOR_OBSERVER_MARKER: Any = "yellow"
COLOR_TARGET_INVISIBLE: Any = (0.5, 0.5, 0.5)   # dim grey
COLOR_TARGET_VISIBLE: Any = (0.0, 1.0, 0.0)     # bright green
COLOR_LOS_LINE: Any = (0.8, 0.8, 0.2)          # yellowish

# Point sizes
POINT_SIZE_OBSERVER: float = 14.0
POINT_SIZE_OBSERVER_ACTIVE: float = 18.0
POINT_SIZE_TARGET: float = 10.0

# Line width for orbits and LOS
ORBIT_LINE_WIDTH: float = 1.0
LOS_LINE_WIDTH: float = 1.5


# ---------------------------------------------------------------------------
# Scene state dataclass
# ---------------------------------------------------------------------------

@dataclass
class SceneState:
    """
    Container for all PyVista actors and data needed by the viewer.

    Attributes
    ----------
    plotter : pv.Plotter
        The PyVista plotter containing the scene.
    dataset : dict
        The viewer dataset as returned by NEBULA_PYVISTA_DATA.build_viewer_dataset.
    earth_actor : Any
        Actor handle for the Earth mesh.
    observer_orbit_actors : dict
        Mapping observer_name → orbit actor.
    target_orbit_actors : dict
        Mapping target_name → orbit actor.
    observer_point_sources : dict
        Mapping observer_name → pv.PolyData with a single point.  The
        point coordinates are updated on each timestep.
    target_point_sources : dict
        Mapping target_name → pv.PolyData with a single point.
    observer_point_actors : dict
        Mapping observer_name → actor handle for the observer marker.
    target_point_actors : dict
        Mapping target_name → actor handle for the target marker.
    los_line_sources : dict
        Mapping (observer_name, target_name) → pv.PolyData representing
        a 2-point line between observer and target.  The endpoints are
        updated on each timestep.
    los_line_actors : dict
        Mapping (observer_name, target_name) → actor handle for the LOS line.
    current_index : int
        Index of the current timestep (for convenience / external state).
    """
    plotter: pv.Plotter
    dataset: Dict[str, Any]

    earth_actor: Any

    observer_orbit_actors: Dict[str, Any] = field(default_factory=dict)
    target_orbit_actors: Dict[str, Any] = field(default_factory=dict)

    observer_point_sources: Dict[str, pv.PolyData] = field(default_factory=dict)
    target_point_sources: Dict[str, pv.PolyData] = field(default_factory=dict)

    observer_point_actors: Dict[str, Any] = field(default_factory=dict)
    target_point_actors: Dict[str, Any] = field(default_factory=dict)

    los_line_sources: Dict[Tuple[str, str], pv.PolyData] = field(default_factory=dict)
    los_line_actors: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    # per-observer LOS color
    observer_color_map: Dict[str, Any] = field(default_factory=dict)
    current_index: int = 0


# ---------------------------------------------------------------------------
# Helper functions to construct meshes / actors
# ---------------------------------------------------------------------------
def _build_observer_color_map(observer_names) -> Dict[str, Any]:
    """
    Assign a distinct color to each observer.

    Colors repeat if there are more observers than palette entries.
    """
    palette = [
        (1.0, 1.0, 0.0),  # yellow
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 0.0, 1.0),  # magenta
        (1.0, 0.5, 0.0),  # orange
        (0.0, 1.0, 0.0),  # green
        (1.0, 0.0, 0.0),  # red
    ]
    color_map: Dict[str, Any] = {}
    for i, name in enumerate(observer_names):
        color_map[name] = palette[i % len(palette)]
    return color_map

def _add_earth_sphere(
    plotter: pv.Plotter,
    radius_km: float = DEFAULT_EARTH_RADIUS_KM,
    color: Any = COLOR_EARTH,
    texture_path: Optional[str] = None,
) -> Any:
    """
    Add a simple Earth sphere at the origin to the plotter.

    If ``texture_path`` is provided, the sphere will be textured with
    that image (e.g., a Blue Marble Earth map).  Otherwise a solid
    color sphere is used.
    """
    sphere = pv.Sphere(
        radius=radius_km,
        center=(0.0, 0.0, 0.0),
        theta_resolution=60,
        phi_resolution=60,
    )

    # If a texture path is provided, generate spherical texture coordinates
    # so that the 2-D Earth map can be wrapped around the sphere.
    if texture_path is not None:
        # Create texture coordinates (u, v) on the sphere surface.
        # We use inplace=False so we get a new mesh with tcoords.
        sphere = sphere.texture_map_to_sphere(inplace=False)
        tex = pv.read_texture(texture_path)
        actor = plotter.add_mesh(sphere, texture=tex)
    else:
        actor = plotter.add_mesh(sphere, color=color, smooth_shading=True)

    return actor




def _add_orbit_polyline(
    plotter: pv.Plotter,
    positions_km: np.ndarray,
    color: Any,
    line_width: float,
) -> Any:
    """
    Add an orbit polyline (curve through all positions) to the plotter.

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter to which the orbit will be added.
    positions_km : (N,3) array
        ECI positions in kilometres along the orbit.
    color : Any
        Orbit line color.
    line_width : float
        Line width in pixels.

    Returns
    -------
    actor : Any
        Actor handle for the orbit line.
    """
    # Use PyVista helper to build a polyline from ordered points.
    line_mesh = pv.lines_from_points(positions_km)
    actor = plotter.add_mesh(line_mesh, color=color, line_width=line_width)
    return actor


def _add_point_marker(
    plotter: pv.Plotter,
    initial_position_km: np.ndarray,
    color: Any,
    point_size: float,
) -> Tuple[pv.PolyData, Any]:
    """
    Add a single-point marker to the plotter.

    The marker is represented as a one-point :class:`pv.PolyData` and
    rendered as a sphere-like point via ``render_points_as_spheres``.

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter to which the marker will be added.
    initial_position_km : (3,) array-like
        Initial XYZ position in kilometres.
    color : Any
        Point color.
    point_size : float
        Point size in pixels.

    Returns
    -------
    point_source : pv.PolyData
        The underlying point cloud with a single point.
    actor : Any
        The actor handle for the marker.
    """
    # Ensure we have a (1,3) array for PolyData.
    p = np.asarray(initial_position_km, dtype=float).reshape(1, 3)

    point_source = pv.PolyData(p)
    actor = plotter.add_mesh(
        point_source,
        render_points_as_spheres=True,
        point_size=point_size,
        color=color,
    )
    return point_source, actor


def _add_los_line(
    plotter: pv.Plotter,
    obs_position_km: np.ndarray,
    tar_position_km: np.ndarray,
    color: Any = COLOR_LOS_LINE,
    line_width: float = LOS_LINE_WIDTH,
) -> Tuple[pv.PolyData, Any]:
    """
    Add a 2-point LOS line between observer and target.

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter to which the line will be added.
    obs_position_km : (3,) array-like
        Observer position in km.
    tar_position_km : (3,) array-like
        Target position in km.
    color : Any, optional
        Line color.
    line_width : float, optional
        Line width in pixels.

    Returns
    -------
    line_source : pv.PolyData
        The underlying line mesh with two points.
    actor : Any
        Actor handle for the LOS line.
    """
    pts = np.vstack(
        [
            np.asarray(obs_position_km, dtype=float).reshape(1, 3),
            np.asarray(tar_position_km, dtype=float).reshape(1, 3),
        ]
    )
    line_mesh = pv.lines_from_points(pts)
    actor = plotter.add_mesh(line_mesh, color=color, line_width=line_width)
    return line_mesh, actor


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def create_scene(
    dataset: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    show_orbits: bool = True,
    show_los_lines: bool = True,
    earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM,
    earth_texture_path: Optional[str] = None,
) -> SceneState:
    """
    Create a PyVista scene from a viewer dataset.

    This function:

        * Creates a :class:`pyvista.Plotter`.
        * Adds an Earth sphere.
        * Adds orbit polylines for all observers and targets.
        * Adds point markers for all observers and targets.
        * Optionally adds LOS line actors for every (observer, target)
          pair for which visibility information exists.

    Parameters
    ----------
    dataset : dict
        Viewer dataset produced by :func:`NEBULA_PYVISTA_DATA.build_viewer_dataset`.
        Must contain at least the keys:

            - "observer_names"
            - "target_names"
            - "obs_positions"
            - "tar_positions"
            - "visibility"
    logger : logging.Logger or None, optional
        Logger for status / diagnostic messages.  If None, a simple
        default logger is created.
    show_orbits : bool, optional
        If True (default), draw orbit polylines.
    show_los_lines : bool, optional
        If True (default), create LOS line actors.  These will be
        selectively shown/hidden during updates.
    earth_radius_km : float, optional
        Earth radius for the Earth sphere, in km.

    Returns
    -------
    scene_state : SceneState
        Object containing the plotter and all relevant actor handles.
    """
    if logger is None:
        logger = _build_default_logger()

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.enable_anti_aliasing()
    plotter.show_axes()  # simple axis triad in the corner

    # Extract basics from dataset.
    observer_names = dataset.get("observer_names", [])
    target_names = dataset.get("target_names", [])
    obs_positions = dataset.get("obs_positions", {})
    tar_positions = dataset.get("tar_positions", {})
    visibility = dataset.get("visibility", {})
    n_times = int(dataset.get("n_times", 0))

    if n_times <= 0:
        raise ValueError("create_scene: dataset['n_times'] must be > 0.")
    #build per-observer LOS color map
    observer_color_map = _build_observer_color_map(observer_names)
    
    logger.info(
        "Creating PyVista scene: %d observers, %d targets, %d timesteps.",
        len(observer_names),
        len(target_names),
        n_times,
    )

    # ------------------------------------------------------------------
    # Earth
    # ------------------------------------------------------------------
    earth_actor = _add_earth_sphere(
        plotter=plotter,
        radius_km=earth_radius_km,
        color=COLOR_EARTH,
        texture_path=earth_texture_path,
    )


    # Prepare state container.
    scene_state = SceneState(
        plotter=plotter,
        dataset=dataset,
        earth_actor=earth_actor,
        observer_color_map=observer_color_map, 
    )

    # ------------------------------------------------------------------
    # Orbits and point markers for observers.
    # ------------------------------------------------------------------
    for obs_name in observer_names:
        if obs_name not in obs_positions:
            logger.warning(
                "create_scene: observer '%s' missing in obs_positions; skipping.",
                obs_name,
            )
            continue

        pos = obs_positions[obs_name]  # (N,3)

        if show_orbits:
            orbit_actor = _add_orbit_polyline(
                plotter=plotter,
                positions_km=pos,
                color=COLOR_OBSERVER_ORBIT,
                line_width=ORBIT_LINE_WIDTH,
            )
            scene_state.observer_orbit_actors[obs_name] = orbit_actor

        # Use initial timestep as initial marker position.
        point_source, point_actor = _add_point_marker(
            plotter=plotter,
            initial_position_km=pos[0],
            color=COLOR_OBSERVER_MARKER,
            point_size=POINT_SIZE_OBSERVER,
        )
        scene_state.observer_point_sources[obs_name] = point_source
        scene_state.observer_point_actors[obs_name] = point_actor

    # ------------------------------------------------------------------
    # Orbits and point markers for targets.
    # ------------------------------------------------------------------
    for tar_name in target_names:
        if tar_name not in tar_positions:
            logger.warning(
                "create_scene: target '%s' missing in tar_positions; skipping.",
                tar_name,
            )
            continue

        pos = tar_positions[tar_name]  # (N,3)

        if show_orbits:
            orbit_actor = _add_orbit_polyline(
                plotter=plotter,
                positions_km=pos,
                color=COLOR_TARGET_ORBIT,
                line_width=ORBIT_LINE_WIDTH,
            )
            scene_state.target_orbit_actors[tar_name] = orbit_actor

        point_source, point_actor = _add_point_marker(
            plotter=plotter,
            initial_position_km=pos[0],
            color=COLOR_TARGET_INVISIBLE,  # will be recolored on update
            point_size=POINT_SIZE_TARGET,
        )
        scene_state.target_point_sources[tar_name] = point_source
        scene_state.target_point_actors[tar_name] = point_actor

    # ------------------------------------------------------------------
    # LOS line actors (one per (observer, target) visibility pair).
    # ------------------------------------------------------------------
    if show_los_lines:
        for (obs_name, tar_name), vis_arr in visibility.items():
            # Skip if we don't have positions for either end.
            if obs_name not in obs_positions or tar_name not in tar_positions:
                continue

            # Initial positions at t=0.
            obs_pos0 = obs_positions[obs_name][0]
            tar_pos0 = tar_positions[tar_name][0]

            # Get this observer's LOS color
            los_color = scene_state.observer_color_map.get(obs_name, COLOR_LOS_LINE)
            
            line_source, line_actor = _add_los_line(
                plotter=plotter,
                obs_position_km=obs_pos0,
                tar_position_km=tar_pos0,
                color=los_color,
                line_width=LOS_LINE_WIDTH,
            )

            # Start with LOS lines hidden; update function will toggle.
            line_actor.SetVisibility(False)

            key = (obs_name, tar_name)
            scene_state.los_line_sources[key] = line_source
            scene_state.los_line_actors[key] = line_actor

    # Set an initial camera view that roughly captures Earth and GEO belt.
    plotter.camera_position = "zy"  # side-on view as a simple default

    logger.info("Scene construction complete.")
    return scene_state


# ---------------------------------------------------------------------------
# Time-step update
# ---------------------------------------------------------------------------

def update_scene_time_index(
    scene_state: SceneState,
    index: int,
    active_observer: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Update all marker positions and visibility highlighting for a timestep.

    This function should be called whenever the viewer changes the
    current timestep (e.g., via a slider).  It:

        * Moves all observer and target markers to the positions
          corresponding to ``index``.
        * Colors each target based on whether it is "visible" from the
          selected ``active_observer`` using the dataset's visibility
          index.
        * Shows/hides LOS lines for the active observer only, and only
          when the visibility array is True at that timestep.
        * Optionally emphasizes the active observer by increasing its
          point size.

    Parameters
    ----------
    scene_state : SceneState
        Scene state returned by :func:`create_scene`.
    index : int
        Time index in ``[0, dataset["n_times"]-1]``.
    active_observer : str or None, optional
        Name of the observer that should be treated as "active" for
        visibility highlighting.  If None, the first observer in
        ``dataset["observer_names"]`` is used (if available).
    logger : logging.Logger or None, optional
        Logger for status / diagnostic messages.  If None, a simple
        default logger is created.

    Raises
    ------
    ValueError
        If ``index`` is out of bounds.
    """
    if logger is None:
        logger = _build_default_logger()

    dataset = scene_state.dataset
    n_times = int(dataset.get("n_times", 0))

    if index < 0 or index >= n_times:
        raise ValueError(
            f"update_scene_time_index: index {index} is out of bounds for "
            f"n_times={n_times}."
        )

    observer_names = dataset.get("observer_names", [])
    target_names = dataset.get("target_names", [])
    obs_positions = dataset.get("obs_positions", {})
    tar_positions = dataset.get("tar_positions", {})
    visibility = dataset.get("visibility", {})

    # Choose default active observer if none specified.
    if active_observer is None:
        if observer_names:
            active_observer = observer_names[0]
        else:
            active_observer = ""
    scene_state.current_index = index

    # ------------------------------------------------------------------
    # Update observer marker positions and styling.
    # ------------------------------------------------------------------
    for obs_name in observer_names:
        if obs_name not in obs_positions:
            continue
        if obs_name not in scene_state.observer_point_sources:
            continue

        pos = obs_positions[obs_name][index]  # (3,)
        source = scene_state.observer_point_sources[obs_name]

        # Update the single point in-place.
        pts = source.points
        pts[0, :] = pos
        source.points = pts  # reassign to ensure PyVista notices

        # Adjust styling if this is the active observer.
        actor = scene_state.observer_point_actors.get(obs_name)
        if actor is not None:
            if obs_name == active_observer:
                # Slightly larger point for active observer.
                actor.prop.point_size = POINT_SIZE_OBSERVER_ACTIVE
            else:
                actor.prop.point_size = POINT_SIZE_OBSERVER

    # ------------------------------------------------------------------
    # Update target marker positions and visibility-based color.
    # ------------------------------------------------------------------
    for tar_name in target_names:
        if tar_name not in tar_positions:
            continue
        if tar_name not in scene_state.target_point_sources:
            continue

        pos = tar_positions[tar_name][index]  # (3,)
        source = scene_state.target_point_sources[tar_name]

        pts = source.points
        pts[0, :] = pos
        source.points = pts

        # Determine visibility with respect to the active observer.
        if active_observer:
            key = (active_observer, tar_name)
            vis_arr = visibility.get(key)
            is_visible = bool(vis_arr[index]) if vis_arr is not None else False
        else:
            # No active observer: treat as not visible.
            is_visible = False

        actor = scene_state.target_point_actors.get(tar_name)
        if actor is not None:
            if is_visible:
                actor.prop.color = COLOR_TARGET_VISIBLE
            else:
                actor.prop.color = COLOR_TARGET_INVISIBLE

    # ------------------------------------------------------------------
    # Update LOS lines: only show ones from the active observer.
    # ------------------------------------------------------------------
    for key, line_source in scene_state.los_line_sources.items():
        obs_name, tar_name = key
        actor = scene_state.los_line_actors.get(key)
        if actor is None:
            continue

        # Only consider lines for the active observer.
        if obs_name != active_observer:
            actor.SetVisibility(False)
            continue

        vis_arr = visibility.get(key)
        is_visible = bool(vis_arr[index]) if vis_arr is not None else False

        if not is_visible:
            actor.SetVisibility(False)
            continue

        # Update endpoints if we have positions for both satellites.
        if obs_name not in obs_positions or tar_name not in tar_positions:
            actor.SetVisibility(False)
            continue

        obs_pos = obs_positions[obs_name][index]
        tar_pos = tar_positions[tar_name][index]

        pts = line_source.points
        pts[0, :] = obs_pos
        pts[1, :] = tar_pos
        line_source.points = pts

        actor.SetVisibility(True)

    # Request a render from the plotter (safe even if called frequently).
    scene_state.plotter.render()
