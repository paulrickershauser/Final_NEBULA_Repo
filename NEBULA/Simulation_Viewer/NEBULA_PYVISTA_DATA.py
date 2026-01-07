"""
NEBULA_PYVISTA_DATA
===================

Lightweight data-preparation helpers for the NEBULA PyVista simulation viewer.

This module is intentionally *read-only* with respect to the NEBULA pipeline:
it does not run LOS, illumination, flux, or pixel computation itself.  Instead,
it assumes that upstream picklers (NEBULA_FLUX_PICKLER, NEBULA_LOS_FLUX_PICKLER,
NEBULA_SCHEDULE_PICKLER, NEBULA_ICRS_PAIR_PICKLER, NEBULA_PIXEL_PICKLER, ...)
have already produced rich observer / target track dictionaries and that those
tracks have been loaded into memory (e.g., by calling
NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs()).

The goal of this module is to:

    * Validate and extract a consistent time axis for all tracks.
    * Extract ECI positions for observers and targets in a simple form.
    * Build per-observer / per-target boolean visibility masks suitable
      for driving a 3-D PyVista animation (e.g., LOS-visible flags).
    * Bundle these into a compact "viewer dataset" dictionary that the
      PyVista viewer can consume without needing to know NEBULA's
      internal pickle structure.

Typical usage (from a higher-level script)
-----------------------------------------

>>> from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER
>>> from Simulation_Viewer import NEBULA_PYVISTA_DATA
>>>
>>> # 1) Ensure pixel-augmented tracks exist and load them.
>>> obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
...     force_recompute=False
... )
>>>
>>> # 2) Build a viewer-ready dataset (using a chosen visibility field).
>>> dataset = NEBULA_PYVISTA_DATA.build_viewer_dataset(
...     observer_tracks=obs_tracks,
...     target_tracks=tar_tracks,
...     visibility_field="los_visible",
... )
>>>
>>> # 3) Pass `dataset` into your PyVista scene / animation code.

All functions in this module are pure helpers: they do not mutate the
input track dictionaries.
"""

from __future__ import annotations

# Standard-library imports
import logging
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import numpy as np


# ---------------------------------------------------------------------------
# Logger helper
# ---------------------------------------------------------------------------

def _build_default_logger() -> logging.Logger:
    """
    Build a simple console logger for this module.

    This is used whenever the caller does not provide a logger explicitly.
    """
    logger = logging.getLogger(__name__)

    # Only attach a handler if none are present yet.  This prevents
    # duplicate messages when the module is imported multiple times.
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Default level can be overridden by user code.
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Core helpers: time axis, positions, visibility
# ---------------------------------------------------------------------------

def extract_common_time_axis(
    observer_tracks: Dict[str, Dict[str, Any]],
    target_tracks: Dict[str, Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Extract and validate a common time axis for all observer and target tracks.

    This helper assumes that each track dictionary contains a ``"times"`` entry
    storing the per-timestep epochs (typically as a list of ``datetime``
    objects).  It verifies that *all* tracks share the same time grid and
    returns that grid as a NumPy array.

    Parameters
    ----------
    observer_tracks : dict
        Dictionary mapping observer name → observer track dictionary, as
        returned by the upstream NEBULA picklers.
    target_tracks : dict
        Dictionary mapping target name → target track dictionary.
    logger : logging.Logger or None, optional
        Logger for status / diagnostic messages.  If None, a simple
        default logger is created.

    Returns
    -------
    times : np.ndarray
        One-dimensional NumPy array of length N containing the common
        time axis.  The dtype is usually ``object`` because the entries
        are ``datetime`` instances, but this function does not assume a
        specific dtype.

    Raises
    ------
    KeyError
        If any track is missing the ``"times"`` key.
    ValueError
        If one or more tracks have a time axis that does not match the
        reference time axis (different length or different values).
    """
    if logger is None:
        logger = _build_default_logger()

    if not observer_tracks:
        raise ValueError(
            "extract_common_time_axis: observer_tracks is empty; "
            "at least one observer is required to define a reference time axis."
        )

    # ------------------------------------------------------------------
    # Step 1: choose a reference time axis from the first observer.
    # ------------------------------------------------------------------
    first_obs_name = next(iter(observer_tracks.keys()))
    first_obs_track = observer_tracks[first_obs_name]

    if "times" not in first_obs_track:
        raise KeyError(
            f"extract_common_time_axis: observer '{first_obs_name}' "
            "is missing the 'times' field."
        )

    reference_times = np.asarray(first_obs_track["times"])
    n_times = reference_times.shape[0]

    logger.info(
        "Using observer '%s' as reference time axis with %d steps.",
        first_obs_name,
        n_times,
    )

    # ------------------------------------------------------------------
    # Step 2: verify that all other observers share this time axis.
    # ------------------------------------------------------------------
    for obs_name, obs_track in observer_tracks.items():
        if obs_name == first_obs_name:
            continue

        if "times" not in obs_track:
            raise KeyError(
                f"extract_common_time_axis: observer '{obs_name}' "
                "is missing the 'times' field."
            )

        times = np.asarray(obs_track["times"])

        if times.shape[0] != n_times:
            raise ValueError(
                f"Observer '{obs_name}' has {times.shape[0]} timesteps "
                f"but reference observer '{first_obs_name}' has {n_times}."
            )

        if not np.array_equal(times, reference_times):
            raise ValueError(
                "Observer '{}' has a 'times' array that does not match the "
                "reference defined by '{}'.  All tracks must share the same "
                "time grid for the viewer to work.".format(
                    obs_name, first_obs_name
                )
            )

    # ------------------------------------------------------------------
    # Step 3: verify that all target tracks share this time axis as well.
    # ------------------------------------------------------------------
    for tar_name, tar_track in target_tracks.items():
        if "times" not in tar_track:
            raise KeyError(
                f"extract_common_time_axis: target '{tar_name}' "
                "is missing the 'times' field."
            )

        times = np.asarray(tar_track["times"])

        if times.shape[0] != n_times:
            raise ValueError(
                f"Target '{tar_name}' has {times.shape[0]} timesteps "
                f"but reference observer '{first_obs_name}' has {n_times}."
            )

        if not np.array_equal(times, reference_times):
            raise ValueError(
                "Target '{}' has a 'times' array that does not match the "
                "reference defined by '{}'.  All tracks must share the same "
                "time grid for the viewer to work.".format(
                    tar_name, first_obs_name
                )
            )

    # Return a copy so downstream code cannot mutate the reference.
    return reference_times.copy()


def extract_positions(
    observer_tracks: Dict[str, Dict[str, Any]],
    target_tracks: Dict[str, Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract ECI positions for all observers and targets.

    This helper looks for the ``"r_eci_km"`` field on each track, which
    is expected to be an ``(N, 3)`` NumPy array of ECI/TEME positions
    in kilometres.  It returns two dictionaries mapping name → array.

    Parameters
    ----------
    observer_tracks : dict
        Dictionary mapping observer name → observer track dict.
    target_tracks : dict
        Dictionary mapping target name → target track dict.
    logger : logging.Logger or None, optional
        Logger for status / diagnostic messages.  If None, a simple
        default logger is created.

    Returns
    -------
    obs_positions : dict
        Dictionary mapping observer name → ``(N, 3)`` NumPy array of
        positions in km.
    tar_positions : dict
        Dictionary mapping target name → ``(N, 3)`` NumPy array of
        positions in km.

    Raises
    ------
    KeyError
        If any track is missing the ``"r_eci_km"`` field.
    ValueError
        If any position array does not have shape ``(N, 3)``.
    """
    if logger is None:
        logger = _build_default_logger()

    obs_positions: Dict[str, np.ndarray] = {}
    tar_positions: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Step 1: extract observer positions.
    # ------------------------------------------------------------------
    for obs_name, obs_track in observer_tracks.items():
        if "r_eci_km" not in obs_track:
            raise KeyError(
                f"extract_positions: observer '{obs_name}' is missing 'r_eci_km'."
            )

        r = np.asarray(obs_track["r_eci_km"])

        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(
                f"extract_positions: observer '{obs_name}' has 'r_eci_km' "
                f"with shape {r.shape!r}; expected (N, 3)."
            )

        obs_positions[obs_name] = r

    # ------------------------------------------------------------------
    # Step 2: extract target positions.
    # ------------------------------------------------------------------
    for tar_name, tar_track in target_tracks.items():
        if "r_eci_km" not in tar_track:
            raise KeyError(
                f"extract_positions: target '{tar_name}' is missing 'r_eci_km'."
            )

        r = np.asarray(tar_track["r_eci_km"])

        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(
                f"extract_positions: target '{tar_name}' has 'r_eci_km' "
                f"with shape {r.shape!r}; expected (N, 3)."
            )

        tar_positions[tar_name] = r

    logger.info(
        "Extracted ECI positions for %d observers and %d targets.",
        len(obs_positions),
        len(tar_positions),
    )

    return obs_positions, tar_positions


def build_visibility_index(
    target_tracks: Dict[str, Dict[str, Any]],
    visibility_field: str = "on_detector_visible_sunlit",
    logger: Optional[logging.Logger] = None,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build a per-observer / per-target visibility index for the viewer.

    This helper inspects each target track, looks for a per-observer
    container under ``target['by_observer']``, and, for each observer
    name, extracts a time-series boolean/0-1 array from the specified
    ``visibility_field``.

    The result is a flat dictionary keyed by ``(observer_name, target_name)``,
    which maps to a NumPy array of length N.  This is convenient for
    quickly answering questions like:

        "At timestep k, which targets are visible from observer O?"

    or, equivalently, for iterating over all targets when updating the
    PyVista scene.

    Parameters
    ----------
    target_tracks : dict
        Dictionary mapping target name → target track dict.  Each target
        is expected to follow the NEBULA multi-observer convention:

            target['by_observer'][obs_name][visibility_field]

    visibility_field : str, optional
        Name of the per-observer field to read.  Typical choices are:

            - "los_visible"
            - "pix_on_detector"             (pixel projection inside FOV)
            - "on_detector_and_visible"     (inside FOV *and* LOS-visible)
            - "on_detector_visible_sunlit"  (inside FOV, LOS-visible,
                                            and illuminated / sunlit)

        The default is "los_visible".

    logger : logging.Logger or None, optional
        Logger for status / diagnostic messages.  If None, a simple
        default logger is created.

    Returns
    -------
    visibility : dict
        Dictionary where the keys are ``(observer_name, target_name)``
        tuples and the values are 1-D NumPy arrays of length N with
        boolean or 0/1 entries.

    Notes
    -----
    - Targets that do not have a ``'by_observer'`` container are
      silently skipped.
    - Observers that do not have the requested ``visibility_field`` are
      also skipped for that target.  This allows you to request more
      specialised fields (like "on_detector_visible_sunlit") even when
      some targets may only have basic LOS flags.
    """
    if logger is None:
        logger = _build_default_logger()

    visibility: Dict[Tuple[str, str], np.ndarray] = {}
    n_pairs = 0

    # ------------------------------------------------------------------
    # Loop over all targets and their per-observer containers.
    # ------------------------------------------------------------------
    for tar_name, tar_track in target_tracks.items():
        if "by_observer" not in tar_track:
            logger.debug(
                "build_visibility_index: target '%s' has no 'by_observer' container; skipping.",
                tar_name,
            )
            continue

        by_obs = tar_track["by_observer"]

        if not isinstance(by_obs, dict):
            logger.warning(
                "build_visibility_index: target '%s' has 'by_observer' that is not a dict; skipping.",
                tar_name,
            )
            continue

        for obs_name, per_obs in by_obs.items():
            if not isinstance(per_obs, dict):
                logger.warning(
                    "build_visibility_index: target '%s', observer '%s' entry is not a dict; skipping.",
                    tar_name,
                    obs_name,
                )
                continue

            if visibility_field not in per_obs:
                logger.debug(
                    "build_visibility_index: target '%s', observer '%s' "
                    "does not have field '%s'; skipping.",
                    tar_name,
                    obs_name,
                    visibility_field,
                )
                continue

            arr = np.asarray(per_obs[visibility_field])

            if arr.ndim != 1:
                raise ValueError(
                    "build_visibility_index: target '{}', observer '{}' "
                    "field '{}' has shape {}; expected a 1-D array "
                    "(N,).".format(tar_name, obs_name, visibility_field, arr.shape)
                )

            # Convert to boolean for viewer consumption.  This works
            # for both bool and 0/1 numeric arrays.
            arr_bool = arr.astype(bool)

            visibility[(obs_name, tar_name)] = arr_bool
            n_pairs += 1

    logger.info(
        "build_visibility_index: collected visibility for %d observer–target pairs "
        "(field '%s').",
        n_pairs,
        visibility_field,
    )

    return visibility


# ---------------------------------------------------------------------------
# High-level helper: build full viewer dataset
# ---------------------------------------------------------------------------

def build_viewer_dataset(
    observer_tracks: Dict[str, Dict[str, Any]],
    target_tracks: Dict[str, Dict[str, Any]],
    visibility_field: str = "los_visible",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Construct a compact dataset dictionary for the PyVista viewer.

    This convenience function ties together the lower-level helpers to
    produce a single dictionary with all information the viewer needs:

        - Common time axis      (``dataset["times"]``)
        - Observer positions    (``dataset["obs_positions"]``)
        - Target positions      (``dataset["tar_positions"]``)
        - Visibility index      (``dataset["visibility"]``)
        - Name lists            (``dataset["observer_names"]``,
                                 ``dataset["target_names"]``)

    The original ``observer_tracks`` and ``target_tracks`` dictionaries
    are also included verbatim for any additional metadata that the
    viewer might want to inspect (COEs, flux values, etc.).

    Parameters
    ----------
    observer_tracks : dict
        Dictionary mapping observer name → observer track dict.
    target_tracks : dict
        Dictionary mapping target name → target track dict.
    visibility_field : str, optional
        Name of the per-observer field to use when building the
        visibility index.  See :func:`build_visibility_index` for
        common choices.  Defaults to ``"los_visible"``.
    logger : logging.Logger or None, optional
        Logger for status / diagnostic messages.  If None, a simple
        default logger is created.

    Returns
    -------
    dataset : dict
        A dictionary with at least the following keys:

            - "times"           : (N,) array of epochs
            - "n_times"         : int, number of timesteps
            - "observer_names"  : list of observer names (sorted)
            - "target_names"    : list of target names (sorted)
            - "obs_positions"   : dict name → (N,3) array (km)
            - "tar_positions"   : dict name → (N,3) array (km)
            - "visibility"      : dict (obs_name, tar_name) → (N,) bool array
            - "observer_tracks" : original observer_tracks dict
            - "target_tracks"   : original target_tracks dict
            - "visibility_field": the field name used to build the index

        Additional entries may be added in the future as the viewer
        evolves (e.g., precomputed per-time target lists, etc.).
    """
    if logger is None:
        logger = _build_default_logger()

    logger.info("Building viewer dataset with visibility field '%s'.", visibility_field)

    # 1) Validate and extract common time axis.
    times = extract_common_time_axis(observer_tracks, target_tracks, logger=logger)
    n_times = times.shape[0]

    # 2) Extract positions.
    obs_positions, tar_positions = extract_positions(
        observer_tracks, target_tracks, logger=logger
    )

    # 3) Build visibility index.
    visibility = build_visibility_index(
        target_tracks, visibility_field=visibility_field, logger=logger
    )

    # 4) Sanity-check that all visibility arrays match n_times.
    for (obs_name, tar_name), arr in visibility.items():
        if arr.shape[0] != n_times:
            raise ValueError(
                "build_viewer_dataset: visibility array for observer '{}', "
                "target '{}' has length {}; expected {} (from time axis).".format(
                    obs_name, tar_name, arr.shape[0], n_times
                )
            )

    # 5) Build sorted name lists for convenience.
    observer_names = sorted(list(observer_tracks.keys()))
    target_names = sorted(list(target_tracks.keys()))

    logger.info(
        "Viewer dataset summary: %d observers, %d targets, %d timesteps, "
        "%d visibility pairs.",
        len(observer_names),
        len(target_names),
        n_times,
        len(visibility),
    )

    dataset: Dict[str, Any] = {
        "times": times,
        "n_times": n_times,
        "observer_names": observer_names,
        "target_names": target_names,
        "obs_positions": obs_positions,
        "tar_positions": tar_positions,
        "visibility": visibility,
        "observer_tracks": observer_tracks,
        "target_tracks": target_tracks,
        "visibility_field": visibility_field,
    }

    return dataset
