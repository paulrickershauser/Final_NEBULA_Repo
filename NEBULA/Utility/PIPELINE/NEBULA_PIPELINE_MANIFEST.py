"""
NEBULA_PIPELINE_MANIFEST
=======================

Declarative manifest for the NEBULA *upstream* pipeline.

This file is the single source of truth for the pipeline manager regarding:
  - canonical stage → entrypoint (module + function)
  - canonical stage → owned output pickle files (relative to NEBULA_OUTPUT_DIR)
  - stage descriptions (for dry-run plan output)

Key design choices (as confirmed)
---------------------------------
1) Recompute is deletion-driven:
   - The manager invalidates a stage by deleting ONLY that stage's owned pickle
     outputs (and downstream stage outputs), never entire directories.

2) STOP_AFTER_STAGE semantics:
   - If STOP_AFTER_STAGE is a valid stage ID, run only through it (inclusive).
   - If STOP_AFTER_STAGE is invalid/None, run the full chain.

3) Stage ID vs directory name:
   - Canonical stage ID is "LOS_FLUX" (underscore), but the on-disk directory
     in your current picklers is "LOSFLUX_SatPickles" (no underscore).
     The manifest handles this explicitly.

Implementation note
-------------------
We avoid importing the heavy pickler modules at import-time. Instead, we store
entrypoints as module/function strings and load them lazily via importlib when
the manager actually executes a stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import importlib

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Utility.PIPELINE.NEBULA_PIPELINE_STAGES import (
    Stage,
    StageLike,
    STAGE_ORDER,
    STAGE_INDEX,
    parse_stage,
)
from Configuration import NEBULA_STAR_CONFIG as NSC

# -----------------------------------------------------------------------------
# Post-PIXELS output layout helpers
# -----------------------------------------------------------------------------
# Your STARS modules use:
#   NEBULA_OUTPUT/STARS/<catalog_name>/...
# where <catalog_name> is NEBULA_STAR_CATALOG.name (e.g., "GAIA_DR3_G").
#
# We compute the catalog subdir once here so StageSpec owned_outputs_rel
# exactly matches what the STARS modules write.
STAR_CATALOG_NAME: str = getattr(NSC.NEBULA_STAR_CATALOG, "name", "").strip()
if not STAR_CATALOG_NAME:
    raise RuntimeError(
        "NEBULA_PIPELINE_MANIFEST: NEBULA_STAR_CATALOG.name is missing/empty "
        "in NEBULA_STAR_CONFIG. Star-stage owned outputs cannot be resolved."
    )

STAR_DIR_REL: str = f"STARS/{STAR_CATALOG_NAME}"
TARGET_FRAMES_DIR_REL: str = "TARGET_PHOTON_FRAMES"
PHOTON_FRAMES_DIR_REL: str = "PHOTON_FRAMES"
PHOTON_FRAME_CATALOG_FILENAME: str = "obs_photon_frame_catalog.pkl"
SCENE_DIR_REL: str = "SCENE"
WINDOW_SOURCES_FILENAME: str = "obs_window_sources.pkl"
WINDOW_SOURCES_FILENAME: str = "obs_window_sources.pkl"
ZODIACAL_LIGHT_DIR_REL: str = "ZODIACAL_LIGHT"
ZODIACAL_LIGHT_FILENAME: str = "obs_zodiacal_light.pkl"

    
# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EntrypointSpec:
    """
    Location of the callable that executes a stage.

    Attributes
    ----------
    module : str
        Python module path (e.g., "Utility.SAT_OBJECTS.NEBULA_SAT_PICKLER").
    function : str
        Callable name in that module (e.g., "sat_object_pickler").
    """

    module: str
    function: str


@dataclass(frozen=True)
class StageSpec:
    """
    Declarative stage specification.

    Attributes
    ----------
    stage : Stage
        Canonical stage enum value.
    description : str
        Human-readable description of what the stage does (shown in dry-run).
    entrypoint : EntrypointSpec
        Module/function pair to invoke for this stage.
    owned_outputs_rel : tuple[str, ...]
        Relative paths (from NEBULA_OUTPUT_DIR) to the pickle files owned by this
        stage. These are the ONLY files the manager deletes when invalidating
        this stage.
    extra_kwargs : dict[str, Any]
        Optional fixed kwargs the manager should pass to the stage entrypoint
        every time (e.g., constant knobs). In the current upstream pipeline,
        this is typically empty because defaults live in the picklers.
    """

    stage: Stage
    description: str
    entrypoint: EntrypointSpec
    owned_outputs_rel: Tuple[str, ...]
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Canonical manifest
# -----------------------------------------------------------------------------
_MANIFEST: Tuple[StageSpec, ...] = (
    StageSpec(
        stage=Stage.BASE,
        description="Build base propagated observer/target tracks (root stage).",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_SAT_PICKLER",
            function="sat_object_pickler",
        ),
        owned_outputs_rel=(
            "BASE_SatPickles/observer_tracks.pkl",
            "BASE_SatPickles/target_tracks.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.LOS,
        description="Attach LOS visibility fields to all observer–target pairs.",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_SAT_LOS_PICKLER",
            function="attach_los_to_all_targets",
        ),
        owned_outputs_rel=(
            "LOS_SatPickles/observer_tracks_with_los.pkl",
            "LOS_SatPickles/target_tracks_with_los.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.ILLUM,
        description="Attach illumination (sunlit/eclipsed, phase, etc.) arrays.",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_SAT_ILL_PICKLER",
            function="attach_illum_to_all_targets",
        ),
        owned_outputs_rel=(
            "ILLUM_SatPickles/observer_tracks_with_los_illum.pkl",
            "ILLUM_SatPickles/target_tracks_with_los_illum.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.FLUX,
        description="Attach radiometry / flux products (per observer–target).",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_FLUX_PICKLER",
            function="attach_flux_to_all_targets",
        ),
        owned_outputs_rel=(
            "FLUX_SatPickles/observer_tracks_with_los_illum_flux.pkl",
            "FLUX_SatPickles/target_tracks_with_los_illum_flux.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.LOS_FLUX,
        description="Attach LOS-gated flux products (combined LOS+ILLUM+FLUX+LOS).",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_LOS_FLUX_PICKLER",
            function="attach_los_flux_to_all_targets",
        ),
        # NOTE: on-disk directory name is LOSFLUX_SatPickles (no underscore)
        owned_outputs_rel=(
            "LOSFLUX_SatPickles/observer_tracks_with_los_illum_flux_los.pkl",
            "LOSFLUX_SatPickles/target_tracks_with_los_illum_flux_los.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.POINTING,
        description="Attach pointing schedule to observers (observer-only output).",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_SCHEDULE_PICKLER",
            function="attach_pointing_to_all_observers",
        ),
        owned_outputs_rel=(
            "POINT_SatPickles/observer_tracks_with_pointing.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.ICRS,
        description="Attach observer–target ICRS pair geometry products.",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_ICRS_PAIR_PICKLER",
            function="attach_icrs_to_all_pairs",
        ),
        owned_outputs_rel=(
            "ICRS_SatPickles/observer_tracks_with_icrs.pkl",
            "ICRS_SatPickles/target_tracks_with_icrs_pairs.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.PIXELS,
        description="Project ICRS pairs to detector pixels and write pixel pickles.",
        entrypoint=EntrypointSpec(
            module="Utility.SAT_OBJECTS.NEBULA_PIXEL_PICKLER",
            function="attach_pixels_to_all_pairs",
        ),
        owned_outputs_rel=(
            "PIXEL_SatPickles/observer_tracks_with_pixels.pkl",
            "PIXEL_SatPickles/target_tracks_with_pixels.pkl",
        ),
    ),
    
    # NEW: build the canonical per-window frames[] grid once
    StageSpec(
        stage=Stage.TARGET_FRAMES,
        description="Build canonical photon-domain frame grid (per observer/window/frame: t_utc, t_exp_s, coarse_index).",
        entrypoint=EntrypointSpec(
            module="Utility.FRAMES.NEBULA_PHOTON_FRAME_BUILDER",
            function="main",
        ),
        owned_outputs_rel=(
            f"{PHOTON_FRAMES_DIR_REL}/{PHOTON_FRAME_CATALOG_FILENAME}",
        ),
    ),
    
    # MODIFIED: consume the frame-grid pickle instead of rebuilding it internally
    StageSpec(
        stage=Stage.TARGET_PHOTONS,
        description="Build per-target photon time series (raw + ranked) for all observers/windows.",
        entrypoint=EntrypointSpec(
            module="Utility.FRAMES.NEBULA_TARGET_PHOTONS",
            function="main",
        ),
        owned_outputs_rel=(
            "TARGET_PHOTON_FRAMES/obs_target_frames_raw.pkl",
            "TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl",
        ),
        extra_kwargs={
            # Relative-to-NEBULA_OUTPUT_DIR is supported by the TARGET_PHOTONS edits below
            "frames_pickle_path": f"{PHOTON_FRAMES_DIR_REL}/{PHOTON_FRAME_CATALOG_FILENAME}",
        },
    ),

    StageSpec(
        stage=Stage.TRACKING_MODE,
        description="Annotate photon windows with tracking_mode (sidereal vs slew) and drift metrics.",
        entrypoint=EntrypointSpec(
            module="Utility.FRAMES.NEBULA_TRACKING_MODE",
            function="main",
        ),
        owned_outputs_rel=(
            "TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_tracking.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.SKY,
        description="Attach sky/FOV selection metadata to ranked target frames.",
        entrypoint=EntrypointSpec(
            module="Utility.STARS.NEBULA_SKY_SELECTOR",
            function="main",
        ),
        owned_outputs_rel=(
            f"{TARGET_FRAMES_DIR_REL}/obs_target_frames_ranked_with_sky.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.GAIA,
        description="Query Gaia DR3 for each window sky cone and cache results.",
        entrypoint=EntrypointSpec(
            module="Utility.STARS.NEBULA_QUERY_GAIA",
            function="main",
        ),
        owned_outputs_rel=(
            f"{STAR_DIR_REL}/obs_gaia_cones.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.STAR_PROJECTION,
        description="Dispatch star projection: sidereal (static) + slew (time-resolved tracks) based on tracking_mode.",
        entrypoint=EntrypointSpec(
            module="Utility.STARS.NEBULA_STAR_PROJECTION_DISPATCHER",
            function="main",
        ),
        owned_outputs_rel=(
            f"{STAR_DIR_REL}/obs_star_projections.pkl",
            f"{STAR_DIR_REL}/obs_star_slew_tracks.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.STAR_PHOTONS,
        description="Convert projected stars into per-star photon time series per window.",
        entrypoint=EntrypointSpec(
            module="Utility.STARS.NEBULA_STAR_PHOTONS",
            function="run_star_photons_pipeline_from_pickles",
        ),
        owned_outputs_rel=(
            f"{STAR_DIR_REL}/obs_star_photons.pkl",
        ),
    ),
    StageSpec(
        stage=Stage.WINDOW_SOURCES,
        description=(
            "Combine per-window targets and projected stars into one robust "
            "observer/window pickle (no per-frame image cubes)."
        ),
        entrypoint=EntrypointSpec(
            module="Utility.SCENE.NEBULA_WINDOW_SOURCES",
            function="main",
        ),
        owned_outputs_rel=(
            f"{SCENE_DIR_REL}/{WINDOW_SOURCES_FILENAME}",
        ),
    ),
    StageSpec(
        stage=Stage.ZODIACAL_LIGHT,
        description=(
            "Compute zodiacal light background for each observer/window using the "
            "WSL/M4OPT backend and write the observer/window zodiacal pickle."
        ),
        entrypoint=EntrypointSpec(
            module="Utility.ZODIACAL_LIGHT.stage.orchestrator",
            function="build_obs_zodiacal_light_for_all_observers",
        ),
        owned_outputs_rel=(
            f"{ZODIACAL_LIGHT_DIR_REL}/{ZODIACAL_LIGHT_FILENAME}",
        ),
    ),
)



# -----------------------------------------------------------------------------
# Indexes / validation
# -----------------------------------------------------------------------------
_MANIFEST_BY_STAGE: Dict[Stage, StageSpec] = {spec.stage: spec for spec in _MANIFEST}


def _validate_manifest() -> None:
    """
    Validate that:
      - every Stage in STAGE_ORDER has exactly one StageSpec,
      - StageSpecs are not missing owned outputs,
      - the manifest contains no extra stages.
    """
    stages_in_manifest = set(_MANIFEST_BY_STAGE.keys())
    stages_in_order = set(STAGE_ORDER)

    missing = stages_in_order - stages_in_manifest
    extra = stages_in_manifest - stages_in_order
    if missing or extra:
        raise RuntimeError(
            "NEBULA_PIPELINE_MANIFEST: Stage coverage mismatch. "
            f"Missing={[s.value for s in sorted(missing, key=lambda x: x.value)]} "
            f"Extra={[s.value for s in sorted(extra, key=lambda x: x.value)]}"
        )

    for s in STAGE_ORDER:
        spec = _MANIFEST_BY_STAGE[s]
        if not spec.owned_outputs_rel:
            raise RuntimeError(
                f"NEBULA_PIPELINE_MANIFEST: Stage '{s.value}' has no owned outputs."
            )


_validate_manifest()


# -----------------------------------------------------------------------------
# Public helpers used by the pipeline manager
# -----------------------------------------------------------------------------
def iter_stage_specs_in_order() -> Tuple[StageSpec, ...]:
    """Return stage specs in canonical pipeline order."""
    return tuple(_MANIFEST_BY_STAGE[s] for s in STAGE_ORDER)


def get_stage_spec(stage: StageLike) -> StageSpec:
    """Return the StageSpec for a given stage (Stage or stage-id string)."""
    s = parse_stage(stage)
    return _MANIFEST_BY_STAGE[s]


def resolve_owned_output_paths(
    stage: StageLike,
    *,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, ...]:
    """
    Resolve owned outputs for a single stage to absolute Paths.

    Parameters
    ----------
    stage : StageLike
        Stage enum or stage-id string.
    output_dir : pathlib.Path, optional
        Base output directory. If None, uses NEBULA_OUTPUT_DIR.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Absolute paths to owned output pickle files for this stage.
    """
    base = output_dir if output_dir is not None else NEBULA_OUTPUT_DIR
    spec = get_stage_spec(stage)
    return tuple(base / Path(rel) for rel in spec.owned_outputs_rel)


def resolve_owned_output_paths_for_stages(
    stages: Iterable[StageLike],
    *,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Resolve owned outputs for multiple stages to a single, ordered list.

    Ordering is by stage-order first, then by each stage's owned_outputs_rel order.
    Duplicate paths are removed while preserving first-seen order.
    """
    base = output_dir if output_dir is not None else NEBULA_OUTPUT_DIR

    # Normalize to Stage for ordering.
    stage_list: List[Stage] = [parse_stage(s) for s in stages]

    # Sort by canonical order to keep plan output stable/predictable.
    stage_list.sort(key=lambda st: STAGE_INDEX[st])

    seen: set[Path] = set()
    out: List[Path] = []
    for st in stage_list:
        spec = _MANIFEST_BY_STAGE[st]
        for rel in spec.owned_outputs_rel:
            p = base / Path(rel)
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def load_entrypoint(spec: EntrypointSpec) -> Callable[..., Any]:
    """
    Lazily import and return the callable described by `spec`.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the function name is not found in the module.
    TypeError
        If the resolved object is not callable.
    """
    module = importlib.import_module(spec.module)
    fn = getattr(module, spec.function)
    if not callable(fn):
        raise TypeError(
            f"Entrypoint '{spec.module}.{spec.function}' resolved to a non-callable "
            f"object of type {type(fn)}."
        )
    return fn


def load_stage_callable(stage: StageLike) -> Callable[..., Any]:
    """Convenience: load the entrypoint callable for a stage."""
    spec = get_stage_spec(stage)
    return load_entrypoint(spec.entrypoint)


__all__ = [
    "EntrypointSpec",
    "StageSpec",
    "iter_stage_specs_in_order",
    "get_stage_spec",
    "resolve_owned_output_paths",
    "resolve_owned_output_paths_for_stages",
    "load_entrypoint",
    "load_stage_callable",
]
