# debug_nebula_outputs.py
# ----------------------------------------------------------------------
# Run each NEBULA pickler stage and print where the pickle files end up.
# This helps verify that all your *_SatPickles folders are wired correctly.
# ----------------------------------------------------------------------

import os
import traceback

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR, ensure_output_directory

from Utility.SAT_OBJECTS import (
    NEBULA_SAT_PICKLER,
    NEBULA_SAT_LOS_PICKLER,
    NEBULA_SAT_ILL_PICKLER,
    NEBULA_FLUX_PICKLER,
    NEBULA_LOS_FLUX_PICKLER,
    NEBULA_SCHEDULE_PICKLER,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def find_pickles(root_dir, filename):
    """
    Recursively search for `filename` under `root_dir` and return full paths.
    """
    hits = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filename in filenames:
            hits.append(os.path.join(dirpath, filename))
    return hits


def print_pickle_locations(label, filenames):
    """
    For a list of expected pickle filenames, print where they were found.
    """
    print(f"\n--- {label} ---")
    for fn in filenames:
        hits = find_pickles(NEBULA_OUTPUT_DIR, fn)
        if hits:
            print(f"{fn}:")
            for h in hits:
                print(f"  {h}")
        else:
            print(f"{fn}: NOT FOUND")


def run_stage(stage_name, func, **kwargs):
    """
    Run a pickler function with given kwargs, handling exceptions nicely.
    """
    print(f"\n=== Running stage: {stage_name} ===")
    try:
        result = func(**kwargs)
        # Most of these return (obs_tracks, tar_tracks)
        if isinstance(result, tuple) and len(result) == 2:
            obs_tracks, tar_tracks = result
            print(
                f"{stage_name} succeeded: "
                f"{len(obs_tracks)} observer tracks, {len(tar_tracks)} target tracks"
            )
        else:
            print(f"{stage_name} returned: {type(result)}")
    except Exception as e:
        print(f"{stage_name} FAILED with error: {e!r}")
        traceback.print_exc()


# ----------------------------------------------------------------------
# Main debug routine
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Make sure NEBULA_OUTPUT_DIR exists
    output_path = ensure_output_directory()
    print(f"NEBULA_OUTPUT_DIR: {output_path}")

    force = False  # force_recompute=True for all stages

    # 1) Base satellite tracks
    run_stage(
        "BASE tracks (NEBULA_SAT_PICKLER.sat_object_pickler)",
        NEBULA_SAT_PICKLER.sat_object_pickler,
        force_recompute=force,
    )
    print_pickle_locations(
        "BASE tracks pickles",
        [
            "observer_tracks.pkl",
            "target_tracks.pkl",
        ],
    )

    # 2) LOS-only tracks
    run_stage(
        "LOS tracks (NEBULA_SAT_LOS_PICKLER.attach_los_to_all_targets)",
        NEBULA_SAT_LOS_PICKLER.attach_los_to_all_targets,
        force_recompute=force,
    )
    print_pickle_locations(
        "LOS pickles",
        [
            "observer_tracks_with_los.pkl",
            "target_tracks_with_los.pkl",
        ],
    )

    # 3) LOS + illumination
    run_stage(
        "ILLUM tracks (NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets)",
        NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets,
        force_recompute=force,
    )
    print_pickle_locations(
        "LOS + ILLUM pickles",
        [
            "observer_tracks_with_los_illum.pkl",
            "target_tracks_with_los_illum.pkl",
        ],
    )

    # 4) LOS + illumination + flux
    run_stage(
        "FLUX tracks (NEBULA_FLUX_PICKLER.attach_flux_to_all_targets)",
        NEBULA_FLUX_PICKLER.attach_flux_to_all_targets,
        force_recompute=force,
    )
    print_pickle_locations(
        "LOS + ILLUM + FLUX pickles",
        [
            "observer_tracks_with_los_illum_flux.pkl",
            "target_tracks_with_los_illum_flux.pkl",
        ],
    )

    # 5) LOS-gated flux
    run_stage(
        "LOS-FLUX tracks (NEBULA_LOS_FLUX_PICKLER.attach_los_flux_to_all_targets)",
        NEBULA_LOS_FLUX_PICKLER.attach_los_flux_to_all_targets,
        force_recompute=force,
    )
    print_pickle_locations(
        "LOS-gated FLUX pickles",
        [
            "observer_tracks_with_los_illum_flux_los.pkl",
            "target_tracks_with_los_illum_flux_los.pkl",
        ],
    )

    # 6) Pointing on observers
    run_stage(
        "POINTING (NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers)",
        NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers,
        force_recompute=force,
    )
    print_pickle_locations(
        "POINTING pickles",
        [
            "observer_tracks_with_pointing.pkl",
        ],
    )

    print("\n=== Debug run complete. Check paths above for folder layout. ===")
