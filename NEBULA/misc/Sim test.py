# from Utility.SAT_OBJECTS import NEBULA_SCHEDULE_PICKLER

# obs_tracks, tar_tracks = NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers(
#     force_recompute=False
# )

from Utility.SAT_OBJECTS import NEBULA_ICRS_PAIR_PICKLER

obs_icrs, tar_icrs = NEBULA_ICRS_PAIR_PICKLER.attach_icrs_to_all_pairs(
    force_recompute=False
)
