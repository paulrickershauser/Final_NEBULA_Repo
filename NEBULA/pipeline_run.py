# # -*- coding: utf-8 -*-
# """
# NEBULA upstream driver

# Assumptions
# -----------
# - You are running this with working directory set to the NEBULA repo root:
#     C:\\Users\\prick\\Desktop\\Research\\NEBULA
#   (i.e., the directory that contains Configuration/ and Utility/)
# """

# from __future__ import annotations


# # -----------------------------------------------------------------------------
# # 1) Run controls (edit these before running)
# #
# # Valid stage IDs (case-insensitive; aliases supported for LOS_FLUX):
# #   "BASE", "LOS", "ILLUM", "FLUX", "LOS_FLUX" (also: "LOSFLUX", "los-flux"),
# #   "POINTING", "ICRS", "PIXELS"
# #
# # RECOMPUTE_FROM_STAGE:
# #   - None          : do not delete anything; reuse existing outputs when possible
# #   - "<STAGE>"     : delete owned pickle outputs for <STAGE> and downstream stages,
# #                     then run forward (deletion-driven invalidation)
# #
# # STOP_AFTER_STAGE:
# #   - None          : run the full chain through PIXELS
# #   - "<STAGE>"     : stop after <STAGE> (inclusive)
# #
# # DRY_RUN:
# #   - True          : print the resolved plan only (no deletion, no execution)
# #   - False         : perform deletion + execute stages
# #
# # INVALIDATE_BEYOND_STOP_AFTER:
# #   - False (recommended): only delete outputs for stages that will run this invocation
# #   - True               : delete from RECOMPUTE_FROM_STAGE onward even if STOP_AFTER_STAGE is earlier
# #
# # FORCE_RECOMPUTE_ON_ENTRYPOINTS:
# #   - False (recommended): rely on deletion-driven recompute
# #   - True               : also pass force_recompute=True into pickler entrypoints (debug/rare)
# # -----------------------------------------------------------------------------
# '''
# "BASE", "LOS", "ILLUM", "FLUX","FLUX_LOS","POINTING","ICRS", "PIXELS","TARGET_FRAMES","TARGET_PHOTONS","TRACKING_MODE","SKY",
#                              "GAIA","STAR_PROJECTION","STAR_PHOTONS","WINDOW_SOURCES"
# '''
# RECOMPUTE_FROM_STAGE = "PIXELS"     # e.g., None, "FLUX", "LOS_FLUX", "PIXELS"
# STOP_AFTER_STAGE = "WINDOW_SOURCES"          # e.g., None, "LOS_FLUX", "ICRS", "PIXELS"
# DRY_RUN = False                       # start True; flip to False once plan looks correct

# INVALIDATE_BEYOND_STOP_AFTER = False
# FORCE_RECOMPUTE_ON_ENTRYPOINTS = False


# def main() -> None:
#     # Import here so the module can be imported without side effects.
#     from Utility.PIPELINE.NEBULA_PIPELINE_MANAGER import run_pipeline

#     run_pipeline(
#         recompute_from_stage=RECOMPUTE_FROM_STAGE,
#         stop_after_stage=STOP_AFTER_STAGE,
#         dry_run=DRY_RUN,
#         invalidate_beyond_stop_after=INVALIDATE_BEYOND_STOP_AFTER,
#         force_recompute_on_entrypoints=FORCE_RECOMPUTE_ON_ENTRYPOINTS,
#     )


# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
"""
NEBULA upstream driver

Assumptions
-----------
- You are running this with working directory set to the NEBULA repo root:
    C:\\Users\\prick\\Desktop\\Research\\NEBULA
  (i.e. the directory that contains Configuration/ and Utility/)
"""

from __future__ import annotations


# -----------------------------------------------------------------------------
# 1) Run controls (edit these before running)
#
# Valid stage IDs (case-insensitive; aliases supported for LOS_FLUX):
#   "BASE", "LOS", "ILLUM", "FLUX", "LOS_FLUX" (also: "LOSFLUX", "los-flux"),
#   "POINTING", "ICRS", "PIXELS",
#   "TARGET_FRAMES", "TARGET_PHOTONS", "TRACKING_MODE", "SKY",
#   "GAIA", "STAR_PROJECTION", "STAR_PHOTONS", "WINDOW_SOURCES", "ZODIACAL_LIGHT"
#
# RECOMPUTE_FROM_STAGE:
#   - None          : do not delete anything; reuse existing outputs when possible
#   - "<STAGE>"     : delete owned pickle outputs for <STAGE> and downstream stages,
#                     then run forward (deletion-driven invalidation)
#
# STOP_AFTER_STAGE:
#   - None          : run the full chain through ZODIACAL_LIGHT
#   - "<STAGE>"     : stop after <STAGE> (inclusive)
#
# DRY_RUN:
#   - True          : print the resolved plan only (no deletion, no execution)
#   - False         : perform deletion + execute stages
#
# INVALIDATE_BEYOND_STOP_AFTER:
#   - False (recommended): only delete outputs for stages that will run this invocation
#   - True               : delete from RECOMPUTE_FROM_STAGE onward even if STOP_AFTER_STAGE is earlier
#
# FORCE_RECOMPUTE_ON_ENTRYPOINTS:
#   - False (recommended): rely on deletion-driven recompute
#   - True               : also pass force_recompute=True into pickler entrypoints (debug/rare)
# -----------------------------------------------------------------------------
'''
"BASE", "LOS", "ILLUM", "FLUX","LOS_FLUX","POINTING","ICRS", "PIXELS","TARGET_FRAMES","TARGET_PHOTONS","TRACKING_MODE","SKY",
                             "GAIA","STAR_PROJECTION","STAR_PHOTONS","WINDOW_SOURCES","ZODIACAL_LIGHT"
'''
RECOMPUTE_FROM_STAGE = "ZODIACAL_LIGHT"     # e.g. None, "FLUX", "LOS_FLUX", "PIXELS"
STOP_AFTER_STAGE = "ZODIACAL_LIGHT"          # e.g. None, "LOS_FLUX", "ICRS", "PIXELS"
DRY_RUN = False                       # start True; flip to False once plan looks correct

INVALIDATE_BEYOND_STOP_AFTER = False
FORCE_RECOMPUTE_ON_ENTRYPOINTS = False


def main() -> None:
    # Import here so the module can be imported without side effects.
    from Utility.PIPELINE.NEBULA_PIPELINE_MANAGER import run_pipeline

    run_pipeline(
        recompute_from_stage=RECOMPUTE_FROM_STAGE,
        stop_after_stage=STOP_AFTER_STAGE,
        dry_run=DRY_RUN,
        invalidate_beyond_stop_after=INVALIDATE_BEYOND_STOP_AFTER,
        force_recompute_on_entrypoints=FORCE_RECOMPUTE_ON_ENTRYPOINTS,
    )


if __name__ == "__main__":
    main()
