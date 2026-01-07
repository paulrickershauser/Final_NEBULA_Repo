# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 09:22:31 2025

@author: prick
"""

    # 5) Optional star-field pipeline: projection (sidereal + slew) + photons
    #
    # This stage:
    #   - Uses ranked_target_frames (with tracking_mode already annotated),
    #   - Uses the Gaia cones/cache products written by NEBULA_QUERY_GAIA,
    #   - Builds per-window star projections (sidereal + slew),
    #   - Converts those into per-frame star photon time series.
    #
    # All three modules (projection, slew projection, photons) are written as
    # standalone pipelines that read the necessary pickles from NEBULA_OUTPUT
    # and Configuration.* and then write their own outputs back under
    # NEBULA_OUTPUT/FRAMES / NEBULA_OUTPUT/STARS (depending on how you've
    # configured them).
    if RUN_STAR_PIPELINE:
        logger.info(
            "sim_test: Running star-field pipeline "
            "(NEBULA_STAR_PROJECTION + NEBULA_STAR_SLEW_PROJECTION + NEBULA_STAR_PHOTONS)."
        )

        # Sidereal star projections: builds obs_star_projections for windows
        # with tracking_mode='sidereal' (and/or all windows, depending on the
        # internal logic of NEBULA_STAR_PROJECTION).
        #
        # Resolve default paths used by the star-field pipeline so we can
        # pass them explicitly to downstream modules that do not infer
        # locations internally.
        star_projection_path = NEBULA_STAR_PROJECTION._resolve_default_output_path()
        obs_tracks_path = NEBULA_STAR_PROJECTION._resolve_default_obs_tracks_path()
        star_slew_output_path = os.path.join(
            NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR,
            "STARS",
            getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG"),
            "obs_star_slew_projections.pkl",
        )

        # main(...) is expected to:
        #   - Load ranked_target_frames + Gaia cones + obs_tracks from disk,
        #   - Build per-window star projection products,
        #   - Write obs_star_projections.pkl,
        #   - Return the in-memory obs_star_projections dict (or None).
        obs_star_projections = NEBULA_STAR_PROJECTION.main(logger=logger)

        # Slew star projections: builds obs_star_slew_projections for windows
        # with tracking_mode='slew', using per-frame WCS to follow the stars
        # across the detector during the motion.
        #
        # main(...) is expected to:
        #   - Load the same ranked_target_frames + Gaia cones + obs_tracks,
        #   - Build per-frame star positions for slewing windows only,
        #   - Write obs_star_slew_projections.pkl,
        #   - Return the in-memory obs_star_slew_projections dict (or None).
        obs_star_slew_projections = NEBULA_STAR_SLEW_PROJECTION.main(
            star_projection_path=star_projection_path,
            obs_tracks_path=obs_tracks_path,
            output_path=star_slew_output_path,
            logger=logger,
        )

        # Star photons: consumes both the sidereal and slew projection
        # products and builds per-star, per-frame photon time series
        # aligned with the target photon frames.
        #
        # main(...) is expected to:
        #   - Load obs_target_frames_ranked.pkl (for windows & frame timing),
        #   - Load obs_star_projections.pkl (sidereal),
        #   - Load obs_star_slew_projections.pkl (slew),
        #   - Build obs_star_photons[obs_name]["windows"][i]["stars"][...],
        #   - Write obs_star_photons.pkl,
        #   - Return the in-memory obs_star_photons dict (or None).
        obs_star_photons = NEBULA_STAR_PHOTONS.run_star_photons_pipeline_from_pickles(
            frames_with_sky_path=NEBULA_STAR_PROJECTION._resolve_default_frames_path(),
            star_projection_sidereal_path=star_projection_path,
            star_projection_slew_path=star_slew_output_path,
            logger=logger,
        )

        # Provide a compact log summary if we actually got an in-memory dict.
        if isinstance(obs_star_photons, dict):
            logger.info(
                "sim_test: Star-photon pipeline produced entries for %d observers.",
                len(obs_star_photons),
            )
            for obs_name, entry in obs_star_photons.items():
                n_windows = len(entry.get("windows", []))
                logger.info(
                    "  Observer '%s': %d star-photon windows.",
                    obs_name,
                    n_windows,
                )
    else:
        logger.info(
            "sim_test: RUN_STAR_PIPELINE=False; skipping star projection and "
            "star-photon pipelines."
        )
