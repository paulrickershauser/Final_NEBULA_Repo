# debug_frames_probe.py
#
# Quick probe: find timesteps where an observer has
# pointing_valid_for_projection == True, and list all targets
# that are on_detector_visible_sunlit at those times.

import numpy as np

from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER  # uses all prior picklers

# ----------------------------------------------------------------------
# 1) Choose observer and load coarse tracks
# ----------------------------------------------------------------------

# Change this if you want a different observer
OBS_NAME = "SBSS (USA 216)"

obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
    force_recompute=False
)

if OBS_NAME not in obs_tracks:
    raise KeyError(f"Observer '{OBS_NAME}' not found in obs_tracks keys: {list(obs_tracks.keys())}")

obs = obs_tracks[OBS_NAME]

times = np.asarray(obs["times"])
mask_valid = np.asarray(obs["pointing_valid_for_projection"], dtype=bool)

if times.shape[0] != mask_valid.shape[0]:
    raise ValueError(
        f"Length mismatch: times={times.shape[0]}, "
        f"pointing_valid_for_projection={mask_valid.shape[0]}"
    )

# ----------------------------------------------------------------------
# 2) Build frame_indices based on observer-level mask
# ----------------------------------------------------------------------

frame_indices = np.where(mask_valid)[0]
frame_times = times[frame_indices]

print(f"Observer: {OBS_NAME}")
print(f"Total coarse timesteps: {len(times)}")
print(f"Timesteps with pointing_valid_for_projection=True: {len(frame_indices)}")
print()

if len(frame_indices) == 0:
    print("No valid pointing timesteps found; nothing to inspect.")
    raise SystemExit

# ----------------------------------------------------------------------
# 3) For the first few frame_indices, list visible/sunlit/on-detector targets
# ----------------------------------------------------------------------

N_FRAMES_TO_SHOW = 5  # tweak as you like

for k, idx in enumerate(frame_indices[:N_FRAMES_TO_SHOW], start=1):
    t_utc = times[idx]

    print("=" * 72)
    print(f"Frame {k}: coarse_index={idx}, time={t_utc.isoformat()}")
    print("Targets with on_detector_visible_sunlit == True:")

    n_sources = 0

    for tar_name, tar_track in tar_tracks.items():
        by_obs = tar_track.get("by_observer", {})
        if OBS_NAME not in by_obs:
            continue

        entry = by_obs[OBS_NAME]

        # Safety: ensure arrays are long enough
        if "on_detector_visible_sunlit" not in entry:
            continue
        if idx >= len(entry["on_detector_visible_sunlit"]):
            continue

        if not entry["on_detector_visible_sunlit"][idx]:
            continue

        # This target contributes to the frame
        x = entry["pix_x"][idx]
        y = entry["pix_y"][idx]

        # Use LOS-gated photon flux at the aperture (per m^2 per s)
        phi = entry.get("rad_photon_flux_g_m2_s_los_only", [None])[idx]

        print(f"  - {tar_name:25s}  pix=({x:8.3f}, {y:8.3f})  "
              f"phi_los_only={phi:.3e} [photons m^-2 s^-1]")

        n_sources += 1

    if n_sources == 0:
        print("  (no targets meet on_detector_visible_sunlit at this time)")

print("=" * 72)
print("Done.")
