"""Step-1 smoketest for NEBULA -> EBS_SDA_SIM bridge radiometry.

What this smoketest verifies
----------------------------
1) We can load the two upstream NEBULA pickles:
   - SCENE/obs_window_sources.pkl
   - ZODIACAL_LIGHT/obs_zodiacal_light.pkl
2) We can resolve an effective collecting area + throughput from NEBULA's
   ACTIVE_SENSOR (or fail loudly with a clear message).
3) We can apply the scalar conversion:
       (ph/m^2/s) -> (ph/s)
   without touching Rachel Oliver's code.

What this smoketest explicitly does NOT do
-----------------------------------------
- PSF/optics blurring
- writing HDF5 frames
- running circuitry.generate_events
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

from NEBULA_EBS_SDA_SIM_RADIOMETRY import (
    phi_ph_m2_s_to_ph_s,
    phi_ph_m2_s_pix_to_ph_s_pix,
    resolve_radiometry_scale_from_sensor,
)


def _load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def main() -> None:
    # Ensure the NEBULA repo root is on sys.path so that
    # `import Configuration....` works even when running with Spyder's
    # `--wdir` pointing at Utility/EVENT_CAMERA.
    this_dir = Path(__file__).resolve().parent
    nebula_root = this_dir.parents[1]  # .../NEBULA
    if str(nebula_root) not in sys.path:
        sys.path.insert(0, str(nebula_root))

    # Resolve NEBULA output directory.
    # Priority:
    #   1) Environment variable NEBULA_OUTPUT_DIR
    #   2) Your repo config (Configuration/NEBULA_PATH_CONFIG.py)
    env_out = os.environ.get("NEBULA_OUTPUT_DIR", "").strip()
    if env_out:
        out_root = Path(env_out).expanduser().resolve()
    else:
        try:
            from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR  # type: ignore
        except Exception:
            from NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR  # type: ignore
        out_root = Path(NEBULA_OUTPUT_DIR).expanduser().resolve()

    p_scene = out_root / "SCENE" / "obs_window_sources.pkl"
    p_zodi = out_root / "ZODIACAL_LIGHT" / "obs_zodiacal_light.pkl"

    if not p_scene.exists():
        raise FileNotFoundError(f"Missing: {p_scene}")
    if not p_zodi.exists():
        raise FileNotFoundError(f"Missing: {p_zodi}")

    obs_window_sources = _load_pickle(p_scene)
    obs_zodi = _load_pickle(p_zodi)

    # Resolve sensor scale.
    try:
        from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR  # type: ignore
    except Exception:
        from NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR  # type: ignore

    scale = resolve_radiometry_scale_from_sensor(ACTIVE_SENSOR)

    print("[STEP1] Radiometry scale:")
    print(f"  aperture_area_m2     = {scale.aperture_area_m2:.6g}")
    print(f"  optical_throughput   = {scale.optical_throughput:.6g}")
    print(f"  effective_area_m2    = {scale.effective_area_m2:.6g}")

    # Pick the first observer and first window.
    obs_name = next(iter(obs_window_sources.keys()))
    w0 = obs_window_sources[obs_name]["windows"][0]

    # --- Stars ---
    star_id = next(iter(w0["stars"].keys()))
    star = w0["stars"][star_id]
    phi_star = np.asarray(star["phi_ph_m2_s"], dtype=float)
    ph_s_star = phi_ph_m2_s_to_ph_s(phi_star, scale=scale)

    print("[STEP1] Example star:")
    print(f"  star_id              = {star_id}")
    print(f"  phi_ph_m2_s (min/max) = {phi_star.min():.6g} / {phi_star.max():.6g}")
    print(f"  ph_s (min/max)        = {ph_s_star.min():.6g} / {ph_s_star.max():.6g}")

    # --- Targets (sparse) ---
    target_id = next(iter(w0["targets"].keys()))
    target = w0["targets"][target_id]
    phi_tgt = np.asarray(target["phi_ph_m2_s"], dtype=float)
    ph_s_tgt = phi_ph_m2_s_to_ph_s(phi_tgt, scale=scale)

    print("[STEP1] Example target:")
    print(f"  target_id            = {target_id}")
    print(f"  phi_ph_m2_s (min/max) = {phi_tgt.min():.6g} / {phi_tgt.max():.6g}")
    print(f"  ph_s (min/max)        = {ph_s_tgt.min():.6g} / {ph_s_tgt.max():.6g}")

    # --- Zodi (per pixel) ---
    z_w0 = obs_zodi[obs_name]["windows"][0]
    plane3 = z_w0["zodi"]["models"]["plane3"]
    coeffs_per_pixel = np.asarray(plane3["coeffs_per_pixel"], dtype=float)
    # coeffs shape is (n_frames, 3). To keep Step-1 purely radiometric,
    # just check scaling on the DC term a0 (first coefficient).
    phi_zodi_a0 = coeffs_per_pixel[:, 0]
    ph_s_zodi_a0 = phi_ph_m2_s_pix_to_ph_s_pix(phi_zodi_a0, scale=scale)

    print("[STEP1] Zodi plane3 a0 coefficient (per pixel):")
    print(f"  phi_ph_m2_s_pix a0 (min/max) = {phi_zodi_a0.min():.6g} / {phi_zodi_a0.max():.6g}")
    print(f"  ph_s_pix a0 (min/max)        = {ph_s_zodi_a0.min():.6g} / {ph_s_zodi_a0.max():.6g}")

    print("[STEP1] PASS: radiometry scaling executed with no exceptions.")


if __name__ == "__main__":
    main()
