"""
NEBULA -> Rachel EBS_SDA_SIM Bridge: Step 2 (Optics + Attribution)

Implements:
  2.1B  Optics kernel generation (SAFT primary; optional impulse_response for comparison)
  2.2   HDF5 frame writer with deterministic ordering (zero-padded dataset names)
  2.3   Rachel pixel_attribution (power_ref policy; no placeholder/zero dicts)
  2.4   Smoke tests

Key design constraints from user:
- Do not modify Rachel's code.
- Use Rachel's optics & attribution implementations.
- Do not assume pickle layout beyond explicit validations.
- Ensure output compatibility with Rachel's CircuitryModel.generate_events.
"""

from __future__ import annotations

import os
import sys
import math
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional, Iterable

import numpy as np
import h5py

# Rachel uses skimage warp for translation/scale of complex fields (real/imag separately)
import skimage.transform as tr
from skimage.transform import warp


# -----------------------------
# Optional: import astropy/synphot only for the power_ref policy
# (vmag_2_irradiance_Gia_G is inside Rachel's RadiometryModel and uses these)
# -----------------------------
try:
    import astropy.units as u
except Exception as e:
    u = None  # type: ignore


# =============================================================================
# 0) Import Rachel radiometry without assuming install layout
# =============================================================================

def _ensure_rachel_on_path() -> None:
    """
    Ensure Rachel's repo root is importable.

    Use environment variable:
      EBS_SDA_SIM_ROOT=/path/to/Rachel/EBS_SDA_SIM

    This avoids assuming local directory structure.
    """
    root = os.environ.get("EBS_SDA_SIM_ROOT", "").strip()
    if root and os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


_ensure_rachel_on_path()

# Rachel radiometry model
from radiometry import RadiometryModel  # noqa: E402


# =============================================================================
# 1) Data contracts (explicit) and validation
# =============================================================================

REQUIRED_TOP_KEYS = {"rows", "cols", "dt_frame_s", "windows"}
REQUIRED_WINDOW_KEYS_SCENE = {"t_utc_iso", "targets", "stars"}
REQUIRED_WINDOW_KEYS_ZODI = {"t_utc_iso", "zodi"}

REQUIRED_SOURCE_KEYS = {"source_name", "x_pix", "y_pix", "phi_ph_m2_s"}


def _assert_keys(d: Dict[str, Any], required: Iterable[str], context: str) -> None:
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"[{context}] Missing required keys: {missing}. Available keys: {list(d.keys())}")


def load_pickles(scene_pkl: str, zodi_pkl: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load NEBULA pickles and validate minimal required structure.
    """
    with open(scene_pkl, "rb") as f:
        scene = pickle.load(f)
    with open(zodi_pkl, "rb") as f:
        zodi = pickle.load(f)

    if not isinstance(scene, dict) or not isinstance(zodi, dict):
        raise TypeError("Expected both pickles to load as dicts keyed by observer name.")

    # Validate each observer
    for obs_name, obs_data in scene.items():
        if not isinstance(obs_data, dict):
            raise TypeError(f"[scene:{obs_name}] Expected dict observer record.")
        _assert_keys(obs_data, REQUIRED_TOP_KEYS, f"scene:{obs_name}")

    for obs_name, obs_data in zodi.items():
        if not isinstance(obs_data, dict):
            raise TypeError(f"[zodi:{obs_name}] Expected dict observer record.")
        _assert_keys(obs_data, REQUIRED_TOP_KEYS, f"zodi:{obs_name}")

    return scene, zodi


def get_window(scene_obs: Dict[str, Any], zodi_obs: Dict[str, Any], window_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract a window from scene + zodi with validation.
    """
    windows_scene = scene_obs["windows"]
    windows_zodi = zodi_obs["windows"]

    if not (0 <= window_idx < len(windows_scene)):
        raise IndexError(f"window_idx={window_idx} out of range for scene windows (n={len(windows_scene)})")
    if not (0 <= window_idx < len(windows_zodi)):
        raise IndexError(f"window_idx={window_idx} out of range for zodi windows (n={len(windows_zodi)})")

    w_scene = windows_scene[window_idx]
    w_zodi = windows_zodi[window_idx]

    if not isinstance(w_scene, dict) or not isinstance(w_zodi, dict):
        raise TypeError("Each window must be a dict.")

    _assert_keys(w_scene, REQUIRED_WINDOW_KEYS_SCENE, f"scene.window[{window_idx}]")
    _assert_keys(w_zodi, REQUIRED_WINDOW_KEYS_ZODI, f"zodi.window[{window_idx}]")

    # Validate time axis agreement by length (do not assume same timestamps, but require same count)
    t_scene = w_scene["t_utc_iso"]
    t_zodi = w_zodi["t_utc_iso"]
    if len(t_scene) != len(t_zodi):
        raise ValueError(f"Scene and zodi windows have different frame counts: {len(t_scene)} vs {len(t_zodi)}")

    return w_scene, w_zodi


# =============================================================================
# 2) Optics config derived from NEBULA sensor config (no hard assumptions)
# =============================================================================

@dataclass(frozen=True)
class OpticsConfig:
    """
    Minimal optics configuration required to reproduce Rachel’s SAFT call path and warp mapping.
    """
    rows: int
    cols: int

    pixel_pitch_m: float
    focal_length_m: float
    aperture_diameter_m: float

    # wavelength used for comparisons/impulse_response sizing;
    # SAFT call path itself is normalized, but we need this to compute airy disk diameter.
    wavelength_m: float

    # SAFT grid / patch sizing
    patch_pix_x: int
    patch_pix_y: int
    gamma_subpix_per_airy: int  # Rachel interactive "num_pix_per_airy_disk"

    # Pupil plane sampling for SAFT input (must be small enough for SAFT compute)
    observation_plane_sampling: int  # Np

    # Derived
    sufficient_airy_disk_discretization: bool
    subpix_per_pix_x: int
    subpix_per_pix_y: int
    Nax: int
    Nay: int
    mx: float
    my: float


def derive_optics_config(
    rows: int,
    cols: int,
    pixel_pitch_m: float,
    focal_length_m: float,
    aperture_diameter_m: float,
    wavelength_m: float,
    patch_pix_x: int = 32,
    patch_pix_y: int = 32,
    gamma_subpix_per_airy: int = 4,
    observation_plane_sampling: int = 256,
) -> OpticsConfig:
    """
    Derive subpixel and SAFT parameters following the logic in Rachel's opticalSystem
    (without interactive prompts). Rachel determines whether the airy disk is sufficiently
    discretized, and if not, chooses a subpixel-per-pixel factor and computes mx,my.

    In opticalSystem, she computes:
      spatial_resolution = 2.44 * (focal_length * wavelength / aperture_diameter)
      if spatial_resolution / max_pixel_dim < 4: insufficient -> enable subpixels
      subpix_per_pix_x = ceil((pixelwidth * gamma) / spatial_resolution), then force even
      mx = sim_num_pix_x * subpix_per_pix_x / gamma
    :contentReference[oaicite:8]{index=8}
    """
    if patch_pix_x % 2 != 0 or patch_pix_y % 2 != 0:
        raise ValueError("patch_pix_x and patch_pix_y must be even to match Rachel’s convention.")

    if observation_plane_sampling % 2 != 0:
        raise ValueError("observation_plane_sampling (Np) must be even.")

    # Airy disk diameter on focal plane (meters): 2.44 * f * lambda / D
    spatial_resolution_m = 2.44 * (focal_length_m * wavelength_m / aperture_diameter_m)

    max_pixel_dim_m = max(pixel_pitch_m, pixel_pitch_m)

    sufficient = not ((spatial_resolution_m / max_pixel_dim_m) < 4)

    if sufficient:
        subpix_x = 1
        subpix_y = 1
    else:
        # replicate opticalSystem logic: ceil((pixelwidth * gamma) / spatial_resolution), then force even:contentReference[oaicite:9]{index=9}
        subpix_x = int(math.ceil((pixel_pitch_m * gamma_subpix_per_airy) / spatial_resolution_m))
        subpix_y = int(math.ceil((pixel_pitch_m * gamma_subpix_per_airy) / spatial_resolution_m))
        if subpix_x % 2 == 1:
            subpix_x += 1
        if subpix_y % 2 == 1:
            subpix_y += 1

    Nax = patch_pix_x * subpix_x
    Nay = patch_pix_y * subpix_y

    # mx,my per opticalSystem: mx = sim_num_pix_x * subpix_per_pix_x / gamma:contentReference[oaicite:10]{index=10}
    mx = (patch_pix_x * subpix_x) / float(gamma_subpix_per_airy)
    my = (patch_pix_y * subpix_y) / float(gamma_subpix_per_airy)

    return OpticsConfig(
        rows=rows,
        cols=cols,
        pixel_pitch_m=pixel_pitch_m,
        focal_length_m=focal_length_m,
        aperture_diameter_m=aperture_diameter_m,
        wavelength_m=wavelength_m,
        patch_pix_x=patch_pix_x,
        patch_pix_y=patch_pix_y,
        gamma_subpix_per_airy=gamma_subpix_per_airy,
        observation_plane_sampling=observation_plane_sampling,
        sufficient_airy_disk_discretization=sufficient,
        subpix_per_pix_x=subpix_x,
        subpix_per_pix_y=subpix_y,
        Nax=Nax,
        Nay=Nay,
        mx=mx,
        my=my,
    )


# =============================================================================
# 3) Step 2.1B: Kernel generation (SAFT primary; impulse optional)
# =============================================================================

def build_pupil_mask(rm: RadiometryModel, cfg: OpticsConfig) -> Tuple[np.ndarray, float]:
    """
    Build a circular pupil mask using Rachel's circ() helper.
    """
    Np = cfg.observation_plane_sampling
    delta_p = cfg.aperture_diameter_m / float(Np)

    # Rachel builds coordinates like arange(-N/2+0.5, N/2+0.5)*delta
    x = (np.arange(-Np / 2 + 0.5, Np / 2 + 0.5) * delta_p).astype(np.float64)
    X, Y = np.meshgrid(x, x)

    pupil = rm.circ(X, Y, cfg.aperture_diameter_m)
    return pupil.astype(np.float64), float(delta_p)


def _warp_complex_to_pixel_patch(
    U_subpix: np.ndarray,
    cfg: OpticsConfig,
    x_rel: float,
    y_rel: float,
) -> np.ndarray:
    """
    Warp complex subpixel field into a pixel-resolution patch, using Rachel's approach:
    AffineTransform(translation=..., scale=1/subpix) + warp(real) + warp(imag)
    
    """
    tx = x_rel - (cfg.patch_pix_x / 2.0)
    ty = y_rel - (cfg.patch_pix_y / 2.0)

    source_transform = tr.AffineTransform(
        translation=[tx, ty],
        scale=[1.0 / cfg.subpix_per_pix_x, 1.0 / cfg.subpix_per_pix_y],
    )

    real_transformed = warp(
        np.real(U_subpix),
        source_transform.inverse,
        output_shape=(cfg.patch_pix_y, cfg.patch_pix_x),
        mode="constant",
        cval=0.0,
        order=1,
    )
    imag_transformed = warp(
        np.imag(U_subpix),
        source_transform.inverse,
        output_shape=(cfg.patch_pix_y, cfg.patch_pix_x),
        mode="constant",
        cval=0.0,
        order=1,
    )

    return real_transformed + 1j * imag_transformed


def precompute_psf_kernel(
    rm: RadiometryModel,
    cfg: OpticsConfig,
    pupil_mask: np.ndarray,
    delta_p: float,
    method: str = "saft",
) -> Dict[str, Any]:
    """
    Precompute a normalized PSF kernel on the *subpixel* grid (U_subpix_scaled),
    plus metadata required for rendering patches.

    Normalization:
    - Build base pupil-plane field corresponding to phi_base = 1 ph/s/m^2.
    - Propagate to subpixel image plane via SAFT (or impulse_response).
    - Warp into a centered pixel patch and compute power_image.
    - Compute scalar alpha = sqrt(power_aperture / power_image) and apply to U_subpix.

    Rachel’s pipeline does the same conceptually via scale_field(),
    after computing power_aperature and power_image.
    """
    phi_base = 1.0  # ph / (s*m^2) as float
    # Pupil-plane field amplitude inside aperture: sqrt(phi_base) [sqrt(ph/s)/m]
    U_pupil = pupil_mask * math.sqrt(phi_base)

    # Power entering aperture (ph/s) using Rachel intensity_calc()
    power_ap = rm.intensity_calc(U_pupil, delta_p, delta_p)

    if method.lower() == "saft":
        # Rachel SAFT call path
        U_subpix = rm.saft(U_pupil, cfg.mx, cfg.my, cfg.Nax, cfg.Nay)
    elif method.lower() == "impulse":
        # Optional kernel for comparison, defined in Rachel code:contentReference[oaicite:16]{index=16}
        paddingx = max(0, cfg.Nax - cfg.observation_plane_sampling)
        paddingy = max(0, cfg.Nay - cfg.observation_plane_sampling)
        U_subpix = rm.impulse_response(
            pupil_mask,
            cfg.wavelength_m,
            cfg.focal_length_m,
            delta_p,
            delta_p,
            paddingx,
            paddingy,
        )
    else:
        raise ValueError("method must be one of: 'saft', 'impulse'")

    # Warp to a centered pixel patch (x_rel=y_rel=patch/2 => tx=ty=0)
    U_patch_center = _warp_complex_to_pixel_patch(
        U_subpix=U_subpix,
        cfg=cfg,
        x_rel=cfg.patch_pix_x / 2.0,
        y_rel=cfg.patch_pix_y / 2.0,
    )

    power_img = rm.intensity_calc(U_patch_center, cfg.pixel_pitch_m, cfg.pixel_pitch_m)

    if power_img <= 0 or not np.isfinite(power_img):
        raise RuntimeError(f"Computed power_img invalid: {power_img}")

    alpha = math.sqrt(power_ap / power_img)

    return {
        "method": method.lower(),
        "phi_base": phi_base,
        "power_aperture_base": float(power_ap),
        "power_image_base": float(power_img),
        "alpha": float(alpha),
        "U_subpix_scaled": U_subpix * alpha,
    }


# =============================================================================
# 4) Step 2.3: power_ref policy + pixel_attribution
# =============================================================================

def compute_power_ref_limiting_mag(
    rm: RadiometryModel,
    limiting_mag: float,
    bandpass_width_A: float,
    aperture_diameter_m: float,
) -> float:
    """
    Compute power_ref (ph/s) using Rachel’s policy:
      irradiance_min = vmag_2_irradiance_Gia_G(...)
      power_min = irradiance_min * (D/2)^2 * pi
    

    Returns float in ph/s.
    """
    if u is None:
        raise ImportError("astropy is required to compute power_ref via Rachel's vmag conversion.")

    irradiance_min = rm.vmag_2_irradiance_Gia_G(
        limiting_mag,
        bandpass_width_A * u.AA,
        aperture_diameter_m,
    )
    power_min = irradiance_min * ((aperture_diameter_m / 2.0) ** 2) * math.pi
    return float(power_min.to(u.ph / u.s).value)


def merge_attribution_with_offset(
    global_dict: Dict[Tuple[int, int], List[List[Any]]],
    local_dict: Dict[Tuple[int, int], List[List[Any]]],
    x_offset: int,
    y_offset: int,
) -> None:
    """
    Merge per-patch pixel_attribution results into a global per-frame attribution dict.

    Rachel’s pixel_attribution uses pixel_key=(xloc, yloc):contentReference[oaicite:18]{index=18}.
    We preserve that convention and offset it back into full-frame coordinates.
    """
    for (x, y), val_list in local_dict.items():
        gx = int(x + x_offset)
        gy = int(y + y_offset)
        key = (gx, gy)
        if key not in global_dict:
            global_dict[key] = []
        global_dict[key].extend(val_list)


# =============================================================================
# 5) Zodi model evaluation (plane3/quad6) without assuming extra pickle fields
# =============================================================================

def make_uv_maps(rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create u,v maps consistent with NEBULA conventions (used previously):
      u in [-1, 1] across x
      v in [-1, 1] across y
    """
    yy, xx = np.mgrid[0:rows, 0:cols]
    u_map = (2.0 * xx / max(cols - 1, 1)) - 1.0
    v_map = (2.0 * yy / max(rows - 1, 1)) - 1.0
    return u_map.astype(np.float64), v_map.astype(np.float64)


def eval_plane3(coeffs_3: np.ndarray, u_map: np.ndarray, v_map: np.ndarray) -> np.ndarray:
    """
    Evaluate plane3 model: c0*1 + c1*u + c2*v
    uv_basis is ['1','u','v'] in pickle.
    """
    c0, c1, c2 = float(coeffs_3[0]), float(coeffs_3[1]), float(coeffs_3[2])
    return (c0 + c1 * u_map + c2 * v_map).astype(np.float64)


def eval_quad6(coeffs_6: np.ndarray, u_map: np.ndarray, v_map: np.ndarray) -> np.ndarray:
    """
    Evaluate quad6 model: c0*1 + c1*u + c2*v + c3*u^2 + c4*u*v + c5*v^2
    uv_basis is ['1','u','v','u2','uv','v2'] in pickle.
    """
    c = [float(x) for x in coeffs_6]
    return (c[0] + c[1] * u_map + c[2] * v_map + c[3] * u_map**2 + c[4] * u_map * v_map + c[5] * v_map**2).astype(np.float64)


# =============================================================================
# 6) Step 2.2/2.3: Frame generation loop
# =============================================================================

def iter_sources(window_scene: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield each source dict from targets and stars, without assuming dict vs list.
    """
    targets = window_scene["targets"]
    stars = window_scene["stars"]

    if isinstance(targets, dict):
        for _, s in targets.items():
            yield s
    elif isinstance(targets, list):
        for s in targets:
            yield s
    else:
        raise TypeError("window_scene['targets'] must be dict or list")

    if isinstance(stars, dict):
        for _, s in stars.items():
            yield s
    elif isinstance(stars, list):
        for s in stars:
            yield s
    else:
        raise TypeError("window_scene['stars'] must be dict or list")


def build_frames_and_attribution(
    scene_window: Dict[str, Any],
    zodi_window: Dict[str, Any],
    rm: RadiometryModel,
    cfg: OpticsConfig,
    psf: Dict[str, Any],
    power_ref_ph_s: float,
    out_h5: str,
    out_att_pkl: str,
    zodi_model: str = "plane3",
    max_frames: Optional[int] = None,
) -> None:
    """
    Main Step 2.2 + Step 2.3 loop:
    - Build each frame photon flux image
    - Write dataset with zero-padded names
    - Build per-frame attribution dict keyed by t_total_sec*u.s (for circuitry compatibility)
    """
    # Validate time axis
    t_utc = scene_window["t_utc_iso"]
    n_frames = len(t_utc)
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    # Zodi coeffs
    zodi = zodi_window["zodi"]
    _assert_keys(zodi, {"models"}, "zodi_window['zodi']")
    models = zodi["models"]
    if zodi_model not in models:
        raise KeyError(f"Requested zodi_model={zodi_model} not found. Available: {list(models.keys())}")

    coeffs = models[zodi_model]["coeffs_per_pixel"]
    if len(coeffs) < n_frames:
        raise ValueError(f"Zodi coeffs length={len(coeffs)} < n_frames={n_frames}")

    # Precompute u,v pixel maps
    u_map, v_map = make_uv_maps(cfg.rows, cfg.cols)

    # Effective collecting area
    aperture_area_m2 = math.pi * (cfg.aperture_diameter_m / 2.0) ** 2

    # Attribution dict keyed by time (Quantity seconds)
    att_by_time: Dict[Any, Dict[Tuple[int, int], List[List[Any]]]] = {}

    # Create output directories
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_att_pkl) or ".", exist_ok=True)

    with h5py.File(out_h5, "w") as hf:
        # Store minimal metadata (non-breaking)
        hf.attrs["rows"] = cfg.rows
        hf.attrs["cols"] = cfg.cols
        hf.attrs["dt_frame_s"] = float(scene_window.get("dt_frame_s", 0.0))

        for i in range(n_frames):
            # ---------------------------------------------------------
            # Step 2.2: initialize per-frame photon flux (ph/s per pixel)
            # ---------------------------------------------------------
            frame_ph = np.zeros((cfg.rows, cfg.cols), dtype=np.float64)

            # Zodi per pixel is irradiance (ph/s/m^2) -> multiply by aperture area
            if zodi_model == "plane3":
                zodi_irr = eval_plane3(np.asarray(coeffs[i]), u_map, v_map)
            else:
                zodi_irr = eval_quad6(np.asarray(coeffs[i]), u_map, v_map)

            frame_ph += zodi_irr * aperture_area_m2

            # ---------------------------------------------------------
            # Step 2.3: per-frame attribution dict
            # ---------------------------------------------------------
            frame_att: Dict[Tuple[int, int], List[List[Any]]] = {}

            # ---------------------------------------------------------
            # Add all sources
            # ---------------------------------------------------------
            for src in iter_sources(scene_window):
                if not isinstance(src, dict):
                    continue

                # Validate minimal required keys per-source
                _assert_keys(src, REQUIRED_SOURCE_KEYS, f"source:{src.get('source_name','<unknown>')}")

                name = src["source_name"]
                x_pix = np.asarray(src["x_pix"], dtype=np.float64)
                y_pix = np.asarray(src["y_pix"], dtype=np.float64)
                phi = np.asarray(src["phi_ph_m2_s"], dtype=np.float64)

                if len(x_pix) <= i or len(y_pix) <= i or len(phi) <= i:
                    raise ValueError(f"Source '{name}' arrays shorter than frame index {i}")

                # Optional on_detector gating if present
                if "on_detector" in src:
                    on_det = np.asarray(src["on_detector"], dtype=bool)
                    if len(on_det) > i and not bool(on_det[i]):
                        continue

                phi_i = float(phi[i])
                if not np.isfinite(phi_i) or phi_i <= 0.0:
                    continue

                # Scale factor relative to phi_base
                scale_amp = math.sqrt(phi_i / float(psf["phi_base"]))

                # Source coordinate in full-frame pixel indices
                xs = float(x_pix[i])
                ys = float(y_pix[i])

                # Patch top-left in global coordinates
                x_start = int(math.floor(xs - cfg.patch_pix_x / 2.0))
                y_start = int(math.floor(ys - cfg.patch_pix_y / 2.0))

                # Relative coordinate inside patch (for Rachel translation formula)
                x_rel = xs - x_start
                y_rel = ys - y_start

                # Warp scaled PSF subpixel field to pixel patch
                U_patch = _warp_complex_to_pixel_patch(
                    U_subpix=psf["U_subpix_scaled"] * scale_amp,
                    cfg=cfg,
                    x_rel=x_rel,
                    y_rel=y_rel,
                )

                # Convert complex field to photon flux (ph/s) per pixel
                # Flux = |U|^2 * pixel_area; pixel_area = pitch^2
                flux_patch = (np.real(U_patch * np.conj(U_patch)) * (cfg.pixel_pitch_m ** 2)).astype(np.float64)

                # Insert patch into frame with bounds checking
                x0 = max(0, x_start)
                y0 = max(0, y_start)
                x1 = min(cfg.cols, x_start + cfg.patch_pix_x)
                y1 = min(cfg.rows, y_start + cfg.patch_pix_y)

                if x1 <= x0 or y1 <= y0:
                    continue

                px0 = x0 - x_start
                py0 = y0 - y_start
                px1 = px0 + (x1 - x0)
                py1 = py0 + (y1 - y0)

                frame_ph[y0:y1, x0:x1] += flux_patch[py0:py1, px0:px1]

                # -----------------------------------------------------
                # Step 2.3: Rachel pixel_attribution on the patch
                # -----------------------------------------------------
                local_att: Dict[Tuple[int, int], List[List[Any]]] = {}
                local_att = rm.pixel_attribution(
                    U_patch,
                    local_att,
                    cfg.pixel_pitch_m,
                    cfg.pixel_pitch_m,
                    power_ref_ph_s,
                    name,
                )
                merge_attribution_with_offset(frame_att, local_att, x_offset=x_start, y_offset=y_start)

            # ---------------------------------------------------------
            # Write frame dataset with zero-padded dataset name
            # ---------------------------------------------------------
            # Rachel uses "ph_flux_time_itr_" prefix
            dset_name = f"ph_flux_time_itr_{i:08d}"
            hf.create_dataset(dset_name, data=frame_ph.astype(np.float32), compression="gzip")

            # Attribution dict key: t_total_sec*u.s (compatible with circuitry’s unit handling)
            if u is not None:
                t_key = (float(i) * float(scene_window.get("dt_frame_s", 0.0))) * u.s
            else:
                # fallback (still usable if their circuitry doesn't require units)
                t_key = float(i) * float(scene_window.get("dt_frame_s", 0.0))

            att_by_time[t_key] = frame_att

    with open(out_att_pkl, "wb") as f:
        pickle.dump(att_by_time, f, protocol=pickle.HIGHEST_PROTOCOL)


# =============================================================================
# 7) Step 2.4 Smoke tests
# =============================================================================

def smoketest_outputs(out_h5: str, out_att_pkl: str, expect_frames: int = 3) -> None:
    """
    Minimal smoke test:
    - Validate HDF5 datasets exist and are finite
    - Validate attribution pickle structure
    """
    with h5py.File(out_h5, "r") as hf:
        keys = sorted([k for k in hf.keys() if "ph_flux_time_itr_" in k])
        if len(keys) < expect_frames:
            raise AssertionError(f"HDF5 contains {len(keys)} frames, expected >= {expect_frames}")
        for k in keys[:expect_frames]:
            arr = hf[k][:]
            if not np.isfinite(arr).all():
                raise AssertionError(f"Non-finite values in frame {k}")
            if arr.shape != (hf.attrs["rows"], hf.attrs["cols"]):
                raise AssertionError(f"Frame {k} has wrong shape {arr.shape}")

    with open(out_att_pkl, "rb") as f:
        att = pickle.load(f)
    if not isinstance(att, dict):
        raise AssertionError("Attribution pickle is not a dict.")
    # Check one entry
    any_key = next(iter(att.keys()))
    any_val = att[any_key]
    if not isinstance(any_val, dict):
        raise AssertionError("Attribution dict entries must be dicts keyed by (x,y).")


# =============================================================================
# 8) CLI entry (optional)
# =============================================================================

def main():
    import argparse

    ap = argparse.ArgumentParser(description="NEBULA Step 2: Optics + Attribution Bridge")
    ap.add_argument("--scene_pkl", required=True)
    ap.add_argument("--zodi_pkl", required=True)
    ap.add_argument("--observer_name", default=None)
    ap.add_argument("--window_idx", type=int, default=0)

    # Sensor/optics parameters (do not assume; user can override)
    ap.add_argument("--pixel_pitch_m", type=float, required=True)
    ap.add_argument("--focal_length_m", type=float, required=True)
    ap.add_argument("--aperture_diameter_m", type=float, required=True)
    ap.add_argument("--wavelength_m", type=float, required=True)

    ap.add_argument("--patch_pix", type=int, default=32)
    ap.add_argument("--gamma_subpix_per_airy", type=int, default=4)
    ap.add_argument("--observation_plane_sampling", type=int, default=256)

    ap.add_argument("--kernel_method", choices=["saft", "impulse"], default="saft")
    ap.add_argument("--also_compute_impulse", action="store_true")

    ap.add_argument("--limiting_mag", type=float, required=True)
    ap.add_argument("--bandpass_width_A", type=float, required=True)

    ap.add_argument("--zodi_model", choices=["plane3", "quad6"], default="plane3")

    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--out_att_pkl", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--smoketest", action="store_true")

    args = ap.parse_args()

    scene, zodi = load_pickles(args.scene_pkl, args.zodi_pkl)

    obs_name = args.observer_name or next(iter(scene.keys()))
    if obs_name not in scene or obs_name not in zodi:
        raise KeyError(f"Observer '{obs_name}' not found in both pickles.")

    scene_obs = scene[obs_name]
    zodi_obs = zodi[obs_name]

    w_scene, w_zodi = get_window(scene_obs, zodi_obs, args.window_idx)

    cfg = derive_optics_config(
        rows=int(scene_obs["rows"]),
        cols=int(scene_obs["cols"]),
        pixel_pitch_m=args.pixel_pitch_m,
        focal_length_m=args.focal_length_m,
        aperture_diameter_m=args.aperture_diameter_m,
        wavelength_m=args.wavelength_m,
        patch_pix_x=args.patch_pix,
        patch_pix_y=args.patch_pix,
        gamma_subpix_per_airy=args.gamma_subpix_per_airy,
        observation_plane_sampling=args.observation_plane_sampling,
    )

    rm = RadiometryModel()

    pupil, delta_p = build_pupil_mask(rm, cfg)

    power_ref = compute_power_ref_limiting_mag(
        rm,
        limiting_mag=args.limiting_mag,
        bandpass_width_A=args.bandpass_width_A,
        aperture_diameter_m=args.aperture_diameter_m,
    )

    # SAFT (primary)
    psf_saft = precompute_psf_kernel(rm, cfg, pupil, delta_p, method="saft")

    # Impulse (optional comparison)
    psf_impulse = None
    if args.also_compute_impulse:
        psf_impulse = precompute_psf_kernel(rm, cfg, pupil, delta_p, method="impulse")

    # Choose PSF method for rendering
    psf_use = psf_saft if args.kernel_method == "saft" else (psf_impulse or psf_saft)

    build_frames_and_attribution(
        scene_window=w_scene,
        zodi_window=w_zodi,
        rm=rm,
        cfg=cfg,
        psf=psf_use,
        power_ref_ph_s=power_ref,
        out_h5=args.out_h5,
        out_att_pkl=args.out_att_pkl,
        zodi_model=args.zodi_model,
        max_frames=args.max_frames,
    )

    if args.smoketest:
        smoketest_outputs(args.out_h5, args.out_att_pkl, expect_frames=min(3, args.max_frames or 3))


if __name__ == "__main__":
    main()
