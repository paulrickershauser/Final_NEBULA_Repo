# -*- coding: utf-8 -*-
"""
NEBULA_EBS_SDA_SIM_STEP2_OPTICS_RUNNER.py

Purpose
-------
Build Rachel-EBS-SDA-SIM-compatible "incident photon-rate frames" (HDF5) and a
pixel attribution dictionary (pickle) from NEBULA's per-window source pickles.

This runner implements NEBULA -> EBS_SDA_SIM "Step 2":
  2.1B  PSF kernel (impulse_response default; SAFT optional hook)
  2.2   Build per-frame ph/s/pixel images and write HDF5 datasets
  2.3   Build attribution dict immediately (no empty dict default)
  2.4   Smoketest / optional call into Rachel's circuitry.generate_events()

Key contracts (Rachel code expectations)
----------------------------------------
- Rachel's circuitry code scans HDF5 datasets containing "ph_flux_time_itr"
  and treats each dataset as a 2D array of incident photon rates [ph/s/pixel].
- Therefore, we write datasets:
      ph_flux_time_itr_00000000
      ph_flux_time_itr_00000001
      ...
  (zero-padded so alphanumeric iteration preserves frame order).

Inputs (NEBULA pickles)
-----------------------
1) WINDOW_SOURCES pickle:
   Default: <NEBULA_OUTPUT>/SCENE/obs_window_sources.pkl
   Contains per-window:
      - stars: per-star arrays of length n_frames (phi_ph_m2_s, x_pix, y_pix, ...)
      - targets: per-target arrays + coarse_indices (sparse / not always length n_frames)

2) ZODIACAL_LIGHT pickle:
   Default: <NEBULA_OUTPUT>/ZODIACAL_LIGHT/obs_zodiacal_light.pkl
   Contains per-window:
      - models['plane3']['coeffs_per_pixel'] : (n_frames, 3)
      - omega_pix['omega_pix_sr_scalar']     : scalar sr/pixel (analytic_scalar mode)

Outputs
-------
- HDF5 frames file: 2D float32 arrays of photons/sec/pixel (rows x cols)
- Attribution pickle: dict keyed by integer frame index (0..n_frames-1), each value is:
      {(y, x): [(source_id, frac), ...], ...}
  where fractions sum to 1 over the pixel's contributing sources (within numerical tolerance).

Notes
-----
- This runner intentionally does NOT modify Rachel's code.
- It uses Rachel's radiometry PSF primitives when available (impulse_response).
"""

# -----------------------------
# Standard library imports
# -----------------------------
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Third-party imports
# -----------------------------
import numpy as np
import h5py

# -----------------------------
# NEBULA imports (your repo)
# -----------------------------
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR  # type: ignore
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR  # type: ignore


# =============================================================================
# Small fail-fast helpers (avoid silent schema drift)
# =============================================================================

_MISSING = object()

def _as_dict(obj: Any, label: str) -> Dict[str, Any]:
    """Fail-fast: enforce dict-like."""
    if isinstance(obj, dict):
        return obj
    # TrackDict-like objects in NEBULA often support .get and key iteration; accept those.
    if hasattr(obj, "get") and hasattr(obj, "keys"):
        return obj  # type: ignore[return-value]
    raise TypeError(f"{label} must be dict-like; got {type(obj).__name__}")

def _req(d: Dict[str, Any], key: str, label: str) -> Any:
    """Fail-fast: require a key."""
    if key not in d:
        raise KeyError(f"Missing required key '{key}' in {label}. Keys={list(d.keys())[:32]}")
    return d[key]

def _track_get(track: Any, key: str, default: Any = _MISSING) -> Any:
    """
    Robust getter for TrackDict-like objects (dict or attribute-like).
    Mirrors NEBULA ZL stage approach.
    """
    if isinstance(track, dict):
        if key in track:
            return track[key]
        if default is not _MISSING:
            return default
        raise KeyError(f"Missing required key in track dict: {key!r}")
    if hasattr(track, "get"):
        try:
            val = track.get(key, _MISSING)  # type: ignore[attr-defined]
            if val is not _MISSING:
                return val
        except TypeError:
            pass
    if hasattr(track, key):
        return getattr(track, key)
    if default is not _MISSING:
        return default
    raise KeyError(f"Missing required key/attribute in track: {key!r}")


# =============================================================================
# Effective area (Step 1 result is used here)
# =============================================================================

def compute_effective_area_m2_from_active_sensor() -> float:
    """
    Compute effective collecting area [m^2] using ACTIVE_SENSOR.
    Uses:
      - aperture_area_m2 if present
      - else aperture_diameter_m if present
      - else raises
    Multiplies by optical_throughput if present, else uses 1.
    """
    # Pull collecting area (prefer explicit area).
    area = getattr(ACTIVE_SENSOR, "aperture_area_m2", None)
    if area is None:
        diam = getattr(ACTIVE_SENSOR, "aperture_diameter_m", None)
        if diam is None:
            raise RuntimeError(
                "ACTIVE_SENSOR missing aperture_area_m2 and aperture_diameter_m; "
                "cannot compute collecting area."
            )
        area = float(np.pi * (float(diam) / 2.0) ** 2)

    # Throughput (optional).
    thr = getattr(ACTIVE_SENSOR, "optical_throughput", None)
    thr_val = 1.0 if thr is None else float(thr)

    return float(area) * thr_val


# =============================================================================
# Zodi plane3 evaluation (per-pixel)
# =============================================================================

def normalized_uv_grid(rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build normalized detector coordinates u,v in [-1, 1] for every pixel center.

    We keep this local and explicit so the runner does not depend on internal
    stage modules being importable in a given Spyder session.
    """
    # Pixel-center convention: x in [0..cols-1], y in [0..rows-1].
    x = (np.arange(cols, dtype=np.float64) + 0.5)
    y = (np.arange(rows, dtype=np.float64) + 0.5)

    # Normalize to [-1, 1].
    u = (2.0 * x / float(cols)) - 1.0
    v = (2.0 * y / float(rows)) - 1.0

    # Meshgrid to (rows, cols).
    uu, vv = np.meshgrid(u, v, copy=False)
    return uu, vv

def eval_plane3_per_pixel(coeffs_per_pixel: np.ndarray, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """
    Evaluate plane3 model per pixel:
        phi = c0 + c1*u + c2*v

    Inputs
    ------
    coeffs_per_pixel : (n_frames, 3)
    uu, vv           : (rows, cols)

    Returns
    -------
    phi : (n_frames, rows, cols)
    """
    c = np.asarray(coeffs_per_pixel, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"plane3 coeffs must be (n_frames,3); got {c.shape}")
    # Broadcast plane coefficients over image grid.
    return (c[:, 0:1, None] + c[:, 1:2, None] * uu[None, :, :] + c[:, 2:3, None] * vv[None, :, :])


# =============================================================================
# PSF kernel (Step 2.1B)
# =============================================================================

def load_rachel_radiometry_model(ebs_pkg_dir: Path):
    """
    Import Rachel's radiometry.py as a library by adding the EBS_SDA_SIM package dir to sys.path.

    Parameters
    ----------
    ebs_pkg_dir : Path
        Directory containing the EBS_SDA_SIM package (the folder with __init__.py).

    Returns
    -------
    radiometry : module
        Imported EBS_SDA_SIM.radiometry module.
    rmtry_model : object
        Instantiated RadiometryModel.
    """
    # Make sure the package directory exists.
    if not ebs_pkg_dir.exists():
        raise FileNotFoundError(str(ebs_pkg_dir))

    # Add parent of package to sys.path (so `import EBS_SDA_SIM` works).
    parent = str(ebs_pkg_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Import.
    import EBS_SDA_SIM.radiometry as rmtry  # type: ignore

    # Instantiate Rachel's model.
    model = rmtry.RadiometryModel()

    return rmtry, model

def build_pupil_mask_circular(*, Np: int, aperture_diameter_m: float) -> Tuple[np.ndarray, float]:
    """
    Build a simple circular pupil mask sampled on an Np x Np grid.

    Returns
    -------
    pupil_mask : (Np, Np) float64 array
        1 inside aperture, 0 outside.
    delta_p : float
        Pupil-plane sample pitch in meters (aperture_diameter/Np).
    """
    # Pupil sample pitch.
    delta_p = float(aperture_diameter_m) / float(Np)

    # Coordinate vectors centered on 0.
    coords = (np.arange(Np, dtype=np.float64) - (Np / 2.0) + 0.5) * delta_p
    xx, yy = np.meshgrid(coords, coords, copy=False)

    # Radius threshold.
    rr = np.sqrt(xx * xx + yy * yy)
    pupil_mask = (rr <= (float(aperture_diameter_m) / 2.0)).astype(np.float64)

    return pupil_mask, delta_p

def psf_kernel_impulse_response(
    *,
    rmtry_model: Any,
    aperture_diameter_m: float,
    wavelength_m: float,
    focal_length_m: float,
    pixel_pitch_m: float,
    padding: int = 8,
) -> np.ndarray:
    """
    Build a PSF kernel using Rachel's RadiometryModel.impulse_response().

    Notes
    -----
    This is the "fast, Rachel-aligned kernel" path.

    Returns
    -------
    kernel : (K, K) float64
        Nonnegative, normalized so sum(kernel)=1.
    """
    # Choose a pupil sampling. Keep it modest to avoid slow kernels.
    # This is a kernel-generation detail; it does not change NEBULA pickles.
    Np = 256

    # Build pupil mask (dimensionless).
    pupil_mask, delta_p = build_pupil_mask_circular(Np=Np, aperture_diameter_m=aperture_diameter_m)

    # Call Rachel's impulse_response:
    # Signature is: impulse_response(self, pupil_mask, wvl, z2, dx, dy, paddingx, paddingy):contentReference[oaicite:6]{index=6}
    # We use dx=dy=delta_p (pupil-plane sampling) and z2=focal_length.
    h = rmtry_model.impulse_response(
        pupil_mask,
        wavelength_m,
        focal_length_m,
        delta_p,
        delta_p,
        padding,
        padding,
    )

    # Convert to intensity kernel.
    # If h is complex, take |h|^2; if real, square.
    h_arr = np.asarray(h)
    if np.iscomplexobj(h_arr):
        psf = (h_arr * np.conjugate(h_arr)).real
    else:
        psf = (h_arr * h_arr).astype(np.float64)

    # Resample PSF to pixel grid if necessary:
    # For a first pass, we assume h is already in the detector sampling of interest.
    # Normalize energy into a unit-sum kernel (dimensionless).
    psf = np.maximum(psf, 0.0)
    s = float(np.sum(psf))
    if s <= 0.0:
        raise RuntimeError("Impulse-response PSF sum is non-positive; cannot normalize.")
    psf /= s

    return psf


# =============================================================================
# Frame synthesis (Step 2.2) + attribution (Step 2.3)
# =============================================================================

def add_kernel_at(
    img: np.ndarray,
    kernel: np.ndarray,
    *,
    x_c: int,
    y_c: int,
    scale: float,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    Add `scale * kernel` into `img` centered at (x_c, y_c).

    Returns
    -------
    (x0, x1, y0, y1), patch
        The patch bounds in img coordinates and the patch data actually added
        (cropped at edges), for optional attribution use.
    """
    rows, cols = img.shape
    kH, kW = kernel.shape

    # Kernel center indices.
    kcx = kW // 2
    kcy = kH // 2

    # Compute image bounds.
    x0 = max(0, x_c - kcx)
    x1 = min(cols, x_c - kcx + kW)
    y0 = max(0, y_c - kcy)
    y1 = min(rows, y_c - kcy + kH)

    # Compute corresponding kernel bounds.
    kx0 = x0 - (x_c - kcx)
    kx1 = kx0 + (x1 - x0)
    ky0 = y0 - (y_c - kcy)
    ky1 = ky0 + (y1 - y0)

    # Crop kernel and add.
    patch = kernel[ky0:ky1, kx0:kx1] * float(scale)
    img[y0:y1, x0:x1] += patch

    return (x0, x1, y0, y1), patch

def build_attribution_from_patches(
    attribution: Dict[Tuple[int, int], List[Tuple[str, float]]],
    *,
    bounds: Tuple[int, int, int, int],
    patch: np.ndarray,
    source_id: str,
    power_ref: float,
) -> None:
    """
    Update a per-frame attribution dict using Rachel's thresholding concept:

      power_thres = 0.1 * power_ref
      mark pixels where patch_value > power_thres
      store fractional contribution patch_value / sum(patch_value) for that source

    Here:
      - patch_value is in ph/s/pixel (because our frames are ph/s/pixel).
      - power_ref must also be in ph/s (collected rate of a limiting reference).

    The attribution dict maps (y,x) -> list[(source_id, frac), ...].

    This is designed to be compatible with later arbiter-side "pick max contributor"
    logic (or per-pixel mixture logic) without requiring changes to Rachel’s circuitry.
    """
    x0, x1, y0, y1 = bounds

    # Threshold policy (Rachel): 10% of reference power.
    power_thres = 0.1 * float(power_ref)

    # If patch has no energy, nothing to attribute.
    total = float(np.sum(patch))
    if total <= 0.0:
        return

    # Find above-threshold pixels.
    yy, xx = np.where(patch > power_thres)
    if yy.size == 0:
        return

    # Fractional contributions over the *entire patch energy* (not only above-threshold),
    # consistent with "percentage = pixel / sum(Image_Watts)" concept.
    # (If you prefer to normalize over only above-threshold pixels, change here.)
    for j in range(yy.size):
        py = int(y0 + yy[j])
        px = int(x0 + xx[j])
        frac = float(patch[yy[j], xx[j]] / total)
        attribution.setdefault((py, px), []).append((str(source_id), frac))


# =============================================================================
# Main Step 2 execution
# =============================================================================

def run_step2_build_frames(
    *,
    window_sources_path: Path,
    zodi_path: Path,
    ebs_pkg_dir: Path,
    output_h5_path: Path,
    output_attr_path: Path,
    observer_name: Optional[str],
    window_index: int,
    psf_mode: str,
    max_frames: Optional[int],
) -> None:
    """
    Execute Step 2 end-to-end and write outputs to disk.
    """
    # -----------------------------
    # Load pickles
    # -----------------------------
    with open(window_sources_path, "rb") as f:
        obs_window_sources = pickle.load(f)
    with open(zodi_path, "rb") as f:
        obs_zodi = pickle.load(f)

    obs_window_sources = _as_dict(obs_window_sources, "obs_window_sources")
    obs_zodi = _as_dict(obs_zodi, "obs_zodiacal_light")

    # -----------------------------
    # Select observer
    # -----------------------------
    if observer_name is None:
        # Deterministic default: first key in sorted order.
        observer_name = sorted(list(obs_window_sources.keys()))[0]

    if observer_name not in obs_window_sources:
        raise KeyError(f"Observer {observer_name!r} not found in obs_window_sources keys.")

    if observer_name not in obs_zodi:
        raise KeyError(f"Observer {observer_name!r} not found in obs_zodiacal_light keys.")

    obs_track = _as_dict(obs_window_sources[observer_name], f"obs_window_sources[{observer_name}]")
    zodi_track = _as_dict(obs_zodi[observer_name], f"obs_zodiacal_light[{observer_name}]")

    # -----------------------------
    # Select window
    # -----------------------------
    windows = _track_get(obs_track, "windows")
    windows = list(windows)
    if window_index < 0 or window_index >= len(windows):
        raise IndexError(f"window_index={window_index} out of range; n_windows={len(windows)}")

    win = _as_dict(windows[window_index], f"windows[{window_index}]")
    rows = int(_track_get(obs_track, "rows"))
    cols = int(_track_get(obs_track, "cols"))
    n_frames = int(_track_get(win, "n_frames"))

    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    # Coarse index mapping (window uses inclusive indices in your schema).
    start_index = int(_track_get(win, "start_index"))
    # end_index exists but is not required for the mapping.
    # end_index = int(_track_get(win, "end_index"))

    # -----------------------------
    # Effective area
    # -----------------------------
    effective_area_m2 = compute_effective_area_m2_from_active_sensor()

    # -----------------------------
    # Zodi plane3 coeffs (per_pixel)
    # -----------------------------
    zodi_windows = _track_get(zodi_track, "windows")
    zodi_windows = list(zodi_windows)

    # Match by window_index if possible; otherwise assume same ordering.
    zodi_win = None
    for zw in zodi_windows:
        zwd = _as_dict(zw, "zodi_window")
        if int(zwd.get("window_index", -999)) == int(win.get("window_index", window_index)):
            zodi_win = zwd
            break
    if zodi_win is None:
        # Fall back to same positional index.
        zodi_win = _as_dict(zodi_windows[window_index], f"zodi_windows[{window_index}]")

    models = _req(_as_dict(_req(zodi_win, "models", "zodi_win"), "zodi_win.models"), "plane3", "zodi_win.models")
    coeffs_pp = _req(_as_dict(models, "plane3_model"), "coeffs_per_pixel", "plane3_model")
    coeffs_pp = np.asarray(coeffs_pp, dtype=np.float64)
    coeffs_pp = coeffs_pp[:n_frames, :]

    # Precompute full-image u,v.
    uu, vv = normalized_uv_grid(rows, cols)

    # Evaluate zodi per-pixel phi [ph m^-2 s^-1 pix^-1] -> multiply by area for ph/s/pix.
    zodi_phi_pp = eval_plane3_per_pixel(coeffs_pp, uu, vv)  # (n_frames, rows, cols)
    zodi_rate_pp = zodi_phi_pp * float(effective_area_m2)

    # -----------------------------
    # Load stars + targets
    # -----------------------------
    stars = _as_dict(_track_get(win, "stars", default={}), "win.stars")
    targets = _as_dict(_track_get(win, "targets", default={}), "win.targets")

    # -----------------------------
    # Build PSF kernel (2.1B)
    # -----------------------------
    rmtry, rmtry_model = load_rachel_radiometry_model(ebs_pkg_dir)

    # Choose reasonable defaults for kernel generation from ACTIVE_SENSOR and NEBULA optical config:
    # - aperture diameter comes from ACTIVE_SENSOR if present, else from area.
    ap_diam = getattr(ACTIVE_SENSOR, "aperture_diameter_m", None)
    if ap_diam is None:
        ap_area = getattr(ACTIVE_SENSOR, "aperture_area_m2", None)
        if ap_area is None:
            raise RuntimeError("ACTIVE_SENSOR must provide aperture_diameter_m or aperture_area_m2.")
        ap_diam = 2.0 * np.sqrt(float(ap_area) / np.pi)

    # Effective wavelength: use Gaia G pivot (~673 nm) if you are using Gaia G-band products.
    # (This runner keeps the value explicit.)
    wavelength_m = 673.0e-9

    # Focal length and pixel pitch should come from ACTIVE_SENSOR if present.
    focal_length_m = float(getattr(ACTIVE_SENSOR, "focal_length"))
    pixel_pitch_m = float(getattr(ACTIVE_SENSOR, "pixel_pitch"))

    # Kernel mode:
    # - "impulse" : Rachel RadiometryModel.impulse_response
    # - "delta"   : no spreading
    psf_mode = psf_mode.strip().lower()
    if psf_mode not in ("impulse", "delta"):
        raise ValueError("psf_mode must be 'impulse' or 'delta' for this runner.")

    if psf_mode == "delta":
        kernel = np.zeros((1, 1), dtype=np.float64)
        kernel[0, 0] = 1.0
    else:
        kernel = psf_kernel_impulse_response(
            rmtry_model=rmtry_model,
            aperture_diameter_m=float(ap_diam),
            wavelength_m=float(wavelength_m),
            focal_length_m=float(focal_length_m),
            pixel_pitch_m=float(pixel_pitch_m),
            padding=8,
        )

    # -----------------------------
    # Attribution reference power (Rachel concept)
    # -----------------------------
    # We need a "power_ref" in the same units as our patches (ph/s/pix).
    # Use the window meta if present: gaia_window_meta.mag_limit_G is a good choice.
    gaia_meta = _track_get(win, "gaia_window_meta", default={})
    gaia_meta = _as_dict(gaia_meta, "gaia_window_meta")

    mag_limit_g = gaia_meta.get("mag_limit_G", None)
    if mag_limit_g is None:
        # Fail fast rather than guessing a magnitude.
        raise KeyError(
            "Missing gaia_window_meta.mag_limit_G in window sources; cannot define power_ref for attribution."
        )

    # Use NEBULA_STAR_PHOTONS conversion logic indirectly would be ideal, but we keep this runner standalone:
    # Define power_ref as the *collected* photon rate for a star at mag_limit_g:
    # power_ref = phi_ph_m2_s(mag_limit_g) * effective_area_m2
    #
    # We reuse NEBULA_STAR_PHOT
