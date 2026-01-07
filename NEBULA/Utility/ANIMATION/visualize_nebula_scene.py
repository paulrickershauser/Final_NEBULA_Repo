"""
visualize_nebula_scene.py

Standalone visualization utility for NEBULA pickles:
- WINDOW_SOURCES (stars + targets) pickle
- ZODIACAL_LIGHT (background model coefficients) pickle

Creates two optional MP4 animations:
1) POINTS: zodiacal background + "point" sources deposited into pixels (no PSF)
2) GAUSSIAN: background + toy Gaussian PSF sources rendered into an image

Key improvements vs v1:
- Prevents "blurry / resolution collapse" in MP4 by:
    * using interpolation='nearest' (no smoothing)
    * using an explicit high-quality ffmpeg writer (CRF/preset)
- Optional target highlight rings (red circles) without altering target brightness
- Targets are rendered identically to stars (same deposition/PSF and same units)
- Colorbar applies to the full rendered image: zodiacal background + stars + targets
- tqdm progress bar while encoding (with ETA) using a manual ffmpeg frame loop
"""

from __future__ import annotations

import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FFMpegWriter

# Optional: progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------
# User settings
# -----------------------------

# Paths
#
# This file is expected at:
#   <NEBULA_ROOT>/Utility/ANIMATION/visualize_nebula_scene.py
# so the repo root is two parents up from this file.
#
# If the shared path config is available, use it; otherwise fall back to
# <NEBULA_ROOT>/NEBULA_OUTPUT.
def _infer_nebula_root() -> Path:
    return Path(__file__).resolve().parents[2]

NEBULA_ROOT = _infer_nebula_root()
if str(NEBULA_ROOT) not in sys.path:
    sys.path.insert(0, str(NEBULA_ROOT))

try:
    from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR as _NEBULA_OUTPUT_DIR  # type: ignore
    NEBULA_OUTPUT_DIR = str(_NEBULA_OUTPUT_DIR)
except Exception:
    NEBULA_OUTPUT_DIR = str(NEBULA_ROOT / "NEBULA_OUTPUT")

WINDOW_SOURCES_PKL = os.path.join(NEBULA_OUTPUT_DIR, "SCENE", "obs_window_sources.pkl")
ZODIACAL_LIGHT_PKL = os.path.join(NEBULA_OUTPUT_DIR, "ZODIACAL_LIGHT", "obs_zodiacal_light.pkl")

ANIMATIONS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "ANIMATIONS")
os.makedirs(ANIMATIONS_DIR, exist_ok=True)


# Observer/window selection
OBSERVER_NAME = "SBSS (USA 216)"
WINDOW_INDEX = 0

# Which zodiacal background model to visualize
# - "plane3": linear plane in (u,v) (3 params)
# - "quad6":  quadratic in (u,v) (6 params)
ZODI_MODEL = "quad6"

# Frame selection
FRAME_START = 0

# Render every Nth simulation frame.
FRAME_STRIDE = 1

# Playback FPS in the MP4
FPS = 20

# Rendering mode toggles
SAVE_POINTS_ANIM = True
SAVE_GAUSS_ANIM = True

# Output filenames (written to NEBULA_OUTPUT/ANIMATIONS by default)
POINTS_ANIM_OUT = os.path.join(ANIMATIONS_DIR, "nebula_points_unified.mp4")
GAUSS_ANIM_OUT = os.path.join(ANIMATIONS_DIR, "nebula_gauss_unified.mp4")

# POINTS source deposition mode (no PSF)
# - "nearest": deposit all source photons into the nearest pixel
# - "bilinear": deposit into a 2x2 neighborhood using bilinear weights (subpixel, still no PSF)
POINTS_DEPOSIT_MODE = "bilinear"

# Gaussian PSF settings (toy model for visualization)
FWHM_PIX = 1.0
PSF_NSIGMA = 4.0

# Performance: cache integrated PSF kernels by quantized subpixel offset.
# This can drastically speed up long renders (many stars × many frames).
# Set PSF_CACHE_STEP=None to disable quantization/caching.
PSF_CACHE_STEP = None  # pixels (None keeps full subpixel precision; set e.g. 0.05 for speed)

# Display settings
IMSHOW_ORIGIN = "upper"       # image-like coordinates: y=0 at top
IMSHOW_CMAP = "gray"
IMSHOW_INTERP = "nearest"     # IMPORTANT: avoids smoothing blur

# Photometric scaling: one scale applies to everything (bg + stars + targets)
USE_LOG_NORM = True

# If None -> auto from data / simple analytic estimates
VMIN = None   # in ph m^-2 frame^-1 pix^-1
VMAX = None   # in ph m^-2 frame^-1 pix^-1

# If auto-scaling: set VMIN relative to zodiacal background
AUTO_VMIN_BG_FACTOR = 0.8

# Target highlight rings (overlay only; does NOT change target brightness)
DRAW_TARGET_CIRCLES = False
TARGET_CIRCLE_SIZE = 120.0   # scatter marker "area" in points^2
TARGET_CIRCLE_LW = 1.6

# Scale bar
SHOW_SCALE_BAR = False
SCALE_BAR_ARCMIN = 10.0

# Figure / encoding quality
#
# Notes:
# - "Blurry as it plays" is usually MP4 compression interacting with fine detail + low bitrate.
# - CRF controls constant-quality H.264:
#     * 0 is lossless (very large files)
#     * ~12-16 is visually near-lossless for this type of content
#     * 18-23 is typical
FFMPEG_CODEC = "libx264"
FFMPEG_CRF = 14
FFMPEG_PRESET = None  # e.g. "slow" (optional; some ffmpeg builds reject -preset)
FFMPEG_BITRATE = None  # kbps; leave None when using CRF
FFMPEG_PIX_FMT = "yuv420p"

# Increase DPI if you want sharper text/axes at the cost of file size and encoding time.
SAVE_DPI = 140


# -----------------------------
# Helpers
# -----------------------------

def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalized_uv_grids(rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build normalized UV grids used by the zodiacal polynomial models.

    u = 2*x/(cols-1) - 1
    v = 2*y/(rows-1) - 1
    """
    x = np.arange(cols, dtype=np.float64)[None, :]
    y = np.arange(rows, dtype=np.float64)[:, None]
    u = 2.0 * x / (cols - 1.0) - 1.0
    v = 2.0 * y / (rows - 1.0) - 1.0
    return u, v


def _uv_grids(rows: int, cols: int, uv_basis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build UV grids consistent with the basis used by the zodiacal polynomial models.

    Supported uv_basis:
    - "normalized_uv": u,v in [-1,1] with the usual mapping u=2*x/(cols-1)-1, v=2*y/(rows-1)-1
    - "pixel_xy":      u=x_pix, v=y_pix in raw pixel coordinates
    """
    b = str(uv_basis).strip().lower()
    if b in {"normalized_uv", "normalized", "uv"}:
        return _normalized_uv_grids(rows, cols)
    if b in {"pixel_xy", "pixel", "xy"}:
        x = np.arange(cols, dtype=np.float64)[None, :]
        y = np.arange(rows, dtype=np.float64)[:, None]
        return x, y
    raise ValueError(f"Unknown uv_basis={uv_basis!r}; expected 'normalized_uv' or 'pixel_xy'")



def _safe_round_pix(x: np.ndarray, y: np.ndarray, cols: int, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Round float pixel-center coordinates to nearest integer pixel indices and clip to sensor bounds.

    NOTE: uses floor(x+0.5) rather than np.round to avoid "banker's rounding" at half-integers and
    to match the coordinate convention used in _add_gaussian_psf.
    """
    xi = np.clip(np.floor(x + 0.5).astype(np.int64), 0, cols - 1)
    yi = np.clip(np.floor(y + 0.5).astype(np.int64), 0, rows - 1)
    return xi, yi



def _arcsec_per_pix_from_omega_pix_sr(omega_pix_sr: float) -> float:
    """
    Compute an approximate angular pixel scale (arcsec/pix) from Ω_pix [sr/pix].

    Assumes (approximately) square pixels and small angles:
        Ω_pix ≈ (theta_pix_rad)^2  =>  theta_pix_rad ≈ sqrt(Ω_pix)
        arcsec/pix = theta_pix_rad * 206264.806...
    """
    if not np.isfinite(omega_pix_sr) or omega_pix_sr <= 0.0:
        raise ValueError(f"omega_pix_sr must be finite and >0; got {omega_pix_sr!r}")
    theta_pix_rad = math.sqrt(float(omega_pix_sr))
    return theta_pix_rad * 206264.80624709636


def _apply_night_sky_axes_style(ax: Any) -> None:
    """
    Make axes look more like a camera frame (dark theme).
    """
    ax.set_facecolor("black")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def _add_scale_bar(
    ax: Any,
    *,
    rows: int,
    cols: int,
    arcsec_per_pix: float,
    length_arcmin: float,
    origin: str = "upper",
) -> None:
    """
    Draw a simple scale bar in pixel coordinates.
    """
    if not (np.isfinite(arcsec_per_pix) and arcsec_per_pix > 0):
        return

    length_arcsec = float(length_arcmin) * 60.0
    length_pix = length_arcsec / arcsec_per_pix

    # Place near bottom-left, in data coords
    x0 = 0.05 * cols
    x1 = x0 + length_pix
    y0 = 0.93 * rows if origin == "upper" else 0.07 * rows

    ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.0, solid_capstyle="butt")
    ax.text(
        0.5 * (x0 + x1),
        y0 - (0.03 * rows if origin == "upper" else -0.03 * rows),
        f"{length_arcmin:g} arcmin",
        color="white",
        ha="center",
        va="top" if origin == "upper" else "bottom",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"),
    )


def _background_map_per_s(coeffs: np.ndarray, u: np.ndarray, v: np.ndarray, model: str) -> np.ndarray:
    """
    Evaluate a polynomial background model on a u,v grid.

    Returns: ph / m^2 / s / pixel
    """
    if model == "plane3":
        c0, c1, c2 = coeffs
        return c0 + c1 * u + c2 * v
    elif model == "quad6":
        c0, c1, c2, c3, c4, c5 = coeffs
        return c0 + c1 * u + c2 * v + c3 * (u * u) + c4 * (u * v) + c5 * (v * v)
    else:
        raise ValueError(f"Unknown background model: {model!r}")


def _sigma_from_fwhm(fwhm_pix: float) -> float:
    return float(fwhm_pix) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


try:
    from scipy.special import erf  # type: ignore
except Exception:  # pragma: no cover
    erf = np.vectorize(math.erf)


def _integrated_gaussian_kernel(
    sigma: float,
    radius: int,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Integrated Gaussian kernel over pixel squares, centered at (dx,dy) offset
    from the central pixel center.

    dx,dy are subpixel offsets in pixel units (typically in [-0.5,0.5)).

    Returns a (2*radius+1, 2*radius+1) array of weights summing to 1.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    ys = np.arange(-radius, radius + 1, dtype=np.float64)

    x0 = xs - 0.5 - dx
    x1 = xs + 0.5 - dx
    y0 = ys - 0.5 - dy
    y1 = ys + 0.5 - dy

    inv = 1.0 / (math.sqrt(2.0) * sigma)

    wx = 0.5 * (erf(x1 * inv) - erf(x0 * inv))
    wy = 0.5 * (erf(y1 * inv) - erf(y0 * inv))

    k = np.outer(wy, wx)
    s = float(np.sum(k))
    if s <= 0:
        return k
    return k / s


def _add_gaussian_psf(
    img: np.ndarray,
    x: float,
    y: float,
    photons_total: float,
    sigma: float,
    nsigma: float,
    *,
    cache: Optional[Dict[Tuple[float, float], np.ndarray]] = None,
    cache_step: Optional[float] = None,
) -> None:
    """
    Add a PSF for a source at subpixel position (x,y) depositing photons_total into img.

    img units are per pixel, so photons_total is split across pixels by the kernel weights.

    Performance note:
    - If cache is provided and cache_step is a positive float, kernels are cached by
      quantized (dx,dy) subpixel offsets. This can greatly speed up long animations.
    """
    rows, cols = img.shape
    if photons_total <= 0:
        return

    # NOTE on coordinate conventions:
    # NEBULA pickles use pixel-center coordinates (0..cols-1, 0..rows-1) where
    # pixel i is centered at i and spans [i-0.5, i+0.5].
    # Use the nearest integer pixel center so dx,dy are in [-0.5, 0.5).
    xc = int(math.floor(x + 0.5))
    yc = int(math.floor(y + 0.5))

    dx = x - xc
    dy = y - yc
    radius = int(math.ceil(nsigma * sigma))

    if cache is not None and cache_step is not None and cache_step > 0:
        dxq = round(float(dx) / cache_step) * cache_step
        dyq = round(float(dy) / cache_step) * cache_step
        key = (float(dxq), float(dyq))
        k = cache.get(key)
        if k is None:
            k = _integrated_gaussian_kernel(sigma=sigma, radius=radius, dx=float(dxq), dy=float(dyq))
            cache[key] = k
    else:
        k = _integrated_gaussian_kernel(sigma=sigma, radius=radius, dx=dx, dy=dy)

    x0 = xc - radius
    y0 = yc - radius
    x1 = xc + radius
    y1 = yc + radius

    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(cols - 1, x1)
    iy1 = min(rows - 1, y1)

    kx0 = ix0 - x0
    ky0 = iy0 - y0
    kx1 = kx0 + (ix1 - ix0) + 1
    ky1 = ky0 + (iy1 - iy0) + 1

    patch = k[ky0:ky1, kx0:kx1]
    img[iy0:iy1 + 1, ix0:ix1 + 1] += photons_total * patch


def _add_bilinear_splat(img: np.ndarray, x: float, y: float, photons_total: float) -> None:
    """
    "No PSF" subpixel deposition using bilinear weights into a 2x2 neighborhood.

    - photons_total is conserved (weights sum to 1 where pixels exist).
    - Units remain ph m^-2 frame^-1 pix^-1 once deposited.
    """
    rows, cols = img.shape
    if photons_total <= 0:
        return

    # Pixel corner indices
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))

    # Fractional offsets in [0,1)
    fx = float(x) - x0
    fy = float(y) - y0

    # The four neighbors
    coords = [
        (y0,     x0,     (1 - fx) * (1 - fy)),
        (y0,     x0 + 1, fx * (1 - fy)),
        (y0 + 1, x0,     (1 - fx) * fy),
        (y0 + 1, x0 + 1, fx * fy),
    ]

    # Clip + renormalize if near edge
    valid = []
    wsum = 0.0
    for yy, xx, w in coords:
        if 0 <= yy < rows and 0 <= xx < cols and w > 0:
            valid.append((yy, xx, w))
            wsum += w
    if wsum <= 0:
        return

    for yy, xx, w in valid:
        img[yy, xx] += photons_total * (w / wsum)


@dataclass
class TargetTrack:
    name: str
    coarse_indices: np.ndarray
    x_pix: np.ndarray
    y_pix: np.ndarray
    flux_ph_m2_frame: np.ndarray


def _target_at_frame(track: TargetTrack, frame_idx: int) -> Optional[Tuple[float, float, float]]:
    """
    Return (x,y,flux) at this frame if target exists in this frame, else None.
    """
    ci = track.coarse_indices
    j = np.searchsorted(ci, frame_idx)
    if j < ci.size and int(ci[j]) == int(frame_idx):
        return float(track.x_pix[j]), float(track.y_pix[j]), float(track.flux_ph_m2_frame[j])
    return None


def _ffmpeg_encoders_text(ffmpeg_exe: str) -> str:
    """Return the text output of `ffmpeg -encoders`, or '' if probing fails."""
    try:
        return subprocess.check_output(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def _ffmpeg_has_encoder(encoders_text: str, encoder_name: str) -> bool:
    # Match whole words to avoid accidental substring matches.
    return bool(re.search(rf"\b{re.escape(encoder_name)}\b", encoders_text))


def _build_writer(fps: int) -> FFMpegWriter:
    """
    Build an ffmpeg writer with a codec that exists in the user's ffmpeg build.

    The default is H.264 via libx264 for quality/size, but many Windows ffmpeg
    distributions ship without libx264 enabled, which causes:
        Unknown encoder 'libx264'

    This function probes available encoders and falls back to broadly available
    encoders with encoder-appropriate quality flags.
    """
    # Respect Matplotlib's configured ffmpeg path if provided; otherwise use PATH.
    ffmpeg_exe = str(plt.rcParams.get("animation.ffmpeg_path", "ffmpeg"))
    if not ffmpeg_exe or ffmpeg_exe == "ffmpeg":
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

    enc_text = _ffmpeg_encoders_text(ffmpeg_exe)

    # Prefer the user-requested codec if it exists; otherwise pick a safe fallback.
    candidates = [
        str(FFMPEG_CODEC),
        "libx264",
        "h264_nvenc",
        "h264_qsv",
        "h264_vaapi",
        "libopenh264",
        "mpeg4",
    ]

    codec: str = "mpeg4"
    if enc_text:
        for c in candidates:
            if c and _ffmpeg_has_encoder(enc_text, c):
                codec = c
                break

    # Base args: enforce a broadly compatible pixel format.
    extra: List[str] = ["-pix_fmt", str(FFMPEG_PIX_FMT)]

    # Encoder-specific quality controls.
    bitrate = FFMPEG_BITRATE
    if codec in ("libx264", "libx264rgb"):
        extra += ["-crf", str(int(FFMPEG_CRF))]
        if FFMPEG_PRESET:
            extra += ["-preset", str(FFMPEG_PRESET)]
    elif codec == "mpeg4":
        # MPEG-4 Part 2: quantizer control; lower is better quality (1..31).
        # q=2 is visually very high quality for synthetic imagery.
        extra += ["-q:v", "2"]
    else:
        # Hardware/OpenH264 encoders: prefer bitrate control if user did not set one.
        if bitrate is None:
            bitrate = 20000  # kbps (~20 Mbps) default

    return FFMpegWriter(
        fps=fps,
        codec=codec,
        bitrate=bitrate,
        extra_args=extra,
    )


def _make_norm(vmin: float, vmax: float):
    if USE_LOG_NORM:
        # LogNorm requires strictly positive vmin.
        vmin = max(float(vmin), 1e-6)
        vmax = max(float(vmax), vmin * 1.01)
        return LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        vmax = max(float(vmax), float(vmin) + 1e-12)
        return Normalize(vmin=float(vmin), vmax=vmax, clip=True)


def _save_with_tqdm(fig, update_func, n_frames: int, out_path: str, fps: int, dpi: int, desc: str) -> None:
    writer = _build_writer(fps=fps)

    use_pbar = tqdm is not None
    pbar = tqdm(total=n_frames, desc=desc, unit="frame") if use_pbar else None

    # IMPORTANT: keep figure facecolor stable across frames for encoding
    savefig_kwargs = dict(facecolor=fig.get_facecolor())

    with writer.saving(fig, out_path, dpi=dpi):
        for k in range(n_frames):
            update_func(k)
            writer.grab_frame(**savefig_kwargs)
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print("[INFO] Loading pickles...")
    ws = _load_pickle(WINDOW_SOURCES_PKL)
    zodi = _load_pickle(ZODIACAL_LIGHT_PKL)

    if OBSERVER_NAME not in ws:
        raise KeyError(f"Observer {OBSERVER_NAME!r} not found in WINDOW_SOURCES pickle.")
    if OBSERVER_NAME not in zodi:
        raise KeyError(f"Observer {OBSERVER_NAME!r} not found in ZODIACAL_LIGHT pickle.")

    ws_track = ws[OBSERVER_NAME]
    z_track = zodi[OBSERVER_NAME]

    ws_win = ws_track["windows"][WINDOW_INDEX]
    z_win = z_track["windows"][WINDOW_INDEX]

    rows = int(ws_track["rows"])
    cols = int(ws_track["cols"])
    n_frames = int(ws_win["n_frames"])

    # Exposure time (prefer per-frame if provided; fallback to constant)
    any_star = next(iter(ws_win["stars"].values()))
    t_exp_s = np.asarray(any_star["t_exp_s"], dtype=np.float64)
    if t_exp_s.size == n_frames:
        t_exp = float(t_exp_s[0])
    else:
        t_exp = float(t_exp_s[0])
        t_exp_s = np.full(n_frames, t_exp, dtype=np.float64)

    # Zodiacal background coefficients
    z_block = z_win["zodi"]
    if ZODI_MODEL not in z_block["models"]:
        raise KeyError(
            f"ZODI_MODEL {ZODI_MODEL!r} not found in zodi['models']. "
            f"Available: {list(z_block['models'].keys())}"
        )

    coeff_key = "coeffs_per_pixel"  # already converted on Windows side
    coeffs_per_s_pix = np.asarray(z_block["models"][ZODI_MODEL][coeff_key], dtype=np.float64)  # (n_frames, n_params)
    if coeffs_per_s_pix.shape[0] != n_frames:
        raise ValueError(
            f"Background coeffs have n_frames={coeffs_per_s_pix.shape[0]} but WINDOW_SOURCES has {n_frames}"
        )

    # Time axis for display (prefer the stage-provided ISO strings)
    times_utc_iso = z_win.get("times_utc_iso", None)

    # Pixel scale for scale bar (derived from Ω_pix)
    omega_pix_block = z_block.get("omega_pix", {}) or {}
    omega_pix_sr_scalar = float(omega_pix_block.get("omega_pix_sr_scalar", float("nan")))
    omega_pix_sr = omega_pix_block.get("omega_pix_sr", None)

    omega_pix_sr_rep = float("nan")
    if np.isfinite(omega_pix_sr_scalar) and omega_pix_sr_scalar > 0.0:
        omega_pix_sr_rep = omega_pix_sr_scalar
    elif omega_pix_sr is not None:
        omega_arr = np.asarray(omega_pix_sr, dtype=np.float64)
        if omega_arr.size > 0 and np.isfinite(omega_arr).any():
            omega_pix_sr_rep = float(np.nanmedian(omega_arr))

    arcsec_per_pix = float("nan")
    if np.isfinite(omega_pix_sr_rep) and omega_pix_sr_rep > 0.0:
        arcsec_per_pix = _arcsec_per_pix_from_omega_pix_sr(omega_pix_sr_rep)
        print(f"[INFO] omega_pix_sr ~ {omega_pix_sr_rep:.6e} sr/pix -> ~{arcsec_per_pix:.2f} arcsec/pix")

    # Precompute u,v grid (must match the basis used for the fitted coefficients)
    uv_basis = str(
        z_block.get("models", {}).get(ZODI_MODEL, {}).get(
            "uv_basis",
            z_block.get("sampling", {}).get("uv_basis", "normalized_uv"),
        )
    )
    u, v = _uv_grids(rows, cols, uv_basis)


    # Stars list
    star_items = list(ws_win["stars"].items())

    # Targets list
    target_tracks: List[TargetTrack] = []
    for name, td in ws_win["targets"].items():
        target_tracks.append(
            TargetTrack(
                name=str(name),
                coarse_indices=np.asarray(td["coarse_indices"], dtype=np.int64),
                x_pix=np.asarray(td["x_pix"], dtype=np.float64),
                y_pix=np.asarray(td["y_pix"], dtype=np.float64),
                flux_ph_m2_frame=np.asarray(td["flux_ph_m2_frame"], dtype=np.float64),
            )
        )

    print(f"[INFO] Observer={OBSERVER_NAME!r} window={WINDOW_INDEX} size={rows}x{cols} frames={n_frames}")
    print(f"[INFO] Stars={len(star_items)} targets={len(target_tracks)} t_exp={t_exp:.6f} s model={ZODI_MODEL}")

    # -----------------------------
    # Stats (same as v1, plus max flux across all frames)
    # -----------------------------
    print("\n[STATS] Stars (total photons per frame, ph m^-2 frame^-1):")
    star_meds = []
    star_maxes = []
    star_global_max = 0.0
    for _, s in star_items:
        f = np.asarray(s["flux_ph_m2_frame"], dtype=np.float64)
        star_meds.append(np.median(f))
        star_maxes.append(np.max(f))
        star_global_max = max(star_global_max, float(np.max(f)))
    star_meds = np.asarray(star_meds)
    star_maxes = np.asarray(star_maxes)
    print(f"  median(star_median_flux) = {np.median(star_meds):.3e}")
    print(f"  min/max(star_median_flux)= {np.min(star_meds):.3e} / {np.max(star_meds):.3e}")
    print(f"  brightest star max flux  = {np.max(star_maxes):.3e}")

    tgt_meds = []
    tgt_global_max = 0.0
    for t in target_tracks:
        if t.flux_ph_m2_frame.size > 0:
            tgt_meds.append(np.median(t.flux_ph_m2_frame))
            tgt_global_max = max(tgt_global_max, float(np.max(t.flux_ph_m2_frame)))
    if tgt_meds:
        tgt_meds = np.asarray(tgt_meds)
        print("\n[STATS] Targets (total photons per frame when present, ph m^-2 frame^-1):")
        print(f"  median(target_median_flux) = {np.median(tgt_meds):.3e}")
        print(f"  min/max(target_median_flux)= {np.min(tgt_meds):.3e} / {np.max(tgt_meds):.3e}")

    # Representative background stats (per pixel, per frame)
    fi_ref = int(FRAME_START)
    bg_ref = _background_map_per_s(coeffs_per_s_pix[fi_ref], u, v, ZODI_MODEL) * float(t_exp_s[fi_ref])  # ph/m^2/frame/pix
    bg_med = float(np.median(bg_ref))
    bg_min = float(np.min(bg_ref))
    bg_max = float(np.max(bg_ref))
    bg_total = float(np.sum(bg_ref))

    print("\n[STATS] Zodiacal background (per pixel per frame, ph m^-2 frame^-1 pix^-1):")
    print(f"  bg_median_pix_frame = {bg_med:.3e}  (min={bg_min:.3e} max={bg_max:.3e})")
    print(f"  bg_total_frame (sum over pixels) = {bg_total:.3e} ph m^-2 frame^-1")

    # Compare sources to background at their subpixel locations (use on-detector sources only)
    active_stars0 = [
        s
        for _, s in star_items
        if ("on_detector" not in s or bool(s["on_detector"][fi_ref]))
    ]
    xs0 = np.array([float(s["x_pix"][fi_ref]) for s in active_stars0], dtype=np.float64)
    ys0 = np.array([float(s["y_pix"][fi_ref]) for s in active_stars0], dtype=np.float64)
    sf0 = np.array([float(s["flux_ph_m2_frame"][fi_ref]) for s in active_stars0], dtype=np.float64)

    b0 = str(uv_basis).strip().lower()
    if b0 in {"normalized_uv", "normalized", "uv"}:
        u0 = (2.0 * xs0 / (cols - 1.0)) - 1.0
        v0 = (2.0 * ys0 / (rows - 1.0)) - 1.0
    else:
        u0, v0 = xs0, ys0

    bg0_local = _background_map_per_s(coeffs_per_s_pix[fi_ref], u0, v0, ZODI_MODEL) * float(t_exp_s[fi_ref])
    ratio0 = sf0 / np.maximum(bg0_local, 1e-30)

    print("\n[STATS] Stars vs background at source location (dimensionless ratio):")
    print(f"  n_on_detector        = {ratio0.size:d}")
    if ratio0.size > 0:
        print(f"  median(star/bg)      = {float(np.median(ratio0)):.1f}")
        print(f"  5–95% range          = {float(np.percentile(ratio0,5)):.1f} .. {float(np.percentile(ratio0,95)):.1f}")
        print(f"  max(star/bg)         = {float(np.max(ratio0)):.1f}")

    # Same metric for targets present in the reference frame
    tx0, ty0, tf0 = [], [], []
    for t in target_tracks:
        hit = _target_at_frame(t, fi_ref)
        if hit is None:
            continue
        x, y, f = hit
        if (x < -0.5) or (x > (cols - 0.5)) or (y < -0.5) or (y > (rows - 0.5)):
            continue
        tx0.append(x)
        ty0.append(y)
        tf0.append(f)

    if len(tf0) > 0:
        tx0 = np.asarray(tx0, dtype=np.float64)
        ty0 = np.asarray(ty0, dtype=np.float64)
        tf0 = np.asarray(tf0, dtype=np.float64)

        if b0 in {"normalized_uv", "normalized", "uv"}:
            ut0 = (2.0 * tx0 / (cols - 1.0)) - 1.0
            vt0 = (2.0 * ty0 / (rows - 1.0)) - 1.0
        else:
            ut0, vt0 = tx0, ty0

        bg_t_local = _background_map_per_s(coeffs_per_s_pix[fi_ref], ut0, vt0, ZODI_MODEL) * float(t_exp_s[fi_ref])
        ratio_t = tf0 / np.maximum(bg_t_local, 1e-30)

        print("\n[STATS] Targets vs background at target location (dimensionless ratio):")
        print(f"  n_present_on_detector = {ratio_t.size:d}")
        print(f"  median(target/bg)     = {float(np.median(ratio_t)):.1f}")
        print(f"  5–95% range           = {float(np.percentile(ratio_t,5)):.1f} .. {float(np.percentile(ratio_t,95)):.1f}")
        print(f"  max(target/bg)        = {float(np.max(ratio_t)):.1f}")


    # Frame list to animate
    frame_idxs = np.arange(FRAME_START, n_frames, FRAME_STRIDE, dtype=np.int64)

    # Frame-to-frame time step for informational overlays (prefer actual timestamps if available)
    dt_ws = float(ws_track["dt_frame_s"])
    dt_frame_s = dt_ws
    dt_from_times: Optional[float] = None
    dt_min_s: Optional[float] = None
    dt_max_s: Optional[float] = None

    if times_utc_iso is not None and len(times_utc_iso) == n_frames and n_frames >= 2:
        try:
            # Strip trailing "Z" (timezone) for numpy datetime64 parsing; diffs are unaffected
            t_arr = np.asarray(
                [(t[:-1] if (isinstance(t, str) and t.endswith("Z")) else str(t)) for t in times_utc_iso],
                dtype="datetime64[ns]",
            )
            dts = np.diff(t_arr).astype("timedelta64[ns]").astype(np.float64) / 1e9
            if dts.size > 0 and np.isfinite(dts).all():
                dt_from_times = float(np.median(dts))
                dt_min_s = float(np.min(dts))
                dt_max_s = float(np.max(dts))
                if dt_from_times > 0:
                    dt_frame_s = dt_from_times
        except Exception:
            pass

    rendered_dt_s = float(FRAME_STRIDE) * dt_frame_s
    speedup = float(FPS) * rendered_dt_s

    print(
        f"\n[INFO] Animating {frame_idxs.size} frames "
        f"(start={FRAME_START}, stride={FRAME_STRIDE}, dt_frame_s={dt_frame_s:g} -> rendered_dt_s={rendered_dt_s:g})"
    )
    if dt_from_times is not None and dt_min_s is not None and dt_max_s is not None:
        print(f"[INFO] dt_frame_s from times_utc_iso: median={dt_from_times:g}s (min={dt_min_s:g}s max={dt_max_s:g}s)")
    else:
        print(f"[INFO] dt_frame_s from WINDOW_SOURCES: {dt_ws:g}s (times_utc_iso unavailable or not parseable)")
    print(f"[INFO] Playback FPS={FPS} -> speedup ≈ {speedup:.1f}× simulation-time per video second")


    # -----------------------------
    # Shared color scaling (applies to bg + stars + targets)
    # -----------------------------
    # Estimate a sensible vmax in per-pixel units
    max_source_flux_total = max(star_global_max, tgt_global_max)

    # For PSF mode, peak pixel is ~ (max kernel weight)*total flux; compute at centered dx=dy=0
    sigma = _sigma_from_fwhm(FWHM_PIX)
    psf_kernel_cache: Dict[Tuple[float, float], np.ndarray] = {}
    radius = int(math.ceil(PSF_NSIGMA * sigma))
    k0 = _integrated_gaussian_kernel(sigma=sigma, radius=radius, dx=0.0, dy=0.0)
    peak_w = float(np.max(k0))  # <= 1
    peak_est_psf = peak_w * max_source_flux_total
    peak_est_delta = max_source_flux_total  # if deposited entirely to one pixel

    # Auto vmin/vmax per animation type if not provided
    vmin_auto = max(1e-6, AUTO_VMIN_BG_FACTOR * bg_min)

    # Use the same vmin for both; vmax differs between points (delta/bilinear) and gaussian
    vmin_points = float(VMIN) if VMIN is not None else vmin_auto
    vmin_gauss = float(VMIN) if VMIN is not None else vmin_auto

    vmax_points = float(VMAX) if VMAX is not None else max(bg_max, peak_est_delta) * 1.05
    vmax_gauss = float(VMAX) if VMAX is not None else max(bg_max, peak_est_psf) * 1.05

    print("\n[INFO] Display scaling (shared units across bg + sources):")
    print(f"  vmin_points = {vmin_points:.3e}, vmax_points = {vmax_points:.3e}  (ph m^-2 frame^-1 pix^-1)")
    print(f"  vmin_gauss  = {vmin_gauss:.3e}, vmax_gauss  = {vmax_gauss:.3e}  (ph m^-2 frame^-1 pix^-1)")
    if USE_LOG_NORM:
        print("  scale type  = LogNorm (log10 axis)")
    else:
        print("  scale type  = Linear")

    # -----------------------------
    # 1) POINTS (unified image: bg + deposited sources)
    # -----------------------------
    if SAVE_POINTS_ANIM:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig1.patch.set_facecolor("black")
        _apply_night_sky_axes_style(ax1)
        ax1.set_title("POINTS: Zodiacal Background + Sources (no PSF)")
        ax1.set_xlabel("x_pix")
        ax1.set_ylabel("y_pix")

        norm1 = _make_norm(vmin_points, vmax_points)

        # Build initial frame image (bg + deposited sources)
        def render_points_frame(fi: int) -> np.ndarray:
            bg = _background_map_per_s(coeffs_per_s_pix[fi], u, v, ZODI_MODEL) * float(t_exp_s[fi])
            img = bg.copy()

            # Stars (gather into arrays)
            active_stars = [
                s
                for _, s in star_items
                if ("on_detector" not in s or bool(s["on_detector"][fi]))
            ]
            xs = np.array([float(s["x_pix"][fi]) for s in active_stars], dtype=np.float64)
            ys = np.array([float(s["y_pix"][fi]) for s in active_stars], dtype=np.float64)
            sf = np.array([float(s["flux_ph_m2_frame"][fi]) for s in active_stars], dtype=np.float64)

            # Targets present at this frame (gathered into arrays so targets and stars deposit identically)
            tx, ty, tf = [], [], []
            for t in target_tracks:
                hit = _target_at_frame(t, fi)
                if hit is None:
                    continue
                x, y, f = hit
                # Skip targets whose centers are off-detector (prevents edge-clipping artifacts)
                if (x < -0.5) or (x > (cols - 0.5)) or (y < -0.5) or (y > (rows - 0.5)):
                    continue
                tx.append(x)
                ty.append(y)
                tf.append(f)

            if len(tf) > 0:
                xs = np.concatenate([xs, np.asarray(tx, dtype=np.float64)])
                ys = np.concatenate([ys, np.asarray(ty, dtype=np.float64)])
                sf = np.concatenate([sf, np.asarray(tf, dtype=np.float64)])

            if POINTS_DEPOSIT_MODE == "nearest":
                xi, yi = _safe_round_pix(xs, ys, cols, rows)
                np.add.at(img, (yi, xi), sf)
            elif POINTS_DEPOSIT_MODE == "bilinear":
                # Bilinear is cheap but not easily vectorized; loop is still OK for ~300 sources/frame
                for x, y, f in zip(xs, ys, sf):
                    _add_bilinear_splat(img, x, y, f)
            else:
                raise ValueError(f"Unknown POINTS_DEPOSIT_MODE: {POINTS_DEPOSIT_MODE!r}")

            return img


        fi0 = int(frame_idxs[0])
        img0 = render_points_frame(fi0)

        im1 = ax1.imshow(
            img0,
            origin=IMSHOW_ORIGIN,
            cmap=IMSHOW_CMAP,
            norm=norm1,
            interpolation=IMSHOW_INTERP,
        )

        cb1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label("ph m$^{-2}$ frame$^{-1}$ pix$^{-1}$", color="white")
        cb1.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb1.ax.get_yticklabels(), color="white")

        if SHOW_SCALE_BAR and np.isfinite(arcsec_per_pix):
            _add_scale_bar(
                ax1,
                rows=rows,
                cols=cols,
                arcsec_per_pix=arcsec_per_pix,
                length_arcmin=SCALE_BAR_ARCMIN,
                origin=IMSHOW_ORIGIN,
            )

        # Optional target rings (overlay only)
        tgt_ring = ax1.scatter(
            [],
            [],
            s=TARGET_CIRCLE_SIZE,
            facecolors="none",
            edgecolors="red",
            linewidths=TARGET_CIRCLE_LW,
            alpha=0.95,
        )
        tgt_ring.set_visible(bool(DRAW_TARGET_CIRCLES))

        # Put metadata above the axes (keeps it off the image)
        fig1.subplots_adjust(top=0.88)

        txt1 = fig1.text(
            0.01,
            0.98,
            "",
            transform=fig1.transFigure,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )


        def update_points(k: int) -> None:
            fi = int(frame_idxs[k])
            img = render_points_frame(fi)
            im1.set_data(img)

            # Active targets for rings
            if DRAW_TARGET_CIRCLES:
                txs, tys = [], []
                for t in target_tracks:
                    hit = _target_at_frame(t, fi)
                    if hit is None:
                        continue
                    x, y, _ = hit
                    txs.append(x)
                    tys.append(y)
                if len(txs) > 0:
                    tgt_ring.set_offsets(np.column_stack([np.asarray(txs), np.asarray(tys)]))
                else:
                    tgt_ring.set_offsets(np.empty((0, 2)))

            t_str = (
                times_utc_iso[fi]
                if (times_utc_iso is not None and len(times_utc_iso) == n_frames)
                else "<time_unknown>"
            )

            # Keep text diagnostic but in unified units
            txt1.set_text(
                f"time_utc={t_str}\n"
                f"frame={fi} (rendered_dt={rendered_dt_s:g}s)\n"
                f"units: ph m^-2 frame^-1 pix^-1\n"
            )

        print(f"[INFO] Saving POINTS animation to {POINTS_ANIM_OUT} ...")
        _save_with_tqdm(
            fig1,
            update_points,
            n_frames=len(frame_idxs),
            out_path=POINTS_ANIM_OUT,
            fps=FPS,
            dpi=SAVE_DPI,
            desc="Encoding POINTS MP4",
        )
        plt.close(fig1)

    # -----------------------------
    # 2) GAUSSIAN PSF (unified image: bg + PSF sources)
    # -----------------------------
    if SAVE_GAUSS_ANIM:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.patch.set_facecolor("black")
        _apply_night_sky_axes_style(ax2)
        ax2.set_title(f"GAUSSIAN: Background + PSF Sources (FWHM={FWHM_PIX:g} px)")
        ax2.set_xlabel("x_pix")
        ax2.set_ylabel("y_pix")

        norm2 = _make_norm(vmin_gauss, vmax_gauss)

        def render_gauss_frame(fi: int) -> np.ndarray:
            bg = _background_map_per_s(coeffs_per_s_pix[fi], u, v, ZODI_MODEL) * float(t_exp_s[fi])
            img = bg.copy()

            # Stars
            for _, s in star_items:
                if ("on_detector" in s) and (not bool(s["on_detector"][fi])):
                    continue
                x = float(s["x_pix"][fi])
                y = float(s["y_pix"][fi])
                f = float(s["flux_ph_m2_frame"][fi])
                _add_gaussian_psf(img, x, y, f, sigma=sigma, nsigma=PSF_NSIGMA, cache=psf_kernel_cache, cache_step=PSF_CACHE_STEP)

            # Targets
            for t in target_tracks:
                hit = _target_at_frame(t, fi)
                if hit is None:
                    continue
                x, y, f = hit
                # Skip targets whose centers are off-detector (prevents edge-clipping artifacts)
                if (x < -0.5) or (x > (cols - 0.5)) or (y < -0.5) or (y > (rows - 0.5)):
                    continue
                _add_gaussian_psf(img, x, y, f, sigma=sigma, nsigma=PSF_NSIGMA, cache=psf_kernel_cache, cache_step=PSF_CACHE_STEP)

            return img


        fi0 = int(frame_idxs[0])
        img_init = render_gauss_frame(fi0)

        im2 = ax2.imshow(
            img_init,
            origin=IMSHOW_ORIGIN,
            cmap=IMSHOW_CMAP,
            norm=norm2,
            interpolation=IMSHOW_INTERP,
        )

        cb2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("ph m$^{-2}$ frame$^{-1}$ pix$^{-1}$", color="white")
        cb2.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb2.ax.get_yticklabels(), color="white")

        if SHOW_SCALE_BAR and np.isfinite(arcsec_per_pix):
            _add_scale_bar(
                ax2,
                rows=rows,
                cols=cols,
                arcsec_per_pix=arcsec_per_pix,
                length_arcmin=SCALE_BAR_ARCMIN,
                origin=IMSHOW_ORIGIN,
            )

        tgt_ring2 = ax2.scatter(
            [],
            [],
            s=TARGET_CIRCLE_SIZE,
            facecolors="none",
            edgecolors="red",
            linewidths=TARGET_CIRCLE_LW,
            alpha=0.95,
        )
        tgt_ring2.set_visible(bool(DRAW_TARGET_CIRCLES))

        # Put metadata above the axes (keeps it off the image)
        fig2.subplots_adjust(top=0.88)

        txt2 = fig2.text(
            0.01,
            0.98,
            "",
            transform=fig2.transFigure,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )


        def update_gauss(k: int) -> None:
            fi = int(frame_idxs[k])
            img = render_gauss_frame(fi)
            im2.set_data(img)

            if DRAW_TARGET_CIRCLES:
                txs, tys = [], []
                for t in target_tracks:
                    hit = _target_at_frame(t, fi)
                    if hit is None:
                        continue
                    x, y, _ = hit
                    txs.append(x)
                    tys.append(y)
                if len(txs) > 0:
                    tgt_ring2.set_offsets(np.column_stack([np.asarray(txs), np.asarray(tys)]))
                else:
                    tgt_ring2.set_offsets(np.empty((0, 2)))

            t_str = (
                times_utc_iso[fi]
                if (times_utc_iso is not None and len(times_utc_iso) == n_frames)
                else "<time_unknown>"
            )

            txt2.set_text(
                f"time_utc={t_str}\n"
                f"frame={fi} (rendered_dt={rendered_dt_s:g}s)\n"
                f"units: ph m^-2 frame^-1 pix^-1\n"
                f"img_max={float(np.max(img)):.2e}\n"
            )

        print(f"[INFO] Saving GAUSSIAN animation to {GAUSS_ANIM_OUT} ...")
        _save_with_tqdm(
            fig2,
            update_gauss,
            n_frames=len(frame_idxs),
            out_path=GAUSS_ANIM_OUT,
            fps=FPS,
            dpi=SAVE_DPI,
            desc="Encoding GAUSSIAN MP4",
        )
        plt.close(fig2)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
