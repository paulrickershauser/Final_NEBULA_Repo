# -*- coding: utf-8 -*-
"""
visualize_nebula_scene.py

Standalone visualization for NEBULA WINDOW_SOURCES + ZODIACAL_LIGHT.

Two rendering modes:
  1) DELTA (no Gaussian): deposit each source into the pixel grid using
     subpixel bilinear weights (unit-consistent "per pixel" image).
  2) GAUSSIAN (toy PSF): deposit each source using a truncated 2D Gaussian PSF
     parameterized by FWHM in pixels.

Key design goals
----------------
- Same units for EVERYTHING in the displayed image:
    ph m^-2 frame^-1 pix^-1
  Background is generated from zodi model coeffs per pixel per second, then
  multiplied by t_exp_s to get per-frame. Stars/targets already provide
  flux_ph_m2_frame (per-frame total); rendering deposits that flux into pixels.
- One shared colorbar/scale applies to background + stars + targets.
- Optional red rings around targets for identification ONLY (does not change
  brightness; targets are rendered identically to stars).
- tqdm progress while saving mp4.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FFMpegWriter

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# User settings
# =============================================================================

# Input pickles (relative to repo root is typical)
WINDOW_SOURCES_PKL = Path("NEBULA_OUTPUT/SCENE/obs_window_sources.pkl")
ZODIACAL_LIGHT_PKL = Path("NEBULA_OUTPUT/ZODIACAL_LIGHT/obs_zodiacal_light.pkl")

# Choose observer/window
OBSERVER_NAME: Optional[str] = None  # None -> first observer found
WINDOW_INDEX = 0

# Zodi model choice (must exist in the zodi pickle)
ZODI_MODEL = "quad6"  # "quad6" or "plane3"

# Output files
OUT_DELTA_MP4 = Path("nebula_delta_over_bg.mp4")
OUT_GAUSS_MP4 = Path("nebula_gauss_over_bg.mp4")

# Enable/disable outputs (two options requested)
SAVE_DELTA_ANIM = True
SAVE_GAUSS_ANIM = True

# Animation controls (NO max-frames cap; full sequence by default)
FRAME_START = 0
FRAME_STOP: Optional[int] = None      # None -> through last frame
FRAME_STRIDE = 1                      # 1 -> render every simulation frame
FPS = 20                              # playback FPS for MP4
DPI = 120                             # output resolution control

# Visualization style
COLORMAP = "gray"                     # camera-like monochrome
ORIGIN = "lower"                      # pixel coordinate convention

# Display scaling: log is strongly recommended for this dynamic range.
DISPLAY_SCALE = "log"                 # "log" or "linear"

# Optional: draw red rings around targets (identification only)
DRAW_TARGET_RINGS = True
TARGET_RING_SIZE = 40                 # marker size (points^2 in matplotlib scatter)
TARGET_RING_LW = 0.8                  # ring line width

# Gaussian PSF settings (used only if SAVE_GAUSS_ANIM)
FWHM_PIX = 2.0                        # tune this
PSF_NSIGMA = 4.0                      # patch radius = ceil(nsigma*sigma)

# Performance knobs
BACKGROUND_CLAMP_NONNEGATIVE = True   # clamp background <0 to 0 (fit artifacts)
BACKGROUND_MODE = "per_frame"         # "per_frame" (accurate) or "median_frame" (faster)


# =============================================================================
# Helpers: loading and validation
# =============================================================================

def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing pickle: {path.resolve()}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _pick_observer(ws: Dict, zl: Dict, observer_name: Optional[str]) -> str:
    if observer_name is not None:
        if observer_name not in ws or observer_name not in zl:
            raise KeyError(f"Observer {observer_name!r} not found in both pickles.")
        return observer_name
    # default: first common observer
    common = [k for k in ws.keys() if k in zl]
    if not common:
        raise RuntimeError("No common observers found between WINDOW_SOURCES and ZODIACAL_LIGHT pickles.")
    return common[0]


def _assert_same_window_shapes(ws_win: dict, zl_win: dict) -> None:
    n1 = int(ws_win["n_frames"])
    n2 = int(zl_win["n_frames"])
    if n1 != n2:
        raise RuntimeError(f"Frame count mismatch: window_sources n_frames={n1}, zodi n_frames={n2}")
    if int(ws_win["window_index"]) != int(zl_win["window_index"]):
        raise RuntimeError("Window index mismatch between pickles.")


# =============================================================================
# Background model evaluation (normalized_uv)
# =============================================================================

def _build_uv_grids(rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build normalized uv grids consistent with your zodi sampling:
      u = 2*x/(cols-1) - 1
      v = 2*y/(rows-1) - 1
    where x in [0..cols-1], y in [0..rows-1].

    Returns float32 arrays: U, V, U2, UV, V2 all shaped (rows, cols).
    """
    x = np.arange(cols, dtype=np.float32)
    y = np.arange(rows, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    U = 2.0 * X / float(cols - 1) - 1.0
    V = 2.0 * Y / float(rows - 1) - 1.0

    U2 = U * U
    V2 = V * V
    UV = U * V
    return U, V, U2, UV, V2


def _eval_bg_map_per_s(
    coeffs: np.ndarray,
    model: str,
    U: np.ndarray,
    V: np.ndarray,
    U2: np.ndarray,
    UV: np.ndarray,
    V2: np.ndarray,
    out: np.ndarray,
    tmp: np.ndarray,
) -> np.ndarray:
    """
    Evaluate background map in units of ph m^-2 s^-1 pix^-1 into `out`.

    model:
      - "plane3":  c0 + c1*U + c2*V
      - "quad6":   c0 + c1*U + c2*V + c3*U2 + c4*UV + c5*V2
    """
    if model == "plane3":
        c0, c1, c2 = [float(x) for x in coeffs]
        out.fill(c0)
        np.multiply(U, c1, out=tmp); out += tmp
        np.multiply(V, c2, out=tmp); out += tmp
        return out

    if model == "quad6":
        c0, c1, c2, c3, c4, c5 = [float(x) for x in coeffs]
        out.fill(c0)
        np.multiply(U,  c1, out=tmp); out += tmp
        np.multiply(V,  c2, out=tmp); out += tmp
        np.multiply(U2, c3, out=tmp); out += tmp
        np.multiply(UV, c4, out=tmp); out += tmp
        np.multiply(V2, c5, out=tmp); out += tmp
        return out

    raise ValueError(f"Unknown model: {model!r}")


# =============================================================================
# Source packing (stars + targets) into frame-aligned arrays
# =============================================================================

def _pack_stars(ws_win: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x  : (n_stars, n_frames) float32
      y  : (n_stars, n_frames) float32
      f  : (n_stars, n_frames) float64  [ph m^-2 frame^-1]  (total per source per frame)
      on : (n_stars, n_frames) bool
      ids: (n_stars,) object (source_id strings)
    """
    stars = ws_win["stars"]
    ids = np.array([s["source_id"] for s in stars.values()], dtype=object)

    x = np.vstack([np.asarray(s["x_pix"], dtype=np.float32) for s in stars.values()])
    y = np.vstack([np.asarray(s["y_pix"], dtype=np.float32) for s in stars.values()])
    f = np.vstack([np.asarray(s["flux_ph_m2_frame"], dtype=np.float64) for s in stars.values()])
    on = np.vstack([np.asarray(s["on_detector"], dtype=bool) for s in stars.values()])
    return x, y, f, on, ids


def _pack_targets(ws_win: dict, n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Targets are sparse (coarse_indices). Build frame-aligned arrays:

      x  : (n_targets, n_frames) float32 (NaN when absent)
      y  : (n_targets, n_frames) float32 (NaN when absent)
      f  : (n_targets, n_frames) float64 (0 when absent)
      on : (n_targets, n_frames) bool
      ids: (n_targets,) object (source_id strings)
    """
    targets = ws_win["targets"]
    ids = np.array([t["source_id"] for t in targets.values()], dtype=object)

    x = np.full((len(targets), n_frames), np.nan, dtype=np.float32)
    y = np.full((len(targets), n_frames), np.nan, dtype=np.float32)
    f = np.zeros((len(targets), n_frames), dtype=np.float64)
    on = np.zeros((len(targets), n_frames), dtype=bool)

    for i, t in enumerate(targets.values()):
        idx = np.asarray(t["coarse_indices"], dtype=np.int64)
        if idx.size == 0:
            continue
        if idx.min() < 0 or idx.max() >= n_frames:
            raise RuntimeError(f"Target {t['source_id']!r}: coarse_indices out of range [0..{n_frames-1}]")
        x[i, idx] = np.asarray(t["x_pix"], dtype=np.float32)
        y[i, idx] = np.asarray(t["y_pix"], dtype=np.float32)
        f[i, idx] = np.asarray(t["flux_ph_m2_frame"], dtype=np.float64)
        on[i, idx] = True

    return x, y, f, on, ids


# =============================================================================
# Rendering: delta (bilinear) and gaussian
# =============================================================================

def _deposit_delta_bilinear(
    img: np.ndarray,  # (rows, cols) float32/float64
    xs: np.ndarray,   # (n_src,) float32
    ys: np.ndarray,   # (n_src,) float32
    fs: np.ndarray,   # (n_src,) float64
) -> None:
    """
    Deposit source flux into pixels using bilinear weights for subpixel accuracy.
    Flux units are preserved: img ends in ph m^-2 frame^-1 pix^-1.

    This is "no Gaussian" but still subpixel-accurate.
    """
    rows, cols = img.shape
    flat = img.ravel()

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)

    # Need x0 in [0..cols-2], y0 in [0..rows-2] for 2x2 deposit
    m = (fs > 0) & (x0 >= 0) & (x0 < cols - 1) & (y0 >= 0) & (y0 < rows - 1)
    if not np.any(m):
        return

    x0 = x0[m]; y0 = y0[m]
    dx = (xs[m] - x0.astype(np.float32)).astype(np.float32)
    dy = (ys[m] - y0.astype(np.float32)).astype(np.float32)
    f = fs[m].astype(np.float64)

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    idx00 = y0 * cols + x0
    idx10 = y0 * cols + (x0 + 1)
    idx01 = (y0 + 1) * cols + x0
    idx11 = (y0 + 1) * cols + (x0 + 1)

    np.add.at(flat, idx00, f * w00)
    np.add.at(flat, idx10, f * w10)
    np.add.at(flat, idx01, f * w01)
    np.add.at(flat, idx11, f * w11)


def _deposit_gaussian_psf(
    img: np.ndarray,          # (rows, cols) float32/float64
    xs: np.ndarray,           # (n_src,) float32
    ys: np.ndarray,           # (n_src,) float32
    fs: np.ndarray,           # (n_src,) float64
    sigma: float,
    nsigma: float,
) -> None:
    """
    Deposit each source using a truncated 2D Gaussian PSF centered at subpixel (x,y).
    Kernel is separable: K(dx,dy) = wy(dy)*wx(dx), normalized to sum=1 for flux conservation.
    """
    rows, cols = img.shape
    r = int(np.ceil(nsigma * sigma))
    if r < 1:
        # sigma very small -> treat as delta-bilinear
        _deposit_delta_bilinear(img, xs, ys, fs)
        return

    # Precompute integer offsets [-r..r]
    offs = np.arange(-r, r + 1, dtype=np.int64)
    offs_f = offs.astype(np.float64)

    for x, y, f in zip(xs, ys, fs):
        if not np.isfinite(x) or not np.isfinite(y) or f <= 0:
            continue

        # Patch integer anchor around floor
        x0 = int(np.floor(float(x)))
        y0 = int(np.floor(float(y)))

        # Patch bounds
        x_idx = x0 + offs
        y_idx = y0 + offs

        # Skip if fully off-sensor
        if x_idx[-1] < 0 or x_idx[0] >= cols or y_idx[-1] < 0 or y_idx[0] >= rows:
            continue

        # Clip patch to sensor bounds
        x_good = (x_idx >= 0) & (x_idx < cols)
        y_good = (y_idx >= 0) & (y_idx < rows)
        if not np.any(x_good) or not np.any(y_good):
            continue

        x_use = x_idx[x_good]
        y_use = y_idx[y_good]

        # Subpixel-centered separable weights:
        # dx = (pixel_x - x), dy = (pixel_y - y)
        dx = (x_use.astype(np.float64) - float(x))
        dy = (y_use.astype(np.float64) - float(y))

        wx = np.exp(-(dx * dx) / (2.0 * sigma * sigma))
        wy = np.exp(-(dy * dy) / (2.0 * sigma * sigma))

        # Outer product -> 2D kernel over clipped patch
        K = wy[:, None] * wx[None, :]
        s = float(K.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        K /= s

        img[np.ix_(y_use, x_use)] += (f * K)


# =============================================================================
# Stats + normalization selection
# =============================================================================

def _compute_display_norm(
    bg_level: float,
    peak_level: float,
    mode: str,
) -> Tuple[object, str]:
    """
    Choose a normalization so that:
      - background is not blown out
      - bright stars don't force everything to black
    """
    units = r"ph m$^{-2}$ frame$^{-1}$ pix$^{-1}$"

    if mode == "log":
        # Keep vmin > 0 for LogNorm.
        vmin = max(bg_level, 1e-3)
        vmax = max(peak_level, vmin * 10.0)
        return LogNorm(vmin=vmin, vmax=vmax), units

    # linear
    vmin = 0.0
    vmax = max(peak_level, 1.0)
    return Normalize(vmin=vmin, vmax=vmax), units


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("[INFO] Loading pickles...")
    ws = _load_pickle(WINDOW_SOURCES_PKL)
    zl = _load_pickle(ZODIACAL_LIGHT_PKL)

    obs = _pick_observer(ws, zl, OBSERVER_NAME)

    ws_obs = ws[obs]
    zl_obs = zl[obs]

    ws_win = ws_obs["windows"][WINDOW_INDEX]
    zl_win = zl_obs["windows"][WINDOW_INDEX]
    _assert_same_window_shapes(ws_win, zl_win)

    rows = int(ws_obs["rows"])
    cols = int(ws_obs["cols"])
    n_frames = int(ws_win["n_frames"])
    dt_frame_s = float(ws_obs["dt_frame_s"])
    t_exp_s = dt_frame_s  # in your data this is constant and equals dt_frame_s

    # Zodi coeffs (per pixel per second)
    zodi = zl_win["zodi"]
    model_dict = zodi["models"]
    if ZODI_MODEL not in model_dict:
        raise KeyError(f"ZODI_MODEL={ZODI_MODEL!r} not found in zodi models: {list(model_dict.keys())}")
    coeffs_per_pixel = np.asarray(model_dict[ZODI_MODEL]["coeffs_per_pixel"], dtype=np.float64)  # (n_frames, k)

    # Build UV grids for model evaluation
    U, V, U2, UV, V2 = _build_uv_grids(rows, cols)
    bg_buf = np.empty((rows, cols), dtype=np.float32)
    tmp = np.empty((rows, cols), dtype=np.float32)

    # Pack sources
    sx, sy, sf, son, star_ids = _pack_stars(ws_win)
    tx, ty, tf, ton, target_ids = _pack_targets(ws_win, n_frames=n_frames)

    n_stars = sx.shape[0]
    n_targets = tx.shape[0]
    print(f"[INFO] omega_pix_sr_scalar = {zodi['omega_pix']['omega_pix_sr_scalar']:.6e} sr/pix")
    print(f"[INFO] Observer={obs!r} window={WINDOW_INDEX} size={rows}x{cols} frames={n_frames}")
    print(f"[INFO] Stars={n_stars} targets={n_targets} t_exp={t_exp_s:.6f} s model={ZODI_MODEL}")

    # Frame range selection (NO max-frames cap)
    stop = FRAME_STOP if FRAME_STOP is not None else n_frames
    if FRAME_START < 0 or FRAME_START >= n_frames:
        raise ValueError(f"FRAME_START out of range: {FRAME_START}")
    if stop <= FRAME_START or stop > n_frames:
        raise ValueError(f"FRAME_STOP out of range: {FRAME_STOP}")
    frame_idx = np.arange(FRAME_START, stop, FRAME_STRIDE, dtype=np.int64)

    rendered_dt_s = dt_frame_s * FRAME_STRIDE
    speedup = rendered_dt_s * FPS
    print(f"[INFO] Animating {len(frame_idx)} frames (start={FRAME_START}, stop={stop}, stride={FRAME_STRIDE})")
    print(f"[INFO] dt_frame_s={dt_frame_s} -> rendered_dt_s={rendered_dt_s} ; FPS={FPS} ; speedup≈{speedup:.1f}×")

    # Basic stats for display scaling:
    # Background level estimate: median background at center (u=v=0 => coefficient c0)
    # In normalized_uv, center corresponds to U=0,V=0 so plane/quad constant term dominates.
    bg_center_per_s = coeffs_per_pixel[:, 0]
    bg_center_per_frame = bg_center_per_s * t_exp_s
    bg_level = float(np.median(bg_center_per_frame))

    # Peak source level estimate depends on render mode:
    star_max = float(np.max(sf))
    targ_max = float(np.max(tf)) if tf.size else 0.0
    src_max = max(star_max, targ_max)

    # For delta-bilinear, a single pixel can approach ~src_max (if near pixel center)
    peak_delta = src_max + bg_level

    # For Gaussian, peak pixel is reduced by approx kernel max; use continuous approx 1/(2πσ²)
    # This is only for choosing a plotting vmax; exact peaks vary with subpixel location.
    sigma = float(FWHM_PIX / 2.354820045)  # FWHM = 2*sqrt(2 ln2)*sigma
    peak_gauss_est = (src_max / max(2.0 * np.pi * sigma * sigma, 1e-9)) + bg_level

    # Choose consistent normalization for both animations
    peak_level_for_norm = peak_delta if SAVE_DELTA_ANIM and not SAVE_GAUSS_ANIM else max(peak_delta, peak_gauss_est)
    norm, units = _compute_display_norm(bg_level=bg_level, peak_level=peak_level_for_norm, mode=DISPLAY_SCALE)

    # Optionally speed background by freezing to median frame
    bg_coeff_median = np.median(coeffs_per_pixel, axis=0) if BACKGROUND_MODE == "median_frame" else None

    def make_bg_frame(i: int) -> np.ndarray:
        if bg_coeff_median is not None:
            c = bg_coeff_median
        else:
            c = coeffs_per_pixel[i]

        _eval_bg_map_per_s(c, ZODI_MODEL, U, V, U2, UV, V2, out=bg_buf, tmp=tmp)
        bg = bg_buf  # per second
        bg *= float(t_exp_s)  # per frame
        if BACKGROUND_CLAMP_NONNEGATIVE:
            np.maximum(bg, 0.0, out=bg)
        return bg

    def make_sources_for_frame(i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          xs, ys, fs : sources to render (stars+targets) for this frame
          target_xy  : (n_present_targets, 2) for optional rings
        """
        # Stars present
        sm = son[:, i]
        xs_s = sx[sm, i]
        ys_s = sy[sm, i]
        fs_s = sf[sm, i]

        # Targets present
        tm = ton[:, i]
        xs_t = tx[tm, i]
        ys_t = ty[tm, i]
        fs_t = tf[tm, i]

        xs_all = np.concatenate([xs_s, xs_t]).astype(np.float32, copy=False)
        ys_all = np.concatenate([ys_s, ys_t]).astype(np.float32, copy=False)
        fs_all = np.concatenate([fs_s, fs_t]).astype(np.float64, copy=False)

        target_xy = np.stack([xs_t, ys_t], axis=1) if xs_t.size else np.zeros((0, 2), dtype=np.float32)
        return xs_all, ys_all, fs_all, target_xy

    def render_frame(i: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build image for a single frame in units ph m^-2 frame^-1 pix^-1.
        Returns (img, target_xy).
        """
        img = make_bg_frame(i).copy()  # float32
        xs_all, ys_all, fs_all, target_xy = make_sources_for_frame(i)

        if mode == "delta":
            _deposit_delta_bilinear(img, xs_all, ys_all, fs_all)
        elif mode == "gaussian":
            _deposit_gaussian_psf(img, xs_all, ys_all, fs_all, sigma=sigma, nsigma=PSF_NSIGMA)
        else:
            raise ValueError(mode)

        return img, target_xy

    def save_movie(mode: str, out_path: Path) -> None:
        print(f"[INFO] Saving {mode} animation to {out_path} ...")
        fig, ax = plt.subplots(figsize=(cols / 120.0, rows / 120.0), dpi=DPI)

        # Initial frame
        img0, target_xy0 = render_frame(int(frame_idx[0]), mode=mode)
        im = ax.imshow(img0, cmap=COLORMAP, origin=ORIGIN, norm=norm)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")

        # Colorbar applies to EVERYTHING because we render a single image in per-pixel units
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(units)

        # Optional target rings (identification only; targets rendered same as stars in the image)
        ring = None
        if DRAW_TARGET_RINGS:
            ring = ax.scatter(
                target_xy0[:, 0],
                target_xy0[:, 1],
                s=TARGET_RING_SIZE,
                facecolors="none",
                edgecolors="red",
                linewidths=TARGET_RING_LW,
            )

        # Title updated per frame
        title = ax.set_title("")

        writer = FFMpegWriter(fps=FPS, metadata={"artist": "NEBULA"}, bitrate=-1)

        # tqdm wrapper
        it = frame_idx
        if tqdm is not None:
            it = tqdm(frame_idx, desc=f"Writing {mode} frames", unit="frame")

        with writer.saving(fig, str(out_path), dpi=DPI):
            for k in it:
                i = int(k)
                img, target_xy = render_frame(i, mode=mode)
                im.set_data(img)

                if ring is not None:
                    ring.set_offsets(target_xy)

                title.set_text(f"{obs} | window={WINDOW_INDEX} | frame={i}/{n_frames-1} | t={i*dt_frame_s:.2f}s")

                writer.grab_frame()

        plt.close(fig)

    if SAVE_DELTA_ANIM:
        save_movie("delta", OUT_DELTA_MP4)

    if SAVE_GAUSS_ANIM:
        save_movie("gaussian", OUT_GAUSS_MP4)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
