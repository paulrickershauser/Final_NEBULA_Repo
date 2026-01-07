"""
NEBULA_ZODIACAL_LIGHT_STAGE.py   (Windows/NEBULA side)

Purpose
-------
This is the Windows-side “stage runner” for generating a standalone
zodiacal-light product:

    NEBULA_OUTPUT/ZODIACAL_LIGHT/obs_zodiacal_light.pkl

It:
1) Loads NEBULA_OUTPUT/SCENE/obs_window_sources.pkl (your “single source of truth”).
2) For each observer/window:
   - Slices the per-frame time/pointing/state arrays using start_index/end_index
     (treated as INCLUSIVE).
   - Builds a set of *sample directions on sky* (ICRS RA/Dec) per frame, using:
       - Preferred: NEBULA’s own WCS/camera assumptions (if available)
       - Fallback: a spherical tangent-plane approximation using boresight+roll+FOV
   - Writes a request payload (JSON + NPZ) into a WSL-visible TMP folder.
   - Calls the WSL worker (which uses m4opt) to evaluate zodiacal background.
   - Fits plane(3) and quadratic(6) coefficients per frame, and optionally stores
     a full 2D map (config-controlled).
3) Writes a pickle containing the compact ZL representation, per observer/window.

Important constraint
--------------------
This module MUST NOT import m4opt (or any WSL-only dependencies). It only:
- orchestrates computation
- performs geometry sampling and curve fitting on Windows
- calls WSL as an external process via wsl.exe

Inputs (from obs_window_sources.pkl)
------------------------------------
For each observer (TrackDict):
- rows, cols, dt_frame_s, sensor_name, catalog_name, catalog_band
- observer_geometry:
    - state_vectors.times : list[datetime] (tz-aware UTC per your confirmation)
    - state_vectors.r_eci_km : (N,3) TEME xyz in km
    - pointing.* arrays: (N,) boresight ra/dec/roll etc
- windows list:
    - start_index, end_index (inclusive), n_frames, start_time/end_time, window_index, ...

Outputs (obs_zodiacal_light.pkl)
--------------------------------
Per observer/window we store:
- alignment provenance (times_utc_iso, boresight arrays, start/end indices)
- zodi samples metadata (units, omega_pix_sr, sample grid definition)
- per-frame fit coefficients:
    - plane3_coeffs: (n_frames,3)
    - quad6_coeffs:  (n_frames,6)
- optional 2D map (downsampled or full, depending on config)
- fit diagnostics (RMS residuals vs samples)

Configuration
-------------
This file supports two config “styles” (to avoid breaking your workflow):

Style A (constants-style; my earlier draft):
    Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG with attributes like:
        WINDOW_SOURCES_PICKLE_REL, ZODIACAL_LIGHT_OUTPUT_PICKLE_REL,
        FIELD_SAMPLE_GRID_SHAPE, EXPORT_FULL_MAP, FULL_MAP_DOWNSAMPLE,
        BANDPASS_MODE_DEFAULT, BANDPASS_PRESETS, etc.

Style B (dataclass-style; your alternate version):
    Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG defines:
        ZODIACAL_LIGHT_CONFIG = ZLConfig()
    with nested configs:
        .bandpass_mode, .bandpass_top_hat, .bandpass_svo, .fit, .wsl, .out, ...

This stage normalizes either style into a single internal “ResolvedStageConfig”.

WSL worker contract
-------------------
The stage calls a WSL worker script that reads a request payload base path and
writes a response payload base path:

    python <worker> <request_base> <response_base>

The worker must run inside WSL and is responsible for importing m4opt and
computing zodiacal light values.

Request schema (minimum)
------------------------
Request meta (JSON):
- schema_version: str
- observer_name: str
- window_index: int
- omega_pix_sr: float
- times_utc_iso: list[str] length n_frames
- bandpass: dict
    mode: "tophat" or "svo"
    ... parameters

Request arrays (NPZ):
- sample_radec_deg: (n_frames, n_samples, 2) float64
- observer_teme_xyz_km: (n_frames, 3) float64  (optional but recommended)

Response arrays (NPZ) expected:
- phi_ph_m2_s_pix: (n_frames, n_samples) float64

Response meta (JSON) expected:
- quantity: "phi_ph_m2_s_pix"
- units: "ph m-2 s-1 pix-1"
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR, SensorConfig

from Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_LIGHT_IO import read_payload, write_payload


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    if logger is not None:
        return logger
    lg = logging.getLogger("NEBULA_ZODIACAL_LIGHT_STAGE")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


# -----------------------------------------------------------------------------
# Config normalization
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedStageConfig:
    """
    Internal, normalized configuration for the stage.

    This is derived from Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG, supporting
    either constants-style or dataclass-style config definitions.

    Attributes
    ----------
    enabled : bool
        Stage enable flag.
    input_pickle_path : Path
        Absolute path to obs_window_sources.pkl
    output_pickle_path : Path
        Absolute path to obs_zodiacal_light.pkl
    tmp_dir : Path
        Absolute path to a Windows directory that is also visible in WSL (/mnt/<drive>/...)
    schema_version : str
        Schema tag stored in outputs and requests.

    Sampling / field export
    -----------------------
    sample_n_u, sample_n_v : int
        Grid size used for coefficient-fitting samples.
    sample_margin_pix : int
        Optional pixel margin excluding edges from sampling.
    store_plane3 : bool
        Whether to compute/store plane3 coefficients.
    store_quad6 : bool
        Whether to compute/store quad6 coefficients.
    export_full_map : bool
        Whether to compute/store a full 2D map.
    full_map_downsample : int
        Downsample factor for full map (1 = full-res).

    Bandpass
    --------
    bandpass_mode : str
        "tophat" or "svo"
    bandpass_presets : dict
        Optional mapping keyed by (catalog_name, catalog_band) to parameters.
        If absent, stage falls back to global defaults.

    WSL invocation
    --------------
    wsl_distro : str
    wsl_python : str
        Executed inside WSL (can include "~/" only if using bash -lc).
    wsl_worker_script_wsl : str
        Worker script path as seen from WSL (e.g., /mnt/c/.../NEBULA_ZODIACAL_LIGHT_WSL_WORKER.py)
    wsl_cwd_wsl : str
        Working directory to cd into in WSL before execution.
    wsl_timeout_s : int
    """
    # Enable/disable
    enabled: bool

    # IO (absolute paths)
    input_pickle_path: Path
    output_pickle_path: Path
    tmp_dir: Path
    schema_version: str
    overwrite: bool

    # Validation behavior
    validation_enabled: bool
    strict_validation: bool

    # Sampling / products
    sample_n_u: int
    sample_n_v: int
    sample_margin_pix: int
    store_plane3: bool
    store_quad6: bool
    export_full_map: bool
    full_map_downsample: int

    # Pixel solid angle policy (Windows-side)
    omega_pix_mode: str                  # "from_sensor_fov" | "explicit"
    omega_pix_sr_explicit: Optional[float]

    # Bandpass routing (Windows-side)
    use_catalog_name: bool
    use_catalog_band: bool
    catalog_name_override: Optional[str]
    catalog_band_override: Optional[str]
    bandpass_by_key: Dict[str, Dict[str, Any]]   # lookup key -> request dict {"mode":..., ...}
    default_bandpass: Dict[str, Any]             # request dict {"mode":..., ...}

    # WSL invocation
    wsl_distro: str
    wsl_python: str
    wsl_worker_script_wsl: str
    wsl_cwd_wsl: str
    wsl_timeout_s: int


def _bandpass_spec_to_request(spec: Any) -> Dict[str, Any]:
    """
    Convert a config BandpassSpec (dataclass-style) into the JSON-serializable
    request dict expected by the WSL backend:
        {"mode":"svo","filter_id":...} or {"mode":"tophat","lambda_min_nm":...,"lambda_max_nm":...,"lambda_eff_nm":...}
    """
    mode = str(getattr(spec, "mode", "")).lower().strip()

    if mode in {"svo", "svo_id"}:
        fid = getattr(spec, "svo_filter_id", None) or getattr(spec, "filter_id", None)
        if not fid:
            raise ValueError("BandpassSpec mode='svo_id' requires svo_filter_id.")
        return {"mode": "svo", "filter_id": str(fid)}

    if mode in {"tophat", "tophat_nm"}:
        # Your config defines top-hat as (center_nm, width_nm)
        center = getattr(spec, "center_nm", None)
        width = getattr(spec, "width_nm", None)
        if center is None or width is None:
            raise ValueError("BandpassSpec mode='tophat_nm' requires center_nm and width_nm.")
        center = float(center)
        width = float(width)
        return {
            "mode": "tophat",
            "lambda_min_nm": center - 0.5 * width,
            "lambda_max_nm": center + 0.5 * width,
            "lambda_eff_nm": center,
        }

    if mode in {"curve_file", "curve"}:
        raise NotImplementedError("BandpassSpec mode='curve_file' is not supported by this stage.")

    raise ValueError(f"Unknown BandpassSpec.mode: {mode!r}")


def resolve_stage_config(
    *,
    logger: Optional[logging.Logger] = None,
) -> ResolvedStageConfig:
    """
    Resolve and normalize the Zodiacal Light stage configuration.

    Supported schemas:
      1) Dataclass instance ACTIVE_ZODIACAL_LIGHT_CONFIG (preferred)
      2) Dataclass instance ZODIACAL_LIGHT_CONFIG (legacy alias)
      3) Legacy constants-only config (kept for backward compatibility)

    No-guesswork policy:
      - No implicit worker discovery: config must specify worker_relpath / cwd_wsl.
      - Bandpass must resolve deterministically; if not found, use default_bandpass (dataclass schema).
    """
    try:
        from Configuration import NEBULA_ZODIACAL_LIGHT_CONFIG as ZLC  # type: ignore
    except Exception as e:
        raise ImportError("ZODIACAL_LIGHT_STAGE: failed to import Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG.") from e

    cfg_obj = None
    if hasattr(ZLC, "ACTIVE_ZODIACAL_LIGHT_CONFIG"):
        cfg_obj = getattr(ZLC, "ACTIVE_ZODIACAL_LIGHT_CONFIG")
    elif hasattr(ZLC, "ZODIACAL_LIGHT_CONFIG"):
        cfg_obj = getattr(ZLC, "ZODIACAL_LIGHT_CONFIG")

    # ----------------------------
    # Dataclass-style schema
    # ----------------------------
    if cfg_obj is not None:
        cfg = cfg_obj

        input_pickle_path = Path(NEBULA_OUTPUT_DIR) / Path(cfg.io.window_sources_relpath)
        output_pickle_path = Path(NEBULA_OUTPUT_DIR) / Path(cfg.io.output_relpath)
        tmp_dir = Path(NEBULA_OUTPUT_DIR) / Path(cfg.io.tmp_dir_relpath)

        wsl_distro = str(cfg.wsl.distro)
        wsl_python = str(cfg.wsl.wsl_python)
        wsl_cwd_wsl = str(cfg.wsl.cwd_wsl)

        worker_rel = str(cfg.wsl.worker_relpath).lstrip("/").replace("\\", "/")
        wsl_worker_script_wsl = wsl_cwd_wsl.rstrip("/") + "/" + worker_rel

        # Sampling
        n_u, n_v = tuple(cfg.field.sample_grid)
        sample_n_u = int(n_u)
        sample_n_v = int(n_v)
        sample_margin_pix = int(cfg.field.sample_margin_pix)

        models_to_store = tuple(str(m) for m in cfg.field.models_to_store)
        store_plane3 = "plane3" in models_to_store
        store_quad6 = "quad6" in models_to_store
        export_full_map = bool(cfg.field.export_map2d)
        full_map_downsample = int(cfg.field.map2d_downsample)

        # Pixel solid angle policy
        omega_pix_mode = str(cfg.omega_pix.mode)
        omega_pix_sr_explicit = getattr(cfg.omega_pix, "omega_pix_sr_explicit", None)

        # Bandpass routing
        use_catalog_name = bool(cfg.bandpass.use_catalog_name)
        use_catalog_band = bool(cfg.bandpass.use_catalog_band)
        catalog_name_override = getattr(cfg.bandpass, "catalog_name_override", None)
        catalog_band_override = getattr(cfg.bandpass, "catalog_band_override", None)

        bandpass_by_key: Dict[str, Dict[str, Any]] = {}
        for k, spec in dict(cfg.bandpass.bandpass_by_catalog_name).items():
            bandpass_by_key[str(k)] = _bandpass_spec_to_request(spec)

        default_bandpass = _bandpass_spec_to_request(cfg.bandpass.default_bandpass)

        validation_enabled = bool(getattr(cfg, "validation_enabled", True))
        strict_validation = bool(getattr(cfg, "strict_validation", False))

        return ResolvedStageConfig(
            enabled=bool(cfg.enabled),
            input_pickle_path=input_pickle_path,
            output_pickle_path=output_pickle_path,
            tmp_dir=tmp_dir,
            schema_version=str(cfg.io.schema_version),
            overwrite=bool(cfg.io.overwrite),
            validation_enabled=validation_enabled,
            strict_validation=strict_validation,
            sample_n_u=sample_n_u,
            sample_n_v=sample_n_v,
            sample_margin_pix=sample_margin_pix,
            store_plane3=store_plane3,
            store_quad6=store_quad6,
            export_full_map=export_full_map,
            full_map_downsample=full_map_downsample,
            omega_pix_mode=omega_pix_mode,
            omega_pix_sr_explicit=None if omega_pix_sr_explicit is None else float(omega_pix_sr_explicit),
            use_catalog_name=use_catalog_name,
            use_catalog_band=use_catalog_band,
            catalog_name_override=None if catalog_name_override is None else str(catalog_name_override),
            catalog_band_override=None if catalog_band_override is None else str(catalog_band_override),
            bandpass_by_key=bandpass_by_key,
            default_bandpass=default_bandpass,
            wsl_distro=wsl_distro,
            wsl_python=wsl_python,
            wsl_worker_script_wsl=wsl_worker_script_wsl,
            wsl_cwd_wsl=wsl_cwd_wsl,
            wsl_timeout_s=int(cfg.wsl.timeout_s),
        )

    # ----------------------------
    # Legacy constants-only schema (kept, but normalized into new fields)
    # ----------------------------
    required = [
        "ZODIACAL_LIGHT_ENABLED",
        "WINDOW_SOURCES_PICKLE_REL",
        "ZODIACAL_LIGHT_OUTPUT_PICKLE_REL",
        "WSL_DISTRO",
        "WSL_PYTHON",
        "WSL_WORKER_SCRIPT_WIN",
        "SCHEMA_VERSION",
    ]
    missing = [k for k in required if not hasattr(ZLC, k)]
    if missing:
        raise AttributeError(
            "ZODIACAL_LIGHT_STAGE: legacy config module is missing required fields: " + ", ".join(missing)
        )

    input_pickle_path = Path(NEBULA_OUTPUT_DIR) / Path(getattr(ZLC, "WINDOW_SOURCES_PICKLE_REL"))
    output_pickle_path = Path(NEBULA_OUTPUT_DIR) / Path(getattr(ZLC, "ZODIACAL_LIGHT_OUTPUT_PICKLE_REL"))
    tmp_dir = Path(NEBULA_OUTPUT_DIR) / Path(getattr(ZLC, "WSL_STAGING_SUBDIR_REL", "SCENE/WSL_ZODIACAL_LIGHT"))

    wsl_distro = str(getattr(ZLC, "WSL_DISTRO"))
    wsl_python = str(getattr(ZLC, "WSL_PYTHON"))
    wsl_cwd_wsl = str(getattr(ZLC, "WSL_CWD_WSL", "/mnt/c/Users/prick/Desktop/Research/NEBULA"))
    wsl_worker_script_wsl = win_path_to_wsl(Path(getattr(ZLC, "WSL_WORKER_SCRIPT_WIN")))

    grid = getattr(ZLC, "FIELD_SAMPLE_GRID_SHAPE", (3, 3))
    sample_n_u = int(grid[0])
    sample_n_v = int(grid[1])
    sample_margin_pix = int(getattr(ZLC, "FIELD_SAMPLE_MARGIN_PIX", 0))

    models = tuple(getattr(ZLC, "FIELD_MODELS_TO_STORE", ("plane3", "quad6")))
    store_plane3 = "plane3" in models
    store_quad6 = "quad6" in models

    export_full_map = bool(getattr(ZLC, "EXPORT_FULL_MAP", False))
    full_map_downsample = int(getattr(ZLC, "FULL_MAP_DOWNSAMPLE", 1))

    # Legacy bandpass (pass through as a “default” only)
    # Expect caller to keep legacy BANDPASS_PRESETS already in backend request format.
    bandpass_by_key = {}
    default_bandpass = dict(getattr(ZLC, "DEFAULT_BANDPASS", {"mode": "svo", "filter_id": "GAIA/GAIA3.G"}))

    return ResolvedStageConfig(
        enabled=bool(getattr(ZLC, "ZODIACAL_LIGHT_ENABLED")),
        input_pickle_path=input_pickle_path,
        output_pickle_path=output_pickle_path,
        tmp_dir=tmp_dir,
        schema_version=str(getattr(ZLC, "SCHEMA_VERSION")),
        overwrite=bool(getattr(ZLC, "OVERWRITE", True)),
        validation_enabled=bool(getattr(ZLC, "VALIDATE_INPUTS", True)),
        strict_validation=bool(getattr(ZLC, "STRICT_VALIDATION", False)),
        sample_n_u=sample_n_u,
        sample_n_v=sample_n_v,
        sample_margin_pix=sample_margin_pix,
        store_plane3=store_plane3,
        store_quad6=store_quad6,
        export_full_map=export_full_map,
        full_map_downsample=full_map_downsample,
        omega_pix_mode=str(getattr(ZLC, "OMEGA_PIX_MODE", "from_sensor_fov")),
        omega_pix_sr_explicit=getattr(ZLC, "OMEGA_PIX_SR_EXPLICIT", None),
        use_catalog_name=bool(getattr(ZLC, "USE_CATALOG_NAME", True)),
        use_catalog_band=bool(getattr(ZLC, "USE_CATALOG_BAND", True)),
        catalog_name_override=getattr(ZLC, "CATALOG_NAME_OVERRIDE", None),
        catalog_band_override=getattr(ZLC, "CATALOG_BAND_OVERRIDE", None),
        bandpass_by_key=bandpass_by_key,
        default_bandpass=default_bandpass,
        wsl_distro=wsl_distro,
        wsl_python=wsl_python,
        wsl_worker_script_wsl=wsl_worker_script_wsl,
        wsl_cwd_wsl=wsl_cwd_wsl,
        wsl_timeout_s=int(getattr(ZLC, "WSL_TIMEOUT_S", 1800)),
    )




# -----------------------------------------------------------------------------
# Windows <-> WSL path translation (string-only)
# -----------------------------------------------------------------------------

def win_path_to_wsl(path: Path) -> str:
    """
    Convert a Windows drive path to a WSL /mnt/<drive>/... path.

    Parameters
    ----------
    path : Path
        Windows path (absolute).

    Returns
    -------
    str
        WSL equivalent.
    """
    p = str(path)
    # e.g., "C:\\Users\\x" -> "/mnt/c/Users/x"
    drive = p[0].lower()
    rest = p[2:].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


# -----------------------------------------------------------------------------
# Core stage entrypoints
# -----------------------------------------------------------------------------

def build_obs_zodiacal_light_for_all_observers(
    *,
    window_sources_pickle_path: Optional[Path] = None,
    output_pickle_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build zodiacal-light products for all observers/windows.

    Parameters
    ----------
    window_sources_pickle_path : Path | None
        Path to obs_window_sources.pkl. If None, taken from config.
    output_pickle_path : Path | None
        Output path for obs_zodiacal_light.pkl. If None, taken from config.
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    obs_zodi : dict
        Dictionary keyed by observer name. Contains per-window ZL products.

    Side effects
    ------------
    Writes output pickle to disk.

    Notes
    -----
    - This is the stage-level orchestrator. It calls the per-window builder.
    - It uses start_index/end_index as inclusive indices into pointing/state arrays.
    """
    lg = _get_logger(logger)
    cfg = resolve_stage_config()
    if not cfg.enabled:
        lg.info("Zodiacal Light stage disabled by config; returning empty dict.")
        return {}

    src_path = window_sources_pickle_path.resolve() if window_sources_pickle_path else cfg.input_pickle_path
    out_path = output_pickle_path.resolve() if output_pickle_path else cfg.output_pickle_path
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"WINDOW_SOURCES pickle not found: {src_path}")

    lg.info("Loading WINDOW_SOURCES pickle: %s", src_path)
    with open(src_path, "rb") as f:
        obs_window_sources: Dict[str, Any] = pickle.load(f)

    obs_zodi: Dict[str, Any] = {}
    created_utc = datetime.now(timezone.utc).isoformat()

    for obs_name, obs_track in obs_window_sources.items():
        obs_entry = _build_obs_entry(obs_name, obs_track, cfg, created_utc, lg)
        obs_zodi[obs_name] = obs_entry

    lg.info("Writing zodiacal-light output pickle: %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(obs_zodi, f, protocol=pickle.HIGHEST_PROTOCOL)

    lg.info("Zodiacal Light stage complete: %d observers.", len(obs_zodi))
    return obs_zodi


def _build_obs_entry(
    obs_name: str,
    obs_track: Any,
    cfg: ResolvedStageConfig,
    created_utc: str,
    lg: logging.Logger,
) -> Dict[str, Any]:
    """
    Build the output dict entry for one observer.

    Inputs
    ------
    obs_name : str
    obs_track : TrackDict-like object from obs_window_sources.pkl
    cfg : ResolvedStageConfig
    created_utc : str
        ISO timestamp for provenance.
    lg : logging.Logger

    Output
    ------
    dict with keys:
      observer_name, sensor_name, rows, cols, dt_frame_s, catalog_name, catalog_band,
      schema_version, created_utc, windows:[...]
    """
    sensor_name = str(obs_track.get("sensor_name", ACTIVE_SENSOR.name))
    catalog_name = str(obs_track.get("catalog_name", "UNKNOWN_CATALOG"))
    catalog_band = str(obs_track.get("catalog_band", "UNKNOWN_BAND"))

    sensor = resolve_sensor(sensor_name)

    rows = int(obs_track.get("rows", sensor.rows))
    cols = int(obs_track.get("cols", sensor.cols))
    dt_frame_s = float(obs_track.get("dt_frame_s"))

    windows = list(obs_track.get("windows", []))
    lg.info(
        "Observer '%s': %d windows (sensor=%s, catalog=%s/%s).",
        obs_name,
        len(windows),
        sensor_name,
        catalog_name,
        catalog_band,
    )

    obs_entry: Dict[str, Any] = {
        "observer_name": obs_name,
        "sensor_name": sensor_name,
        "rows": rows,
        "cols": cols,
        "dt_frame_s": dt_frame_s,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "schema_version": cfg.schema_version,
        "created_utc": created_utc,
        "windows": [],
    }

    # Pixel solid angle for converting background surface brightness -> per pixel.
    omega_pix_sr = compute_pixel_solid_angle_sr(sensor, cfg)

    for w in windows:
        wrec = build_window_zodi_product(
            obs_name=obs_name,
            obs_track=obs_track,
            window=w,
            sensor=sensor,
            omega_pix_sr=omega_pix_sr,
            cfg=cfg,
            logger=lg,
        )
        obs_entry["windows"].append(wrec)

    return obs_entry



def build_window_zodi_product(
    *,
    obs_name: str,
    obs_track: Any,
    window: Dict[str, Any],
    sensor: SensorConfig,
    omega_pix_sr: float,
    cfg: ResolvedStageConfig,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build zodiacal light product for a single window.

    Parameters
    ----------
    obs_name : str
        Observer name.
    obs_track : Any
        TrackDict-like structure from obs_window_sources.pkl for this observer.
    window : dict
        One window entry from obs_track["windows"].
    sensor : SensorConfig
        Sensor configuration (rows/cols/FOV).
    omega_pix_sr : float
        Pixel solid angle in steradians per pixel.
    cfg : ResolvedStageConfig
        Normalized stage config.
    logger : logging.Logger | None

    Returns
    -------
    window_record : dict
        Output record for this window, including fitted coefficients and optional map.

    Steps performed
    ---------------
    1) Slice per-frame arrays using start_index/end_index (inclusive).
    2) Generate sample pixel grid(s).
    3) Convert sample pixels -> sample sky directions (RA/Dec) per frame.
    4) Write request payload -> call WSL worker -> read response samples.
    5) Fit plane3/quad6 per frame; optionally build map2d.
    """
    lg = _get_logger(logger)

    window_index = int(window.get("window_index", -1))
    start_index = int(window.get("start_index", -1))
    end_index = int(window.get("end_index", -1))
    n_frames_expected = int(window.get("n_frames", -1))

    # 1) Slice time, pointing, and TEME position arrays
    times_dt, boresight_ra_deg, boresight_dec_deg, boresight_roll_deg, r_teme_km = slice_window_frame_data(
        obs_track=obs_track,
        start_index=start_index,
        end_index=end_index,
        n_frames_expected=n_frames_expected,
    )

    n_frames = len(times_dt)

    # 2) Build sample pixel grids
    fit_grid = make_sample_pixel_grid(
        rows=sensor.rows,
        cols=sensor.cols,
        n_u=cfg.sample_n_u,
        n_v=cfg.sample_n_v,
        margin_pix=cfg.sample_margin_pix,
    )
    fit_x_pix, fit_y_pix = fit_grid["x_pix"], fit_grid["y_pix"]
    fit_u_norm, fit_v_norm = fit_grid["u_norm"], fit_grid["v_norm"]

    # Optional full map grid
    map_grid = None
    if cfg.export_full_map:
        map_grid = make_full_map_pixel_grid(
            rows=sensor.rows,
            cols=sensor.cols,
            downsample=max(1, cfg.full_map_downsample),
            margin_pix=0,
        )

    # 3) Compute sample RA/Dec per frame for the fit grid (and optionally full map grid)
    fit_radec = compute_sample_radec_deg(
        boresight_ra_deg=boresight_ra_deg,
        boresight_dec_deg=boresight_dec_deg,
        boresight_roll_deg=boresight_roll_deg,
        x_pix=fit_x_pix,
        y_pix=fit_y_pix,
        times_utc=times_dt,
        sensor=sensor,
        prefer_nebula_wcs=True,
        strict=cfg.strict_validation,
        logger=lg,
    )


    map_radec = None
    if map_grid is not None:
        map_radec = compute_sample_radec_deg(
            boresight_ra_deg=boresight_ra_deg,
            boresight_dec_deg=boresight_dec_deg,
            boresight_roll_deg=boresight_roll_deg,
            x_pix=map_grid["x_pix"],
            y_pix=map_grid["y_pix"],
            times_utc=times_dt,
            sensor=sensor,
            prefer_nebula_wcs=True,
            strict=cfg.strict_validation,
            logger=lg,
        )


    # 4) Build request payload(s) and call WSL worker
    bandpass = build_bandpass_dict(
        catalog_name=str(obs_track.get("catalog_name", "UNKNOWN")),
        catalog_band=str(obs_track.get("catalog_band", "UNKNOWN")),
        cfg=cfg,
    )
    times_utc_iso = [t.isoformat() for t in times_dt]

    # Fit-grid request/response
    req_base_fit, resp_base_fit = make_tmp_bases(cfg.tmp_dir, obs_name, window_index, suffix="fit")
    req_meta_fit = {
        "schema_version": cfg.schema_version,
        "observer_name": obs_name,
        "window_index": window_index,
        "start_index": start_index,
        "end_index": end_index,
        "n_frames": n_frames,
        "omega_pix_sr": float(omega_pix_sr),
        "bandpass": bandpass,
        "times_utc_iso": times_utc_iso,
    }
    req_arrays_fit = {
        "sample_radec_deg": np.asarray(fit_radec, dtype=np.float64),
        "observer_teme_xyz_km": np.asarray(r_teme_km, dtype=np.float64),
    }
    write_payload(req_base_fit, req_meta_fit, req_arrays_fit, compress=True, overwrite=True, forbid_object_arrays=True)

    run_wsl_worker(
        request_base_win=req_base_fit,
        response_base_win=resp_base_fit,
        cfg=cfg,
        logger=lg,
    )

    resp_meta_fit, resp_arrays_fit = read_payload(resp_base_fit, allow_pickle=False, forbid_object_arrays=True)
    if "phi_ph_m2_s_pix" not in resp_arrays_fit:
        raise KeyError(f"WSL response missing required array 'phi_ph_m2_s_pix' for window {window_index}.")
    phi_fit = np.asarray(resp_arrays_fit["phi_ph_m2_s_pix"], dtype=np.float64)  # (n_frames, n_samples)

    if phi_fit.shape != (n_frames, fit_x_pix.size):
        raise ValueError(
            f"WSL response phi shape {phi_fit.shape} != expected {(n_frames, fit_x_pix.size)} "
            f"for window {window_index}."
        )

    # Optional map-grid request/response
    phi_map = None
    if map_radec is not None and map_grid is not None:
        req_base_map, resp_base_map = make_tmp_bases(cfg.tmp_dir, obs_name, window_index, suffix="map")
        req_meta_map = dict(req_meta_fit)
        req_meta_map.update({"sample_kind": "map2d", "downsample": int(map_grid["downsample"])})
        req_arrays_map = {
            "sample_radec_deg": np.asarray(map_radec, dtype=np.float64),
            "observer_teme_xyz_km": np.asarray(r_teme_km, dtype=np.float64),
        }
        write_payload(req_base_map, req_meta_map, req_arrays_map, compress=True, overwrite=True, forbid_object_arrays=True)

        run_wsl_worker(
            request_base_win=req_base_map,
            response_base_win=resp_base_map,
            cfg=cfg,
            logger=lg,
        )

        resp_meta_map, resp_arrays_map = read_payload(resp_base_map, allow_pickle=False, forbid_object_arrays=True)
        if "phi_ph_m2_s_pix" not in resp_arrays_map:
            raise KeyError(f"WSL response missing required array 'phi_ph_m2_s_pix' (map) for window {window_index}.")
        phi_map_flat = np.asarray(resp_arrays_map["phi_ph_m2_s_pix"], dtype=np.float64)
        if phi_map_flat.shape != (n_frames, map_grid["x_pix"].size):
            raise ValueError(
                f"Map phi shape {phi_map_flat.shape} != expected {(n_frames, map_grid['x_pix'].size)}"
            )
        # reshape to (n_frames, n_v, n_u) using the stored grid shape
        n_u = int(map_grid["n_u"])
        n_v = int(map_grid["n_v"])
        phi_map = phi_map_flat.reshape((n_frames, n_v, n_u))

    # 5) Fit plane3 / quad6 coefficients per frame
    coeff_plane = None
    coeff_quad = None
    diagnostics: Dict[str, Any] = {}

    if cfg.store_plane3:
        coeff_plane, rms_plane = fit_plane3(phi_fit, fit_u_norm, fit_v_norm)
        diagnostics["plane3_rms_per_frame"] = rms_plane

    if cfg.store_quad6:
        coeff_quad, rms_quad = fit_quad6(phi_fit, fit_u_norm, fit_v_norm)
        diagnostics["quad6_rms_per_frame"] = rms_quad

    # Summary diagnostics
    if "plane3_rms_per_frame" in diagnostics:
        diagnostics["plane3_rms_median"] = float(np.median(diagnostics["plane3_rms_per_frame"]))
    if "quad6_rms_per_frame" in diagnostics:
        diagnostics["quad6_rms_median"] = float(np.median(diagnostics["quad6_rms_per_frame"]))

    # Output window record
    window_record: Dict[str, Any] = {
        "window_index": window_index,
        "start_index": start_index,
        "end_index": end_index,
        "start_time": window.get("start_time"),
        "end_time": window.get("end_time"),
        "n_frames": n_frames,
        # Alignment provenance (useful for sanity checks)
        "times_utc_iso": np.asarray(times_utc_iso, dtype="U"),
        "boresight_ra_deg": np.asarray(boresight_ra_deg, dtype=np.float64),
        "boresight_dec_deg": np.asarray(boresight_dec_deg, dtype=np.float64),
        "boresight_roll_deg": np.asarray(boresight_roll_deg, dtype=np.float64),
        # Zodiacal outputs
        "zodi": {
            "quantity": "phi_ph_m2_s_pix",
            "units": "ph m-2 s-1 pix-1",
            "omega_pix_sr": float(omega_pix_sr),
            "bandpass": bandpass,
            "sample_fit_grid": {
                "n_u": int(cfg.sample_n_u),
                "n_v": int(cfg.sample_n_v),
                "margin_pix": int(cfg.sample_margin_pix),
                "x_pix": np.asarray(fit_x_pix, dtype=np.float64),
                "y_pix": np.asarray(fit_y_pix, dtype=np.float64),
                "u_norm": np.asarray(fit_u_norm, dtype=np.float64),
                "v_norm": np.asarray(fit_v_norm, dtype=np.float64),
            },
            "models": {
                "plane3": None if coeff_plane is None else {"coeffs": coeff_plane},
                "quad6": None if coeff_quad is None else {"coeffs": coeff_quad},
                "map2d": None if phi_map is None else {
                    "downsample": int(map_grid["downsample"]) if map_grid else 1,
                    "n_u": int(map_grid["n_u"]) if map_grid else None,
                    "n_v": int(map_grid["n_v"]) if map_grid else None,
                    "phi_ph_m2_s_pix_map": phi_map,
                },
            },
        },
        "fit_diagnostics": diagnostics,
    }

    return window_record


# -----------------------------------------------------------------------------
# Slicing and sensor helpers
# -----------------------------------------------------------------------------

def resolve_sensor(sensor_name: str) -> SensorConfig:
    """
    Resolve a SensorConfig by name, defaulting to ACTIVE_SENSOR if unknown.

    Parameters
    ----------
    sensor_name : str

    Returns
    -------
    SensorConfig
    """
    if sensor_name in SENSOR_CONFIGS:
        return SENSOR_CONFIGS[sensor_name]
    return ACTIVE_SENSOR


def slice_window_frame_data(
    *,
    obs_track: Any,
    start_index: int,
    end_index: int,
    n_frames_expected: int,
) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice per-frame data arrays for a window using inclusive indexing.

    Parameters
    ----------
    obs_track : Any
        Observer TrackDict from obs_window_sources.pkl
    start_index, end_index : int
        Inclusive indices into:
          - observer_geometry["state_vectors"]["times"]
          - observer_geometry["state_vectors"]["r_eci_km"]
          - observer_geometry["pointing"][...]
    n_frames_expected : int
        Window's n_frames. Used for validation.

    Returns
    -------
    times_dt : list[datetime], length n_frames
    boresight_ra_deg : np.ndarray, shape (n_frames,)
    boresight_dec_deg : np.ndarray, shape (n_frames,)
    roll_deg : np.ndarray, shape (n_frames,)
    r_teme_km : np.ndarray, shape (n_frames,3)

    Raises
    ------
    ValueError
        If slice lengths do not match n_frames_expected.
    """
    og = obs_track["observer_geometry"]
    sv = og["state_vectors"]
    pt = og["pointing"]

    # Inclusive slicing: [start_index, end_index]
    sl = slice(start_index, end_index + 1)

    times_dt = list(sv["times"][sl])
    r_teme_km = np.asarray(sv["r_eci_km"][sl], dtype=np.float64)

    boresight_ra_deg = np.asarray(pt["pointing_boresight_ra_deg"][sl], dtype=np.float64)
    boresight_dec_deg = np.asarray(pt["pointing_boresight_dec_deg"][sl], dtype=np.float64)
    roll_deg = np.asarray(pt["pointing_boresight_roll_deg"][sl], dtype=np.float64)

    n_frames = len(times_dt)
    if n_frames_expected > 0 and n_frames != n_frames_expected:
        raise ValueError(
            f"Window slice length mismatch: got {n_frames} frames from indices "
            f"[{start_index},{end_index}] but window says n_frames={n_frames_expected}."
        )
    if r_teme_km.shape != (n_frames, 3):
        raise ValueError(f"r_eci_km slice has shape {r_teme_km.shape}, expected {(n_frames,3)}.")

    return times_dt, boresight_ra_deg, boresight_dec_deg, roll_deg, r_teme_km


def compute_pixel_solid_angle_sr(sensor: SensorConfig, cfg: ResolvedStageConfig) -> float:
    """
    Compute pixel solid angle (sr/pixel) according to cfg, with a small-angle approximation fallback.

    Parameters
    ----------
    sensor : SensorConfig
        Must provide fov_deg and cols (assumes square pixels and symmetric scale).
    cfg : ResolvedStageConfig
        If it exposes omega-pixel controls, they are honored:
          - omega_pix_mode: "from_sensor_fov" or "explicit"
          - omega_pix_sr_explicit: float

        If these attributes are absent (legacy cfg), defaults to "from_sensor_fov".

    Returns
    -------
    float
        omega_pix_sr in steradians per pixel.
    """
    mode = str(getattr(cfg, "omega_pix_mode", "from_sensor_fov"))

    if mode == "explicit":
        omega = getattr(cfg, "omega_pix_sr_explicit", None)
        if omega is None:
            raise ValueError("omega_pix_mode='explicit' requires omega_pix_sr_explicit to be set.")
        return float(omega)

    if mode == "from_sensor_fov":
        pixel_scale_deg = float(sensor.fov_deg) / float(sensor.cols)
        pixel_scale_rad = math.radians(pixel_scale_deg)
        return float(pixel_scale_rad * pixel_scale_rad)

    raise ValueError(f"Unknown omega_pix_mode: {mode!r}")



# -----------------------------------------------------------------------------
# Sample grid generation
# -----------------------------------------------------------------------------

def make_sample_pixel_grid(
    *,
    rows: int,
    cols: int,
    n_u: int,
    n_v: int,
    margin_pix: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create a modest sampling grid for fitting plane/quadratic coefficients.

    Parameters
    ----------
    rows, cols : int
        Detector shape.
    n_u, n_v : int
        Number of sample points across the detector in x (u) and y (v).
        Example: n_u=3, n_v=3 -> 9 samples.
    margin_pix : int
        Exclude pixels within this margin of the detector edges.

    Returns
    -------
    dict with arrays:
      x_pix : (n_samples,) float64
      y_pix : (n_samples,) float64
      u_norm : (n_samples,) float64   normalized [-1,1] coordinate
      v_norm : (n_samples,) float64
      n_u, n_v : scalar arrays not returned (kept in cfg)

    Notes
    -----
    Flatten order is row-major on the (v,u) meshgrid:
      v index (y) varies slow, u index (x) varies fast.
    """
    if n_u < 2 or n_v < 2:
        raise ValueError("n_u and n_v should be >= 2 for stable fitting.")
    if margin_pix < 0:
        raise ValueError("margin_pix must be >= 0.")

    x0 = float(margin_pix)
    x1 = float(cols - 1 - margin_pix)
    y0 = float(margin_pix)
    y1 = float(rows - 1 - margin_pix)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("margin_pix too large; no sampling area remains.")

    x_lin = np.linspace(x0, x1, int(n_u), dtype=np.float64)
    y_lin = np.linspace(y0, y1, int(n_v), dtype=np.float64)

    xx, yy = np.meshgrid(x_lin, y_lin)  # shapes (n_v, n_u)
    x_pix = xx.reshape(-1)
    y_pix = yy.reshape(-1)

    u_norm, v_norm = normalized_pixel_coords(x_pix, y_pix, rows=rows, cols=cols)

    return {
        "x_pix": x_pix,
        "y_pix": y_pix,
        "u_norm": u_norm,
        "v_norm": v_norm,
    }


def make_full_map_pixel_grid(
    *,
    rows: int,
    cols: int,
    downsample: int = 1,
    margin_pix: int = 0,
) -> Dict[str, Any]:
    """
    Create a full (or downsampled) pixel grid for exporting a 2D zodiacal map.

    Parameters
    ----------
    rows, cols : int
        Detector shape.
    downsample : int
        1 => every pixel; 2 => every other pixel, etc.
    margin_pix : int
        Optional margin exclusion.

    Returns
    -------
    dict with:
      x_pix : (n_samples,) float64
      y_pix : (n_samples,) float64
      n_u : int   number of x samples
      n_v : int   number of y samples
      downsample : int

    Notes
    -----
    The returned x_pix/y_pix are flattened in row-major order (v slow, u fast).
    The stage reshapes the returned phi array back into (n_frames, n_v, n_u).
    """
    downsample = max(1, int(downsample))

    xs = np.arange(margin_pix, cols - margin_pix, downsample, dtype=np.int32)
    ys = np.arange(margin_pix, rows - margin_pix, downsample, dtype=np.int32)

    if xs.size == 0 or ys.size == 0:
        raise ValueError("Full-map grid is empty (check margin/downsample).")

    xx, yy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    x_pix = xx.reshape(-1)
    y_pix = yy.reshape(-1)

    return {
        "x_pix": x_pix,
        "y_pix": y_pix,
        "n_u": int(xs.size),
        "n_v": int(ys.size),
        "downsample": downsample,
    }


def normalized_pixel_coords(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to normalized coordinates in [-1,1] centered on detector.

    Parameters
    ----------
    x_pix, y_pix : np.ndarray
        Flattened pixel coordinates.
    rows, cols : int
        Detector shape.

    Returns
    -------
    (u_norm, v_norm) : (np.ndarray, np.ndarray)
        Normalized coordinates where:
          u_norm = (x - (cols-1)/2) / ((cols-1)/2)
          v_norm = (y - (rows-1)/2) / ((rows-1)/2)
    """
    cx = 0.5 * (float(cols) - 1.0)
    cy = 0.5 * (float(rows) - 1.0)
    sx = cx if cx != 0.0 else 1.0
    sy = cy if cy != 0.0 else 1.0
    u_norm = (x_pix - cx) / sx
    v_norm = (y_pix - cy) / sy
    return u_norm.astype(np.float64), v_norm.astype(np.float64)


# -----------------------------------------------------------------------------
# Pixel -> sky direction conversion
# -----------------------------------------------------------------------------

def compute_sample_radec_deg(
    *,
    boresight_ra_deg: np.ndarray,
    boresight_dec_deg: np.ndarray,
    boresight_roll_deg: np.ndarray,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    times_utc: Sequence[Any],
    sensor: SensorConfig,
    prefer_nebula_wcs: bool = True,
    strict: bool = False,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute ICRS sample directions (RA/Dec) for each frame and each sample pixel.

    Preferred:
      - Use NEBULA_WCS.build_wcs_for_observer(...) pixel->world

    Fallback:
      - compute_radec_with_tangent_plane(...) using boresight+roll+pixel scale
        with SkyCoord.directional_offset_by(). 
    """
    lg = _get_logger(logger)

    x = np.asarray(x_pix, dtype=np.float64).reshape(-1)
    y = np.asarray(y_pix, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x_pix and y_pix must have the same number of samples.")

    n_frames = int(len(boresight_ra_deg))
    if len(boresight_dec_deg) != n_frames or len(boresight_roll_deg) != n_frames:
        raise ValueError("Pointing arrays must all have the same length.")
    if len(times_utc) != n_frames:
        raise ValueError("times_utc must have length n_frames (match pointing arrays).")

    sample_xy_pix = np.stack([x, y], axis=1)  # (n_samples,2)

    if prefer_nebula_wcs:
        try:
            return try_compute_radec_with_nebula_wcs(
                sample_xy_pix=sample_xy_pix,
                boresight_ra_deg=np.asarray(boresight_ra_deg, dtype=np.float64),
                boresight_dec_deg=np.asarray(boresight_dec_deg, dtype=np.float64),
                boresight_roll_deg=np.asarray(boresight_roll_deg, dtype=np.float64),
                times_utc=times_utc,
                sensor=sensor,
                logger=lg,
            )
        except Exception as e:
            if strict:
                raise RuntimeError("ZODIACAL_LIGHT_STAGE: NEBULA WCS pixel->world failed (strict=True).") from e
            lg.warning("NEBULA WCS pixel->world failed; falling back to tangent-plane approximation. %r", e)

    return compute_radec_with_tangent_plane(
        boresight_ra_deg=np.asarray(boresight_ra_deg, dtype=np.float64),
        boresight_dec_deg=np.asarray(boresight_dec_deg, dtype=np.float64),
        boresight_roll_deg=np.asarray(boresight_roll_deg, dtype=np.float64),
        x_pix=x,
        y_pix=y,
        sensor=sensor,
    )


def try_compute_radec_with_nebula_wcs(
    *,
    sample_xy_pix: np.ndarray,
    boresight_ra_deg: np.ndarray,
    boresight_dec_deg: np.ndarray,
    boresight_roll_deg: np.ndarray,
    times_utc: Sequence[Any],
    sensor: SensorConfig,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute per-frame RA/Dec for detector sample points using NEBULA's WCS.

    No-guesswork behavior:
      - Uses one import path: NEBULA_WCS.build_wcs_for_observer
      - If import or pixel->world fails, raises.
    """
    lg = _get_logger(logger)

    if sample_xy_pix.ndim != 2 or sample_xy_pix.shape[1] != 2:
        raise ValueError("sample_xy_pix must have shape (n_points, 2).")

    n_frames = int(len(boresight_ra_deg))
    if len(boresight_dec_deg) != n_frames or len(boresight_roll_deg) != n_frames:
        raise ValueError("Pointing arrays must all have the same length.")
    if len(times_utc) != n_frames:
        raise ValueError("times_utc must have length n_frames (match pointing arrays).")

    # Canonical import path for your repo (you have NEBULA_WCS.py).
    from NEBULA_WCS import build_wcs_for_observer  # type: ignore

    track_stub: Dict[str, Any] = {
        "times": list(times_utc),
        "pointing_boresight_ra_deg": np.asarray(boresight_ra_deg, dtype=np.float64),
        "pointing_boresight_dec_deg": np.asarray(boresight_dec_deg, dtype=np.float64),
        "pointing_boresight_roll_deg": np.asarray(boresight_roll_deg, dtype=np.float64),
    }

    wcs_obj = build_wcs_for_observer(track_stub, sensor_config=sensor)

    if isinstance(wcs_obj, list):
        wcs_list = wcs_obj
    else:
        wcs_list = [wcs_obj] * n_frames

    if len(wcs_list) != n_frames:
        raise ValueError(f"build_wcs_for_observer returned {len(wcs_list)} WCS objects, expected {n_frames}.")

    x = sample_xy_pix[:, 0]
    y = sample_xy_pix[:, 1]

    out = np.full((n_frames, sample_xy_pix.shape[0], 2), np.nan, dtype=np.float64)
    for i, w in enumerate(wcs_list):
        ra, dec = _wcs_pixel_to_world_deg(w, x, y)
        out[i, :, 0] = ra
        out[i, :, 1] = dec

    lg.debug("Computed sample RA/Dec via NEBULA WCS for %d frames.", n_frames)
    return out




def _apply_single_wcs(wcs_obj: Any, x_pix: np.ndarray, y_pix: np.ndarray, n_frames: int) -> Optional[np.ndarray]:
    """
    Apply one WCS object to sample pixels. If it cannot convert, return None.
    """
    try:
        ra, dec = _wcs_pixel_to_world_deg(wcs_obj, x_pix, y_pix)
        radec = np.stack([ra, dec], axis=-1)  # (n_samples,2)
        radec = np.repeat(radec[None, :, :], n_frames, axis=0)  # (n_frames,n_samples,2)
        return radec
    except Exception:
        return None


def _apply_wcs_list(wcs_list: Sequence[Any], x_pix: np.ndarray, y_pix: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply a per-frame list of WCS objects. If any frame fails, return None.
    """
    n_frames = len(wcs_list)
    n_samples = x_pix.size
    out = np.empty((n_frames, n_samples, 2), dtype=np.float64)

    for i, w in enumerate(wcs_list):
        try:
            ra, dec = _wcs_pixel_to_world_deg(w, x_pix, y_pix)
            out[i, :, 0] = ra
            out[i, :, 1] = dec
        except Exception:
            return None
    return out


def _wcs_pixel_to_world_deg(wcs_obj: Any, x_pix: np.ndarray, y_pix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt to convert pixel coords to world coords using common WCS interfaces.

    Supported patterns (best effort)
    --------------------------------
    1) wcs_obj.pixel_to_world(x, y) -> SkyCoord
    2) wcs_obj.wcs.all_pix2world(x, y, origin) -> ra, dec (deg)

    Returns
    -------
    (ra_deg, dec_deg) : np.ndarray, np.ndarray
    """
    if hasattr(wcs_obj, "pixel_to_world"):
        sc = wcs_obj.pixel_to_world(x_pix, y_pix)
        if hasattr(sc, "ra") and hasattr(sc, "dec"):
            return np.asarray(sc.ra.deg, dtype=np.float64), np.asarray(sc.dec.deg, dtype=np.float64)

    if hasattr(wcs_obj, "wcs") and hasattr(wcs_obj.wcs, "all_pix2world"):
        ra, dec = wcs_obj.wcs.all_pix2world(x_pix, y_pix, 0)
        return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)

    raise AttributeError("Unsupported WCS object: cannot find pixel_to_world or wcs.all_pix2world.")


def compute_radec_with_tangent_plane(
    *,
    boresight_ra_deg: np.ndarray,
    boresight_dec_deg: np.ndarray,
    boresight_roll_deg: np.ndarray,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    sensor: SensorConfig,
) -> np.ndarray:
    """
    Fallback conversion from pixel coords to RA/Dec using a spherical offset model.

    Parameters
    ----------
    boresight_ra_deg, boresight_dec_deg, boresight_roll_deg : (n_frames,)
    x_pix, y_pix : (n_samples,)
    sensor : SensorConfig
        Uses sensor.fov_deg and sensor.cols to derive approximate pixel scale.

    Returns
    -------
    radec : (n_frames, n_samples, 2)

    Method (physics / geometry)
    ---------------------------
    1) Compute detector-plane offsets (u_det, v_det) in radians from pixel deltas:
         du = (x - cx) * pixel_scale_rad
         dv = (y - cy) * pixel_scale_rad
       where pixel_scale_rad ≈ radians(fov_deg / cols)

    2) Rotate offsets by roll about boresight:
         [u, v] = R(roll) * [du, dv]
       (sign convention depends on NEBULA’s roll definition; this is a best-effort.)

    3) Convert (u, v) to a position angle and separation:
         sep = hypot(u, v)
         pa  = atan2(u, v)   # east of north

    4) Apply spherical offset from the boresight SkyCoord:
         coord = boresight.directional_offset_by(pa, sep)

    This is robust for modest fields of view and avoids flat-sky RA/cos(dec) pitfalls.
    """
    n_frames = int(boresight_ra_deg.size)
    n_samples = int(x_pix.size)

    cx = 0.5 * (sensor.cols - 1)
    cy = 0.5 * (sensor.rows - 1)

    pixel_scale_rad = math.radians(float(sensor.fov_deg) / float(sensor.cols))

    du = (x_pix - cx) * pixel_scale_rad
    dv = (y_pix - cy) * pixel_scale_rad

    # Precompute sample separation (depends only on du/dv after roll, which depends on roll per frame).
    out = np.empty((n_frames, n_samples, 2), dtype=np.float64)

    for i in range(n_frames):
        r = math.radians(float(boresight_roll_deg[i]))
        cr = math.cos(r)
        sr = math.sin(r)

        # rotate detector offsets by roll
        u_off = cr * du - sr * dv
        v_off = sr * du + cr * dv

        sep = np.hypot(u_off, v_off) * u.rad
        pa = np.arctan2(u_off, v_off) * u.rad  # east of north

        bore = SkyCoord(ra=float(boresight_ra_deg[i]) * u.deg, dec=float(boresight_dec_deg[i]) * u.deg, frame="icrs")
        pts = bore.directional_offset_by(pa, sep)

        out[i, :, 0] = pts.ra.deg
        out[i, :, 1] = pts.dec.deg

    return out


# -----------------------------------------------------------------------------
# Bandpass request builder
# -----------------------------------------------------------------------------

# def build_bandpass_dict(*, catalog_name: str, catalog_band: str, cfg: ResolvedStageConfig) -> Dict[str, Any]:
    """
    Build the 'bandpass' dict sent to the WSL backend, keyed off catalog_name/band.

    Parameters
    ----------
    catalog_name, catalog_band : str
        Values from obs_track["catalog_name"] and obs_track["catalog_band"].
    cfg : ResolvedStageConfig

    Returns
    -------
    dict
        bandpass specification (JSON-serializable).

    Behavior
    --------
    If cfg.bandpass_presets contains a key (catalog_name, catalog_band), we use it.
    Otherwise we fall back to a conservative Gaia-G tophat default if bandpass_mode=tophat,
    or a default SVO filter id if bandpass_mode=svo.

    You can tighten this policy once you finalize your config structure.
    """
    key = (str(catalog_name), str(catalog_band))
    preset = cfg.bandpass_presets.get(key)
    if not preset:
        raise ValueError(
            "ZODIACAL_LIGHT_STAGE: no bandpass preset configured for "
            f"(catalog_name={catalog_name!r}, catalog_band={catalog_band!r}). "
            "Add an entry to cfg.bandpass.defaults (or legacy BANDPASS_PRESETS)."
        )

    mode = str(cfg.bandpass_mode).lower().strip()

    if mode == "tophat" and "tophat" in preset:
        d = dict(preset["tophat"])
        return {"mode": "tophat", **d}

    if mode == "svo" and "svo" in preset:
        d = dict(preset["svo"])
        if not d.get("filter_id"):
            raise ValueError(f"ZODIACAL_LIGHT_STAGE: preset {key} missing svo.filter_id.")
        return {"mode": "svo", **d}

    # If cfg.bandpass_mode does not match, allow a single available choice.
    if len(preset) == 1:
        only_key = next(iter(preset.keys()))
        if only_key == "tophat":
            d = dict(preset["tophat"])
            return {"mode": "tophat", **d}
        if only_key == "svo":
            d = dict(preset["svo"])
            if not d.get("filter_id"):
                raise ValueError(f"ZODIACAL_LIGHT_STAGE: preset {key} missing svo.filter_id.")
            return {"mode": "svo", **d}

    raise ValueError(
        f"ZODIACAL_LIGHT_STAGE: cfg.bandpass_mode={cfg.bandpass_mode!r} does not match "
        f"available presets for {key}: {sorted(preset.keys())}."
    )
def build_bandpass_dict(*, catalog_name: str, catalog_band: str, cfg: ResolvedStageConfig) -> Dict[str, Any]:
    """
    Build the 'bandpass' dict sent to the WSL backend.

    Lookup policy (deterministic):
      - Apply overrides if set.
      - Build candidate keys depending on cfg.use_catalog_name / cfg.use_catalog_band:
          * if both:  [f"{name}:{band}", name, band]
          * name only: [name]
          * band only: [band]
      - First match in cfg.bandpass_by_key wins.
      - Otherwise, fall back to cfg.default_bandpass.
    """
    name = cfg.catalog_name_override or str(catalog_name)
    band = cfg.catalog_band_override or str(catalog_band)

    candidates: List[str] = []
    if cfg.use_catalog_name and cfg.use_catalog_band:
        candidates.extend([f"{name}:{band}", name, band])
    elif cfg.use_catalog_name:
        candidates.append(name)
    elif cfg.use_catalog_band:
        candidates.append(band)

    for k in candidates:
        if k in cfg.bandpass_by_key:
            bp = dict(cfg.bandpass_by_key[k])
            return _validate_bandpass_request_dict(bp)

    bp = dict(cfg.default_bandpass)
    return _validate_bandpass_request_dict(bp)


def _validate_bandpass_request_dict(bp: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(bp.get("mode", "")).lower().strip()

    if mode == "svo":
        fid = bp.get("filter_id")
        if not fid:
            raise ValueError("Bandpass dict mode='svo' requires filter_id.")
        return {"mode": "svo", "filter_id": str(fid)}

    if mode == "tophat":
        lo = bp.get("lambda_min_nm")
        hi = bp.get("lambda_max_nm")
        if lo is None or hi is None:
            raise ValueError("Bandpass dict mode='tophat' requires lambda_min_nm and lambda_max_nm.")
        out = {
            "mode": "tophat",
            "lambda_min_nm": float(lo),
            "lambda_max_nm": float(hi),
        }
        if bp.get("lambda_eff_nm") is not None:
            out["lambda_eff_nm"] = float(bp["lambda_eff_nm"])
        return out

    raise ValueError(f"Unknown bandpass dict mode: {mode!r}")


# -----------------------------------------------------------------------------
# WSL call (no m4opt import; pure subprocess)
# -----------------------------------------------------------------------------

def run_wsl_worker(
    *,
    request_base_win: Path,
    response_base_win: Path,
    cfg: ResolvedStageConfig,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Execute the WSL worker script to compute zodiacal samples.

    Parameters
    ----------
    request_base_win : Path
        Windows base path (no extension) for request payload.
    response_base_win : Path
        Windows base path (no extension) for response payload.
    cfg : ResolvedStageConfig
        WSL invocation settings are taken from this config.
    logger : logging.Logger | None

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the worker fails (non-zero return code) or times out.

    Notes
    -----
    This function calls:
        wsl.exe -d <distro> -- bash -lc "cd <cwd> && <python> <worker> <req_wsl> <resp_wsl>"
    """
    lg = _get_logger(logger)

    req_json = Path(str(request_base_win) + ".json")
    req_npz = Path(str(request_base_win) + ".npz")
    if not req_json.exists() or not req_npz.exists():
        raise FileNotFoundError(f"Request payload missing: {req_json} / {req_npz}")

    request_base_wsl = win_path_to_wsl(request_base_win.resolve())
    response_base_wsl = win_path_to_wsl(response_base_win.resolve())

    # Use bash -lc to support "~/" and standard Ubuntu env behavior.
    bash_cmd = f"cd {shlex_quote(cfg.wsl_cwd_wsl)} && {shlex_quote(cfg.wsl_python)} {shlex_quote(cfg.wsl_worker_script_wsl)} {shlex_quote(request_base_wsl)} {shlex_quote(response_base_wsl)}"
    cmd = ["wsl.exe", "-d", cfg.wsl_distro, "--", "bash", "-lc", bash_cmd]

    lg.info("WSL worker call: obs payload '%s' -> '%s'", request_base_win.name, response_base_win.name)

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(cfg.wsl_timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"WSL worker timed out after {cfg.wsl_timeout_s}s.\n{e}") from e

    if p.returncode != 0:
        raise RuntimeError(
            "WSL worker failed.\n"
            f"Return code: {p.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- stdout ---\n{p.stdout}\n"
            f"--- stderr ---\n{p.stderr}\n"
        )

    resp_json = Path(str(response_base_win) + ".json")
    resp_npz = Path(str(response_base_win) + ".npz")
    if not resp_json.exists() or not resp_npz.exists():
        raise RuntimeError(f"WSL worker returned success but response payload missing: {resp_json} / {resp_npz}")


def shlex_quote(s: str) -> str:
    """
    Minimal POSIX shell quoting for bash -lc strings.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Quoted string safe for bash parsing.
    """
    return "'" + s.replace("'", "'\"'\"'") + "'"


def make_tmp_bases(tmp_dir: Path, obs_name: str, window_index: int, suffix: str) -> Tuple[Path, Path]:
    """
    Create deterministic request/response base paths under tmp_dir.

    Parameters
    ----------
    tmp_dir : Path
        Windows temp directory.
    obs_name : str
    window_index : int
    suffix : str
        e.g., "fit" or "map"

    Returns
    -------
    (req_base, resp_base) : (Path, Path)
        Base paths WITHOUT extensions.
    """
    safe_obs = "".join(c if c.isalnum() or c in "._-" else "_" for c in obs_name).strip("_")
    req_base = tmp_dir / f"zodi_req__{safe_obs}__w{window_index:04d}__{suffix}"
    resp_base = tmp_dir / f"zodi_resp__{safe_obs}__w{window_index:04d}__{suffix}"
    return req_base, resp_base


# -----------------------------------------------------------------------------
# Fitting (plane3 and quad6)
# -----------------------------------------------------------------------------

def fit_plane3(phi: np.ndarray, u_norm: np.ndarray, v_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit plane coefficients per frame:
        y = c0 + c1*u + c2*v

    Parameters
    ----------
    phi : np.ndarray
        Samples, shape (n_frames, n_samples)
    u_norm, v_norm : np.ndarray
        Normalized sample coordinates, shape (n_samples,)

    Returns
    -------
    coeffs : np.ndarray
        Shape (n_frames, 3)
    rms_per_frame : np.ndarray
        Shape (n_frames,), RMS residuals against the samples.

    Notes
    -----
    Uses a shared design matrix across frames and solves with a pseudo-inverse for speed.
    """
    y = np.asarray(phi, dtype=np.float64)
    u_ = np.asarray(u_norm, dtype=np.float64).reshape(-1)
    v_ = np.asarray(v_norm, dtype=np.float64).reshape(-1)

    if y.ndim != 2:
        raise ValueError("phi must have shape (n_frames, n_samples)")
    if u_.size != y.shape[1] or v_.size != y.shape[1]:
        raise ValueError("u_norm/v_norm must match phi's n_samples")

    X = np.stack([np.ones_like(u_), u_, v_], axis=1)  # (n_samples, 3)
    pinv = np.linalg.pinv(X)  # (3, n_samples)
    coeffs = y @ pinv.T  # (n_frames, 3)

    yhat = coeffs @ X.T
    resid = y - yhat
    rms = np.sqrt(np.mean(resid * resid, axis=1))

    return coeffs.astype(np.float64), rms.astype(np.float64)


def fit_quad6(phi: np.ndarray, u_norm: np.ndarray, v_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit quadratic coefficients per frame:
        y = c0 + c1*u + c2*v + c3*u^2 + c4*u*v + c5*v^2

    Parameters
    ----------
    phi : np.ndarray
        Samples, shape (n_frames, n_samples)
    u_norm, v_norm : np.ndarray
        Normalized sample coordinates, shape (n_samples,)

    Returns
    -------
    coeffs : np.ndarray
        Shape (n_frames, 6)
    rms_per_frame : np.ndarray
        Shape (n_frames,), RMS residuals against the samples.
    """
    y = np.asarray(phi, dtype=np.float64)
    u_ = np.asarray(u_norm, dtype=np.float64).reshape(-1)
    v_ = np.asarray(v_norm, dtype=np.float64).reshape(-1)

    if y.ndim != 2:
        raise ValueError("phi must have shape (n_frames, n_samples)")
    if u_.size != y.shape[1] or v_.size != y.shape[1]:
        raise ValueError("u_norm/v_norm must match phi's n_samples")

    X = np.stack(
        [
            np.ones_like(u_),
            u_,
            v_,
            u_ * u_,
            u_ * v_,
            v_ * v_,
        ],
        axis=1,
    )  # (n_samples, 6)

    pinv = np.linalg.pinv(X)  # (6, n_samples)
    coeffs = y @ pinv.T  # (n_frames, 6)

    yhat = coeffs @ X.T
    resid = y - yhat
    rms = np.sqrt(np.mean(resid * resid, axis=1))

    return coeffs.astype(np.float64), rms.astype(np.float64)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entrypoint for manual runs.

    Parameters
    ----------
    argv : sequence[str] | None
        If None, argparse uses sys.argv.

    Returns
    -------
    int
        Exit code (0 success).
    """
    ap = argparse.ArgumentParser(description="NEBULA Zodiacal Light stage (Windows orchestrator)")
    ap.add_argument("--window_sources", type=str, default=None, help="Path to obs_window_sources.pkl (optional)")
    ap.add_argument("--output", type=str, default=None, help="Output pickle path (optional)")
    args = ap.parse_args(argv)

    lg = _get_logger(None)

    src = Path(args.window_sources).resolve() if args.window_sources else None
    out = Path(args.output).resolve() if args.output else None

    build_obs_zodiacal_light_for_all_observers(
        window_sources_pickle_path=src,
        output_pickle_path=out,
        logger=lg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
