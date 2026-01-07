"""
NEBULA_ZODIACAL_LIGHT_CONFIG.py

Central configuration for the Zodiacal Light (ZL) product pipeline.

This module is intentionally "configuration-only":
- No orchestration logic
- No m4opt imports
- No execution entrypoints

Design intent (NEBULA framing)
------------------------------
1) The WSL backend (m4opt) provides a sky-background quantity in a surface-brightness
   convention (normalized per 1 arcsec^2 patch of sky). We treat that as "native"
   provenance and store it as per_arcsec2.

2) We also store a canonical radiance-like quantity per steradian (per_sr) and
   store a convenience "per pixel" value (per_pixel) computed via a nominal
   pixel solid angle Ω_pix derived from ACTIVE_SENSOR geometry.

3) Pixel solid angle Ω_pix is treated as an *optical geometry* property:
   - In this configuration we compute a single scalar Ω_pix using pixel_pitch/focal_length.
   - This is a small-angle, distortion-ignorant approximation.
   - If you later introduce distortion-aware plate scale, the correct evolution is to
     compute Ω_pix from the same distortion-aware camera model/WCS used for projections.

Fail-fast policy
----------------
This config is intended to "fail fast" when wired incorrectly:
- If a non-active sensor is requested, raise immediately.
- If Ω_pix is needed but sensor geometry is invalid, raise.
- If bandpass spec is incomplete/invalid, raise.
- If map2d_downsample is invalid, raise.

Exports
-------
- ZODIACAL_LIGHT_CONFIG : the single active dataclass instance used by the pipeline
- Helper functions used by the Windows stage:
    resolve_*_path(), get_sensor_config(), pixel_solid_angle_sr(), pixel_solid_angle_arcsec2(),
    convert_per_arcsec2_to_per_sr(), convert_per_sr_to_per_pixel(), validate_config()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple
import math

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR, SensorConfig


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

BandpassMode = Literal["svo_id", "tophat_nm"]
ZodiSpatialModel = Literal["plane3", "quad6", "map2d"]
ZodiQuantity = Literal["per_arcsec2", "per_sr", "per_pixel"]

# As requested: keep ONLY geometry-based Ω_pix.
PixelSolidAngleMode = Literal["from_sensor_geometry"]


# -----------------------------------------------------------------------------
# Angular conversion constants
# -----------------------------------------------------------------------------
# 1 arcsec = pi / (180 * 3600) radians.
ARCSEC_TO_RAD: float = math.pi / (180.0 * 3600.0)

# 1 arcsec^2 expressed as steradians (sr).
ARCSEC2_TO_SR: float = ARCSEC_TO_RAD * ARCSEC_TO_RAD  # ≈ 2.35044e-11 sr / arcsec^2


# -----------------------------------------------------------------------------
# Bandpass specification
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BandpassSpec:
    """
    Defines how the WSL backend should construct a bandpass curve for integration.

    Supported modes (config-only; actual implementation lives in the WSL backend):
    - "svo_id":
        Use an SVO Filter Profile Service filter identifier (e.g., "GAIA/GAIA3.G").
        Required fields:
            svo_filter_id

    - "tophat_nm":
        Use a simple top-hat defined in nanometers.
        Required fields:
            center_nm, width_nm

    Notes
    -----
    - This module does not fetch or validate any external resources.
    - Validation here is structural: required fields must be present and finite.
    """
    mode: BandpassMode

    # SVO
    svo_filter_id: Optional[str] = None

    # Top-hat (nm)
    center_nm: Optional[float] = None
    width_nm: Optional[float] = None

    description: str = ""


@dataclass(frozen=True)
class ZLCatalogBandpassConfig:
    """
    Declares the bandpass used for ZL integration.

    IMPORTANT:
    - This configuration is authoritative for the ZL pipeline.
    - The stage should not "discover" bandpass from obs_window_sources.pkl if you
      already define it here.
    - If you choose to validate pickle metadata, treat these fields as expected
      values and raise if the pickle disagrees.

    Fields
    ------
    catalog_name_expected, catalog_band_expected:
        Optional "expected" values for fail-fast checks against scene metadata.

    bandpass:
        The bandpass specification the WSL backend will use.
    """
    catalog_name_expected: Optional[str] = "GAIA_DR3_G"
    catalog_band_expected: Optional[str] = "G"

    bandpass: BandpassSpec = BandpassSpec(
        mode="svo_id",
        svo_filter_id="GAIA/GAIA3.G",
        description="Gaia (E)DR3 G passband via SVO FPS",
    )


# -----------------------------------------------------------------------------
# Fit / field sampling
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZLFieldModelConfig:
    """
    Spatial representation controls for how ZL is represented downstream.

    models_to_store:
        Which compact per-frame models to store. Common choice is ("plane3","quad6")
        so you can compare fidelity/residuals.

    export_map2d:
        Whether to also store a full 2D map per frame. This can be large.

    map2d_downsample:
        If export_map2d is True, downsample factor applied to the exported map.
        - 1 means "store full resolution"
        - 2 means "store every other pixel" in each axis, etc.

    sample_grid:
        The (n_u, n_v) grid used to sample ZL on the detector before fitting.
        Example: (5,5) => 25 samples per frame.

    sample_margin_pix:
        Optional margin (in pixels) to keep samples away from the very edge.

    normalized_uv:
        If True, fitting uses normalized detector coordinates u,v in [-1,1] instead
        of raw pixel indices. A typical mapping is:

            u = 2 * x/(cols-1) - 1
            v = 2 * y/(rows-1) - 1

        This improves numerical conditioning of polynomial least-squares fits and
        makes coefficients more comparable across sensors.
    """
    models_to_store: Sequence[ZodiSpatialModel] = ("plane3", "quad6")

    export_map2d: bool = False
    map2d_downsample: int = 1

    sample_grid: Tuple[int, int] = (5, 5)
    sample_margin_pix: int = 0
    normalized_uv: bool = True


# -----------------------------------------------------------------------------
# Unit / output conventions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZLUnitsConfig:
    """
    Controls which quantities are stored in the output product.

    store_native_per_arcsec2:
        Store the WSL backend's native convention per arcsec^2.

    derived_outputs:
        Additional quantities to compute and store.
        - "per_sr"    : derived from per_arcsec2 via ARCSEC2_TO_SR
        - "per_pixel" : derived from per_sr via Ω_pix (sr/pixel)

    Project requirement:
    - Store all three values (per_arcsec2 + per_sr + per_pixel).
    """
    store_native_per_arcsec2: bool = True
    derived_outputs: Sequence[ZodiQuantity] = ("per_sr", "per_pixel")


# -----------------------------------------------------------------------------
# Pixel solid angle Ω_pix (sr/pixel)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZLPixelSolidAngleConfig:
    """
    How to compute the nominal per-pixel solid angle Ω_pix (sr/pixel).

    Only supported mode:
    - "from_sensor_geometry":
        Ω_pix ≈ (pixel_pitch / focal_length)^2

    This uses ACTIVE_SENSOR.pixel_pitch and ACTIVE_SENSOR.focal_length (meters).
    """
    mode: PixelSolidAngleMode = "from_sensor_geometry"


# -----------------------------------------------------------------------------
# WSL invocation settings (used by the Windows stage)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZLWSLConfig:
    """
    Windows -> WSL execution settings.

    worker_relpath:
        Repo-relative path to the WSL worker script.
    """
    distro: str = "Ubuntu-24.04"
    wsl_python: str = "~/venvs/nebula_m4opt/bin/python"
    worker_relpath: str = "Utility/ZODIACAL_LIGHT/NEBULA_ZODIACAL_LIGHT_WSL_WORKER.py"
    cwd_wsl: str = "/mnt/c/Users/prick/Desktop/Research/NEBULA"
    timeout_s: int = 1800


# -----------------------------------------------------------------------------
# IO paths and schema
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZLIOConfig:
    """
    Input/Output locations relative to NEBULA_OUTPUT_DIR.
    """
    window_sources_relpath: str = "SCENE/obs_window_sources.pkl"
    tmp_dir_relpath: str = "TMP/ZODIACAL_LIGHT"
    output_relpath: str = "ZODIACAL_LIGHT/obs_zodiacal_light.pkl"

    schema_version: str = "0.1"
    overwrite: bool = True


# -----------------------------------------------------------------------------
# Top-level config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZodiacalLightConfig:
    """
    Top-level configuration object for the Zodiacal Light pipeline.

    This configuration is anchored to ACTIVE_SENSOR. The stage may optionally
    validate obs_window_sources.pkl metadata against sensor_name_expected and
    catalog.*_expected, but the stage must not select sensor/band dynamically.
    """
    enabled: bool = True

    observer_position_frame: Literal["TEME"] = "TEME"

    sensor_name_expected: str = ACTIVE_SENSOR.name

    catalog: ZLCatalogBandpassConfig = ZLCatalogBandpassConfig()
    field: ZLFieldModelConfig = ZLFieldModelConfig()
    units: ZLUnitsConfig = ZLUnitsConfig()
    omega_pix: ZLPixelSolidAngleConfig = ZLPixelSolidAngleConfig()
    wsl: ZLWSLConfig = ZLWSLConfig()
    io: ZLIOConfig = ZLIOConfig()

    bridge_format: Literal["npz+json", "pickle", "hdf5"] = "npz+json"


# Single global instance (NEBULA pattern)
ZODIACAL_LIGHT_CONFIG = ZodiacalLightConfig()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def resolve_window_sources_path(cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG) -> Path:
    """Resolve the absolute path to obs_window_sources.pkl."""
    p = Path(cfg.io.window_sources_relpath)
    return p if p.is_absolute() else (Path(NEBULA_OUTPUT_DIR) / p)


def resolve_tmp_dir(cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG) -> Path:
    """Resolve the absolute tmp staging directory used for Windows<->WSL payloads."""
    p = Path(cfg.io.tmp_dir_relpath)
    return p if p.is_absolute() else (Path(NEBULA_OUTPUT_DIR) / p)


def resolve_output_path(cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG) -> Path:
    """Resolve the absolute output path for the standalone ZL pickle."""
    p = Path(cfg.io.output_relpath)
    return p if p.is_absolute() else (Path(NEBULA_OUTPUT_DIR) / p)


def get_sensor_config(sensor_name: Optional[str] = None) -> SensorConfig:
    """
    Return ACTIVE_SENSOR only.

    Fail-fast behavior:
    - If sensor_name is provided and does not match ACTIVE_SENSOR.name, raise.
    """
    if sensor_name is None:
        return ACTIVE_SENSOR
    if str(sensor_name) != str(ACTIVE_SENSOR.name):
        raise RuntimeError(
            "NEBULA_ZODIACAL_LIGHT_CONFIG: requested sensor_name does not match ACTIVE_SENSOR. "
            f"requested={sensor_name!r} ACTIVE_SENSOR.name={ACTIVE_SENSOR.name!r}"
        )
    return ACTIVE_SENSOR


def pixel_solid_angle_sr(
    sensor: SensorConfig,
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> float:
    """
    Compute nominal Ω_pix (sr/pixel) from sensor geometry:
        Ω_pix ≈ (pixel_pitch / focal_length)^2
    """
    if cfg.omega_pix.mode != "from_sensor_geometry":
        raise RuntimeError(f"Unsupported omega_pix.mode={cfg.omega_pix.mode!r}")

    p = float(sensor.pixel_pitch)
    f = float(sensor.focal_length)

    if not math.isfinite(p) or p <= 0.0:
        raise RuntimeError(f"sensor.pixel_pitch is invalid; expected finite > 0, got {sensor.pixel_pitch!r}")
    if not math.isfinite(f) or f <= 0.0:
        raise RuntimeError(f"sensor.focal_length is invalid; expected finite > 0, got {sensor.focal_length!r}")

    s = p / f
    return s * s


def pixel_solid_angle_arcsec2(
    sensor: SensorConfig,
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> float:
    """
    Compute nominal Ω_pix in arcsec^2 / pixel:
        Ω_pix_arcsec2 = Ω_pix_sr / ARCSEC2_TO_SR
    """
    return pixel_solid_angle_sr(sensor=sensor, cfg=cfg) / ARCSEC2_TO_SR


def convert_per_arcsec2_to_per_sr(phi_per_arcsec2: float) -> float:
    """
    Convert per_arcsec2 -> per_sr:
        phi_per_sr = phi_per_arcsec2 / ARCSEC2_TO_SR
    """
    x = float(phi_per_arcsec2)
    if not math.isfinite(x):
        raise RuntimeError(f"convert_per_arcsec2_to_per_sr: input not finite: {phi_per_arcsec2!r}")
    return x / ARCSEC2_TO_SR


def convert_per_sr_to_per_pixel(phi_per_sr: float, omega_pix_sr: float) -> float:
    """
    Convert per_sr -> per_pixel:
        phi_per_pixel = phi_per_sr * omega_pix_sr
    """
    x = float(phi_per_sr)
    o = float(omega_pix_sr)
    if not math.isfinite(x):
        raise RuntimeError(f"convert_per_sr_to_per_pixel: phi_per_sr not finite: {phi_per_sr!r}")
    if not math.isfinite(o) or o <= 0.0:
        raise RuntimeError(f"convert_per_sr_to_per_pixel: omega_pix_sr invalid: {omega_pix_sr!r}")
    return x * o


def validate_config(cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG) -> None:
    """
    Validate configuration structure and raise immediately on invalid settings.

    Intended call site: at the top of the Windows ZL stage.

    Validates:
    - field.sample_grid positive
    - field.sample_margin_pix >= 0
    - field.map2d_downsample >= 1
    - bandpass structural requirements
    - if per_pixel requested, sensor geometry supports Ω_pix
    """
    # Field sampling
    n_u, n_v = cfg.field.sample_grid
    if int(n_u) <= 0 or int(n_v) <= 0:
        raise RuntimeError(f"field.sample_grid invalid: {cfg.field.sample_grid!r}")
    if int(cfg.field.sample_margin_pix) < 0:
        raise RuntimeError(f"field.sample_margin_pix invalid: {cfg.field.sample_margin_pix!r}")
    if int(cfg.field.map2d_downsample) < 1:
        raise RuntimeError(f"field.map2d_downsample invalid: {cfg.field.map2d_downsample!r}")

    # Units sanity
    if not cfg.units.store_native_per_arcsec2:
        raise RuntimeError("units.store_native_per_arcsec2 must be True for your stated requirement.")
    if "per_sr" not in cfg.units.derived_outputs or "per_pixel" not in cfg.units.derived_outputs:
        raise RuntimeError("units.derived_outputs must include both 'per_sr' and 'per_pixel' for your stated requirement.")

    # Bandpass validation (structural)
    bp = cfg.catalog.bandpass
    if bp.mode == "svo_id":
        if not bp.svo_filter_id or not str(bp.svo_filter_id).strip():
            raise RuntimeError("BandpassSpec(mode='svo_id') requires svo_filter_id to be set.")
    elif bp.mode == "tophat_nm":
        if bp.center_nm is None or bp.width_nm is None:
            raise RuntimeError("BandpassSpec(mode='tophat_nm') requires center_nm and width_nm.")
        c = float(bp.center_nm)
        w = float(bp.width_nm)
        if (not math.isfinite(c)) or (not math.isfinite(w)) or w <= 0.0:
            raise RuntimeError(f"Invalid tophat bandpass center/width: center_nm={bp.center_nm!r}, width_nm={bp.width_nm!r}")
    else:
        raise RuntimeError(f"Unsupported BandpassSpec.mode={bp.mode!r}")

    # Ω_pix check if per_pixel requested
    if "per_pixel" in cfg.units.derived_outputs:
        _ = pixel_solid_angle_sr(sensor=ACTIVE_SENSOR, cfg=cfg)
