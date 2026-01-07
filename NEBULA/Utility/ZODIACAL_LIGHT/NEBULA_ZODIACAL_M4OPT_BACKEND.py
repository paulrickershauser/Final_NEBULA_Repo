# -*- coding: utf-8 -*-
"""
NEBULA_ZODIACAL_M4OPT_BACKEND.py  (WSL-only)

Purpose
-------
This module is the *only* place in the NEBULA zodiacal-light integration that
imports and calls M4OPT. It is intended to run inside WSL (Ubuntu-24.04 with
m4opt==2.0.1).

It computes zodiacal light (diffuse sky background) for a set of sample
line-of-sight directions, returning values suitable for NEBULA ingestion.

Key design choice (architecture)
--------------------------------
- Windows/NEBULA computes the *sample directions* (RA/Dec on the sky) using
  NEBULA’s own WCS/sensor geometry.
- WSL/M4OPT evaluates the zodiacal background at those directions/times and
  returns a compact numeric array.

This avoids duplicating NEBULA camera geometry inside WSL and keeps the WSL
worker’s job “physics only.”

What quantity is returned?
--------------------------
This backend returns:

    phi_ph_m2_s_sr : photons / m^2 / s / sr   (band-integrated)

This matches the “entrance pupil” level:
- NOT multiplied by collecting area
- NOT multiplied by throughput
- NOT multiplied by QE

In M4OPT, sky background models follow a surface-brightness convention integrated
over a reference solid angle (BACKGROUND_SOLID_ANGLE). The underlying synphot
countrate(background, bandpass) is therefore normalized to that reference patch.
NEBULA normalizes this to *per steradian* by dividing by BACKGROUND_SOLID_ANGLE.

Instrument-specific conversions are intentionally NOT performed here. The Windows
stage derives, as needed:
- per pixel:    phi_ph_m2_s_pix     = phi_ph_m2_s_sr * omega_pix_sr
- per arcsec^2: phi_ph_m2_s_arcsec2 = phi_ph_m2_s_sr * ARCSEC2_TO_SR

Request/response contract
-------------------------
This backend is called by a small WSL entrypoint script (worker) that handles
file I/O. The entrypoint passes a plain dict `req` with only built-in types and
numpy arrays.

Required request fields
-----------------------
req = {
  "schema_version": str,

  # Time axis: one timestamp per NEBULA frame
  "times_utc_iso": list[str] | np.ndarray[str] shape (n_frames,),

  # Sample directions per frame (ICRS)
  # Either:
  "sample_ra_deg":  np.ndarray[float] shape (n_frames, n_samples),
  "sample_dec_deg": np.ndarray[float] shape (n_frames, n_samples),
  # or:
  "sample_radec_deg": np.ndarray[float] shape (n_frames, n_samples, 2),

  # Bandpass definition
  "bandpass": {
      # Mode is parsed case-insensitively and supports aliases in _parse_bandpass():
      #   - {"tophat", "tophat_nm"}
      #   - {"svo", "svo_id"}
      "mode": str,

      # if mode is tophat-like:
      "lambda_min_nm": float,
      "lambda_max_nm": float,
      # or config-friendly:
      "center_nm": float,
      "width_nm": float,

      # optional:
      "lambda_eff_nm": float | None,

      # if mode is svo-like:
      "filter_id": str,
      # or config-friendly:
      "svo_filter_id": str,
  },

  # Observer location (fail-fast: no fallbacks).
  #
  # Preferred:
  # "observer_itrs_xyz_m": np.ndarray[float] shape (n_frames, 3)
  #
  # NEBULA stage-provided (assumed TEME, despite the ECI name):
  # "observer_eci_xyz_km": np.ndarray[float] shape (n_frames, 3)
}

Returned response fields
------------------------
resp = {
  "schema_version": str,
  "quantity": "phi_ph_m2_s_sr",
  "units": "ph m-2 s-1 sr-1",
  "phi_ph_m2_s_sr": np.ndarray[float64] shape (n_frames, n_samples),
  "meta": { ... }  # small metadata for debugging/auditing
}

Notes
-----
- This file intentionally contains no Windows-specific code (no wsl.exe calls,
  no path translations). It is pure compute logic.
- Chunking support is included to avoid very large memory spikes if you later
  increase sample density.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from astropy import units as u
from astropy.coordinates import EarthLocation, ITRS, SkyCoord
from astropy.time import Time

# synphot is a dependency of m4opt; used for constructing a simple top-hat bandpass
from synphot import Empirical1D, SpectralElement

# M4OPT imports (WSL-only)
from m4opt.synphot import observing, bandpass_from_svo  # type: ignore
from m4opt.synphot.background import ZodiacalBackground  # type: ignore
from m4opt.synphot._math import countrate  # type: ignore (private but stable in 2.0.1)
from m4opt.synphot.background._core import BACKGROUND_SOLID_ANGLE  # type: ignore



BandpassMode = Literal["tophat", "tophat_nm", "svo", "svo_id"]


@dataclass(frozen=True)
class BandpassSpec:
    """
    Parsed bandpass request.

    Attributes
    ----------
    mode : {"tophat","svo"}
        Bandpass mode.
    lambda_min_nm, lambda_max_nm : float | None
        For tophat bandpass: wavelength endpoints in nm.
    lambda_eff_nm : float | None
        Optional effective wavelength in nm (not required for countrate).
    filter_id : str | None
        For SVO bandpass: SVO filter id (e.g., "GAIA/GAIA3.G").
    """
    mode: BandpassMode
    lambda_min_nm: Optional[float] = None
    lambda_max_nm: Optional[float] = None
    lambda_eff_nm: Optional[float] = None
    filter_id: Optional[str] = None


def compute_zodi_for_window_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute zodiacal background for a single NEBULA window request.

    Parameters
    ----------
    req : dict
        Plain request dictionary (see module docstring). Must include:
        - times_utc_iso
        - sample RA/Dec arrays (either sample_ra_deg+sample_dec_deg OR sample_radec_deg)
        - bandpass spec
        - observer location (observer_itrs_xyz_m or observer_eci_xyz_km)

    Returns
    -------
    resp : dict
        Response dictionary with:
        - phi_ph_m2_s_sr : np.ndarray[float64], shape (n_frames, n_samples)
          Band-integrated zodiacal photon rate at the entrance pupil, per m^2 per s per sr.
        - quantity / units strings and small meta for debugging.

    Raises
    ------
    KeyError, ValueError, TypeError
        If required fields are missing or have incompatible shapes.
    """
    schema_version = str(req.get("schema_version", "unknown"))

    times = _parse_times(req)
    ra_deg, dec_deg = _parse_sample_radec(req)

    bandpass_spec = _parse_bandpass(req)
    bp = _build_bandpass_model(bandpass_spec)

    # Fail-fast: observer location must be provided; no geocenter fallback.
    observer_location, observer_location_mode = _build_observer_location(req, times)

    # Evaluate M4OPT zodiacal background at (time, direction).
    # We vectorize by flattening (n_frames, n_samples) -> (n_total,).
    phi = _eval_zodiacal_photon_rate_per_m2_s_sr(
        times=times,
        observer_location=observer_location,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        bandpass=bp,
    )

    # Build response
    resp: Dict[str, Any] = {
        "schema_version": schema_version,
        "quantity": "phi_ph_m2_s_sr",
        "units": "ph m-2 s-1 sr-1",
        "phi_ph_m2_s_sr": phi,
        "meta": {
            "n_frames": int(phi.shape[0]),
            "n_samples": int(phi.shape[1]),
            "bandpass_mode": bandpass_spec.mode,
            "background_solid_angle_sr": float(BACKGROUND_SOLID_ANGLE.to_value(u.sr)),
            "observer_location_mode": observer_location_mode,
        },
    }
    return resp



# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def _parse_times(req: Dict[str, Any]) -> Time:
    """
    Parse frame times from ISO strings into an astropy Time array.

    Input
    -----
    req["times_utc_iso"] : list[str] or np.ndarray[str], shape (n_frames,)

    Output
    ------
    times : astropy.time.Time, shape (n_frames,), scale='utc'
    """
    if "times_utc_iso" not in req:
        raise KeyError("Request missing required key: 'times_utc_iso'")

    t = req["times_utc_iso"]
    if isinstance(t, np.ndarray):
        t_list = t.tolist()
    else:
        t_list = list(t)

    # astropy can parse ISO-8601 strings including timezone offsets.
    # We enforce UTC scale for consistency.
    times = Time(t_list, scale="utc")
    if times.ndim != 1:
        # Defensive: we expect one time per frame
        times = times.flatten()
    return times


def _parse_sample_radec(req: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse sample directions (ICRS RA/Dec) from the request.

    Accepted forms
    --------------
    A) Separate arrays:
        sample_ra_deg  : (n_frames, n_samples)
        sample_dec_deg : (n_frames, n_samples)

    B) Combined array:
        sample_radec_deg : (n_frames, n_samples, 2)  where [:,:,0]=RA, [:,:,1]=Dec

    Returns
    -------
    ra_deg, dec_deg : np.ndarray[float64]
        Each with shape (n_frames, n_samples)
    """
    if "sample_radec_deg" in req:
        arr = np.asarray(req["sample_radec_deg"], dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(
                "sample_radec_deg must have shape (n_frames, n_samples, 2)"
            )
        ra_deg = arr[:, :, 0]
        dec_deg = arr[:, :, 1]
        return ra_deg, dec_deg

    if "sample_ra_deg" not in req or "sample_dec_deg" not in req:
        raise KeyError(
            "Request must include either 'sample_radec_deg' or both "
            "'sample_ra_deg' and 'sample_dec_deg'."
        )

    ra_deg = np.asarray(req["sample_ra_deg"], dtype=np.float64)
    dec_deg = np.asarray(req["sample_dec_deg"], dtype=np.float64)

    if ra_deg.shape != dec_deg.shape:
        raise ValueError(
            f"sample_ra_deg shape {ra_deg.shape} != sample_dec_deg shape {dec_deg.shape}"
        )
    if ra_deg.ndim != 2:
        raise ValueError("Sample RA/Dec arrays must have shape (n_frames, n_samples)")

    return ra_deg, dec_deg


def _parse_bandpass(req: Dict[str, Any]) -> BandpassSpec:
    """
    Parse the bandpass spec from request['bandpass'].

    Accepted schemas (backwards-compatible)
    --------------------------------------
    This backend is intentionally tolerant to avoid "schema drift" between the
    Windows stage config and the WSL backend. The request must include:

        req["bandpass"]["mode"]

    Supported mode values (case-insensitive):

      - {"tophat", "tophat_nm"}:
          A simple top-hat bandpass in nanometers.

          Accepted parameterizations:
            A) Explicit endpoints:
                lambda_min_nm, lambda_max_nm  (preferred)
            B) Center/width:
                center_nm, width_nm           (config-friendly)

      - {"svo", "svo_id"}:
          An SVO Filter Profile Service filter identifier.

          Accepted keys:
            filter_id       (backend-native)
            svo_filter_id   (config-friendly)

    Returns
    -------
    BandpassSpec
        Normalized to:
          - mode in {"tophat","svo"}
          - tophat uses lambda_min_nm/lambda_max_nm
          - svo uses filter_id
    """
    bp = req.get("bandpass", None)
    if bp is None or not isinstance(bp, dict):
        raise KeyError("Request missing required dict key: 'bandpass'")

    mode_raw = str(bp.get("mode", "")).strip().lower()

    # Normalize mode aliases.
    if mode_raw in ("tophat", "tophat_nm"):
        mode = "tophat"
    elif mode_raw in ("svo", "svo_id"):
        mode = "svo"
    else:
        raise ValueError(
            "bandpass.mode must be one of {'tophat','tophat_nm','svo','svo_id'} "
            f"(case-insensitive). Got: {mode_raw!r}"
        )

    if mode == "tophat":
        lam_min = bp.get("lambda_min_nm", None)
        lam_max = bp.get("lambda_max_nm", None)

        if lam_min is None or lam_max is None:
            # Config-friendly parameterization: center/width.
            center = bp.get("center_nm", None)
            width = bp.get("width_nm", None)
            if center is None or width is None:
                raise KeyError(
                    "tophat bandpass requires either (lambda_min_nm, lambda_max_nm) "
                    "or (center_nm, width_nm)"
                )
            c = float(center)
            w = float(width)
            if not np.isfinite(c) or not np.isfinite(w):
                raise ValueError("center_nm and width_nm must be finite")
            if w <= 0:
                raise ValueError("width_nm must be > 0")
            lam_min = c - 0.5 * w
            lam_max = c + 0.5 * w

        lam_min_f = float(lam_min)
        lam_max_f = float(lam_max)
        if not np.isfinite(lam_min_f) or not np.isfinite(lam_max_f):
            raise ValueError("lambda_min_nm and lambda_max_nm must be finite")
        if lam_max_f <= lam_min_f:
            raise ValueError("tophat requires lambda_max_nm > lambda_min_nm")

        lam_eff = bp.get("lambda_eff_nm", None)
        return BandpassSpec(
            mode="tophat",
            lambda_min_nm=lam_min_f,
            lambda_max_nm=lam_max_f,
            lambda_eff_nm=None if lam_eff is None else float(lam_eff),
        )

    # mode == "svo"
    fid = bp.get("filter_id", None)
    if not fid:
        fid = bp.get("svo_filter_id", None)

    if not fid:
        raise KeyError("svo bandpass requires filter_id (or svo_filter_id)")

    return BandpassSpec(mode="svo", filter_id=str(fid))


def _require_float(req: Dict[str, Any], key: str) -> float:
    """
    Fetch a required float from the request with a clear error message.
    """
    if key not in req:
        raise KeyError(f"Request missing required key: '{key}'")
    try:
        return float(req[key])
    except Exception as e:
        raise TypeError(f"Request key '{key}' must be a float-like value") from e


# -----------------------------------------------------------------------------
# Bandpass construction
# -----------------------------------------------------------------------------

def _build_bandpass_model(spec: BandpassSpec):
    """
    Build a synphot-compatible bandpass model suitable for m4opt.synphot.countrate().

    For "svo": uses m4opt.synphot.bandpass_from_svo(filter_id).
    For "tophat": constructs a simple rectangular transmission using synphot Empirical1D.

    Returns
    -------
    bandpass_model : synphot.SpectralElement or astropy.modeling.Model
        A 1D model mapping wavelength -> dimensionless transmission.
    """
    if spec.mode == "svo":
        # M4OPT handles download/caching internally.
        return bandpass_from_svo(spec.filter_id)  # type: ignore[arg-type]

    if spec.mode == "tophat":
        assert spec.lambda_min_nm is not None and spec.lambda_max_nm is not None
        lam_min = float(spec.lambda_min_nm) * u.nm
        lam_max = float(spec.lambda_max_nm) * u.nm
        if lam_max <= lam_min:
            raise ValueError("tophat lambda_max_nm must be > lambda_min_nm")

        # Build a minimal “box” curve:
        #   transmission = 0 below min
        #   transmission = 1 inside
        #   transmission = 0 above max
        points = u.Quantity([lam_min, lam_min, lam_max, lam_max])
        lookup = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        return SpectralElement(Empirical1D, points=points, lookup_table=lookup)

    raise ValueError(f"Unsupported bandpass mode: {spec.mode}")


# -----------------------------------------------------------------------------
# Observer location handling
# -----------------------------------------------------------------------------

def _build_observer_location(req: Dict[str, Any], times: Time) -> Tuple[EarthLocation, str]:
    """
    Build an EarthLocation for the observing state.

    Fail-fast contract (no fallbacks)
    --------------------------------
    Exactly one of the following must be provided:

    A) req["observer_itrs_xyz_m"] : ndarray shape (n_frames,3)
       Interpreted as ITRS geocentric X,Y,Z in meters.

    B) req["observer_eci_xyz_km"] : ndarray shape (n_frames,3)
       Assumed TEME geocentric X,Y,Z in kilometers (despite the ECI name).
       Converted to ITRS using astropy coordinates at the provided obstime.

    Legacy key policy
    -----------------
    - If req["observer_teme_xyz_km"] is present, this function raises and instructs
      the caller to use "observer_eci_xyz_km" instead.

    Returns
    -------
    (EarthLocation, mode_str)
        mode_str is one of {"itrs_xyz_m","eci_xyz_km_assumed_teme"}.

    Raises
    ------
    KeyError, ValueError
        If required position is missing or shapes do not match.
    RuntimeError
        If TEME->ITRS conversion fails (no fallback).
    """
    if "observer_teme_xyz_km" in req:
        raise KeyError(
            "Request includes legacy key 'observer_teme_xyz_km'. "
            "Use 'observer_eci_xyz_km' (assumed TEME, km) instead."
        )

    has_itrs = "observer_itrs_xyz_m" in req
    has_eci = "observer_eci_xyz_km" in req

    # EDIT 1: enforce “exactly one key”
    if has_itrs and has_eci:
        raise KeyError(
            "Observer location is ambiguous: provide exactly one of "
            "'observer_itrs_xyz_m' or 'observer_eci_xyz_km' (got both)."
        )
    if not has_itrs and not has_eci:
        raise KeyError(
            "Request must include observer location: either 'observer_itrs_xyz_m' (meters, ITRS) "
            "or 'observer_eci_xyz_km' (km, assumed TEME). No fallbacks are allowed."
        )

    if has_itrs:
        xyz_m = np.asarray(req["observer_itrs_xyz_m"], dtype=np.float64)
        if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
            raise ValueError("observer_itrs_xyz_m must have shape (n_frames, 3)")
        if xyz_m.shape[0] != times.size:
            raise ValueError(
                f"observer_itrs_xyz_m n_frames={xyz_m.shape[0]} does not match times n_frames={times.size}"
            )

        # EDIT 2: finite-value validation
        if not np.isfinite(xyz_m).all():
            raise ValueError("observer_itrs_xyz_m contains NaN/Inf values (fail-fast).")

        loc = EarthLocation.from_geocentric(
            xyz_m[:, 0] * u.m,
            xyz_m[:, 1] * u.m,
            xyz_m[:, 2] * u.m,
        )
        return loc, "itrs_xyz_m"

    # has_eci
    xyz_km = np.asarray(req["observer_eci_xyz_km"], dtype=np.float64)
    if xyz_km.ndim != 2 or xyz_km.shape[1] != 3:
        raise ValueError("observer_eci_xyz_km must have shape (n_frames, 3)")
    if xyz_km.shape[0] != times.size:
        raise ValueError(
            f"observer_eci_xyz_km n_frames={xyz_km.shape[0]} does not match times n_frames={times.size}"
        )

    # EDIT 2: finite-value validation
    if not np.isfinite(xyz_km).all():
        raise ValueError("observer_eci_xyz_km contains NaN/Inf values (fail-fast).")

    # Assumption (per NEBULA stage contract): observer_eci_xyz_km is TEME.
    from astropy.coordinates import TEME

    # EDIT 3: wrap TEME->ITRS failures with a more actionable message
    try:
        teme = TEME(
            x=xyz_km[:, 0] * u.km,
            y=xyz_km[:, 1] * u.km,
            z=xyz_km[:, 2] * u.km,
            obstime=times,
        )
        itrs = teme.transform_to(ITRS(obstime=times))
    except Exception as e:
        raise RuntimeError(
            "Failed to transform 'observer_eci_xyz_km' (assumed TEME, km) to ITRS. "
            "This conversion is time-dependent and may require valid IERS Earth-orientation data "
            "(astropy.utils.iers). No fallback is permitted."
        ) from e

    loc = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
    return loc, "eci_xyz_km_assumed_teme"


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

def _eval_zodiacal_photon_rate_per_m2_s_sr(
    *,
    times: Time,
    observer_location: EarthLocation,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    bandpass: Any,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """
    Evaluate zodiacal background and return band-integrated photon rate per m^2 per s per sr.

    Parameters
    ----------
    times : astropy.time.Time, shape (n_frames,)
        Frame times in UTC scale.
    observer_location : astropy.coordinates.EarthLocation
        Observer location. Vectorized locations are allowed.
    ra_deg, dec_deg : np.ndarray
        Sample directions, shape (n_frames, n_samples)
    bandpass : synphot SpectralElement or model
        Bandpass model mapping wavelength -> transmission.
    chunk_size : int
        Flattened evaluation chunk size to limit peak memory.

    Returns
    -------
    phi : np.ndarray[float64], shape (n_frames, n_samples)
        Zodiacal photon rate at entrance pupil level:
          photons / m^2 / s / sr  (band-integrated)

    Implementation details
    ----------------------
    - m4opt.synphot._math.countrate(model, bandpass) yields a photon count rate per unit
      collecting area for the background model, integrated over M4OPT’s reference solid
      angle (BACKGROUND_SOLID_ANGLE).
    - We convert to per-steradian via:
          phi_sr = countrate(...) / BACKGROUND_SOLID_ANGLE
      and leave any per-pixel conversion (multiplying by Ω_pix) to the Windows stage.
    """
    if ra_deg.shape != dec_deg.shape:
        raise ValueError("ra_deg and dec_deg must have identical shapes")
    if ra_deg.ndim != 2:
        raise ValueError("ra_deg/dec_deg must have shape (n_frames, n_samples)")

    n_frames, n_samples = ra_deg.shape

    # Flatten time+direction arrays for vectorized evaluation.
    ra_flat = ra_deg.reshape(-1)
    dec_flat = dec_deg.reshape(-1)

    # Repeat each frame time across its n_samples.
    # Example: times=[t0,t1], n_samples=3 => obstime=[t0,t0,t0,t1,t1,t1]
    frame_idx = np.repeat(np.arange(n_frames), n_samples)
    obstime = times[frame_idx]

    # SkyCoord for all sample rays (ICRS)
    target_coord = SkyCoord(
        ra=ra_flat * u.deg,
        dec=dec_flat * u.deg,
        frame="icrs",
    )

    # M4OPT validates that (observer_location, target_coord, obstime) shapes are broadcastable.
    # Since we flatten (n_frames, n_samples) -> (n_total,), we must also expand the observer
    # location from (n_frames,) -> (n_total,) unless it is a scalar EarthLocation.
    loc_shape = getattr(observer_location, "shape", ())
    observer_is_scalar = (loc_shape == ())
    if not observer_is_scalar:
        if len(loc_shape) != 1 or int(loc_shape[0]) != int(n_frames):
            raise ValueError(
                "observer_location must be scalar or have shape (n_frames,). "
                f"Got shape={loc_shape!r} for n_frames={n_frames}."
            )

        # Repeat geocentric coordinates per sample so each flattened ray has a matching location.
        x_m = observer_location.x.to(u.m)
        y_m = observer_location.y.to(u.m)
        z_m = observer_location.z.to(u.m)
        x_rep = np.repeat(np.asarray(x_m.value, dtype=np.float64), n_samples) * u.m
        y_rep = np.repeat(np.asarray(y_m.value, dtype=np.float64), n_samples) * u.m
        z_rep = np.repeat(np.asarray(z_m.value, dtype=np.float64), n_samples) * u.m

    # Prepare output buffer
    out = np.empty((n_frames * n_samples,), dtype=np.float64)

    zb = ZodiacalBackground()

    n_total = n_frames * n_samples
    for i0 in range(0, n_total, chunk_size):
        i1 = min(i0 + chunk_size, n_total)

        if observer_is_scalar:
            loc_chunk = observer_location
        else:
            loc_chunk = EarthLocation.from_geocentric(
                x_rep[i0:i1],
                y_rep[i0:i1],
                z_rep[i0:i1],
            )

        with observing(
            observer_location=loc_chunk,
            target_coord=target_coord[i0:i1],
            obstime=obstime[i0:i1],
        ):
            rate_per_m2 = countrate(zb, bandpass)  # units ~ Hz / m^2

            # Convert to per-steradian by dividing by the reference solid angle.
            rate_per_m2_per_sr = rate_per_m2 / BACKGROUND_SOLID_ANGLE

            vals = rate_per_m2_per_sr.to_value(u.Hz / (u.m**2 * u.sr))
            if np.shape(vals) not in [(), (i1 - i0,)]:
                raise RuntimeError(
                    "Unexpected countrate shape; expected scalar or (chunk_len,).\n"
                    f"  got shape={np.shape(vals)!r}\n"
                    f"  chunk_len={(i1 - i0)}\n"
                )
            out[i0:i1] = vals

    return out.reshape((n_frames, n_samples))
