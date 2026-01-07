"""
NEBULA_WINDOW_SOURCES
====================

Pipeline stage: WINDOW_SOURCES

Build a combined, per-observer / per-window pickle that contains:

- Window-level metadata and TARGET time series copied from:
    TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_sky.pkl

- STAR photon time series for stars that are on the sensor copied from:
    STARS/<catalog>/obs_star_photons.pkl

- Optional (lightweight) Gaia query/provenance metadata copied from:
    STARS/<catalog>/obs_gaia_cones.pkl

Important constraints (per your spec)
-------------------------------------
- No per-frame image cubes/frames are stored in this pickle.
- Only stars that are projected onto the sensor are included.
- Metadata copied from Gaia/sky sources must be lightweight (no large arrays).

Output
------
NEBULA_OUTPUT_DIR/SCENE/obs_window_sources.pkl
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Configuration import NEBULA_STAR_CONFIG as NSC


WINDOW_SOURCES_SCHEMA_VERSION: str = "0.2"
WINDOW_SOURCES_FILENAME: str = "obs_window_sources.pkl"

# Observer geometry source (for multi-observer scene assembly)
OBSERVER_TRACKS_DIR_REL: str = "PIXEL_SatPickles"
OBSERVER_TRACKS_FILENAME: str = "observer_tracks_with_pixels.pkl"

# Optional: per-observer frame catalog (used to recover sensor_name if later stages dropped it)
PHOTON_FRAMES_DIR_REL: str = "PHOTON_FRAMES"
OBSERVER_FRAMES_CATALOG_FILENAME: str = "obs_photon_frame_catalog.pkl"

TARGET_FRAMES_DIR_REL: str = "TARGET_PHOTON_FRAMES"
TARGETS_WITH_SKY_FILENAME: str = "obs_target_frames_ranked_with_sky.pkl"

SCENE_DIR_REL: str = "SCENE"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _dump_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), datetime))


def _prune_meta(
    obj: Any,
    *,
    max_list_len: int = 64,
    max_str_len: int = 10_000,
    mode: str = "annotate",
    report: Optional[Dict[str, Any]] = None,
    path: str = "",
) -> Any:
    """
    Recursively prune an object to keep metadata lightweight *without silent loss*.

    Modes
    -----
    - mode="drop":
        Preserves the prior behavior:
          * Drops numpy arrays entirely (returns None)
          * Drops lists/tuples longer than max_list_len (returns None)
          * Truncates long strings to max_str_len
        In dict/list contexts, values that prune to None are omitted.

    - mode="annotate" (default):
        Instead of silently dropping/truncating, replaces pruned content with a small,
        structured sentinel that describes what was removed and why:
          * ndarray -> {"__pruned__": True, "kind": "ndarray", "shape": ..., "dtype": ..., "size": ...}
          * long list/tuple -> {"__pruned__": True, "kind": "list"/"tuple", "orig_len": ..., "max_list_len": ..., "head": [...]}
          * long string -> {"__truncated__": True, "kind": "string", "orig_len": ..., "max_str_len": ..., "sha256": ..., "prefix": ..., "suffix": ...}
        This keeps metadata bounded while making pruning fully transparent.

    Notes
    -----
    This function is intended ONLY for provenance/lightweight metadata. It must not be
    applied to the core science payloads (targets/stars time-series arrays).
    """

    def _record(kind: str, detail: Dict[str, Any]) -> None:
        if report is None:
            return
        bucket = report.setdefault(kind, {"count": 0, "examples": []})
        bucket["count"] = int(bucket.get("count", 0)) + 1
        examples = bucket.setdefault("examples", [])
        # cap examples to keep report lightweight
        if isinstance(examples, list) and len(examples) < 25:
            examples.append({"path": path, **detail})

    # 1) numpy arrays: never embed in metadata; either drop or annotate
    if isinstance(obj, np.ndarray):
        _record("ndarray", {"shape": list(obj.shape), "dtype": str(obj.dtype), "size": int(obj.size)})
        if mode == "drop":
            return None
        return {
            "__pruned__": True,
            "kind": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "size": int(obj.size),
        }

    # 2) scalars: keep; but annotate long strings instead of silently truncating
    if _is_scalar(obj):
        if isinstance(obj, str) and len(obj) > max_str_len:
            h = hashlib.sha256(obj.encode("utf-8", errors="replace")).hexdigest()
            _record("long_string", {"orig_len": len(obj), "max_str_len": int(max_str_len), "sha256": h})
            if mode == "drop":
                return obj[:max_str_len]
            # bounded previews (still lightweight) + hash for full integrity tracking
            prefix_len = min(2000, len(obj))
            suffix_len = min(2000, len(obj))
            return {
                "__truncated__": True,
                "kind": "string",
                "orig_len": len(obj),
                "max_str_len": int(max_str_len),
                "sha256": h,
                "prefix": obj[:prefix_len],
                "suffix": obj[-suffix_len:] if suffix_len > 0 else "",
            }
        return obj

    # 3) sequences: annotate long lists/tuples instead of dropping silently
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        if n > max_list_len:
            _record("long_list", {"orig_len": n, "max_list_len": int(max_list_len), "type": type(obj).__name__})
            if mode == "drop":
                return None
            head = []
            for i, x in enumerate(obj[:max_list_len]):
                px = _prune_meta(
                    x,
                    max_list_len=max_list_len,
                    max_str_len=max_str_len,
                    mode=mode,
                    report=report,
                    path=f"{path}[{i}]",
                )
                head.append(px)
            return {
                "__pruned__": True,
                "kind": type(obj).__name__,
                "orig_len": n,
                "max_list_len": int(max_list_len),
                "head": head,
            }

        pruned = []
        for i, x in enumerate(obj):
            px = _prune_meta(
                x,
                max_list_len=max_list_len,
                max_str_len=max_str_len,
                mode=mode,
                report=report,
                path=f"{path}[{i}]",
            )
            if px is None and mode == "drop":
                continue
            pruned.append(px)
        return type(obj)(pruned)

    # 4) dicts: recurse and keep keys; in drop mode omit None; in annotate mode keep sentinels
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            subpath = f"{path}.{key}" if path else key
            pv = _prune_meta(
                v,
                max_list_len=max_list_len,
                max_str_len=max_str_len,
                mode=mode,
                report=report,
                path=subpath,
            )
            if pv is None and mode == "drop":
                continue
            out[key] = pv
        return out

    # 5) fallback: repr(), annotated if it must be truncated
    s = repr(obj)
    if len(s) > max_str_len:
        h = hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()
        _record("repr_truncated", {"orig_len": len(s), "max_str_len": int(max_str_len), "sha256": h, "type": type(obj).__name__})
        if mode == "drop":
            return s[:max_str_len]
        prefix_len = min(2000, len(s))
        suffix_len = min(2000, len(s))
        return {
            "__truncated__": True,
            "kind": "repr",
            "type": type(obj).__name__,
            "orig_len": len(s),
            "max_str_len": int(max_str_len),
            "sha256": h,
            "prefix": s[:prefix_len],
            "suffix": s[-suffix_len:] if suffix_len > 0 else "",
        }
    return s


def _gaia_window_key(window: Mapping[str, Any]) -> Tuple[int, int, int]:
    """
    Match windows by sky center (RA/Dec) and radius, rounded to micro-degrees.

    This avoids relying on list ordering (Gaia cone windows do not have window_index).
    """
    ra = float(window.get("sky_center_ra_deg", float("nan")))
    dec = float(window.get("sky_center_dec_deg", float("nan")))
    rad = float(window.get("sky_radius_deg", window.get("query_radius_deg", float("nan"))))
    return (int(round(ra * 1e6)), int(round(dec * 1e6)), int(round(rad * 1e6)))


def _build_gaia_window_lookup(gaia_obs: Mapping[str, Any]) -> Dict[Tuple[int, int, int], Mapping[str, Any]]:
    lookup: Dict[Tuple[int, int, int], Mapping[str, Any]] = {}
    for w in gaia_obs.get("windows", []) or []:
        lookup[_gaia_window_key(w)] = w
    return lookup


def _prune_gaia_obs_meta(gaia_obs: Mapping[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Keep only small, provenance-like fields at the observer level.
    keep = {
        "catalog_name",
        "catalog_table",
        "release",
        "band",
        "mode",
        "mag_limit_sensor_G",
        "query_config",
        "run_meta",
    }
    out: Dict[str, Any] = {}
    for k in keep:
        if k in gaia_obs:
            out[k] = _prune_meta(gaia_obs[k], mode="annotate", report=report, path=f"gaia_meta.{k}")
    return out


def _prune_gaia_window_meta(
    gaia_window: Mapping[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
    window_index: Optional[int] = None,
) -> Dict[str, Any]:
    # Keep scalar-ish fields and small dicts; drop/annotate all Gaia per-row arrays.
    keep = {
        "status",
        "error_message",
        "n_rows",
        "mag_limit_G",
        "query_radius_deg",
        "sky_center_ra_deg",
        "sky_center_dec_deg",
        "sky_radius_deg",
        "query_meta",
    }
    out: Dict[str, Any] = {}
    for k in keep:
        if k in gaia_window:
            out[k] = _prune_meta(
                gaia_window[k],
                mode="annotate",
                report=report,
                path=f"gaia_window_meta[{window_index}].{k}",
            )
    return out


def _filter_on_sensor_stars(stars: Mapping[str, Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    """Keep only stars with any on_detector==True."""
    out: Dict[str, Mapping[str, Any]] = {}
    for sid, sdict in stars.items():
        on = sdict.get("on_detector", None)
        if isinstance(on, np.ndarray):
            if not bool(np.any(on)):
                continue
        elif on is False:
            continue
        out[str(sid)] = sdict
    return out


def _fail_or_warn(*, strict: bool, warnings: list[str], msg: str, logger: Optional[Any] = None) -> None:
    """Record a validation issue, optionally raising if strict."""
    warnings.append(msg)
    if logger is not None:
        logger.warning(msg)
    if strict:
        raise ValueError(msg)


def _contains_ndarray(obj: Any) -> bool:
    """Return True if any numpy.ndarray is found recursively inside obj."""
    if isinstance(obj, np.ndarray):
        return True
    if isinstance(obj, dict):
        return any(_contains_ndarray(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_ndarray(v) for v in obj)
    return False


def _validate_targets_window(
    *,
    win: Mapping[str, Any],
    warnings: list[str],
    strict: bool,
    logger: Optional[Any],
) -> None:
    targets = win.get("targets", None)
    if not isinstance(targets, dict):
        _fail_or_warn(strict=strict, warnings=warnings, msg="WINDOW_SOURCES validation: window['targets'] is missing or not a dict.", logger=logger)
        return

    n_targets = win.get("n_targets", None)
    if isinstance(n_targets, int) and n_targets != len(targets):
        _fail_or_warn(
            strict=strict,
            warnings=warnings,
            msg=f"WINDOW_SOURCES validation: n_targets={n_targets} but len(targets)={len(targets)} for window_index={win.get('window_index')}.",
            logger=logger,
        )

    start_index = win.get("start_index", None)
    end_index = win.get("end_index", None)

    for tid, tdict in targets.items():
        if not isinstance(tdict, dict):
            _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: target '{tid}' is not a dict.", logger=logger)
            continue

        # Basic identity checks
        st = tdict.get("source_type", None)
        if st is not None and st != "target":
            _fail_or_warn(
                strict=False,
                warnings=warnings,
                msg=f"WINDOW_SOURCES validation: target '{tid}' has source_type='{st}' (expected 'target').",
                logger=logger,
            )

        ci = tdict.get("coarse_indices", None)
        if ci is None or not isinstance(ci, np.ndarray):
            _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: target '{tid}' missing coarse_indices ndarray.", logger=logger)
            continue
        if ci.ndim != 1:
            _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: target '{tid}' coarse_indices not 1D.", logger=logger)
            continue
        if ci.size > 1 and not bool(np.all(ci[1:] >= ci[:-1])):
            _fail_or_warn(
                strict=False,
                warnings=warnings,
                msg=f"WINDOW_SOURCES validation: target '{tid}' coarse_indices are not nondecreasing (window_index={win.get('window_index')}).",
                logger=logger,
            )
        if isinstance(start_index, int) and isinstance(end_index, int) and ci.size > 0:
            if int(ci.min()) < int(start_index) or int(ci.max()) > int(end_index):
                _fail_or_warn(
                    strict=False,
                    warnings=warnings,
                    msg=f"WINDOW_SOURCES validation: target '{tid}' coarse_indices out of [{start_index}, {end_index}] (min={int(ci.min())}, max={int(ci.max())}).",
                    logger=logger,
                )

        # Shape consistency across target arrays (only for 1D ndarrays)
        lengths = {}
        for k, v in tdict.items():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                lengths[k] = int(v.shape[0])
        if lengths:
            expected = lengths.get("coarse_indices", None)
            if expected is None:
                expected = next(iter(lengths.values()))
            mismatched = {k: n for k, n in lengths.items() if n != expected}
            if mismatched:
                _fail_or_warn(
                    strict=strict,
                    warnings=warnings,
                    msg=f"WINDOW_SOURCES validation: target '{tid}' 1D array length mismatch; expected {expected}, got {mismatched}.",
                    logger=logger,
                )


def _validate_stars_window(
    *,
    win: Mapping[str, Any],
    rows: Optional[int],
    cols: Optional[int],
    dt_frame_s: Optional[float],
    warnings: list[str],
    strict: bool,
    logger: Optional[Any],
) -> None:
    stars = win.get("stars", None)
    if stars is None:
        _fail_or_warn(strict=strict, warnings=warnings, msg="WINDOW_SOURCES validation: window['stars'] is missing.", logger=logger)
        return
    if not isinstance(stars, dict):
        _fail_or_warn(strict=strict, warnings=warnings, msg="WINDOW_SOURCES validation: window['stars'] is not a dict.", logger=logger)
        return

    n_stars = win.get("n_stars", None)
    if isinstance(n_stars, int) and n_stars != len(stars):
        _fail_or_warn(
            strict=False,
            warnings=warnings,
            msg=f"WINDOW_SOURCES validation: n_stars={n_stars} but len(stars)={len(stars)} for window_index={win.get('window_index')}.",
            logger=logger,
        )

    n_frames = win.get("n_frames", None)
    if not isinstance(n_frames, int) or n_frames <= 0:
        _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: invalid n_frames={n_frames} for window_index={win.get('window_index')}.", logger=logger)
        return

    # Check cadence/time bounds using one representative star (avoids O(n_stars) datetime work).
    rep = None
    for _, sdict in stars.items():
        rep = sdict
        break
    if isinstance(rep, dict):
        t_utc = rep.get("t_utc", None)
        if isinstance(t_utc, np.ndarray) and t_utc.size >= 2:
            try:
                dt = (t_utc[1] - t_utc[0]).total_seconds()
                if dt_frame_s is not None and abs(float(dt) - float(dt_frame_s)) > 1e-6:
                    _fail_or_warn(
                        strict=False,
                        warnings=warnings,
                        msg=f"WINDOW_SOURCES validation: star t_utc cadence {dt} != dt_frame_s {dt_frame_s} (window_index={win.get('window_index')}).",
                        logger=logger,
                    )
            except Exception:
                _fail_or_warn(
                    strict=False,
                    warnings=warnings,
                    msg=f"WINDOW_SOURCES validation: could not compute t_utc cadence for window_index={win.get('window_index')}.",
                    logger=logger,
                )

        # Window time bounds match (if present)
        wt0 = win.get("start_time", None)
        wt1 = win.get("end_time", None)
        if wt0 is not None and wt1 is not None and isinstance(t_utc, np.ndarray) and t_utc.size > 0:
            try:
                if t_utc[0] != wt0 or t_utc[-1] != wt1:
                    _fail_or_warn(
                        strict=False,
                        warnings=warnings,
                        msg=f"WINDOW_SOURCES validation: window start/end_time do not match star t_utc endpoints (window_index={win.get('window_index')}).",
                        logger=logger,
                    )
            except Exception:
                pass

    required_arrays = ("t_utc", "t_exp_s", "x_pix", "y_pix", "phi_ph_m2_s", "flux_ph_m2_frame", "mag_G", "on_detector")

    for sid, sdict in stars.items():
        if not isinstance(sdict, dict):
            _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: star '{sid}' is not a dict.", logger=logger)
            continue

        st = sdict.get("source_type", None)
        if st is not None and st != "star":
            _fail_or_warn(
                strict=False,
                warnings=warnings,
                msg=f"WINDOW_SOURCES validation: star '{sid}' has source_type='{st}' (expected 'star').",
                logger=logger,
            )

        gaia_id = sdict.get("gaia_source_id", None)
        if isinstance(gaia_id, np.ndarray):
            # should be scalar, not array
            _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: star '{sid}' gaia_source_id is ndarray (expected scalar).", logger=logger)
        elif isinstance(gaia_id, np.integer):
            if gaia_id.dtype.itemsize < 8:
                _fail_or_warn(
                    strict=strict,
                    warnings=warnings,
                    msg=f"WINDOW_SOURCES validation: star '{sid}' gaia_source_id dtype is {gaia_id.dtype} (expected int64-compatible).",
                    logger=logger,
                )
        elif isinstance(gaia_id, int):
            # Python int is fine (arbitrary precision)
            pass
        elif gaia_id is not None:
            _fail_or_warn(strict=False, warnings=warnings, msg=f"WINDOW_SOURCES validation: star '{sid}' gaia_source_id is type {type(gaia_id)}.", logger=logger)

        # Key consistency (warning only)
        if gaia_id is not None:
            if str(gaia_id) != str(sid) and str(sdict.get("source_id", "")) != str(sid):
                _fail_or_warn(
                    strict=False,
                    warnings=warnings,
                    msg=f"WINDOW_SOURCES validation: star dict key '{sid}' does not match gaia_source_id/source_id.",
                    logger=logger,
                )

        # Required array shapes
        for k in required_arrays:
            v = sdict.get(k, None)
            if not isinstance(v, np.ndarray):
                _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: star '{sid}' missing ndarray '{k}'.", logger=logger)
                continue
            if v.ndim != 1:
                _fail_or_warn(strict=strict, warnings=warnings, msg=f"WINDOW_SOURCES validation: star '{sid}' array '{k}' is not 1D.", logger=logger)
                continue
            if int(v.shape[0]) != int(n_frames):
                _fail_or_warn(
                    strict=strict,
                    warnings=warnings,
                    msg=f"WINDOW_SOURCES validation: star '{sid}' array '{k}' length {int(v.shape[0])} != n_frames {n_frames}.",
                    logger=logger,
                )

        on = sdict.get("on_detector", None)
        if isinstance(on, np.ndarray) and on.size > 0 and not bool(np.any(on)):
            _fail_or_warn(
                strict=strict,
                warnings=warnings,
                msg=f"WINDOW_SOURCES validation: star '{sid}' has on_detector all-False but is included in stars dict.",
                logger=logger,
            )

        # Optional geometry checks (warning only; only when on_detector is True)
        if isinstance(on, np.ndarray) and on.size == n_frames and rows is not None and cols is not None:
            try:
                x = sdict.get("x_pix", None)
                y = sdict.get("y_pix", None)
                if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                    mask = on.astype(bool)
                    if mask.any():
                        if not np.isfinite(x[mask]).all() or not np.isfinite(y[mask]).all():
                            _fail_or_warn(
                                strict=False,
                                warnings=warnings,
                                msg=f"WINDOW_SOURCES validation: star '{sid}' has non-finite x/y while on_detector=True.",
                                logger=logger,
                            )
                        if (x[mask] < 0).any() or (x[mask] >= float(cols)).any() or (y[mask] < 0).any() or (y[mask] >= float(rows)).any():
                            _fail_or_warn(
                                strict=False,
                                warnings=warnings,
                                msg=f"WINDOW_SOURCES validation: star '{sid}' has x/y out of bounds while on_detector=True (rows={rows}, cols={cols}).",
                                logger=logger,
                            )
            except Exception:
                pass


def _validate_window_sources(
    *,
    obs_name: str,
    obs_out: Mapping[str, Any],
    logger: Optional[Any],
    strict: bool,
) -> None:
    """Validate one observer block for structural consistency (lightweight)."""
    windows = obs_out.get("windows", None)
    if not isinstance(windows, list):
        raise ValueError(f"WINDOW_SOURCES validation: observer '{obs_name}' windows missing or not a list.")

    # Ensure gaia_meta is lightweight (no arrays)
    if "gaia_meta" in obs_out and _contains_ndarray(obs_out.get("gaia_meta")):
        raise ValueError(f"WINDOW_SOURCES validation: observer '{obs_name}' gaia_meta contains numpy arrays (should be pruned).")

    rows = obs_out.get("rows", None)
    cols = obs_out.get("cols", None)
    dt_frame_s = obs_out.get("dt_frame_s", None)

    total_warnings = 0
    for w in windows:
        if not isinstance(w, dict):
            raise ValueError(f"WINDOW_SOURCES validation: observer '{obs_name}' contains non-dict window.")
        if "frames" in w:
            raise ValueError(f"WINDOW_SOURCES validation: observer '{obs_name}' window_index={w.get('window_index')} contains forbidden key 'frames'.")

        win_warn: list[str] = []
        n_frames = w.get("n_frames", None)
        if not isinstance(n_frames, int) or n_frames <= 0:
            _fail_or_warn(
                strict=True,
                warnings=win_warn,
                msg=f"WINDOW_SOURCES validation: invalid n_frames={n_frames} for observer '{obs_name}' window_index={w.get('window_index')}.",
                logger=logger,
            )

        _validate_targets_window(win=w, warnings=win_warn, strict=strict, logger=logger)
        _validate_stars_window(win=w, rows=rows, cols=cols, dt_frame_s=dt_frame_s, warnings=win_warn, strict=strict, logger=logger)

        # Attach lightweight integrity output
        integ = w.get("integrity", None)
        if not isinstance(integ, dict):
            integ = {}
            w["integrity"] = integ
        integ["warnings"] = win_warn
        total_warnings += len(win_warn)

    # Observer-level validation summary
    rm = obs_out.get("run_meta", None)
    if isinstance(rm, dict):
        rm["validation"] = {"enabled": True, "strict": bool(strict), "warnings_total": int(total_warnings)}



def _safe_get_sensor_name(
    *,
    obs_name: str,
    targets_with_sky_obs: Mapping[str, Any],
    observer_tracks_obs: Optional[Mapping[str, Any]],
    observer_frames_catalog_obs: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Best-effort recovery of observer sensor_name from any available upstream stage."""

    # Preferred: explicit sensor_name attached to the targets-with-sky payload (if present)
    sensor_name = targets_with_sky_obs.get("sensor_name")
    if isinstance(sensor_name, str) and sensor_name.strip():
        return sensor_name.strip()

    # Next: photon frame catalog (earlier stage retains sensor_name reliably)
    if observer_frames_catalog_obs:
        sensor_name = observer_frames_catalog_obs.get("sensor_name")
        if isinstance(sensor_name, str) and sensor_name.strip():
            return sensor_name.strip()

    # Next: observer tracks (may include sensor_name depending on the build)
    if observer_tracks_obs:
        sensor_name = observer_tracks_obs.get("sensor_name")
        if isinstance(sensor_name, str) and sensor_name.strip():
            return sensor_name.strip()

    return None


def _extract_observer_geometry(
    *,
    obs_tracks: Optional[Mapping[str, Any]],
    sensor_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Extract a compact but information-complete observer geometry package.

    Intended to make obs_window_sources.pkl self-contained for multi-observer use without
    re-loading the PIXELS observer-track pickle.
    """

    if not obs_tracks:
        return None

    # Scalars / orbit elements
    orbit_keys = [
        "a_km",
        "e",
        "inc_rad",
        "raan_rad",
        "argp_rad",
        "mean_motion_rad_per_min",
        "mean_anomaly_epoch_rad",
        "p_km",
    ]
    orbit_elements: Dict[str, Any] = {k: obs_tracks[k] for k in orbit_keys if k in obs_tracks}

    # State vectors / geometry time series
    state_keys = [
        "r_eci_km",
        "v_eci_km_s",
        "icrs_x_km",
        "icrs_y_km",
        "icrs_z_km",
        "nu_rad",
    ]
    state_vectors: Dict[str, Any] = {k: obs_tracks[k] for k in state_keys if k in obs_tracks}

    # Pointing / attitude time series and flags
    pointing: Dict[str, Any] = {}
    for k, v in obs_tracks.items():
        if k.startswith("pointing_"):
            pointing[k] = v

    # -------------------------------------------------------------------------
    # Roll handling / standardization
    # -------------------------------------------------------------------------
    # Per the current pointing scheduler contract, roll MUST be provided as a
    # time series on the observer track under:
    #   - "pointing_boresight_roll_deg" : np.ndarray shape (N,)
    #
    # This stage enforces that contract and will NOT emit any scalar roll values
    # or legacy roll field names.
    #
    # N is inferred from pointing_boresight_ra_deg (preferred), then
    # pointing_boresight_dec_deg, then times.

    def _infer_n_steps() -> Optional[int]:
        for _k in ("pointing_boresight_ra_deg", "pointing_boresight_dec_deg", "times"):
            if _k in obs_tracks and obs_tracks[_k] is not None:
                try:
                    return int(len(obs_tracks[_k]))
                except Exception:
                    pass
        return None

    n_steps = _infer_n_steps()

    # Remove legacy / ambiguous roll keys if present.
    for _legacy in ("roll_deg", "pointing_roll_deg", "boresight_roll_deg", "pointing_roll"):
        pointing.pop(_legacy, None)

    # Validate RA/Dec lengths if present.
    if n_steps is not None:
        for _k in ("pointing_boresight_ra_deg", "pointing_boresight_dec_deg"):
            if _k in pointing and pointing[_k] is not None:
                _arr = np.asarray(pointing[_k]).reshape(-1)
                if _arr.size != n_steps:
                    raise ValueError(
                        f"{_k} length {_arr.size} does not match expected n_steps={n_steps}"
                    )

    # Require roll time series whenever boresight RA/Dec are present.
    roll_val = pointing.get(
        "pointing_boresight_roll_deg",
        obs_tracks.get("pointing_boresight_roll_deg"),
    )
    if roll_val is None:
        if ("pointing_boresight_ra_deg" in pointing) or ("pointing_boresight_dec_deg" in pointing):
            raise KeyError(
                "Expected observer track to contain 'pointing_boresight_roll_deg' as a time series "
                "matching the boresight RA/Dec arrays."
            )
    else:
        roll_series = np.asarray(roll_val, dtype=float)
        if roll_series.ndim == 0:
            raise TypeError(
                "pointing_boresight_roll_deg must be a time series array (shape (N,)); "
                "a scalar roll value is not permitted."
            )
        roll_series = roll_series.reshape(-1)
        if n_steps is not None and roll_series.size != n_steps:
            raise ValueError(
                f"pointing_boresight_roll_deg length {roll_series.size} does not match expected n_steps={n_steps}"
            )
        pointing["pointing_boresight_roll_deg"] = roll_series

    # Optional: full timebase from observer tracks (useful for indexing and debugging)
    if "times" in obs_tracks:
        # May be a list[datetime] or a numpy/object array.
        state_vectors["times"] = obs_tracks["times"]

    return {
        "sensor_name": sensor_name,
        "orbit_elements": orbit_elements,
        "state_vectors": state_vectors,
        "pointing": pointing,
    }

def main(*, logger: Optional[Any] = None, force_recompute: bool = False, validate: bool = True, strict: bool = False) -> Path:
    """
    Entrypoint for the WINDOW_SOURCES stage.

    Parameters
    ----------
    logger : logging.Logger | None
        Pipeline logger (optional).
    force_recompute : bool
        If True, overwrite output even if it exists.
    validate : bool
        If True, run structural and consistency validation on the combined output.
    strict : bool
        If True, validation failures raise exceptions; otherwise they are logged as warnings.

    Returns
    -------
    Path
        Path to the output pickle.
    """

    # Resolve catalog path (align with manifest and existing STAR_* stages)
    star_catalog_name = getattr(NSC.NEBULA_STAR_CATALOG, "name", "").strip()
    if not star_catalog_name:
        raise RuntimeError(
            "WINDOW_SOURCES: NEBULA_STAR_CATALOG.name is missing/empty in NEBULA_STAR_CONFIG."
        )

    star_dir_rel = f"STARS/{star_catalog_name}"

    targets_with_sky_path = NEBULA_OUTPUT_DIR / TARGET_FRAMES_DIR_REL / TARGETS_WITH_SKY_FILENAME
    star_photons_path = NEBULA_OUTPUT_DIR / star_dir_rel / "obs_star_photons.pkl"
    gaia_cones_path = NEBULA_OUTPUT_DIR / star_dir_rel / "obs_gaia_cones.pkl"  # optional
    out_path = NEBULA_OUTPUT_DIR / SCENE_DIR_REL / WINDOW_SOURCES_FILENAME

    observer_tracks_path = NEBULA_OUTPUT_DIR / OBSERVER_TRACKS_DIR_REL / OBSERVER_TRACKS_FILENAME  # optional but recommended
    observer_frames_catalog_path = NEBULA_OUTPUT_DIR / PHOTON_FRAMES_DIR_REL / OBSERVER_FRAMES_CATALOG_FILENAME  # optional

    if out_path.exists() and not force_recompute:
        if logger:
            logger.info(f"WINDOW_SOURCES: output exists; skipping: {out_path}")
        return out_path

    if logger:
        logger.info("WINDOW_SOURCES: loading inputs")
        logger.info(f"  targets_with_sky: {targets_with_sky_path}")
        logger.info(f"  star_photons:     {star_photons_path}")
        logger.info(f"  gaia_cones:       {gaia_cones_path} (optional)")

    targets_with_sky = _load_pickle(targets_with_sky_path)
    star_photons = _load_pickle(star_photons_path)

    gaia_cones: Optional[Mapping[str, Any]] = None
    if gaia_cones_path.exists():
        try:
            gaia_cones = _load_pickle(gaia_cones_path)
        except Exception as e:
            if logger:
                logger.warning(f"WINDOW_SOURCES: failed to load gaia_cones; continuing without it: {e}")
            gaia_cones = None

    observer_tracks: Optional[Mapping[str, Any]] = None
    if observer_tracks_path.exists():
        try:
            observer_tracks = _load_pickle(observer_tracks_path)
        except Exception as e:
            if logger:
                logger.warning(
                    f"WINDOW_SOURCES: failed to load observer_tracks; continuing without it: {e}"
                )
            observer_tracks = None

    observer_frames_catalog: Optional[Mapping[str, Any]] = None
    if observer_frames_catalog_path.exists():
        try:
            observer_frames_catalog = _load_pickle(observer_frames_catalog_path)
        except Exception as e:
            if logger:
                logger.warning(
                    f"WINDOW_SOURCES: failed to load observer_frames_catalog; continuing without it: {e}"
                )
            observer_frames_catalog = None

    # Build combined structure
    combined: Dict[str, Any] = {}

    # iterate observers from targets_with_sky (canonical windows)
    for obs_name, t_obs in (targets_with_sky or {}).items():
        s_obs = (star_photons or {}).get(obs_name, None)
        g_obs = (gaia_cones or {}).get(obs_name, None) if gaia_cones else None

        # Track any metadata pruning so nothing is silently lost.
        prune_report: Dict[str, Any] = {}

        # Optional: attach observer geometry (orbit elements, state vectors, pointing) for multi-observer scenes
        obs_track_key = t_obs.get("observer_name", obs_name)
        o_tracks_obs = observer_tracks.get(obs_track_key) if observer_tracks else None
        frames_catalog_obs = (
            observer_frames_catalog.get(obs_track_key) if observer_frames_catalog else None
        )

        sensor_name = _safe_get_sensor_name(
            obs_name=obs_name,
            targets_with_sky_obs=t_obs,
            observer_tracks_obs=o_tracks_obs,
            observer_frames_catalog_obs=frames_catalog_obs,
        )

        observer_geometry = _extract_observer_geometry(
            obs_tracks=o_tracks_obs,
            sensor_name=sensor_name,
        )

        # Observer-level meta: keep it small, but include enough to validate consistency.
        obs_out: Dict[str, Any] = {
            "observer_name": t_obs.get("observer_name", obs_name),
            "sensor_name": sensor_name,
            "observer_geometry": observer_geometry,
            "rows": t_obs.get("rows"),
            "cols": t_obs.get("cols"),
            "dt_frame_s": t_obs.get("dt_frame_s"),
            "run_meta": {
                "schema_version": WINDOW_SOURCES_SCHEMA_VERSION,
                "builder": "Utility.SCENE.NEBULA_WINDOW_SOURCES.main",
                "created_utc": _utc_now_iso(),
                "inputs_rel": {
                    "targets_with_sky": f"{TARGET_FRAMES_DIR_REL}/{TARGETS_WITH_SKY_FILENAME}",
                    "star_photons": f"{star_dir_rel}/obs_star_photons.pkl",
                    "gaia_cones": f"{star_dir_rel}/obs_gaia_cones.pkl" if gaia_cones_path.exists() else None,
                    "observer_tracks": f"{OBSERVER_TRACKS_DIR_REL}/{OBSERVER_TRACKS_FILENAME}" if observer_tracks_path.exists() else None,
                    "observer_frames_catalog": f"{PHOTON_FRAMES_DIR_REL}/{OBSERVER_FRAMES_CATALOG_FILENAME}" if observer_frames_catalog_path.exists() else None,
                },
                "notes": [],
                # Annotated pruning report (counts + a small set of example paths)
                "prune_report": prune_report,
            },
            "windows": [],
        }

        # Record validation settings in provenance notes (lightweight)
        if isinstance(obs_out.get("run_meta"), dict):
            obs_out["run_meta"].setdefault("notes", []).append(
                f"validation_enabled={validate}; strict={strict}"
            )
            obs_out["run_meta"].setdefault("notes", []).append(
                "meta_prune_mode=annotate; max_list_len=64; max_str_len=10000"
            )

        if s_obs:
            obs_out["catalog_name"] = s_obs.get("catalog_name")
            obs_out["catalog_band"] = s_obs.get("catalog_band")
            # Optional: surface the star-photons run meta (annotated if anything would be pruned/truncated)
            obs_out["star_photons_meta"] = _prune_meta(
                s_obs.get("run_meta", {}),
                mode="annotate",
                report=prune_report,
                path="star_photons_meta",
            )

        if g_obs:
            obs_out["gaia_meta"] = _prune_gaia_obs_meta(g_obs, report=prune_report)

        # Star windows lookup by window_index (present in star_photons windows)
        star_win_by_index: Dict[Any, Mapping[str, Any]] = {}
        if s_obs:
            for sw in s_obs.get("windows", []) or []:
                star_win_by_index[sw.get("window_index")] = sw

        # Gaia windows lookup by (ra, dec, radius) key
        gaia_win_lookup = _build_gaia_window_lookup(g_obs) if g_obs else {}

        for t_win in t_obs.get("windows", []) or []:
            # Shallow copy: preserves numpy arrays without duplicating them in memory.
            win_out: Dict[str, Any] = dict(t_win)

            wi = t_win.get("window_index", None)

            # Attach stars (filtering to on-sensor only)
            stars_out: Dict[str, Mapping[str, Any]] = {}
            if s_obs:
                s_win = star_win_by_index.get(wi)
                if s_win is None:
                    # fallback: match by time/index bounds if window_index is absent or mismatch
                    for sw in s_obs.get("windows", []) or []:
                        if (
                            sw.get("start_index") == t_win.get("start_index")
                            and sw.get("end_index") == t_win.get("end_index")
                        ):
                            s_win = sw
                            break
                if s_win:
                    stars_out = _filter_on_sensor_stars(s_win.get("stars", {}) or {})

            win_out["stars"] = stars_out
            win_out["n_stars"] = int(len(stars_out))

            # Optional Gaia window metadata (lightweight, annotated)
            if g_obs:
                g_key = _gaia_window_key(t_win)
                g_win = gaia_win_lookup.get(g_key)
                if g_win is None:
                    # fallback to index alignment only when lengths match
                    t_idx = int(wi) if isinstance(wi, int) else None
                    if t_idx is not None:
                        g_windows = g_obs.get("windows", []) or []
                        if 0 <= t_idx < len(g_windows):
                            g_win = g_windows[t_idx]
                if g_win is not None:
                    win_out["gaia_window_meta"] = _prune_gaia_window_meta(
                        g_win,
                        report=prune_report,
                        window_index=int(wi) if isinstance(wi, int) else None,
                    )

            obs_out["windows"].append(win_out)

        combined[obs_name] = obs_out

    if logger:
        logger.info(f"WINDOW_SOURCES: writing output: {out_path}")

    if validate:
        if logger:
            logger.info("WINDOW_SOURCES: validating combined output")
        for obs_name, obs_out in combined.items():
            _validate_window_sources(obs_name=obs_name, obs_out=obs_out, logger=logger, strict=strict)

    _dump_pickle(combined, out_path)
    return out_path
