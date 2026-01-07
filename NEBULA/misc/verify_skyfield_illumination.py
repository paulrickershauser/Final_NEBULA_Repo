"""
verify_skyfield_illumination.py

Verification script to compare NEBULA's Skyfield-based illumination
pipeline against Skyfield's own EarthSatellite.is_sunlit() for:

  Observer: SBSS (USA 216)
  Target:   TDRS 3

Time grid is built from NEBULA_TIME_CONFIG.DEFAULT_TIME_WINDOW and
DEFAULT_PROPAGATION_STEPS, and the DE ephemeris is loaded from:

  NEBULA_INPUT_DIR / "NEBULA_EPHEMERIS" / "de440s.bsp"

Run this from the NEBULA project root, e.g. in Spyder:

  %runfile "C:/Users/prick/Desktop/Research/NEBULA/verify_skyfield_illumination.py"

The key output is a mismatch count between:

  A) TDRS3_sat.at(t).is_sunlit(eph)
  B) is_sunlit(ICRF.from_time_and_frame_vectors(t, TEME, r,v))

If A and B agree bit-for-bit, then the TEME→ICRF→is_sunlit usage
inside your NEBULA illumination pipeline is probably correct, and any
disagreement with your stored `illum_is_sunlit` would point to how
you’re building the state vectors (times, frame, etc.), not the call
to Skyfield.

There is also an optional helper:

  compare_nebula_track_to_earthsat(track, line1, line2, eph_path=None)

which you can call from an interactive session once you have a NEBULA
track dict (e.g., TDRS 3) with keys ['times', 'illum_is_sunlit'].
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Optional

from datetime import datetime, timedelta, timezone

import numpy as np

from skyfield.api import load, EarthSatellite  # Timescale, loader, TLE → satellite
from skyfield.sgp4lib import TEME             # TEME frame object
from skyfield.positionlib import ICRF         # Generic ICRF position class
from skyfield.units import Distance, Velocity # Distance/Velocity wrappers

# ----------------------------------------------------------------------
# Make sure NEBULA root is on sys.path so we can import Configuration.*
# (Assumes this file lives in the NEBULA root directory.)
# ----------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
NEBULA_ROOT = THIS_FILE.parent
if str(NEBULA_ROOT) not in sys.path:
    sys.path.insert(0, str(NEBULA_ROOT))

from Configuration.NEBULA_TIME_CONFIG import (
    DEFAULT_TIME_WINDOW,
    DEFAULT_PROPAGATION_STEPS,
)
from Configuration.NEBULA_PATH_CONFIG import NEBULA_INPUT_DIR


# ----------------------------------------------------------------------
# Hard-coded TLEs for SBSS (observer) and TDRS 3 (target)
# ----------------------------------------------------------------------
SBSS_NAME = "SBSS (USA 216)"
SBSS_LINE1 = (
    "1 37168U 10048A   25312.90026619  .00008279  00000+0  41891-3 0  9995"
)
SBSS_LINE2 = (
    "2 37168  97.7591 188.9739 0083334 241.7147 321.3794 15.14254838819106"
)

TDRS_NAME = "TDRS 3"
TDRS_LINE1 = (
    "1 19548U 88091B   25312.45027976 -.00000302  00000+0  00000+0 0  9990"
)
TDRS_LINE2 = (
    "2 19548  12.7717 342.4579 0040433 340.4110 198.1740  1.00266379123173"
)

# Ephemeris filename used in NEBULA
EPHEMERIS_FILENAME = "de440s.bsp"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def build_time_grid_from_config() -> list[datetime]:
    """
    Build a list of timezone-aware UTC datetimes using NEBULA_TIME_CONFIG.

    Uses:
      DEFAULT_TIME_WINDOW.start_utc / end_utc  (ISO strings)
      DEFAULT_PROPAGATION_STEPS.base_dt_s      (seconds between samples)

    Returns
    -------
    times : list of datetime (timezone=UTC)
        [t0, t0 + dt, ..., t_end] inclusive.
    """
    start_str = DEFAULT_TIME_WINDOW.start_utc
    end_str = DEFAULT_TIME_WINDOW.end_utc
    dt_s = float(DEFAULT_PROPAGATION_STEPS.base_dt_s)

    start_dt = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)

    total_seconds = (end_dt - start_dt).total_seconds()
    n_steps = int(round(total_seconds / dt_s)) + 1

    times = [start_dt + timedelta(seconds=i * dt_s) for i in range(n_steps)]
    return times


def build_earthsatellites(ts):
    """
    Construct Skyfield EarthSatellite objects for SBSS and TDRS.

    Parameters
    ----------
    ts : skyfield.api.Timescale

    Returns
    -------
    sbss_sat, tdrs_sat : EarthSatellite
    """
    sbss_sat = EarthSatellite(SBSS_LINE1, SBSS_LINE2, SBSS_NAME, ts)
    tdrs_sat = EarthSatellite(TDRS_LINE1, TDRS_LINE2, TDRS_NAME, ts)
    return sbss_sat, tdrs_sat


def compute_sunlit_via_frame_vectors(
    sat: EarthSatellite,
    t,
    eph,
) -> np.ndarray:
    """
    Your NEBULA-style pipeline in a minimal, controlled setting.

    Steps:
      1. Get satellite state in the TEME frame:
           pos_teme, vel_teme = sat.at(t).frame_xyz_and_velocity(TEME)
      2. Build an ICRF position from those TEME vectors:
           pos_icrf = ICRF.from_time_and_frame_vectors(t, TEME,
                                                      pos_teme, vel_teme)
      3. Ask Skyfield whether those ICRF positions are in sunlight:
           sunlit = pos_icrf.is_sunlit(eph)

    Parameters
    ----------
    sat : EarthSatellite
        Target satellite (e.g., TDRS 3).
    t : skyfield.api.Time
        Time array built by ts.from_datetimes(times).
    eph : skyfield.jpllib.SpiceKernel / JPL ephemeris
        Loaded DE ephemeris (e.g. de440s.bsp) containing at least
        Earth and Sun.

    Returns
    -------
    np.ndarray (bool)
        True where satellite is sunlit, False where eclipsed.
    """
    # TEME position+velocity for the satellite at each time step
    pos_teme, vel_teme = sat.at(t).frame_xyz_and_velocity(TEME)

    # Build ICRF positions from the TEME vectors
    pos_icrf = ICRF.from_time_and_frame_vectors(
        t,
        TEME,
        pos_teme,   # Distance in TEME
        vel_teme,   # Velocity in TEME
    )

    # Let Skyfield do the actual shadow geometry
    sunlit = pos_icrf.is_sunlit(eph)

    # Cast to plain numpy bool array for easy comparison
    return np.asarray(sunlit, dtype=bool)


def summarize_boolean_comparison(
    label_a: str,
    a: np.ndarray,
    label_b: str,
    b: np.ndarray,
    times: Optional[Sequence[datetime]] = None,
    max_print: int = 20,
) -> None:
    """
    Print a simple mismatch summary between two boolean arrays.

    Parameters
    ----------
    label_a, label_b : str
        Human-readable names for the two series being compared.
    a, b : np.ndarray (bool)
        Boolean arrays of equal length.
    times : sequence of datetime, optional
        If provided, first a few mismatches will print the corresponding
        UTC timestamps.
    max_print : int
        Maximum number of mismatch rows to print.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {label_a} has {a.shape}, {label_b} has {b.shape}"
        )

    mismatches = np.nonzero(a != b)[0]
    n_total = a.size
    n_mismatch = mismatches.size

    print(f"\n=== Boolean comparison: {label_a} vs {label_b} ===")
    print(f"Total timesteps   : {n_total}")
    print(f"Total mismatches  : {n_mismatch}")
    if n_total > 0:
        frac = n_mismatch / float(n_total)
        print(f"Global agreement  : {100.0 * (1.0 - frac):.3f}%")

    if n_mismatch == 0:
        print("All timesteps agree exactly.\n")
        return

    print(f"\nFirst {min(max_print, n_mismatch)} mismatches:")
    header = " idx |  A  |  B "
    if times is not None:
        header += "| UTC time"
    print(header)
    print("-" * len(header))

    for k, idx in enumerate(mismatches[:max_print]):
        row = f"{idx:4d} | {int(a[idx])!s:>3} | {int(b[idx])!s:>3}"
        if times is not None:
            row += f" | {times[idx].isoformat()}"
        print(row)
    print()


# ----------------------------------------------------------------------
# Optional helper: compare any NEBULA track's illum_is_sunlit to EarthSatellite
# ----------------------------------------------------------------------
def compare_nebula_track_to_earthsat(
    track: dict,
    line1: str,
    line2: str,
    eph_path: Optional[Path] = None,
) -> None:
    """
    Compare an existing NEBULA illumination array against EarthSatellite.

    Use this inside an interactive session AFTER you have run NEBULA
    and have a track dict in memory, e.g.:

        from verify_skyfield_illumination import compare_nebula_track_to_earthsat
        tdrs_track = tar_tracks["TDRS 3"]     # however you obtain it

        compare_nebula_track_to_earthsat(
            track=tdrs_track,
            line1=TDRS_LINE1,
            line2=TDRS_LINE2,
        )

    Requirements for `track`:
      - track["times"]           : list/array of datetime (UTC)
      - track["illum_is_sunlit"] : array-like of bool, same length

    Parameters
    ----------
    track : dict
        NEBULA track dictionary for a single satellite.
    line1, line2 : str
        TLE lines for that satellite.
    eph_path : Path, optional
        Path to the DE ephemeris. If None, uses NEBULA_INPUT_DIR /
        "NEBULA_EPHEMERIS" / "de440s.bsp".
    """
    times = list(track["times"])
    nebula_sunlit = np.asarray(track["illum_is_sunlit"], dtype=bool)

    if eph_path is None:
        eph_path = (
            Path(NEBULA_INPUT_DIR)
            / "NEBULA_EPHEMERIS"
            / EPHEMERIS_FILENAME
        )

    ts = load.timescale()
    t = ts.from_datetimes(times)
    eph = load(str(eph_path))

    sat = EarthSatellite(line1, line2, track.get("name", "Unknown"), ts)

    earthsat_sunlit = np.asarray(sat.at(t).is_sunlit(eph), dtype=bool)

    summarize_boolean_comparison(
        label_a="NEBULA illum_is_sunlit",
        a=nebula_sunlit,
        label_b="EarthSatellite.is_sunlit",
        b=earthsat_sunlit,
        times=times,
    )


# ----------------------------------------------------------------------
# Main verification: TDRS-3 EarthSatellite vs TEME→ICRF pipeline
# ----------------------------------------------------------------------
def main():
    # 1) Build time grid from NEBULA config
    times = build_time_grid_from_config()

    # 2) Load timescale + ephemeris
    ts = load.timescale()
    t = ts.from_datetimes(times)

    eph_path = (
        Path(NEBULA_INPUT_DIR)
        / "NEBULA_EPHEMERIS"
        / EPHEMERIS_FILENAME
    )
    eph = load(str(eph_path))

    # 3) Build EarthSatellite objects
    sbss_sat, tdrs_sat = build_earthsatellites(ts)

    # 4) "Ground truth": Skyfield's direct method
    tdrs_sunlit_direct = np.asarray(
        tdrs_sat.at(t).is_sunlit(eph), dtype=bool
    )

    # 5) NEBULA-style method: TEME vectors → ICRF → is_sunlit
    tdrs_sunlit_via_vectors = compute_sunlit_via_frame_vectors(
        tdrs_sat,
        t,
        eph,
    )

    # 6) Compare and print summary
    summarize_boolean_comparison(
        label_a="EarthSatellite.at(t).is_sunlit",
        a=tdrs_sunlit_direct,
        label_b="ICRF.from_time_and_frame_vectors(...).is_sunlit",
        b=tdrs_sunlit_via_vectors,
        times=times,
    )

    # (Optional) you can also quickly check SBSS illumination consistency:
    sbss_sunlit_direct = np.asarray(sbss_sat.at(t).is_sunlit(eph), dtype=bool)
    sbss_sunlit_via_vectors = compute_sunlit_via_frame_vectors(
        sbss_sat,
        t,
        eph,
    )
    summarize_boolean_comparison(
        label_a="SBSS EarthSatellite.is_sunlit",
        a=sbss_sunlit_direct,
        label_b="SBSS TEME→ICRF→is_sunlit",
        b=sbss_sunlit_via_vectors,
        times=times,
    )


if __name__ == "__main__":
    main()
