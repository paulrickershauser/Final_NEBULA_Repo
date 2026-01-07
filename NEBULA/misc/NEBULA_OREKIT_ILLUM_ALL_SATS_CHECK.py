# -*- coding: utf-8 -*-
"""
NEBULA_OREKIT_ILLUM_ALL_SATS_CHECK.py

Compare NEBULA's Skyfield-based illumination flags
    track["illum_is_sunlit"]
for *all targets* produced by NEBULA_SAT_ILL_PICKLER against Orekit's
EclipseDetector evaluated on NEBULA's own r,v tracks.

For each target satellite:
    - Compute Orekit eclipse function g(t) from r_eci_km, v_eci_km_s.
    - Convert to a boolean Orekit "is_sunlit" flag via g >= 0.
    - Compare against NEBULA's illum_is_sunlit.
    - Print mismatch counts and agreement percentage.

At the end:
    - Rank satellites by mismatch count (descending).
    - Print a global summary over all targets.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np

# ----------------------------------------------------------------------
# NEBULA imports
# ----------------------------------------------------------------------
# This will:
#   1) Ensure LOS-enhanced tracks exist,
#   2) Attach Skyfield illumination arrays (illum_is_sunlit, etc.),
#   3) Return two dicts of serialised track fields:
#        observer_tracks_with_illum, target_tracks_with_illum
from Utility.SAT_OBJECTS import NEBULA_SAT_ILL_PICKLER

# ----------------------------------------------------------------------
# Orekit setup
# ----------------------------------------------------------------------
import orekit

# Initialize the JVM. If it's already running, this may either be a no-op
# or raise; in the latter case we just ignore the error because the VM
# is already up.
try:
    orekit.initVM()
except Exception as e:
    print(f"[INFO] orekit.initVM() raised {e!r} - JVM probably already started.")

from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate

# Point Orekit at your orekit-data directory
setup_orekit_curdir(
    r"C:\Users\prick\Desktop\Research\NEBULA\Input\NEBULA_OREKIT_DATA\orekit-data-main"
)

from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory
from org.orekit.propagation.events import EclipseDetector
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.hipparchus.geometry.euclidean.threed import Vector3D


# ----------------------------------------------------------------------
# Orekit context + shadow function g(state)
# ----------------------------------------------------------------------
def build_orekit_context() -> Dict[str, Any]:
    """
    Initialize core Orekit objects needed for eclipse evaluation.

    Returns
    -------
    ctx : dict
        Dictionary with:
            "utc"         : TimeScale (UTC)
            "eme2000"     : inertial Frame
            "itrf"        : Earth-fixed frame (ITRF 2010)
            "sun"         : Celestial body Sun
            "earth_shape" : OneAxisEllipsoid WGS-84 Earth
            "eclipse_det" : EclipseDetector (Sun occulted by Earth)
    """
    utc = TimeScalesFactory.getUTC()
    eme2000 = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # WGS-84 Earth ellipsoid in ITRF
    earth_shape = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    # Sun body
    sun = CelestialBodyFactory.getSun()

    # EclipseDetector(sun, sun_radius, earth_shape):
    #  g(state) < 0  => in umbra (eclipsed)
    #  g(state) >= 0 => sunlit
    eclipse_det = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape)

    return {
        "utc": utc,
        "eme2000": eme2000,
        "itrf": itrf,
        "earth_shape": earth_shape,
        "sun": sun,
        "eclipse_det": eclipse_det,
    }


def compute_orekit_shadow_from_track(
    track: Dict[str, Any],
    ctx: Dict[str, Any],
) -> np.ndarray:
    """
    Compute Orekit eclipse-detector g(state) using NEBULA's own r,v track
    (converted to meters) rather than TLE propagation.

    Parameters
    ----------
    track : dict
        Single-satellite track dict as returned by NEBULA_SAT_ILL_PICKLER.
        Must contain:
            "times"       : list of datetime (UTC)
            "r_eci_km"    : (N,3) position vectors, km, NEBULA inertial frame
            "v_eci_km_s"  : (N,3) velocity vectors, km/s
    ctx : dict
        Context from build_orekit_context().

    Returns
    -------
    g_vals : np.ndarray, shape (N,)
        EclipseDetector g(state) for each NEBULA state:
            g < 0  => in umbra (eclipsed)
            g >= 0 => sunlit
    """
    times = list(track["times"])
    r_eci_km = np.asarray(track["r_eci_km"], dtype=float)
    v_eci_km_s = np.asarray(track["v_eci_km_s"], dtype=float)

    if r_eci_km.shape != v_eci_km_s.shape:
        raise ValueError("r_eci_km and v_eci_km_s shapes differ")

    if r_eci_km.ndim != 2 or r_eci_km.shape[1] != 3:
        raise ValueError("r_eci_km and v_eci_km_s must be N×3 arrays")

    eclipse_det = ctx["eclipse_det"]
    frame = ctx["eme2000"]
    utc = ctx["utc"]

    g_vals = np.zeros(r_eci_km.shape[0], dtype=float)

    for i, (dt, r_km, v_km_s) in enumerate(zip(times, r_eci_km, v_eci_km_s)):
        # Convert datetime to Orekit AbsoluteDate
        date = datetime_to_absolutedate(dt)

        # Convert km -> m
        r_m = r_km * 1000.0
        v_m_s = v_km_s * 1000.0

        # Orekit PVCoordinates in EME2000
        pv = PVCoordinates(
            Vector3D(float(r_m[0]), float(r_m[1]), float(r_m[2])),
            Vector3D(float(v_m_s[0]), float(v_m_s[1]), float(v_m_s[2])),
        )

        # Build orbit & state (μ only affects dynamics, not instantaneous geometry)
        orbit = CartesianOrbit(pv, frame, date, Constants.WGS84_EARTH_MU)
        state = SpacecraftState(orbit)

        # Eclipse function value
        g_vals[i] = eclipse_det.g(state)

    return g_vals


# ----------------------------------------------------------------------
# Main comparison: NEBULA illum_is_sunlit vs Orekit g>=0 for *all* targets
# ----------------------------------------------------------------------
def compare_all_targets(force_recompute_illum: bool = False) -> None:
    """
    Ensure NEBULA illumination pickles exist, then compare illum_is_sunlit
    against Orekit EclipseDetector for every target track.

    Parameters
    ----------
    force_recompute_illum : bool, optional
        If True, force NEBULA_SAT_ILL_PICKLER to recompute illumination
        for all targets. If False (default), it will reuse existing
        illumination pickles if present.
    """
    # 1) Ensure illumination pickles exist and load them
    #    - obs_tracks_with_illum  : dict of observer track dicts
    #    - tgt_tracks_with_illum  : dict of target track dicts
    obs_tracks, tgt_tracks = NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets(
        force_recompute=force_recompute_illum
    )

    print(f"Loaded {len(obs_tracks)} observers and {len(tgt_tracks)} targets "
          f"from NEBULA_SAT_ILL_PICKLER.")

    # 2) Build Orekit context once
    ctx = build_orekit_context()

    # 3) Per-target statistics
    stats: List[Dict[str, Any]] = []

    print("\n=== Per-target NEBULA vs Orekit illumination comparison ===")
    print("Name                           |   N   | mismatches | agreement [%]")
    print("-----------------------------------------------------------------")

    for name in sorted(tgt_tracks.keys()):
        track = tgt_tracks[name]

        # NEBULA Skyfield-based illumination flags
        nebula_is_sunlit = np.asarray(track["illum_is_sunlit"], dtype=bool)

        # Orekit eclipse function g(state) for this track
        g_vals = compute_orekit_shadow_from_track(track, ctx)
        orekit_is_sunlit = g_vals >= 0.0  # g < 0 => eclipsed, g >= 0 => lit

        if nebula_is_sunlit.shape != orekit_is_sunlit.shape:
            raise ValueError(
                f"Length mismatch for {name}: "
                f"illum_is_sunlit {nebula_is_sunlit.shape}, "
                f"Orekit flag {orekit_is_sunlit.shape}"
            )

        N = nebula_is_sunlit.size
        mismatches = int(np.count_nonzero(nebula_is_sunlit != orekit_is_sunlit))
        agreement = 100.0 * (1.0 - mismatches / N) if N > 0 else 100.0

        print(f"{name:30s} | {N:5d} | {mismatches:10d} | {agreement:11.3f}")

        stats.append(
            {
                "name": name,
                "N": N,
                "mismatches": mismatches,
                "agreement": agreement,
            }
        )

    if not stats:
        print("\nNo target tracks found. Check NEBULA_SAT_ILL_PICKLER config.")
        return

    # 4) Rank satellites by mismatch count
    print("\n=== Targets ranked by mismatch count (descending) ===")
    print("Name                           | mismatches |   N   | agreement [%]")
    print("-----------------------------------------------------------------")

    for s in sorted(stats, key=lambda d: d["mismatches"], reverse=True):
        print(
            f"{s['name']:30s} | {s['mismatches']:10d} | {s['N']:5d} | "
            f"{s['agreement']:11.3f}"
        )

    # 5) Global summary over all targets
    total_N = sum(s["N"] for s in stats)
    total_mismatches = sum(s["mismatches"] for s in stats)
    global_agreement = 100.0 * (1.0 - total_mismatches / total_N) if total_N > 0 else 100.0

    print("\n=== Global summary over all targets ===")
    print(f"Total timesteps : {total_N}")
    print(f"Total mismatches: {total_mismatches}")
    print(f"Global agreement: {global_agreement:11.3f} %")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Set to True if you want to force a fresh illumination recompute
    # before comparing against Orekit.
    compare_all_targets(force_recompute_illum=False)
