# -*- coding: utf-8 -*-
"""
NEBULA_OREKIT_ILLUM_CHECK.py

Four-way GEO illumination cross-check for TDRS 3:

    • NEBULA (Skyfield)   : track["illum_is_sunlit"]
    • poliastro           : eclipse_function(time, r, v)
    • Orekit-TLE          : Orekit TLE propagator + EclipseDetector
    • Orekit-track        : Orekit EclipseDetector fed by NEBULA r,v
    • Skyfield EarthSat   : skyfield.api.EarthSatellite.is_sunlit

It reuses helpers from NEBULA_POLIASTRO_ILLUM_CHECK so you get the
same track loading and poliastro setup you already used.

Assumptions
-----------
- This script lives in the NEBULA repo root, alongside:
    NEBULA_POLIASTRO_ILLUM_CHECK.py
- Orekit Python wrapper is installed (JCC-based `orekit` package).
- An `orekit-data` directory (or zip) is present in the working dir,
  compatible with `orekit.pyhelpers.setup_orekit_curdir()`.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# NEBULA + poliastro helpers
# ----------------------------------------------------------------------
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
import NEBULA_POLIASTRO_ILLUM_CHECK as poli_chk
import matplotlib.pyplot as plt

# We will reuse:
#   - poli_chk.load_losflux_pickles()
#   - poli_chk.get_track()
#   - poli_chk.load_de440s()
#   - poli_chk.compute_poliastro_shadow_values()
#   - poli_chk.summarize_boolean_comparison()

# Names must match the ones you used in NEBULA_POLIASTRO_ILLUM_CHECK
TARGET_NAME = poli_chk.TARGET_NAME  # "TDRS 3"

# TDRS 3 TLE (same epoch + lines you pasted in chat)
TDRS3_LINE1 = "1 19548U 88091B   25312.45027976 -.00000302  00000+0  00000+0 0  9990"
TDRS3_LINE2 = "2 19548  12.7717 342.4579 0040433 340.4110 198.1740  1.00266379123173"

# ----------------------------------------------------------------------
# Orekit setup
# ----------------------------------------------------------------------
import orekit
orekit.initVM()  # OK to call once at import time

from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate
setup_orekit_curdir(r"C:\Users\prick\Desktop\Research\NEBULA\Input\NEBULA_OREKIT_DATA\orekit-data-main")  # looks for orekit-data in current working dir

from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory
from org.orekit.propagation.events import EclipseDetector
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.hipparchus.geometry.euclidean.threed import Vector3D


def build_orekit_context() -> dict:
    """
    Initialize core Orekit objects needed for eclipse evaluation.

    Returns
    -------
    ctx : dict
        Dictionary with:
            utc          : TimeScale (UTC)
            eme2000      : inertial Frame
            itrf         : Earth-fixed frame (ITRF 2010)
            sun          : Celestial body Sun
            earth_shape  : OneAxisEllipsoid WGS-84 Earth
            eclipse_det  : EclipseDetector (Sun occulted by Earth ellipsoid)
    """
    utc = TimeScalesFactory.getUTC()
    eme2000 = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # WGS-84 Earth model in ITRF
    earth_shape = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    # Sun body
    sun = CelestialBodyFactory.getSun()

    # EclipseDetector(sun, sun_radius, earth_shape):
    # g(state) < 0 inside umbra, > 0 outside. (Total eclipse by default)
    eclipse_det = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape)

    return {
        "utc": utc,
        "eme2000": eme2000,
        "itrf": itrf,
        "earth_shape": earth_shape,
        "sun": sun,
        "eclipse_det": eclipse_det,
    }


# ----------------------------------------------------------------------
# Orekit Option A: internal TLE propagation
# ----------------------------------------------------------------------
def compute_orekit_shadow_from_tle(
    times,
    ctx: dict,
) -> np.ndarray:
    """
    Compute Orekit eclipse-detector g(state) evaluating a TLE-propagated
    TDRS 3 state at each NEBULA timestamp.

    Parameters
    ----------
    times : sequence of datetime
        List/array of timezone-aware Python datetimes (UTC) from NEBULA track.
    ctx : dict
        Context from build_orekit_context().

    Returns
    -------
    g_vals : np.ndarray, shape (N,)
        EclipseDetector g(state) at each time.
        Sign convention (Orekit):
            g < 0  => in umbra (eclipsed)
            g >= 0 => sunlit
    """
    eclipse_det = ctx["eclipse_det"]
    utc = ctx["utc"]

    # Build TLE and propagator
    tle = TLE(TDRS3_LINE1, TDRS3_LINE2)
    propagator = TLEPropagator.selectExtrapolator(tle)

    g_vals = np.zeros(len(times), dtype=float)

    for i, dt in enumerate(times):
        # Convert Python datetime -> Orekit AbsoluteDate (UTC).
        # New pyhelpers signature only needs the datetime.
        date = datetime_to_absolutedate(dt)
    
        # Propagate TLE to this date
        state = propagator.propagate(date)
    
        # Evaluate eclipse function
        g_vals[i] = eclipse_det.g(state)

    return g_vals


# ----------------------------------------------------------------------
# Orekit Option B: use NEBULA r,v track
# ----------------------------------------------------------------------
def compute_orekit_shadow_from_nebula_track(
    track: dict,
    ctx: dict,
) -> np.ndarray:
    """
    Compute Orekit eclipse-detector g(state) using NEBULA's own r,v track
    (converted to meters) rather than Orekit TLE propagation.

    Parameters
    ----------
    track : dict
        Single-satellite track from NEBULA, as returned by
        NEBULA_POLIASTRO_ILLUM_CHECK.get_track(...).
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

    if r_eci_km.shape[1] != 3:
        raise ValueError("r_eci_km and v_eci_km_s must be N×3 arrays")

    eclipse_det = ctx["eclipse_det"]
    frame = ctx["eme2000"]
    # utc = ctx["utc"]  # no longer needed with new datetime_to_absolutedate()
    
    g_vals = np.zeros(r_eci_km.shape[0], dtype=float)
    
    for i, (dt, r_km, v_km_s) in enumerate(zip(times, r_eci_km, v_eci_km_s)):
        # Convert datetime to AbsoluteDate (UTC assumed from tz-aware datetime)
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
# Skyfield EarthSatellite.is_sunlit from the same TDRS 3 TLE
# ----------------------------------------------------------------------
def compute_skyfield_sunlit_from_tle(
    times,
    eph,
    ts,
) -> np.ndarray:
    """
    Use Skyfield's EarthSatellite.is_sunlit() as a separate baseline
    for TDRS 3, using the same TLE and DE440s ephemeris as NEBULA.
    """
    from skyfield.api import EarthSatellite  # local import to avoid extra deps if unused

    # Build Skyfield Time array for the requested datetimes
    t_sky = ts.from_datetimes(times)

    # EarthSatellite from the same TDRS-3 TLE
    sat = EarthSatellite(TDRS3_LINE1, TDRS3_LINE2, "TDRS 3", ts)
    sat_at_t = sat.at(t_sky)

    # IMPORTANT: pass the *ephemeris* (kernel) to is_sunlit, not the sun body
    is_sunlit = sat_at_t.is_sunlit(eph)

    return np.asarray(is_sunlit, dtype=bool)



# ----------------------------------------------------------------------
# Main comparison driver
# ----------------------------------------------------------------------
# def main() -> None:
    # # 1) Load NEBULA LOS+illum track (reusing your previous helper)
    # obs_tracks, tar_tracks = poli_chk.load_losflux_pickles()
    # tar_track = poli_chk.get_track(tar_tracks, TARGET_NAME)

    # times = list(tar_track["times"])
    # nebula_is_sunlit = np.asarray(tar_track["illum_is_sunlit"], dtype=bool)

    # print(f"Loaded track '{TARGET_NAME}' with {nebula_is_sunlit.size} timesteps.")
    # print(
    #     f"NEBULA illum_is_sunlit: "
    #     f"{nebula_is_sunlit.sum()} True, "
    #     f"{nebula_is_sunlit.size - nebula_is_sunlit.sum()} False"
    # )

    # # 2) Compute poliastro shadow function on the same grid
    # eph, ts = poli_chk.load_de440s()
    # shadow_vals = poli_chk.compute_poliastro_shadow_values(tar_track, eph, ts)

    # # Based on your previous run and poliastro docs, we interpret:
    # #   shadow_vals > 0  => in umbra (eclipsed)
    # #   shadow_vals < 0  => sunlit
    # poli_is_sunlit = shadow_vals < 0.0

    # # 3) Skyfield EarthSatellite.is_sunlit baseline from TLE
    # skyfield_is_sunlit = compute_skyfield_sunlit_from_tle(times, eph, ts)

    # # 4) Orekit context + both shadow variants
    # ctx = build_orekit_context()

    # orekit_g_tle = compute_orekit_shadow_from_tle(times, ctx)
    # orekit_tle_is_sunlit = orekit_g_tle >= 0.0  # g < 0 => shadow, g >= 0 => lit

    # orekit_g_track = compute_orekit_shadow_from_nebula_track(tar_track, ctx)
    # orekit_track_is_sunlit = orekit_g_track >= 0.0

    # # 5) Run pairwise comparisons using your existing summarizer
    # #    (We pass times+shadow_vals only when poliastro is involved so you see g-values.)

    # def summarize(label_a, arr_a, label_b, arr_b, with_shadow=False):
    #     if with_shadow:
    #         poli_chk.summarize_boolean_comparison(
    #             label_a,
    #             arr_a,
    #             label_b,
    #             arr_b,
    #             times=times,
    #             shadow_vals=shadow_vals,
    #         )
    #     else:
    #         poli_chk.summarize_boolean_comparison(
    #             label_a,
    #             arr_a,
    #             label_b,
    #             arr_b,
    #             times=times,
    #             shadow_vals=None,
    #         )

    # # NEBULA vs poliastro  (you've already seen this, but now it's in the 4-way context)
    # summarize("NEBULA illum_is_sunlit", nebula_is_sunlit,
    #           "poliastro (shadow<0 → sunlit)", poli_is_sunlit,
    #           with_shadow=True)

    # # NEBULA vs Orekit (TLE)
    # summarize("NEBULA illum_is_sunlit", nebula_is_sunlit,
    #           "Orekit-TLE (g>=0 → sunlit)", orekit_tle_is_sunlit)

    # # NEBULA vs Orekit (track r,v)
    # summarize("NEBULA illum_is_sunlit", nebula_is_sunlit,
    #           "Orekit-track (g>=0 → sunlit)", orekit_track_is_sunlit)

    # # poliastro vs Orekit-track
    # summarize("poliastro (shadow<0 → sunlit)", poli_is_sunlit,
    #           "Orekit-track (g>=0 → sunlit)", orekit_track_is_sunlit,
    #           with_shadow=True)

    # # poliastro vs Orekit-TLE
    # summarize("poliastro (shadow<0 → sunlit)", poli_is_sunlit,
    #           "Orekit-TLE (g>=0 → sunlit)", orekit_tle_is_sunlit,
    #           with_shadow=True)

    # # Skyfield EarthSatellite vs NEBULA (should be ~identical by construction)
    # summarize("Skyfield EarthSatellite.is_sunlit", skyfield_is_sunlit,
    #           "NEBULA illum_is_sunlit", nebula_is_sunlit)

    # # Skyfield EarthSatellite vs Orekit-track
    # summarize("Skyfield EarthSatellite.is_sunlit", skyfield_is_sunlit,
    #           "Orekit-track (g>=0 → sunlit)", orekit_track_is_sunlit)

    # # Skyfield EarthSatellite vs poliastro
    # summarize("Skyfield EarthSatellite.is_sunlit", skyfield_is_sunlit,
    #           "poliastro (shadow<0 → sunlit)", poli_is_sunlit,
    #           with_shadow=True)

    # # ------------------------------------------------------------------
    # # 6) Plot illumination flags vs time for all models
    # # ------------------------------------------------------------------
    # # Convert times to hours from start for a nice x-axis
    # t0 = times[0]
    # t_hours = np.array([(t - t0).total_seconds() / 3600.0 for t in times])

    # # Convert booleans to ints for plotting (0/1), then vertically offset
    # nebula_y     = nebula_is_sunlit.astype(int)         + 0
    # skyfield_y   = skyfield_is_sunlit.astype(int)       + 2
    # orekit_tr_y  = orekit_track_is_sunlit.astype(int)   + 4
    # orekit_tle_y = orekit_tle_is_sunlit.astype(int)     + 6
    # poli_y       = poli_is_sunlit.astype(int)           + 8

    # plt.figure(figsize=(12, 6))

    # # Use step plots so the 0/1 changes are clear over time
    # plt.step(t_hours, nebula_y,     where="post", label="NEBULA (Skyfield)")
    # plt.step(t_hours, skyfield_y,   where="post", label="Skyfield EarthSatellite")
    # plt.step(t_hours, orekit_tr_y,  where="post", label="Orekit (NEBULA track)")
    # plt.step(t_hours, orekit_tle_y, where="post", label="Orekit (TLE)")
    # plt.step(t_hours, poli_y,       where="post", label="poliastro eclipse_function")

    # # Label the y-axis ticks so the offsets make sense
    # plt.yticks(
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     [
    #         "NEBULA: dark",    "NEBULA: sunlit",
    #         "Skyfield: dark",  "Skyfield: sunlit",
    #         "Orekit tr: dark", "Orekit tr: sunlit",
    #         "Orekit TLE: dark","Orekit TLE: sunlit",
    #         "poliastro: dark", "poliastro: sunlit",
    #     ],
    # )

    # plt.xlabel("Time since start [hours]")
    # plt.title(f"Illumination comparison for {TARGET_NAME}")
    # plt.grid(True, axis="x", linestyle=":")
    # plt.legend(loc="upper right")
    # plt.tight_layout()
    # plt.show()

    # # ------------------------------------------------------------------
    # # 7) Optional: mismatch indicator plot (NEBULA vs each)
    # # ------------------------------------------------------------------
    # mismatch_poli   = (nebula_is_sunlit != poli_is_sunlit).astype(int)
    # mismatch_ore_tr = (nebula_is_sunlit != orekit_track_is_sunlit).astype(int)
    # mismatch_ore_tl = (nebula_is_sunlit != orekit_tle_is_sunlit).astype(int)

    # plt.figure(figsize=(12, 4))
    # plt.step(t_hours, mismatch_poli,   where="post", label="NEBULA vs poliastro")
    # plt.step(t_hours, mismatch_ore_tr, where="post", label="NEBULA vs Orekit-track")
    # plt.step(t_hours, mismatch_ore_tl, where="post", label="NEBULA vs Orekit-TLE")

    # plt.xlabel("Time since start [hours]")
    # plt.ylabel("Mismatch (0/1)")
    # plt.title(f"Illumination mismatches vs NEBULA for {TARGET_NAME}")
    # plt.grid(True, axis="x", linestyle=":")
    # plt.legend(loc="upper right")
    # plt.tight_layout()
    # plt.show()
def main() -> None:
    """
    Minimal driver: compute Orekit's eclipse function g(state) for the
    NEBULA track of TARGET_NAME, and print time, g, and sunlit flag.

    Convention:
        g(state) < 0 -> in Earth's shadow (eclipsed)
        g(state) >= 0 -> sunlit
    """
    # 1) Load NEBULA track for the GEO target
    obs_tracks, tar_tracks = poli_chk.load_losflux_pickles()
    tar_track = poli_chk.get_track(tar_tracks, TARGET_NAME)

    times = list(tar_track["times"])

    print(f"Loaded track '{TARGET_NAME}' with {len(times)} timesteps.")

    # 2) Build Orekit context (frames, Earth, Sun, EclipseDetector, etc.)
    ctx = build_orekit_context()

    # 3) Evaluate Orekit eclipse function for each NEBULA state
    orekit_g_track = compute_orekit_shadow_from_nebula_track(tar_track, ctx)
    orekit_track_is_sunlit = orekit_g_track >= 0.0  # g < 0 => shadow, g >= 0 => lit

    # 4) Basic summary
    n = len(times)
    n_lit = int(orekit_track_is_sunlit.sum())
    n_dark = n - n_lit
    print(f"Orekit sunlit counts: {n_lit} True, {n_dark} False")

    # 5) Print a table of time, g, and sunlit flag for inspection
    print()
    print(" idx | UTC time                 | g_track [m]        | sunlit")
    print("---------------------------------------------------------------")
    for idx, (t, g, sunlit) in enumerate(zip(times, orekit_g_track, orekit_track_is_sunlit)):
        # sunlit as 1/0 to keep it compact
        print(f"{idx:4d} | {t.isoformat()} | {g: .6e} | {int(sunlit)}")



if __name__ == "__main__":
    main()
