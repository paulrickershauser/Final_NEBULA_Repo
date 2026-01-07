"""
NEBULA_POLIASTRO_ILLUM_CHECK.py

Independent illumination sanity check for NEBULA using poliastro.

This script:

1) Loads NEBULA LOS+illumination pickles from
       NEBULA_OUTPUT/LOSFLUX_SatPickles/
   which are produced by NEBULA_LOS_FLUX_PICKLER / SCHEDULE_PICKLER.

2) Extracts the observer and target tracks for:
       Observer: "SBSS (USA 216)"
       Target:   "TDRS 3"

3) For EACH timestep, uses poliastro.core.events.eclipse_function
   (Escobal-style conical shadow function) to compute a continuous
   "shadow value" based on:
       - Satellite state (r, v) in km / km/s, Earth-centered
       - Sun position relative to Earth in km (from DE440s via Skyfield)
       - Sun and Earth radii

   See poliastro docs:
       https://docs.poliastro.space/en/stable/autoapi/poliastro/core/events/index.html

4) Compares the SIGN of the shadow function against NEBULA's
   Skyfield-based illumination flag "illum_is_sunlit" for TDRS 3.
   Because poliastro's sign convention changed once, we compute TWO
   booleans:

       poliastro_sunlit_pos = (shadow_val > 0)
       poliastro_sunlit_neg = (shadow_val < 0)

   and report mismatch statistics for BOTH possibilities.

If one sign choice agrees almost perfectly with NEBULA, then your
illumination logic (Skyfield usage, frames, etc.) is consistent with
poliastro's independent eclipse model.

Place this file in the NEBULA project root and run from Spyder:

    %runfile "C:/Users/prick/Desktop/Research/NEBULA/NEBULA_POLIASTRO_ILLUM_CHECK.py" --wdir
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional,Sequence

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from poliastro.bodies import Earth, Sun
from poliastro.core.events import eclipse_function  # conical shadow model

from skyfield.api import load
from skyfield.api import Loader

# NEBULA configuration: paths for input/output
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR, NEBULA_INPUT_DIR


# ---------------------------------------------------------------------
# User-tunable constants
# ---------------------------------------------------------------------

# These must match the keys used in your pickled track dicts
OBSERVER_NAME = "SBSS (USA 216)"
TARGET_NAME   = "TDRS 3"

# Which pickle set do we want?  Here we match your log output:
#   NEBULA_OUTPUT/LOSFLUX_SatPickles/observer_tracks_with_los_illum_flux_los.pkl
#   NEBULA_OUTPUT/LOSFLUX_SatPickles/target_tracks_with_los_illum_flux_los.pkl
LOSFLUX_SUBDIR = "LOSFLUX_SatPickles"
OBS_PICKLE_NAME = "observer_tracks_with_los_illum_flux_los.pkl"
TAR_PICKLE_NAME = "target_tracks_with_los_illum_flux_los.pkl"

# Ephemeris filename under NEBULA_INPUT_DIR / "NEBULA_EPHEMERIS"
EPHEMERIS_FILENAME = "de440s.bsp"

# Toggle plotting
MAKE_PLOT = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_losflux_pickles() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load NEBULA LOS+illum+flux+LOS track pickles from NEBULA_OUTPUT.

    Returns
    -------
    observer_tracks : dict
        Mapping observer name → track dict.
    target_tracks : dict
        Mapping target name → track dict.
    """
    losflux_dir = Path(NEBULA_OUTPUT_DIR) / LOSFLUX_SUBDIR

    obs_path = losflux_dir / OBS_PICKLE_NAME
    tar_path = losflux_dir / TAR_PICKLE_NAME

    if not obs_path.exists():
        raise FileNotFoundError(f"Observer pickle not found: {obs_path}")
    if not tar_path.exists():
        raise FileNotFoundError(f"Target pickle not found: {tar_path}")

    with obs_path.open("rb") as f:
        observer_tracks = pickle.load(f)

    with tar_path.open("rb") as f:
        target_tracks = pickle.load(f)

    return observer_tracks, target_tracks


def get_track(track_dict: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    Convenience: fetch a single satellite track dict by name.
    """
    if name not in track_dict:
        raise KeyError(
            f"Track '{name}' not found in keys: {sorted(track_dict.keys())}"
        )
    return track_dict[name]


def load_de440s():
    """
    Load DE440s ephemeris from the standard NEBULA input location.

    Returns
    -------
    eph : skyfield.jpllib.SpiceKernel
    ts  : skyfield.api.Timescale
    """
    eph_dir = Path(NEBULA_INPUT_DIR) / "NEBULA_EPHEMERIS"
    eph_path = eph_dir / EPHEMERIS_FILENAME
    if not eph_path.exists():
        raise FileNotFoundError(
            f"DE440s ephemeris not found at {eph_path}. "
            f"Check NEBULA_INPUT_DIR / NEBULA_EPHEMERIS."
        )

    # Use a dedicated Loader so Skyfield's cache lives under NEBULA_INPUT_DIR
    loader = Loader(str(NEBULA_INPUT_DIR))
    ts = loader.timescale()
    eph = load(str(eph_path))  # direct load is fine as well

    return eph, ts


def compute_poliastro_shadow_values(
    track: Dict[str, Any],
    eph,
    ts,
) -> np.ndarray:
    """
    Evaluate poliastro's eclipse_function for each timestep in a NEBULA track.

    Parameters
    ----------
    track : dict
        NEBULA track dict for a single satellite. Must contain:
            - "times"       : list of timezone-aware datetime objects (UTC)
            - "r_eci_km"    : array (N,3) of Earth-centered position [km]
            - "v_eci_km_s"  : array (N,3) of Earth-centered velocity [km/s]
    eph : skyfield.jpllib.SpiceKernel
        JPL ephemeris containing Earth and Sun.
    ts : skyfield.api.Timescale
        Skyfield timescale used to build Time objects from datetimes.

    Returns
    -------
    shadow_values : np.ndarray of shape (N,)
        Continuous shadow function from poliastro.core.events.eclipse_function.
        The sign convention can differ by version, so we do NOT assume that
        positive means shadow or light here. We'll compare both possibilities
        downstream.
    """
    times = list(track["times"])
    r_eci = np.asarray(track["r_eci_km"], dtype=float)
    v_eci = np.asarray(track["v_eci_km_s"], dtype=float)

    if r_eci.shape != v_eci.shape:
        raise ValueError(
            f"r_eci_km shape {r_eci.shape} != v_eci_km_s shape {v_eci.shape}"
        )

    n_steps = r_eci.shape[0]
    if n_steps != len(times):
        raise ValueError(
            f"len(times)={len(times)} but r_eci has {n_steps} rows."
        )

    # Gravitational parameter μ of Earth in km^3 / s^2
    k_E = Earth.k.to(u.km**3 / u.s**2).value
    # Radii (km)
    R_E = Earth.R.to(u.km).value
    R_S = Sun.R.to(u.km).value

    # Sun position relative to Earth for each timestep, in km.
    earth = eph["earth"]
    sun = eph["sun"]
    t = ts.from_datetimes(times)
    # Vector Earth→Sun in km, shape (3, N)
    r_sun_eci_km = earth.at(t).observe(sun).position.km

    shadow_vals = np.empty(n_steps, dtype=float)

    for i in range(n_steps):
        # Satellite state wrt Earth: [x, y, z, vx, vy, vz]
        u_vec = np.concatenate((r_eci[i], v_eci[i]))

        # Sun position wrt Earth at this timestep (3,)
        r_sec = r_sun_eci_km[:, i]

        # Eclipse function from poliastro: Escobal-style conical shadow
        # k_E : Earth μ
        # u_  : sat state wrt Earth
        # r_sec : Sun position wrt Earth
        # R_sec : Sun radius
        # R_primary : Earth radius
        shadow_vals[i] = eclipse_function(
            k_E,
            u_vec,
            r_sec,
            R_S,
            R_E,
            umbra=True,
        )

    return shadow_vals


def summarize_boolean_comparison(
    label_a: str,
    a: np.ndarray,
    label_b: str,
    b: np.ndarray,
    times: Optional[Sequence] = None,
    shadow_vals: Optional[np.ndarray] = None,
    max_print: int = 80,
) -> None:
    """
    Simple mismatch summary between two boolean arrays, with optional
    detailed listing of when mismatches occur.

    Parameters
    ----------
    label_a, label_b : str
        Names of the two boolean series.
    a, b : array-like of bool
        Boolean arrays of equal length.
    times : sequence of datetime, optional
        If provided, mismatch table will include the UTC timestamp.
    shadow_vals : array-like of float, optional
        If provided, mismatch table will include the poliastro
        shadow_function value at each mismatch.
    max_print : int
        Maximum number of mismatches to list.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {label_a} {a.shape} vs {label_b} {b.shape}"
        )

    mism_idx = np.nonzero(a ^ b)[0]
    total = a.size
    n_mism = mism_idx.size
    agreement = 100.0 * (1.0 - n_mism / total) if total > 0 else 0.0

    print(f"\n=== Boolean comparison: {label_a} vs {label_b} ===")
    print(f"Total timesteps  : {total}")
    print(f"Total mismatches : {n_mism}")
    print(f"Global agreement : {agreement:.3f} %")

    if n_mism == 0:
        print("All timesteps agree exactly.")
        return

    # Header for detailed table
    header = " idx "
    if times is not None:
        header += "| UTC time                 "
    header += "| A | B "
    if shadow_vals is not None:
        header += "| shadow_val"
    print("\nFirst {} mismatches:".format(min(max_print, n_mism)))
    print(header)
    print("-" * len(header))

    for k, i in enumerate(mism_idx[:max_print]):
        row = f"{i:4d} "
        if times is not None:
            row += f"| {times[i].isoformat():<24} "
        row += f"| {int(a[i])} | {int(b[i])} "
        if shadow_vals is not None:
            row += f"| {shadow_vals[i]: .6e}"
        print(row)

# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def main():
    # 1) Load NEBULA LOS+illum+flux+LOS track pickles
    observer_tracks, target_tracks = load_losflux_pickles()

    obs_track = get_track(observer_tracks, OBSERVER_NAME)
    tar_track = get_track(target_tracks, TARGET_NAME)

    # 2) Load DE440s and Skyfield timescale
    eph, ts = load_de440s()

    # 3) Grab NEBULA illumination flags for the target
    nebula_is_sunlit = np.asarray(
        tar_track["illum_is_sunlit"], dtype=bool
    )

    print(
        f"Loaded track '{TARGET_NAME}' with "
        f"{nebula_is_sunlit.size} timesteps."
    )
    print(
        f"NEBULA illum_is_sunlit: "
        f"{nebula_is_sunlit.sum()} True, "
        f"{nebula_is_sunlit.size - nebula_is_sunlit.sum()} False"
    )
    
    times = list(tar_track["times"])

    # 4) Compute poliastro shadow function for each timestep
    shadow_vals = compute_poliastro_shadow_values(tar_track, eph, ts)

    # 5) Derive two possible boolean interpretations of the shadow value
    poliastro_sunlit_pos = shadow_vals > 0.0   # "sunlit if > 0"
    poliastro_sunlit_neg = shadow_vals < 0.0   # "sunlit if < 0"

    # 6) Compare against NEBULA's Skyfield-based result
    summarize_boolean_comparison(
        "NEBULA illum_is_sunlit",
        nebula_is_sunlit,
        "poliastro (shadow>0 → sunlit)",
        poliastro_sunlit_pos,
        times=times,
        shadow_vals=shadow_vals,
    )

    summarize_boolean_comparison(
        "NEBULA illum_is_sunlit",
        nebula_is_sunlit,
        "poliastro (shadow<0 → sunlit)",
        poliastro_sunlit_neg,
        times=times,
        shadow_vals=shadow_vals,
    )


    # 7) Optional: plot shadow value vs time for visual inspection
    if MAKE_PLOT:
        times = list(tar_track["times"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, shadow_vals, label="poliastro shadow_function")
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

        # Overlay NEBULA sunlit periods as a simple marker (0/1 scaled)
        # This is just to see if the zero crossings line up.
        scaled_nebula = nebula_is_sunlit.astype(float)
        # Shift slightly for visibility if necessary
        ax.scatter(
            times,
            scaled_nebula,
            s=5,
            alpha=0.5,
            label="NEBULA illum_is_sunlit (0/1)",
        )

        ax.set_title(f"poliastro shadow_function vs time for {TARGET_NAME}")
        ax.set_xlabel("UTC time")
        ax.set_ylabel("shadow_function value (dimensionless)")
        ax.legend(loc="best")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
