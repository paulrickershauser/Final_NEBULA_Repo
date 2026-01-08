"""NEBULA -> EBS_SDA_SIM (Rachel Oliver) bridge: Step 1 (Radiometry).

This module is intentionally *small* and *opinionated*.

Goal
----
Convert NEBULA's photon *flux* quantities at the entrance pupil
    phi_ph_m2_s  [photons / m^2 / s]
into the photon *rate* quantity expected by Rachel Oliver's circuitry model
input frames:
    ph_s  [photons / s]

Important: QE is applied later
-----------------------------
Rachel's `circuitry.ebCircuitSim.photo_current_photons_per_sec()` multiplies
incident photons/sec by `eta` (QE) and the electron charge. Therefore the
HDF5 frames we will eventually hand to `generate_events()` must be:

    photons / second / pixel   (pre-QE)

This Step-1 module does *not*:
- build pixel frames (Step 2/3),
- apply PSF / optics (Step 2),
- call the circuitry model (Step 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class RadiometryScale:
    """Scalar radiometry factors used to convert flux -> rate."""

    aperture_area_m2: float
    optical_throughput: float

    @property
    def effective_area_m2(self) -> float:
        """Effective collecting area (area * throughput)."""

        return float(self.aperture_area_m2) * float(self.optical_throughput)


def resolve_radiometry_scale_from_sensor(
    sensor: Any,
    *,
    default_optical_throughput: float = 1.0,
) -> RadiometryScale:
    """Resolve collecting area + throughput from a NEBULA-like sensor object.

    This function is designed to work with NEBULA's SensorConfig, but it will
    accept *any* object that provides:

      - aperture_area_m2 (preferred) OR aperture_diameter_m
      - optical_throughput (optional)

    Fail-fast: if the area cannot be resolved, this raises.
    """

    # 1) Collecting area
    area_m2: Optional[float] = getattr(sensor, "aperture_area_m2", None)
    if area_m2 is None:
        diam_m = getattr(sensor, "aperture_diameter_m", None)
        if diam_m is None:
            raise ValueError(
                "Sensor must provide aperture_area_m2 or aperture_diameter_m to compute collecting area."
            )
        diam_m_f = float(diam_m)
        if (not math.isfinite(diam_m_f)) or diam_m_f <= 0.0:
            raise ValueError(f"Invalid aperture_diameter_m={diam_m!r}")
        area_m2 = math.pi * (0.5 * diam_m_f) ** 2
    else:
        area_m2 = float(area_m2)

    if (not math.isfinite(area_m2)) or area_m2 <= 0.0:
        raise ValueError(f"Invalid collecting area (m^2) resolved from sensor: {area_m2!r}")

    # 2) Throughput
    throughput = getattr(sensor, "optical_throughput", None)
    if throughput is None:
        throughput_f = float(default_optical_throughput)
    else:
        throughput_f = float(throughput)

    if (not math.isfinite(throughput_f)) or (throughput_f < 0.0) or (throughput_f > 1.0):
        raise ValueError(f"Invalid optical_throughput={throughput!r} (must be in [0,1])")

    return RadiometryScale(aperture_area_m2=float(area_m2), optical_throughput=float(throughput_f))


def phi_ph_m2_s_to_ph_s(phi_ph_m2_s: Any, *, scale: RadiometryScale) -> np.ndarray:
    """Convert photon flux [ph / m^2 / s] -> photon rate [ph / s].

    Parameters
    ----------
    phi_ph_m2_s:
        Scalar or array-like photon flux at the entrance pupil.
    scale:
        RadiometryScale containing collecting area and throughput.

    Returns
    -------
    np.ndarray
        Photon rate array (same shape as input) in photons / second.

    Notes
    -----
    This performs only the *scalar* flux->rate conversion:

        ph/s = (ph/m^2/s) * (m^2) * (dimensionless throughput)

    It does NOT distribute photons across pixels.
    """

    phi = np.asarray(phi_ph_m2_s, dtype=np.float64)
    if np.any(phi < 0):
        raise ValueError("phi_ph_m2_s contains negative values; expected non-negative photon flux.")
    return phi * scale.effective_area_m2


def phi_ph_m2_s_pix_to_ph_s_pix(phi_ph_m2_s_pix: Any, *, scale: RadiometryScale) -> np.ndarray:
    """Convert per-pixel photon flux [ph / m^2 / s / pix] -> [ph / s / pix]."""

    phi_pix = np.asarray(phi_ph_m2_s_pix, dtype=np.float64)
    if np.any(phi_pix < 0):
        raise ValueError(
            "phi_ph_m2_s_pix contains negative values; expected non-negative photon flux per pixel."
        )
    return phi_pix * scale.effective_area_m2
