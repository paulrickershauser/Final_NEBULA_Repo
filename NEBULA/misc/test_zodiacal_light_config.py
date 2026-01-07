"""
test_zodiacal_light_config.py

Smoke-test for Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG.

This script checks:
- The module imports cleanly
- validate_config() passes
- Ω_pix computations are finite and positive
- unit conversion helpers behave as expected
- get_sensor_config() fails fast on mismatched sensor_name
"""

from __future__ import annotations

import math

from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR
from Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG import (
    ZODIACAL_LIGHT_CONFIG,
    validate_config,
    get_sensor_config,
    pixel_solid_angle_sr,
    pixel_solid_angle_arcsec2,
    convert_per_arcsec2_to_per_sr,
    convert_per_sr_to_per_pixel,
)


def main() -> None:
    # 1) Structural validation (fail-fast)
    validate_config(ZODIACAL_LIGHT_CONFIG)
    print("validate_config(): OK")

    # 2) ACTIVE_SENSOR enforcement
    s = get_sensor_config()
    assert s is ACTIVE_SENSOR, "get_sensor_config() did not return ACTIVE_SENSOR"
    print(f"get_sensor_config(): OK (ACTIVE_SENSOR.name={ACTIVE_SENSOR.name!r})")

    # 3) Fail-fast mismatch check
    try:
        _ = get_sensor_config(sensor_name="__INTENTIONAL_MISMATCH__")
        raise AssertionError("get_sensor_config() did not raise on mismatched sensor_name")
    except RuntimeError:
        print("get_sensor_config(mismatch): OK (raised RuntimeError)")

    # 4) Ω_pix computations
    omega_sr = pixel_solid_angle_sr(ACTIVE_SENSOR, ZODIACAL_LIGHT_CONFIG)
    omega_a2 = pixel_solid_angle_arcsec2(ACTIVE_SENSOR, ZODIACAL_LIGHT_CONFIG)

    assert math.isfinite(omega_sr) and omega_sr > 0.0
    assert math.isfinite(omega_a2) and omega_a2 > 0.0

    print(f"Ω_pix (sr/pix):      {omega_sr:.6e}")
    print(f"Ω_pix (arcsec^2/pix): {omega_a2:.6e}")

    # 5) Unit conversion sanity for a dummy value
    # Use a known positive dummy "native" value: 1 ph m^-2 s^-1 arcsec^-2
    phi_arcsec2 = 1.0
    phi_sr = convert_per_arcsec2_to_per_sr(phi_arcsec2)
    phi_pix = convert_per_sr_to_per_pixel(phi_sr, omega_sr)

    assert math.isfinite(phi_sr) and phi_sr > 0.0
    assert math.isfinite(phi_pix) and phi_pix > 0.0

    print(f"Dummy conversion: {phi_arcsec2} per_arcsec2 -> {phi_sr:.6e} per_sr -> {phi_pix:.6e} per_pixel")

    print("All ZL config smoke tests: PASS")


if __name__ == "__main__":
    main()
