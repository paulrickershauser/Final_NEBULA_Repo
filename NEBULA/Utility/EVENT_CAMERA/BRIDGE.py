"""
step2_psf_bridge_imported.py
============================

This module implements **Step 2** of the NEBULA→EBS_SDA_SIM bridge by
converting NEBULA’s per-source photon fluxes into per‑pixel photon‑rate
frames and an attribution dictionary that are fully compatible with
Rachel Oliver’s event‑based sensor simulator.

Unlike a simple delta‑function PSF, this implementation calls
functions from Rachel’s `radiometry.py` to construct a sinc‑based
point‑spread function (PSF) and uses her attribution helper to build
the per‑frame pixel labels.  The goals of this module are:

* **Read NEBULA window data.**  We load `obs_window_sources.pkl`
  and `obs_zodiacal_light.pkl` from the NEBULA output directory.
* **Compute a diffraction‑limited PSF** using
  `RadiometryModel.central_point_source_sinc_func` defined in
  Rachel’s radiometry module【743779307818715†L2345-L2384】.  This
  function creates a complex field at the pupil by multiplying
  sinc functions in the x and y directions and scales the amplitude
  according to the wavelength, propagation distance and source power
  (photons s⁻¹ m⁻²).  The sinc pattern represents the diffraction
  limit for a uniformly illuminated rectangular aperture.
* **Scale the PSF to conserve energy** using
  `RadiometryModel.scale_field`【743779307818715†L4118-L4177】.  This
  method multiplies the field by a constant so that the integrated
  power in the image plane equals the collected power at the aperture.
  Energy conservation is critical when depositing PSFs onto the
  detector so that the total photon count matches the incident flux.
* **Add zodiacal light** by evaluating the plane‑3 model in
  `obs_zodiacal_light.pkl` on a normalised pixel grid.  The resulting
  per‑pixel photon rate is multiplied by the telescope collecting area,
  matching the radiometric conversion described in Rachel’s thesis【146279661363492†L4010-L4040】.
* **Generate an attribution dictionary** using Rachel’s
  `RadiometryModel.pixel_attribution` routine【743779307818715†L4179-L4247】.
  This function labels detector pixels where a source contributes at
  least 10 % of the limiting magnitude energy, matching the
  attribution format expected by her circuitry simulator.
* **Write the results** to an HDF5 file and a pickle file.  The
  HDF5 file contains datasets named `ph_flux_time_itr_000000` etc.,
  each storing a 2‑D array of photon rates (photons s⁻¹ per pixel).
  The naming convention follows the expectation of
  `circuitry.generate_events`【998765787415542†L98-L107】.

This module is intended to be run from within the `Utility/EVENT_CAMERA`
directory of the NEBULA project.  It depends on the presence of the
EBS_SDA_SIM source code.  The script dynamically adds the
`EBS_SDA_SIM-main/EBS_SDA_SIM` directory to `sys.path` so that it can
import Rachel’s `radiometry` module and use its functions directly.

Usage example (from the command line):

```
python step2_psf_bridge_imported.py \
    --nebula-src "../../NEBULA_OUTPUT/SCENE/obs_window_sources.pkl" \
    --nebula-zodi "../../NEBULA_OUTPUT/ZODIACAL_LIGHT/obs_zodiacal_light.pkl" \
    --ebs-dir "../../EBS_SDA_SIM-main/EBS_SDA_SIM" \
    --out-hdf5 "frames_psf_rachel.h5" \
    --out-pkl "att_dict_psf_rachel.pkl" \
    --kernel-size 21
```

This command produces an HDF5 file with PSF‑spread photon‑rate images
and a corresponding attribution dictionary using Rachel’s sinc‑based
PSF.  The kernel size controls the footprint of the PSF in pixels.

"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import h5py  # type: ignore
import numpy as np  # type: ignore


@dataclass
class SensorConfig:
    """Minimal sensor configuration for NEBULA→EBS_SDA_SIM bridging.

    This dataclass holds the optical parameters necessary to
    convert incident photon flux (photons s⁻¹ m⁻²) into total photon
    rate per source and to construct the PSF.  It mirrors the
    parameters defined in NEBULA’s `SensorConfig` dataclass【655218767494603†L45-L56】.

    Attributes
    ----------
    aperture_diameter_m : float
        Physical diameter of the telescope aperture in metres.
    focal_length_m : float
        Focal length of the optical system in metres.
    pixel_pitch_m : float
        Physical size of one detector pixel in metres.
    """

    aperture_diameter_m: float
    focal_length_m: float
    pixel_pitch_m: float

    @property
    def collecting_area_m2(self) -> float:
        """Return the effective collecting area of the aperture in m².

        The collecting area is computed as \(\pi (D/2)^2\).  In
        Rachel’s work this area is used to convert a flux density
        (photons s⁻¹ m⁻²) into total photon rate per source【146279661363492†L4010-L4040】.
        """
        radius = self.aperture_diameter_m / 2.0
        return float(np.pi * radius * radius)


def add_ebs_to_syspath(ebs_dir: Path) -> None:
    """Add the EBS_SDA_SIM source directory to `sys.path`.

    Parameters
    ----------
    ebs_dir : Path
        Path to the `EBS_SDA_SIM-main/EBS_SDA_SIM` directory.
    """
    ebs_path = str(ebs_dir.resolve())
    if ebs_path not in sys.path:
        sys.path.insert(0, ebs_path)


def load_ebs_radiometry(ebs_dir: Path):
    """Import and return Rachel’s radiometry module.

    This helper adds the EBS directory to `sys.path` and imports the
    `radiometry` module.  It returns the `RadiometryModel` class so
    that callers can instantiate it and access functions such as
    `central_point_source_sinc_func`, `scale_field` and
    `pixel_attribution`.

    Parameters
    ----------
    ebs_dir : Path
        Path to the directory containing `radiometry.py`.

    Returns
    -------
    RadiometryModel
        The class implementing Rachel’s radiometry functions.
    """
    add_ebs_to_syspath(ebs_dir)
    # Import the module after sys.path modification.
    import radiometry  # type: ignore
    return radiometry.RadiometryModel


def normalised_uv_grid(cols: int, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute NEBULA’s normalised \(u,v\) pixel grid.

    NEBULA defines the zodiacal light model on a grid of normalised
    coordinates in the range [-1, 1] along each axis.  This function
    constructs such a grid of shape `(rows, cols)` where each element
    is given by:

    .. math::
        u = \frac{2 x}{(N_x - 1)} - 1,
        \quad
        v = \frac{2 y}{(N_y - 1)} - 1.

    Parameters
    ----------
    cols : int
        Number of detector columns (x dimension).
    rows : int
        Number of detector rows (y dimension).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of shape `(rows, cols)` containing the u and v values.
    """
    x = np.arange(cols, dtype=np.float64)
    y = np.arange(rows, dtype=np.float64)
    xv, yv = np.meshgrid(x, y)
    u = (2.0 * xv / (cols - 1.0)) - 1.0
    v = (2.0 * yv / (rows - 1.0)) - 1.0
    return u, v


def evaluate_plane3(coeffs: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate a plane‑3 zodiacal model at each pixel.

    The plane‑3 model expresses zodiacal flux per pixel as

    .. math::
        \phi(u, v) = a_0 + a_1 u + a_2 v,

    where `(a0, a1, a2) = coeffs`.  The resulting value has units of
    photons s⁻¹ m⁻² per pixel (per unit collecting area) when the
    coefficients come from NEBULA’s zodiacal light stage.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (3,) containing `(a0, a1, a2)`.
    u : np.ndarray
        Normalised u grid.
    v : np.ndarray
        Normalised v grid.

    Returns
    -------
    np.ndarray
        Array of shape `(rows, cols)` giving \(\phi(u,v)\) values.
    """
    a0, a1, a2 = coeffs
    return a0 + a1 * u + a2 * v


def build_psf_kernel(
    rad_model: object,
    kernel_size: int,
    sensor: SensorConfig,
    wavelength_m: float,
    Dz_m: float,
    P_ref: float,
) -> np.ndarray:
    """Construct a normalised PSF kernel using Rachel’s sinc function.

    This function generates a PSF patch centred on the origin by
    calling `central_point_source_sinc_func` from Rachel’s
    radiometry module【743779307818715†L2345-L2384】.  The resulting
    complex field is converted to intensity, scaled to conserve
    energy using `scale_field`【743779307818715†L4118-L4177】, and
    normalised so that the sum of all elements is one.  The kernel
    is used to distribute a point source’s photon rate across
    neighbouring pixels.

    Parameters
    ----------
    rad_model : object
        An instance of `RadiometryModel` imported from Rachel’s
        radiometry code.
    kernel_size : int
        The width and height of the PSF patch in pixels (must be odd).
    sensor : SensorConfig
        Sensor parameters including pixel pitch and aperture diameter.
    wavelength_m : float
        Central wavelength of the passband in metres.
    Dz_m : float
        Propagation distance (focal length) in metres.
    P_ref : float
        Reference power per unit area to scale the field (photons s⁻¹ m⁻²).

    Returns
    -------
    np.ndarray
        A normalised PSF kernel of shape `(kernel_size, kernel_size)`.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    half = kernel_size // 2
    # Create a grid of physical distances from the centre in metres.
    coords = (np.arange(-half, half + 1, dtype=np.float64) * sensor.pixel_pitch_m)
    x1, y1 = np.meshgrid(coords, coords)
    r1 = np.sqrt(x1**2 + y1**2)
    # Convert scalar parameters to astropy quantities to satisfy radiometry API.
    import astropy.units as u  # type: ignore
    wvl = wavelength_m * u.m
    k = (2.0 * np.pi / wavelength_m) * (1.0 / u.m)
    Dz = Dz_m * u.m
    D2 = sensor.aperture_diameter_m * u.m
    P = P_ref * (u.ph / (u.s * u.m**2))
    # Compute the complex field using Rachel’s sinc function.
    Ur = rad_model.central_point_source_sinc_func(
        x1 * u.m, y1 * u.m, r1 * u.m, Dz, D2, wvl, k, P
    )
    # Convert to intensity (photons/s/m^2 per m^2) by multiplying with its conjugate
    intensity = (Ur * np.conjugate(Ur)).real
    # Multiply by pixel area to get photons/s per pixel before scaling
    dx = sensor.pixel_pitch_m
    dy = sensor.pixel_pitch_m
    intensity *= (dx * dy)
    # Use Rachel’s scale_field to enforce energy conservation.  The
    # reference power is P * collecting_area (total photons/s collected),
    # and the power of the raw PSF is sum(intensity).
    P_total = P * sensor.collecting_area_m2
    power_aperature = P_total  # photons/s at aperture
    power_image = intensity.sum() * (u.ph / u.s)
    scaled_field, _ = rad_model.scale_field(
        Ur, dx * u.m, dy * u.m, power_aperature, power_image
    )
    scaled_intensity = (scaled_field * np.conjugate(scaled_field)).real * (dx * dy)
    # Normalise the kernel so its sum is one.  This ensures that
    # scaling by the source photon rate yields correct per-pixel
    # photon rates.
    kernel = scaled_intensity / np.sum(scaled_intensity)
    return np.array(kernel, dtype=np.float64)


def process_frames_with_psf(
    window_sources: Dict,
    zodi_data: Dict,
    sensor: SensorConfig,
    rad_model: object,
    kernel: np.ndarray,
    limiting_mag_g: float,
    out_hdf5: Path,
    out_pkl: Path,
    max_frames: int | None = None,
) -> None:
    """Process NEBULA sources and write PSF‑spread photon‑rate frames.

    This function loops over each time frame in the NEBULA window,
    constructs a photon‑rate image by adding contributions from
    zodiacal light, stars and targets, applies the PSF kernel around
    each source’s pixel position and builds an attribution dictionary
    using Rachel’s `pixel_attribution` helper【743779307818715†L4179-L4247】.

    Parameters
    ----------
    window_sources : dict
        The loaded contents of `obs_window_sources.pkl` for a
        particular observer and window.  This dict must contain
        `windows[0]['stars']`, `windows[0]['targets']` and
        `windows[0]['n_frames']` among other keys.
    zodi_data : dict
        The loaded contents of `obs_zodiacal_light.pkl` for the same
        observer.  Must contain plane‑3 zodiacal coefficients.
    sensor : SensorConfig
        Sensor parameters used for radiometric conversion and grid
        spacing.
    rad_model : object
        An instance of `RadiometryModel`.  Its methods
        `pixel_attribution`, `central_point_source_sinc_func` and
        `scale_field` are used.
    kernel : np.ndarray
        Precomputed PSF kernel normalised to sum to one.
    limiting_mag_g : float
        Limiting magnitude used to compute the reference power for
        attribution.  The irradiance corresponding to this magnitude
        is converted to photons s⁻¹ m⁻² via
        `RadiometryModel.vmag_2_irradiance_Gia_G` and multiplied by
        the collecting area【146279661363492†L4010-L4040】.
    out_hdf5 : Path
        Path to the HDF5 file to create.
    out_pkl : Path
        Path to the pickle file to write the attribution dictionary.
    max_frames : int or None, optional
        If provided, only process the first `max_frames` frames.
    """
    # Extract window and source information
    # The NEBULA pickle is keyed by observer name; select the first track
    # dictionary and then index its windows list.  See the NEBULA
    # pickle structure description.
    track_dict = window_sources if "windows" in window_sources else next(iter(window_sources.values()))
    window = track_dict["windows"][0]
    n_frames = int(window["n_frames"])
    rows = int(track_dict["rows"])
    cols = int(track_dict["cols"])
    dt_frame_s = float(track_dict["dt_frame_s"])
    stars = window["stars"]
    targets = window["targets"]

    # Limit number of frames if requested
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    # Precompute zodiacal grid and reference power for attribution
    u_grid, v_grid = normalised_uv_grid(cols, rows)
    
    # Extract zodiacal coefficients.  The zodiacal pickle has the same
    # observer‑keyed structure as the window pickle: get the first
    # observer entry and then index its windows list.
    zodi_track = zodi_data if "windows" in zodi_data else next(iter(zodi_data.values()))
    zodi_coeffs = zodi_track["windows"][0]["zodi"]["models"]["plane3"]["coeffs_per_pixel"]

    # Compute reference irradiance and power for limiting magnitude
    import astropy.units as u  # type: ignore
    # Rachel's vmag_2_irradiance_Gia_G expects three arguments:
    # magnitude, bandwidth and aperture diameter:contentReference[oaicite:1]{index=1}.
    # Pass a nominal bandwidth of 1 nm and the telescope diameter.
    bandwidth = 1.0 * u.nm
    aperture_diam = sensor.aperture_diameter_m * u.m
    irradiance_lim = rad_model.vmag_2_irradiance_Gia_G(limiting_mag_g, bandwidth, aperture_diam)
    power_ref = irradiance_lim * sensor.collecting_area_m2 #* (u.ph / u.s)

    # Prepare output files
    with h5py.File(out_hdf5, "w") as hf:
        attribution_dict: Dict = {}
        # Use tqdm for progress reporting if available.  If tqdm is not
        # installed, fall back to a simple range iterator.  This mirrors
        # the progress bar usage in earlier bridging scripts and gives
        # feedback on long runs without affecting functionality.
        try:
            from tqdm import tqdm  # type: ignore
            frame_iter: Iterable[int] = tqdm(
                range(n_frames),
                desc="Building frames with PSF",
                unit="frame",
            )
        except Exception:
            frame_iter = range(n_frames)
        # Iterate over frames
        for i in frame_iter:
            frame = np.zeros((rows, cols), dtype=np.float64)
            # Add zodiacal background for this frame
            coeffs = zodi_coeffs[i]
            phi_zodi = evaluate_plane3(coeffs, u_grid, v_grid)
            frame += phi_zodi * sensor.collecting_area_m2
            # Prepare attribution map for this time
            irr_assoc: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
            # Helper function to deposit a PSF patch for each source
            def deposit_source(
                    source_name: str,
                    photon_rate: float,
                    x_pix: float,
                    y_pix: float,
                ) -> None:
                    """Deposit a PSF‑scaled contribution onto the frame and update
                    attribution.
        
                    This helper adds a single source’s photon flux to the image frame
                    using the precomputed PSF kernel.  It also constructs a
                    contribution field with appropriate physical units before
                    passing it to Rachel’s ``pixel_attribution``.  Without units,
                    ``pixel_attribution`` will compute an ``Image_Watts`` array
                    with units derived solely from the pixel dimensions, leading
                    to a mismatch with the reference power and triggering a
                    ``UnitConversionError``.  By attaching units to the photon
                    rate and pixel dimensions here, the ratio used in
                    ``pixel_attribution`` becomes dimensionless and the comparison
                    against a numeric threshold works correctly.
                    """
                    nonlocal frame, irr_assoc
                    # Round source location to the nearest pixel
                    x_int = int(round(x_pix))
                    y_int = int(round(y_pix))
                    half = kernel.shape[0] // 2
                    # Compute patch bounds on the detector
                    x0 = x_int - half
                    y0 = y_int - half
                    x1 = x0 + kernel.shape[1]
                    y1 = y0 + kernel.shape[0]
                    # Compute the overlapping region indices
                    kx0 = max(0, -x0)
                    ky0 = max(0, -y0)
                    kx1 = kernel.shape[1] - max(0, x1 - cols)
                    ky1 = kernel.shape[0] - max(0, y1 - rows)
                    fx0 = max(0, x0)
                    fy0 = max(0, y0)
                    fx1 = min(cols, x1)
                    fy1 = min(rows, y1)
                    # Extract the kernel subset and scale by photon rate
                    sub_k = kernel[ky0:ky1, kx0:kx1]
                    frame[fy0:fy1, fx0:fx1] += sub_k * photon_rate
                    # Build attribution: create a zero array and insert the scaled
                    # kernel contribution, then attach photon flux units (photons/s)
                    contrib = np.zeros((rows, cols), dtype=np.float64)
                    contrib[fy0:fy1, fx0:fx1] = sub_k * photon_rate
                    contrib_qty = contrib * (u.ph / u.s)
                    # Pixel dimensions with units
                    dx_qty = sensor.pixel_pitch_m * u.m
                    dy_qty = sensor.pixel_pitch_m * u.m
                    # Compute intensity (photons/s/m²) and take the square root to
                    # obtain the field amplitude with correct units.  The small
                    # constant prevents division by zero while preserving units.
                    intensity_qty = contrib_qty / (dx_qty * dy_qty + 1e-30 * u.m**2)
                    Uin_qty = np.sqrt(intensity_qty)
                    # Call Rachel’s pixel_attribution with quantities so that the
                    # ratio Image_Watts / power_ref is dimensionless.
                    irr_assoc = rad_model.pixel_attribution(
                        Uin_qty, irr_assoc, dx_qty, dy_qty, power_ref, source_name
                    )


            # Add contributions from stars
            for sid, info in stars.items():
                if info["on_detector"][i]:
                    ph_rate = float(info["phi_ph_m2_s"][i]) * sensor.collecting_area_m2
                    deposit_source(f"star:{sid}", ph_rate, info["x_pix"][i], info["y_pix"][i])
            # Add contributions from targets (sparse time series)
            for tid, tinfo in targets.items():
                coarse_idx = tinfo["coarse_indices"]
                matches = np.where(coarse_idx == i)[0]
                for j in matches:
                    ph_rate = float(tinfo["phi_ph_m2_s"][j]) * sensor.collecting_area_m2
                    deposit_source(f"target:{tid}", ph_rate, tinfo["x_pix"][j], tinfo["y_pix"][j])
            # Write the frame to HDF5 with zero‑padded index【998765787415542†L98-L107】
            dset_name = f"ph_flux_time_itr_{i:06d}"
            hf.create_dataset(dset_name, data=frame.astype(np.float32), compression="gzip")
            # Store attribution for this time (use seconds since start)
            time_key = i * dt_frame_s
            attribution_dict[time_key] = irr_assoc
        # After loop, pickle the attribution dictionary
        with open(out_pkl, "wb") as pf:
            pickle.dump(attribution_dict, pf)


def main() -> None:
    """Command-line interface for the PSF bridge script.

    This function parses arguments and orchestrates the processing of
    NEBULA window data into EBS_SDA_SIM-compatible radiometry outputs.
    It imports Rachel’s radiometry module from the specified EBS
    directory, builds a PSF kernel based on sensor parameters, and
    writes an HDF5 frames file and attribution pickle.
    """
    parser = argparse.ArgumentParser(description="Convert NEBULA sources to EBS radiometry using Rachel’s PSF")
    parser.add_argument("--nebula-src", required=True, help="Path to obs_window_sources.pkl")
    parser.add_argument("--nebula-zodi", required=True, help="Path to obs_zodiacal_light.pkl")
    parser.add_argument("--ebs-dir", required=True, help="Path to EBS_SDA_SIM-main/EBS_SDA_SIM directory")
    parser.add_argument("--out-hdf5", required=True, help="Path for output HDF5 file")
    parser.add_argument("--out-pkl", required=True, help="Path for attribution pickle")
    parser.add_argument("--kernel-size", type=int, default=21, help="Odd size of the PSF kernel in pixels")
    parser.add_argument("--wavelength-m", type=float, default=550e-9, help="Central wavelength in metres (default 550nm)")
    parser.add_argument("--limiting-mag", type=float, default=13.5, help="Limiting G-band magnitude for attribution")
    args = parser.parse_args()
    # Load NEBULA pickles
    nebula_src = Path(args.nebula_src).resolve()
    nebula_zodi = Path(args.nebula_zodi).resolve()
    with open(nebula_src, "rb") as f:
        window_sources = pickle.load(f)
    with open(nebula_zodi, "rb") as f:
        zodi_data = pickle.load(f)
    # Import radiometry module
    rad_model_class = load_ebs_radiometry(Path(args.ebs_dir))
    rad_model = rad_model_class()
    # Define sensor parameters (Gen‑3 VGA-CD sensor from NEBULA)
    sensor = SensorConfig(
        aperture_diameter_m=0.085 / 1.4,
        focal_length_m=0.085,
        pixel_pitch_m=15e-6,
    )
    # Build PSF kernel using Rachel’s function
    # Use a unit‑tagged radiance (photons per second per m²) for P_ref
    import astropy.units as u  # type: ignore
    kernel = build_psf_kernel(
        rad_model,
        kernel_size=args.kernel_size,
        sensor=sensor,
        wavelength_m=args.wavelength_m,
        Dz_m=sensor.focal_length_m,
        P_ref=1.0 * (u.ph / (u.s * u.m**2)),
    )
    # Write frames and attribution
    process_frames_with_psf(
        window_sources=window_sources,
        zodi_data=zodi_data,
        sensor=sensor,
        rad_model=rad_model,
        kernel=kernel,
        limiting_mag_g=args.limiting_mag,
        out_hdf5=Path(args.out_hdf5),
        out_pkl=Path(args.out_pkl),
    )


if __name__ == "__main__":
    import sys
    sys.argv = [
        sys.argv[0],
        "--nebula-src", r"C:\Users\prick\Desktop\Research\NEBULA\NEBULA_OUTPUT\SCENE\obs_window_sources.pkl",
        "--nebula-zodi", r"C:\Users\prick\Desktop\Research\NEBULA\NEBULA_OUTPUT\ZODIACAL_LIGHT\obs_zodiacal_light.pkl",
        "--ebs-dir", r"C:\Users\prick\Desktop\Research\NEBULA\EBS_SDA_SIM-main\EBS_SDA_SIM",
        "--out-hdf5", r"C:\Users\prick\Desktop\Research\NEBULA\NEBULA_OUTPUT\EBS_SIM\frames_psf_rachel.h5",
        "--out-pkl", r"C:\Users\prick\Desktop\Research\NEBULA\NEBULA_OUTPUT\EBS_SIM\att_dict_psf_rachel.pkl",
    ]
    main()
