# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 07:44:25 2021

@author: rache
"""

import numpy as np
from numpy import inf
import h5py
import os
import pdb
import pickle
import scipy.optimize as spo
import scipy.special as sps
import astropy.constants as constants
import scipy.linalg as splinalg
import scipy.interpolate as interpolate
import scipy.fft as fft
import matplotlib.pyplot as plt
import skimage.transform as tr
from matplotlib import ticker
from skyfield.api import load
from astropy import units as u
import synphot


class RadiometryModel(object):
    """
    This class contains all variables and functions needed to perform the
    calculation of incident energy on the event-based sensor pixel.

    Args:
        kwargs:
            user specified values, specifically satellite information


    Attributes:


    """

    def __init__(self, source="point", **kwargs):
        self.source = source
        # This selection will determine the subsequent equations to calculate
        # irradiance on the sensor. Automatically selects point, no arguement
        # is given

        return

    def __str__(self):
        """
        String Representation of the Radiometric Outputs

        When the command "print" is used this method will return the values
        contained in the object

        """
        return

    def calc_sat_point_intensity(self, L, A_surf, rho):
        """
        Calculate the intensity of a point source. Watts sent per steradian.

        Parameters
        ----------
        L : float
            Incident energy on an object that will have some portion reflected in the relevant hemisphere of the observer. [W/m^2]
        A_surf : list
            Area of object reflecting the indicdent energy at a particular reflectivity. [m^2]
        rho : list
            Reflectivity of object reflecting incident energy.

        Returns
        -------
        I : float
            Energy sent into hemisphere assuming a lambertian reflection. [W/sr]

        """
        # Total Energy reflected by the satellite
        Phi = np.sum(L * np.multiply(A_surf, rho))  # [W]
        # Assume satellite is an isotropic point source only reflecting into this hemisphere
        I = Phi / (2 * np.pi)  # [photon/sr/s]
        return I

    def calc_sat_point_irradiance(self, I, R, D, thetaR=0):
        """
        Calculate the irradiance of a satellite at the distance of Earth.

        Parameters
        ----------
        I : float
            Intensity of incoming light. [W/sr]
        R : float
            Propagation distance between satellite and observer. [m]
        D : float
            Diameter of the collecting aperature. [m]
        thetaR : float, optional
            Off axis angle of propagation. [rad] The default is 0.

        Returns
        -------
        E_lambda : float
            Irradiance of the satellite at the distance of the observer. [W/m^2]

        """
        # Total energy recieved per steradian, following reverse square law
        E_lambda = (I * np.cos(thetaR)) / R**2  # Irradiance or the flux
        # Multiply by area pf interest to calculate power revieved
        Area_lobe = (D / 2) ** 2 * np.pi  # Area collecting aperature (m^2)
        sat_power = E_lambda * Area_lobe
        return E_lambda

    def vmag_2_irradiance_Johnson_V(self, vmags, D):
        """
        Calculate the irradiance of a star based on its visual magnitude at the
        distance of Earth.

        Parameters
        ----------
        vmags : float
            Visual magnitude of the modeled star given by a star catalog.
        D : float
            Diameter of the collecting aperature. [m]

        Returns
        -------
        star_irradiance : float
            Irradiance of the star. [W/cm^2]

        """
        # The hipparcos catalog magnitudes are Johnson V
        # Using the calculations outlined in Astronomical Photometry by Arne Haden and Ronald Kaitchuck
        # The V magnitudes are converted to flux via the band's zero point at 5500 Angstroms.
        # By multiplying by the bandwidth 1000 Angstroms and the collecting area for the propogation (with diameter = DROI)
        # the power to define the point source in the propagation is met.
        Flux = (
            10 ** (-0.4 * (np.array(vmags) + 38.52)) * u.W / (u.cm**2 * u.Angstrom)
        )  # Watts/(cm^2*Angstrom)
        Bandwidth = (
            880 * u.Angstrom
        )  # Angstrom, the Vband FWHM is 88 nm or 880 angstroms
        Area_lobe = (D / 2) ** 2 * np.pi  # Area collecting aperature (m^2)
        # cm2_m2 = 100*100 # Unit conversion between 1/cm^2 to 1/m^2
        star_power = Flux * Bandwidth * Area_lobe  # cm2_m2
        # Irradiance is not multiplied by the area of the lobe
        star_irradiance = Flux * Bandwidth
        # Ensure units in W/m^2
        star_irradiance = star_irradiance.to(u.W / (u.m**2))
        return star_irradiance

    def vmag_2_irradiance_Gia_G(self, vmags, bandwidth, D):
        ## Fix Me: Add star attenuation factor due to Gaia passband
        # Use the Vega Magnitude system for Gia, so start with the zero magnitude flux
        Vega_Zero_Mag_Flux = 3.016e3 * u.Jy
        # Use it to scale the flux value produced by the star magnitudes
        Flux = Vega_Zero_Mag_Flux * 10 ** (-0.4 * np.array(vmags))
        # Convert the Jansky units into Photons per second per cm squared per Angstrom
        effective_filter_wavelength = 0.673 * u.um
        Flux = synphot.units.convert_flux(
            effective_filter_wavelength, Flux, synphot.units.PHOTLAM
        )
        # Irradiance is not multiplied by the area of the lobe
        star_irradiance = (Flux * bandwidth).to(u.ph / (u.s * u.m**2))
        # # Ensure units in W/m^2
        # star_irradiance = star_irradiance.to(u.W/(u.m**2))
        return star_irradiance

    def frame_generation_all_steps(
        self,
        prop_parameters,
        observer,
        sat_props,
        sat_obs,
        sat_array,
        t_array,
        sim_name="filename",
        pscrn_file_name="filename",
        watt_file_name=0,
        plot_psf=False,
        time_slice=False,
        time_slice_length=0,
    ):

        # Unpack Propagation Parameters
        nscrns = prop_parameters["nscrns"]
        N = prop_parameters["N"]
        N_total_n = prop_parameters["N_total_n"]
        N_total_pscrn = prop_parameters["N_total_pscrn"]
        z = prop_parameters["z"]
        L = prop_parameters["L"]
        delta_z = prop_parameters["delta_z"]
        delta_n = prop_parameters["delta_n"]  # Desired grid spacing
        delta_n_pscrn = prop_parameters["deltan_pscrn"]  # Phase screen grid spacing
        scale_pscrn = prop_parameters[
            "scale_pscrn"
        ]  # Ratio of desired grid spacing to phase screen spacing
        timescale = prop_parameters["timescale"]
        time_check = timescale
        psn_itr = 0

        # Calculate grid spacing for aperature projection for finding
        # the right subsections of the extended phase screens
        # For the desired grid spacing
        D_fpga_proj = (
            2
            * (np.flip(z) + observer.focal_length)
            * np.tan(np.deg2rad(observer.AFOV.value) / 2)
        )  # Diameter of optic projected
        slope = z[-1] / (D_fpga_proj[0] / 2 - observer.aperature_diameter / 2)
        Dn = observer.aperature_diameter + 2 * z / slope
        N_total_aperature_prop = np.ceil(np.flip(Dn) / delta_n)
        for i in np.arange(0, nscrns + 2):
            if N_total_aperature_prop[0, i] % 2 != 0:
                N_total_aperature_prop[0, i] = N_total_aperature_prop[0, i] + 1
        # Define the center of the propagation planes
        xc_prop = N_total_n / 2
        yc_prop = N_total_n / 2
        # Same calculations for the phase screen spacing, but for each set
        N_total_aperature_prop_pscrn = {}
        xc_pscrn = {}
        yc_pscrn = {}
        for i, psn_key in enumerate(list(N_total_pscrn.keys())):
            N_total_aperature_prop_pscrn_itr = np.ceil(
                np.flip(Dn)[1 : nscrns + 1] / np.stack(delta_n_pscrn[psn_key], axis=0)
            )
            for j in np.arange(0, nscrns):
                if N_total_aperature_prop_pscrn_itr[j] % 2 != 0:
                    N_total_aperature_prop_pscrn_itr[j] = (
                        N_total_aperature_prop_pscrn_itr[j] + 1
                    )
            N_total_aperature_prop_pscrn[psn_key] = N_total_aperature_prop_pscrn_itr
            # Define the center of the phase screen planes
            xc_pscrn[psn_key] = np.array(N_total_pscrn[psn_key]) / 2
            yc_pscrn[psn_key] = np.array(N_total_pscrn[psn_key]) / 2
        del D_fpga_proj
        del slope
        del Dn
        del N_total_aperature_prop_pscrn_itr
        # Define the center of the detector plane
        xc = observer.num_x_pix / 2
        yc = observer.num_y_pix / 2

        # Make (x,y) grid for the detector plots
        x_vector = (
            np.arange(0, observer.num_x_pix, 1, dtype=np.float32) * observer.pixel_width
        )
        y_vector = (
            np.arange(0, observer.num_y_pix, 1, dtype=np.float32)
            * observer.pixel_height
        )
        xdet, ydet = np.meshgrid(x_vector, y_vector, copy=False)

        # Use preselected observation plane sampling of the aperature to define the
        # observation (pupil) plane and aperature mask
        Np = observer.observation_plane_sampling
        delta_p = observer.aperature_diameter / Np
        x_vector = np.arange(-Np / 2 + 0.5, Np / 2 + 0.5, 1, dtype=np.float32) * delta_p
        y_vector = np.arange(-Np / 2 + 0.5, Np / 2 + 0.5, 1, dtype=np.float32) * delta_p
        xp, yp = np.meshgrid(x_vector, y_vector, copy=False)
        pupil_mask = self.circ(xp, yp, observer.aperature_diameter)
        # Define the airy disk plane
        if observer.sufficient_airy_disk_discretization == False:
            Nax = int(
                observer.subpixel_per_pixel_x_direction
                * observer.image_plane_pixels_simulated_x_direction
            )
            Nay = int(
                observer.subpixel_per_pixel_y_direction
                * observer.image_plane_pixels_simulated_y_direction
            )
            delta_ax = observer.pixel_width / observer.subpixel_per_pixel_x_direction
            delta_ay = observer.pixel_height / observer.subpixel_per_pixel_y_direction
            subpix_per_pix_x = observer.subpixel_per_pixel_x_direction
            subpix_per_pix_y = observer.subpixel_per_pixel_y_direction
        else:
            Nax = observer.image_plane_pixels_simulated_x_direction
            Nay = observer.image_plane_pixels_simulated_y_direction
            delta_ax = observer.pixel_width
            delta_ay = observer.pixel_height
            subpix_per_pix_x = 0
            subpix_per_pix_y = 0
        x_vector = (
            np.arange(-Nax / 2 + 0.5, Nax / 2 + 0.5, 1, dtype=np.float32) * delta_ax
        )
        y_vector = (
            np.arange(-Nay / 2 + 0.5, Nay / 2 + 0.5, 1, dtype=np.float32) * delta_ay
        )
        xa, ya = np.meshgrid(x_vector, y_vector, copy=False)

        # Make a polar coordinate grid for the source plane for defining the initial point sources
        x_vector = (
            np.arange(-N / 2 + 0.5, N / 2 + 0.5, 1) * delta_n[0, 0]
        )  # Does not need to be full size of propagation distance, just the size of the subset for the point source definition
        y_vector = np.arange(-N / 2 + 0.5, N / 2 + 0.5, 1) * delta_n[0, 0]
        x1, y1 = np.meshgrid(x_vector, y_vector, copy=False)
        r1, theta1 = self.cart2pol(x1, y1)
        del theta1
        del x_vector
        del y_vector

        # Create an Observation Plane Mask for Each Phase Screen
        # Since each phase screen is the same dimension, N, the super gaussian
        # is the same because it is based on percentage of array not actual dimensions
        # np.arange(-N/2,N/2,1)*delta_n[0,0]
        sg = self.super_gaussian(x1, y1, N, N, delta_n[0, 0], delta_n[0, 0])

        # Calculate intensity of satellite point sources assuming they are constant
        I_sat = {}
        # Make calculation for each satellite in the FOV
        for i, sat_key in enumerate(list(sat_props.keys())):
            A_surf = sat_props[sat_key]["satellite approx projected area"]
            rho = np.array(sat_props[sat_key]["surface rho properties"])
            # Calculate Point Source Power with Equations
            # FIXME: Convert to Photons/s/m^2 and move this into an optical system parameter eventually (Not high priority to move to other part of sim)
            # E_ext = 1366 * u.W / u.m**2  # W/m^2 Solar Irradiance
            E_ext = 3.2681601435982546e18 * u.ph/(u.s*u.m**2) # Photons/s/m^2 Solar Irradiance
            I = self.calc_sat_point_intensity(E_ext, A_surf, rho)
            I_sat[sat_key] = I

        # Calculate the minimum power collected by the aperature to generate events via
        # the limiting magnitude
        irradiance_min = self.vmag_2_irradiance_Gia_G(
            observer.limiting_magnitude, observer.bandpass, observer.aperature_diameter
        )
        power_min = (
            irradiance_min * (observer.aperature_diameter / 2) ** 2 * np.pi
        )  # Before QE*T is applied, but that is applied to all values in the circuitry code

        # Load earth from Skyfield to retrieve Ra and Dec
        planets = load("de421.bsp")
        earth = planets["earth"]

        # Create an HDF5 File to store the PSFs
        directory = os.getcwd()
        if watt_file_name == 0:
            while isinstance(watt_file_name, str) == False:
                watt_file_name = input(
                    "Please input a file name for the file that will hold the final images: "
                )
        newdirectory = directory + "/" + sim_name + "/FinalImageFields/"
        if not os.path.exists(newdirectory):
            os.makedirs(newdirectory)
        full_name = newdirectory + watt_file_name + ".hdf5"
        # Initialize HDF5 File
        hf = h5py.File(full_name, "w")
        hf.close()

        # Create a dictionary to store the different timesteps, attribuation dictionaries
        attribution_dict = {}

        # Calculate timestep
        t_total = t_array.utc[:, 1] - t_array.utc[:, 0]
        # Convert difference into seconds
        if t_total[1] != 0:
            # If there are months in the time object covert it to days
            t_total_days = self.month2day(t_total)
            t_step = np.ceil(
                (
                    (
                        t_total[0] * u.year
                        + t_total_days
                        + t_total[3] * u.hour
                        + t_total[4] * u.minute
                        + t_total[5] * u.second
                    )
                    / timescale
                ).decompose()
            )
        else:
            # Add the total time and divide by the timescale for each phase screen
            # to obtain the total number of iterations of phase screens needed
            t_step = np.ceil(
                (
                    (
                        t_total[0] * u.year
                        + t_total[2] * u.day
                        + t_total[3] * u.hour
                        + t_total[4] * u.minute
                        + t_total[5] * u.second
                    )
                    / timescale
                ).decompose()
            )

        for i, time in enumerate(t_array):

            # Check which set of phase screens should be utilized
            t_total = t_array.utc[:, i] - t_array.utc[:, 0]
            # Convert difference into seconds
            if t_total[1] != 0:
                # If there are months in the time object covert it to days
                t_total_days = self.month2day(t_total)
                t_total_sec = (
                    t_total[0] * u.year
                    + t_total_days
                    + t_total[3] * u.hour
                    + t_total[4] * u.minute
                    + t_total[5] * u.second
                ).decompose()
            else:
                # Add the total time and divide by the timescale for each phase screen
                # to obtain the total number of iterations of phase screens needed
                t_total_sec = (
                    t_total[0] * u.year
                    + t_total[2] * u.day
                    + t_total[3] * u.hour
                    + t_total[4] * u.minute
                    + t_total[5] * u.second
                ).decompose()
            if t_total_sec > time_check:
                psn_itr += 1
                time_check += timescale

            if time_slice == True:
                # Only produce frames once for each time slice duration
                # These frames will be used repeatedly in the circuit analysis
                if i == 0:
                    # initialize time slice check
                    time_slice_check = 0
                else:
                    time_slice_check += t_step
                    if time_slice_check < time_slice_length:
                        continue
                    else:
                        # Reset the check
                        time_slice_check = 0

            # Create a dictionary to store the association of irradiance to a
            # particular point source
            irradiance_association = {}

            # Pull out properties and point sources unique to this timestep
            if isinstance(observer.optics_star_dict[i], (int, np.integer)) == True:
                star_idx = observer.optics_star_dict[i]
                star_objs = observer.optics_star_dict[star_idx]
            else:
                star_objs = observer.optics_star_dict[i]

            star_mags = observer.optics_star_mag_dict[i]
            star_ids = observer.optics_star_id_dict[i]
            wcs_proj = observer.optics_wcs_dict[i]
            detector_ra_dec = observer.optics_pointing_dict[i]

            # pdb.set_trace()
            # Calculate the Ra/Dec positions of stars at this timestep
            star_ras, star_decs, star_dists = earth.at(time).observe(star_objs).radec()
            # Calculate the flux sent by stars at this timestep
            star_fluxes = self.vmag_2_irradiance_Gia_G(
                star_mags, observer.bandpass, observer.aperature_diameter
            )

            for j in np.arange(0, len(star_mags) + len(list(sat_props.keys()))):

                # Ra and Dec and Power of the point source are saved differently for stars and satellites
                if j >= len(star_mags):
                    # Pull out one satellite at a time
                    sat_number = j - len(star_mags)
                    sat_key = list(sat_props.keys())[sat_number]
                    # Check if this satellite is observable at this time step
                    if sat_obs[sat_key][i] != True:
                        continue
                    source_name = sat_key
                    # Replace slashes and spaces with underscores in the source name
                    source_name = source_name.replace("/", "_")
                    source_name = source_name.replace("\\", "_")
                    source_name = source_name.replace(" ", "_")
                    # This propagation iteration, use the satellite
                    ra_vec = (
                        np.linspace(
                            sat_array[sat_key][i][0].value,
                            detector_ra_dec[0].value,
                            nscrns + 2,
                        )
                        * detector_ra_dec[0].unit
                    )
                    dec_vec = (
                        np.linspace(
                            sat_array[sat_key][i][1].value,
                            detector_ra_dec[1].value,
                            nscrns + 2,
                        )
                        * detector_ra_dec[1].unit
                    )
                    # Calculate the angle between sensor pointing and satellite
                    cart_sat = [
                        np.sin(np.pi * u.rad - np.deg2rad(sat_array[sat_key][i][1]))
                        * np.cos(np.deg2rad(sat_array[sat_key][i][0])),
                        np.sin(np.pi * u.rad - np.deg2rad(sat_array[sat_key][i][1]))
                        * np.sin(np.deg2rad(sat_array[sat_key][i][0])),
                        np.cos(np.pi * u.rad - np.deg2rad(sat_array[sat_key][i][1])),
                    ]
                    cart_det = [
                        np.sin(np.pi * u.rad - np.deg2rad(detector_ra_dec[1]))
                        * np.cos(np.deg2rad(detector_ra_dec[0])),
                        np.sin(np.pi * u.rad - np.deg2rad(detector_ra_dec[1]))
                        * np.sin(np.deg2rad(detector_ra_dec[0])),
                        np.cos(np.pi * u.rad - np.deg2rad(detector_ra_dec[1])),
                    ]
                    theta_r = np.arccos(np.dot(cart_sat, cart_det))

                    # Define the point source
                    # The power for this point
                    power = self.calc_sat_point_irradiance(
                        I_sat[sat_key],
                        sat_array[sat_key][i][2],
                        observer.aperature_diameter,
                        theta_r,
                    )
                else:
                    # The point source is a star
                    source_name = star_ids[j]
                    # For each star find the center of the phase screens (ra,dec)
                    ra_vec = (
                        np.linspace(
                            star_ras._degrees[j], detector_ra_dec[0].value, nscrns + 2
                        )
                        * detector_ra_dec[0].unit
                    )
                    dec_vec = (
                        np.linspace(
                            star_decs._degrees[j], detector_ra_dec[1].value, nscrns + 2
                        )
                        * detector_ra_dec[0].unit
                    )
                    # Define the point source
                    # The power for this point
                    power = star_fluxes[j]

                # Covert (ra,dec) values to (x,y) locations on the focal plane
                ra_dec_vec = np.transpose(np.stack((ra_vec, dec_vec)))
                x_y_detector_vec = wcs_proj.wcs_world2pix(ra_dec_vec, 0)

                x_y_prop_vec = []
                x_y_pscrn_vec = []
                for l, pixel in enumerate(x_y_detector_vec):
                    # Calculate ratio of focal plane the point is from the center
                    x_ratio = (pixel[0] - xc) / observer.num_x_pix
                    y_ratio = (pixel[1] - yc) / observer.num_y_pix
                    # Calculate the (x,y) cell in the projected plane
                    x_pix = (x_ratio * N_total_aperature_prop[0, l]) + xc_prop[
                        0, l
                    ]  # If the (x,y) position is in a pixel of the phase screen, we want the floor to capture that volume.
                    y_pix = (y_ratio * N_total_aperature_prop[0, l]) + yc_prop[0, l]
                    x_y_prop_vec.append((x_pix, y_pix))
                    # Calculate the (x,y) cell in the phase screen
                    if (
                        l != 0 and l != len(z) - 1
                    ):  # source plane and observation plane do not have phase screens
                        x_pix = (
                            x_ratio * N_total_aperature_prop_pscrn[psn_itr][l - 1]
                        ) + xc_pscrn[psn_itr][
                            l - 1
                        ]  # If the (x,y) position is in a pixel of the phase screen, we want the floor to capture that volume.
                        y_pix = (
                            y_ratio * N_total_aperature_prop_pscrn[psn_itr][l - 1]
                        ) + yc_pscrn[psn_itr][l - 1]
                        x_y_pscrn_vec.append((x_pix, y_pix))
                # The first tuple in the (x,y) propagation vector will help define the point source
                rp, thp = self.cart2pol(
                    x_y_prop_vec[0][0] * delta_n[0, 0],
                    x_y_prop_vec[0][1] * delta_n[0, 0],
                )
                # pdb.set_trace()
                # Create the Point Source, use central source and recenter on the (x,y) projection pixel after the propagation
                if i == 0:
                    Uin = self.central_point_source_sinc_func(
                        x1,
                        y1,
                        r1,
                        L,
                        observer.aperature_diameter,
                        observer.wavelength,
                        observer.wavenumber,
                        power,
                        time=t_total_sec,
                        source_name=str(source_name),
                        sim_name=sim_name,
                        plot_source=plot_psf,
                    )
                else:
                    Uin = self.central_point_source_sinc_func(
                        x1,
                        y1,
                        r1,
                        L,
                        observer.aperature_diameter,
                        observer.wavelength,
                        observer.wavenumber,
                        power,
                        time=t_total_sec,
                        source_name=str(source_name),
                        sim_name=sim_name,
                        plot_source=False,
                    )
                # Run this point source through the split step propagation
                xn, yn, Uout = self.angular_spec_multi_prop(
                    Uin,
                    observer.wavenumber,
                    delta_n,
                    delta_n_pscrn,
                    scale_pscrn,
                    z,
                    delta_z,
                    sg,
                    pscrn_file_name,
                    psn_itr,
                    x_y_prop_vec,
                    x_y_pscrn_vec,
                    xc_prop,
                    yc_prop,
                )

                # Plot observation fields if requested
                if plot_psf == True and i == 0:
                    self.plot_field(
                        Uout,
                        xn,
                        yn,
                        t_total_sec,
                        str(source_name),
                        "Observation Plane",
                        sim_name,
                    )

                # After propagation determine the energy intercepted by the aperature
                # masked_aperature = self.field_mask(Uout, xn, yn, observer.aperature_diameter)
                # power_aperature = self.intensity_calc(masked_aperature, delta_n[0,-1], delta_n[0,-1])
                # del masked_aperature
                # Magnify the field and convolve field with impulse response
                # Uout, xout, yout = self.magnify_field(Uout, xn, yn, f, f)
                # Resample the field at the aperature to the specified discretization
                Uout = self.resample_field(Uout, xn, yn, xp, yp)
                # Multiply the field by the pupil mask
                Uout = Uout * pupil_mask
                # Plot masked observation field if requested
                if plot_psf == True and i == 0:
                    self.plot_field(
                        Uout,
                        xp,
                        yp,
                        t_total_sec,
                        str(source_name),
                        "Masked Observation Plane",
                        sim_name,
                    )
                # Determine the energy intercepted by the aperature
                power_aperature = self.intensity_calc(Uout, delta_p, delta_p)
                # Take the Semi-Analytical Fourier Transform as described by Soumner to
                # propagate to the image plane
                Uout = self.saft(Uout, observer.mx, observer.my, Nax, Nay)
                # Plot masked observation field if requested
                if plot_psf == True and i == 0:
                    self.plot_field(
                        Uout,
                        xa,
                        ya,
                        t_total_sec,
                        str(source_name),
                        "Local Image Plane",
                        sim_name,
                    )
                # Create a transform that moves the center of the modeled airy disk
                # pixels to the center of the WCS projection of the object's (RA,DEC)
                # Include the scaling factor that transforms the subpixels back to super
                # pixels
                H = tr.AffineTransform(
                    translation=[
                        x_y_detector_vec[0][0]
                        - observer.image_plane_pixels_simulated_x_direction / 2,
                        x_y_detector_vec[0][1]
                        - observer.image_plane_pixels_simulated_y_direction / 2,
                    ],
                    scale=[1 / subpix_per_pix_x, 1 / subpix_per_pix_y],
                )
                # Transform the image using the warp function of skimage.
                # It does not work with complex 128, so this transformation is broken'
                # into components
                Image_real = tr.warp(
                    np.real(Uout),
                    H.inverse,
                    output_shape=(observer.num_y_pix, observer.num_x_pix),
                )
                Image_comp = tr.warp(
                    np.imag(Uout),
                    H.inverse,
                    output_shape=(observer.num_y_pix, observer.num_x_pix),
                )
                Image_all = (Image_real + Image_comp * 1j) * Uout.unit
                del Image_real
                del Image_comp

                # Determine the total energy on the final image
                power_image = self.intensity_calc(
                    Image_all, observer.pixel_width, observer.pixel_height
                )
                if power_image.value == 0:
                    # The point source does not have any energy on the focal plane. Move to the next source.
                    continue
                # Scale the field in the image to have the correct power, transmission factor is applied in circuitry code with QE
                Uout, power_image_scaled = self.scale_field(
                    Image_all,
                    observer.pixel_width,
                    observer.pixel_height,
                    power_aperature,
                    power_image,
                )
                # Compare the scaled field to the original power to assign attribution
                # to specific pixels
                irradiance_association = self.pixel_attribution(
                    Uout,
                    irradiance_association,
                    observer.pixel_width,
                    observer.pixel_height,
                    power_min,
                    source_name,
                )

                # Plot final point spread functions if requested
                if plot_psf == True and i == 0:
                    self.plot_field(
                        Uout,
                        xdet,
                        ydet,
                        t_total_sec,
                        str(source_name),
                        "Image Plane",
                        sim_name,
                    )
                if j == 0:
                    Usum = Uout
                else:
                    Usum += Uout

            # Plot the full summation of fields if requested
            if plot_psf == True and i % 1000 == 0:
                self.plot_field(
                    Usum,
                    xdet,
                    ydet,
                    t_total_sec,
                    "All Sources",
                    "Image Plane",
                    sim_name,
                )
            pdb.set_trace()
            # Final Incident Photon Flux
            Flux_final = (
                np.multiply(Usum, np.conjugate(Usum))
                * observer.pixel_width
                * observer.pixel_height
            ).real.astype(np.float32)*u.ph/u.s
            # Multiply by fill factor of the photodiode of the pixel area
            # Watt_final = Watt_final*detector_fill_factor
            # Save the field in the HDF5 format
            # TODO: Make sure the input into the hdf5 format takes QE into account for star flux, the system QE applies to all sources, but the stars have the additional component based on their spectrum.
            hf = h5py.File(full_name, "r+")
            field_name = "ph_flux_time_itr_" + str(t_total_sec)
            att_name = "attribution_time_itr" + str(t_total_sec)
            hf.create_dataset(
                field_name, data=Flux_final, compression="gzip", compression_opts=9
            )
            hf.create_dataset(
                att_name,
                data=irradiance_association,
                compression="gzip",
                compression_opts=9,
            )
            hf.close()
            # Save the dictionary connecting for this timestep
            attribution_dict[t_total_sec] = irradiance_association
            # Delete PSF to save memory
            del Flux_final

        # Save the attribution dictionary for this simulation
        attribution_full_name = newdirectory + "attribution_dictionary"
        with open(attribution_full_name, "wb") as picklefile:
            pickle.dump(attribution_dict, picklefile)

        return full_name, attribution_dict

    def frame_generation_one_step_lock(
        self,
        prop_parameters,
        observer,
        sat_props,
        sat_obs,
        sat_array,
        t_array,
        t_step,
        lock,
        hdf5_name,
        attr_name,
        sim_name="filename",
        pscrn_file_name="filename",
        plot_psf=False,
    ):

        # Unpack Propagation Parameters
        nscrns = prop_parameters["nscrns"]
        N = prop_parameters["N"]
        N_total_n = prop_parameters["N_total_n"]
        N_total_pscrn = prop_parameters["N_total_pscrn"]
        z = prop_parameters["z"]
        L = prop_parameters["L"]
        delta_z = prop_parameters["delta_z"]
        delta_n = prop_parameters["delta_n"]  # Desired grid spacing
        delta_n_pscrn = prop_parameters["deltan_pscrn"]  # Phase screen grid spacing
        scale_pscrn = prop_parameters[
            "scale_pscrn"
        ]  # Ratio of desired grid spacing to phase screen spacing
        timescale = prop_parameters["timescale"]
        time_check = timescale
        psn_itr = 0

        # Calculate grid spacing for aperature projection for finding
        # the right subsections of the extended phase screens
        # For the desired grid spacing
        D_fpga_proj = (
            2
            * (np.flip(z) + observer.focal_length)
            * np.tan(np.deg2rad(observer.AFOV) / 2)
        )  # Diameter of optic projected
        slope = z[-1] / (D_fpga_proj[0] / 2 - observer.aperature_diameter / 2)
        Dn = observer.aperature_diameter + 2 * z / slope
        N_total_aperature_prop = np.ceil(np.flip(Dn) / delta_n)
        for i in np.arange(0, nscrns + 2):
            if N_total_aperature_prop[0, i] % 2 != 0:
                N_total_aperature_prop[0, i] = N_total_aperature_prop[0, i] + 1
        # Define the center of the propagation planes
        xc_prop = N_total_n / 2
        yc_prop = N_total_n / 2
        # Same calculations for the phase screen spacing, but for each set
        N_total_aperature_prop_pscrn = {}
        xc_pscrn = {}
        yc_pscrn = {}
        for i, psn_key in enumerate(list(N_total_pscrn.keys())):
            N_total_aperature_prop_pscrn_itr = np.ceil(
                np.flip(Dn)[1 : nscrns + 1] / np.stack(delta_n_pscrn[psn_key], axis=0)
            )
            for j in np.arange(0, nscrns):
                if N_total_aperature_prop_pscrn_itr[j] % 2 != 0:
                    N_total_aperature_prop_pscrn_itr[j] = (
                        N_total_aperature_prop_pscrn_itr[j] + 1
                    )
            N_total_aperature_prop_pscrn[psn_key] = N_total_aperature_prop_pscrn_itr
            # Define the center of the phase screen planes
            xc_pscrn[psn_key] = np.array(N_total_pscrn[psn_key]) / 2
            yc_pscrn[psn_key] = np.array(N_total_pscrn[psn_key]) / 2
        del D_fpga_proj
        del slope
        del Dn
        del N_total_aperature_prop_pscrn_itr
        # Define the center of the detector plane
        xc = observer.num_x_pix / 2
        yc = observer.num_y_pix / 2

        # Make (x,y) grid for the detector plots
        x_vector = (
            np.arange(0, observer.num_x_pix, 1, dtype=np.float32) * observer.pixel_width
        )
        y_vector = (
            np.arange(0, observer.num_y_pix, 1, dtype=np.float32)
            * observer.pixel_height
        )
        xdet, ydet = np.meshgrid(x_vector, y_vector, copy=False)

        # Use preselected observation plane sampling of the aperature to define the
        # observation (pupil) plane and aperature mask
        Np = observer.observation_plane_sampling
        delta_p = observer.aperature_diameter / Np
        x_vector = np.arange(-Np / 2 + 0.5, Np / 2 + 0.5, 1, dtype=np.float32) * delta_p
        y_vector = np.arange(-Np / 2 + 0.5, Np / 2 + 0.5, 1, dtype=np.float32) * delta_p
        xp, yp = np.meshgrid(x_vector, y_vector, copy=False)
        pupil_mask = self.circ(xp, yp, observer.aperature_diameter)
        # Define the airy disk plane
        if observer.sufficient_airy_disk_discretization == False:
            Nax = int(
                observer.subpixel_per_pixel_x_direction
                * observer.image_plane_pixels_simulated_x_direction
            )
            Nay = int(
                observer.subpixel_per_pixel_y_direction
                * observer.image_plane_pixels_simulated_y_direction
            )
            delta_ax = observer.pixel_width / observer.subpixel_per_pixel_x_direction
            delta_ay = observer.pixel_height / observer.subpixel_per_pixel_y_direction
            subpix_per_pix_x = observer.subpixel_per_pixel_x_direction
            subpix_per_pix_y = observer.subpixel_per_pixel_y_direction
        else:
            Nax = observer.image_plane_pixels_simulated_x_direction
            Nay = observer.image_plane_pixels_simulated_y_direction
            delta_ax = observer.pixel_width
            delta_ay = observer.pixel_height
            subpix_per_pix_x = 0
            subpix_per_pix_y = 0
        x_vector = (
            np.arange(-Nax / 2 + 0.5, Nax / 2 + 0.5, 1, dtype=np.float32) * delta_ax
        )
        y_vector = (
            np.arange(-Nay / 2 + 0.5, Nay / 2 + 0.5, 1, dtype=np.float32) * delta_ay
        )
        xa, ya = np.meshgrid(x_vector, y_vector, copy=False)

        # Make a polar coordinate grid for the source plane for defining the initial point sources
        x_vector = (
            np.arange(-N / 2 + 0.5, N / 2 + 0.5, 1) * delta_n[0, 0]
        )  # Does not need to be full size of propagation distance, just the size of the subset for the point source definition
        y_vector = np.arange(-N / 2 + 0.5, N / 2 + 0.5, 1) * delta_n[0, 0]
        x1, y1 = np.meshgrid(x_vector, y_vector, copy=False)
        r1, theta1 = self.cart2pol(x1, y1)
        del theta1

        # Create an Observation Plane Mask for Each Phase Screen
        # Since each phase screen is the same dimension, N, the super gaussian
        # is the same because it is based on percentage of array not actual dimensions
        # np.arange(-N/2,N/2,1)*delta_n[0,0]
        sg = self.super_gaussian(x1, y1, N, N, delta_n[0, 0], delta_n[0, 0])
        del x_vector
        del y_vector

        # Calculate intensity of satellite point sources assuming they are constant
        I_sat = {}
        # Make calculation for each satellite in the FOV
        for i, sat_key in enumerate(list(sat_props.keys())):
            A_surf = sat_props[sat_key]["satellite approx projected area"]
            rho = np.array(sat_props[sat_key]["surface rho properties"])
            # Calculate Point Source Power with Equations
            # E_ext = 1366 * u.W / u.m**2  # W/m^2 Solar Irradiance
            E_ext = 3.2681601435982546e18 * u.ph/(u.s*u.m**2) # Photons/s/m^2 Solar Irradiance
            I = self.calc_sat_point_intensity(E_ext, A_surf, rho)
            I_sat[sat_key] = I

        # Calculate the minimum power collected by the aperature to generate events via
        # the limiting magnitude
        irradiance_min = self.vmag_2_irradiance_Gia_G(
            observer.limiting_magnitude, observer.bandpass, observer.aperature_diameter
        )
        power_min = (
            irradiance_min * (observer.aperature_diameter / 2) ** 2 * np.pi
        )  # Before QE*T is applied, but that is applied to all values in the circuitry code

        # Load earth from Skyfield to retrieve Ra and Dec
        planets = load("de421.bsp")
        earth = planets["earth"]

        # Check which set of phase screens should be utilized
        time = t_array[t_step]
        t_total = t_array.utc[:, t_step] - t_array.utc[:, 0]
        # Convert difference into seconds
        if t_total[1] != 0:
            # If there are months in the time object covert it to days
            t_total_days = self.month2day(t_total)
            t_total_sec = (
                t_total[0] * u.year
                + t_total_days
                + t_total[3] * u.hour
                + t_total[4] * u.minute
                + t_total[5] * u.second
            ).decompose()
        else:
            # Add the total time and divide by the timescale for each phase screen
            # to obtain the total number of iterations of phase screens needed
            t_total_sec = (
                t_total[0] * u.year
                + t_total[2] * u.day
                + t_total[3] * u.hour
                + t_total[4] * u.minute
                + t_total[5] * u.second
            ).decompose()
        if t_total_sec > time_check:
            psn_itr += 1
            time_check += timescale

        # Create a dictionary to store the association of irradiance to a
        # particular point source
        irradiance_association = {}

        # Pull out properties and point sources unique to this timestep
        if isinstance(observer.optics_star_dict[t_step], (int, np.integer)) == True:
            star_idx = observer.optics_star_dict[t_step]
            star_objs = observer.optics_star_dict[star_idx]
        else:
            star_objs = observer.optics_star_dict[t_step]

        star_mags = observer.optics_star_mag_dict[t_step]
        star_ids = observer.optics_star_id_dict[t_step]
        wcs_proj = observer.optics_wcs_dict[t_step]
        detector_ra_dec = observer.optics_pointing_dict[t_step]

        # pdb.set_trace()
        # Calculate the Ra/Dec positions of stars at this timestep
        star_ras, star_decs, star_dists = earth.at(time).observe(star_objs).radec()
        # Calculate the flux sent by stars at this timestep
        star_fluxes = self.vmag_2_irradiance_Gia_G(
            star_mags, observer.bandpass, observer.aperature_diameter
        )

        for j in np.arange(0, len(star_mags) + len(list(sat_props.keys()))):

            # Ra and Dec and Power of the point source are saved differently for stars and satellites
            if j >= len(star_mags):
                # Pull out one satellite at a time
                sat_number = j - len(star_mags)
                sat_key = list(sat_props.keys())[sat_number]
                # Check if this satellite is observable at this time step
                if sat_obs[sat_key][t_step] != True:
                    continue
                source_name = sat_key
                # Replace slashes and spaces with underscores in the source name
                source_name = source_name.replace("/", "_")
                source_name = source_name.replace("\\", "_")
                source_name = source_name.replace(" ", "_")
                # This propagation iteration, use the satellite
                ra_vec = (
                    np.linspace(
                        sat_array[sat_key][t_step][0].value,
                        detector_ra_dec[0].value,
                        nscrns + 2,
                    )
                    * detector_ra_dec[0].unit
                )
                dec_vec = (
                    np.linspace(
                        sat_array[sat_key][t_step][1].value,
                        detector_ra_dec[1].value,
                        nscrns + 2,
                    )
                    * detector_ra_dec[1].unit
                )
                # Calculate the angle between sensor pointing and satellite
                cart_sat = [
                    np.sin(np.pi * u.rad - np.deg2rad(sat_array[sat_key][t_step][1]))
                    * np.cos(np.deg2rad(sat_array[sat_key][t_step][0])),
                    np.sin(np.pi * u.rad - np.deg2rad(sat_array[sat_key][t_step][1]))
                    * np.sin(np.deg2rad(sat_array[sat_key][t_step][0])),
                    np.cos(np.pi * u.rad - np.deg2rad(sat_array[sat_key][t_step][1])),
                ]
                cart_det = [
                    np.sin(np.pi * u.rad - np.deg2rad(detector_ra_dec[1]))
                    * np.cos(np.deg2rad(detector_ra_dec[0])),
                    np.sin(np.pi * u.rad - np.deg2rad(detector_ra_dec[1]))
                    * np.sin(np.deg2rad(detector_ra_dec[0])),
                    np.cos(np.pi * u.rad - np.deg2rad(detector_ra_dec[1])),
                ]
                theta_r = np.arccos(np.dot(cart_sat, cart_det))

                # Define the point source
                # The power for this point
                power = self.calc_sat_point_irradiance(
                    I_sat[sat_key],
                    sat_array[sat_key][t_step][2],
                    observer.aperature_diameter,
                    theta_r,
                )
            else:
                # The point source is a star
                source_name = star_ids[j]
                # For each star find the center of the phase screens (ra,dec)
                ra_vec = (
                    np.linspace(
                        star_ras._degrees[j], detector_ra_dec[0].value, nscrns + 2
                    )
                    * detector_ra_dec[0].unit
                )
                dec_vec = (
                    np.linspace(
                        star_decs._degrees[j], detector_ra_dec[1].value, nscrns + 2
                    )
                    * detector_ra_dec[0].unit
                )
                # Define the point source
                # The power for this point
                power = star_fluxes[j]

            # Covert (ra,dec) values to (x,y) locations on the focal plane
            ra_dec_vec = np.transpose(np.stack((ra_vec, dec_vec)))
            x_y_detector_vec = wcs_proj.wcs_world2pix(ra_dec_vec, 0)
            # Use the size of the projected focal planes to determine the (x,y)
            # locations in each of the screens that the light is propagating through
            x_y_prop_vec = []
            x_y_pscrn_vec = []
            for l, pixel in enumerate(x_y_detector_vec):
                # Calculate ratio of focal plane the point is from the center
                x_ratio = (pixel[0] - xc) / observer.num_x_pix
                y_ratio = (pixel[1] - yc) / observer.num_y_pix
                # Calculate the (x,y) cell in the projected plane
                x_pix = (x_ratio * N_total_aperature_prop[0, l]) + xc_prop[
                    0, l
                ]  # If the (x,y) position is in a pixel of the phase screen, we want the floor to capture that volume.
                y_pix = (y_ratio * N_total_aperature_prop[0, l]) + yc_prop[0, l]
                x_y_prop_vec.append((x_pix, y_pix))
                # Calculate the (x,y) cell in the phase screen
                if (
                    l != 0 and l != len(z) - 1
                ):  # source plane and observation plane do not have phase screens
                    x_pix = (
                        x_ratio * N_total_aperature_prop_pscrn[psn_itr][l - 1]
                    ) + xc_pscrn[psn_itr][
                        l - 1
                    ]  # If the (x,y) position is in a pixel of the phase screen, we want the floor to capture that volume.
                    y_pix = (
                        y_ratio * N_total_aperature_prop_pscrn[psn_itr][l - 1]
                    ) + yc_pscrn[psn_itr][l - 1]
                    x_y_pscrn_vec.append((x_pix, y_pix))
            # The first tuple in the (x,y) propagation vector will help define the point source
            rp, thp = self.cart2pol(
                x_y_prop_vec[0][0] * delta_n[0, 0], x_y_prop_vec[0][1] * delta_n[0, 0]
            )

            # Create the Point Source, use central source and recenter on the (x,y) projection pixel after the propagation
            if i == 0:
                Uin = self.central_point_source_sinc_func(
                    x1,
                    y1,
                    r1,
                    L,
                    observer.aperature_diameter,
                    observer.wavelength,
                    observer.wavenumber,
                    power,
                    time=t_total_sec,
                    source_name=str(source_name),
                    sim_name=sim_name,
                    plot_source=plot_psf,
                )
            else:
                Uin = self.central_point_source_sinc_func(
                    x1,
                    y1,
                    r1,
                    L,
                    observer.aperature_diameter,
                    observer.wavelength,
                    observer.wavenumber,
                    power,
                    time=t_total_sec,
                    source_name=str(source_name),
                    sim_name=sim_name,
                    plot_source=False,
                )

            # Run this point source through the split step propagation
            xn, yn, Uout = self.angular_spec_multi_prop(
                Uin,
                observer.wavenumber,
                delta_n,
                delta_n_pscrn,
                scale_pscrn,
                z,
                delta_z,
                sg,
                pscrn_file_name,
                psn_itr,
                x_y_prop_vec,
                x_y_pscrn_vec,
                xc_prop,
                yc_prop,
            )
            # Resample the field at the aperature to the specified discretization
            Uout = self.resample_field(Uout, xn, yn, xp, yp)
            # Multiply the field by the pupil mask
            Uout = Uout * pupil_mask
            # Determine the energy intercepted by the aperature
            power_aperature = self.intensity_calc(Uout, delta_p, delta_p)
            # Take the Semi-Analytical Fourier Transform as described by Soumner to
            # propagate to the image plane
            Uout = self.saft(Uout, observer.mx, observer.my, Nax, Nay)
            # Create a transform that moves the center of the modeled airy disk
            # pixels to the center of the WCS projection of the object's (RA,DEC)
            # Include the scaling factor that transforms the subpixels back to super
            # pixels
            H = tr.AffineTransform(
                translation=[
                    x_y_detector_vec[0][0]
                    - observer.image_plane_pixels_simulated_x_direction / 2,
                    x_y_detector_vec[0][1]
                    - observer.image_plane_pixels_simulated_y_direction / 2,
                ],
                scale=[1 / subpix_per_pix_x, 1 / subpix_per_pix_y],
            )
            # Transform the image using the warp function of skimage.
            # It does not work with complex 128, so this transformation is broken'
            # into components
            Image_real = tr.warp(
                np.real(Uout),
                H.inverse,
                output_shape=(observer.num_y_pix, observer.num_x_pix),
            )
            Image_comp = tr.warp(
                np.imag(Uout),
                H.inverse,
                output_shape=(observer.num_y_pix, observer.num_x_pix),
            )
            Image_all = (Image_real + Image_comp * 1j) * Uout.unit
            del Image_real
            del Image_comp
            # Determine the total energy on the final image
            power_image = self.intensity_calc(
                Image_all, observer.pixel_width, observer.pixel_height
            )
            if power_image.value == 0:
                # The point source does not have any energy on the focal plane. Move to the next source.
                continue
            # Scale the field in the image to have the correct power
            Uout, power_image_scaled = self.scale_field(
                Image_all,
                observer.pixel_width,
                observer.pixel_height,
                power_aperature,
                power_image,
            )
            # Compare the scaled field to the original power to assign attribution
            # to specific pixels
            irradiance_association = self.pixel_attribution(
                Uout,
                irradiance_association,
                observer.pixel_width,
                observer.pixel_height,
                power_min,
                source_name,
            )

            # Plot final point spread functions if requested
            if plot_psf == True and i == 0:
                self.plot_field(
                    Uout,
                    xdet,
                    ydet,
                    t_total_sec,
                    str(source_name),
                    "Image Plane",
                    sim_name,
                )

            if j == 0:
                Usum = Uout
            else:
                Usum += Uout

        # Plot the full summation of fields if requested
        if plot_psf == True and t_step % 1000 == 0:
            self.plot_field(
                Usum, xdet, ydet, t_total_sec, "All Sources", "Image Plane", sim_name
            )

        # Final Incident Flux
        Flux_final = (
            np.multiply(Usum, np.conjugate(Usum))
            * observer.pixel_width
            * observer.pixel_height
        ).real.astype(np.float32)
        # print("Flux units: {} ".format(Flux_final.unit))
        # Multiply by fill factor of the photodiode of the pixel area
        # Watt_final = Watt_final*detector_fill_factor

        # Save the outputs
        # Acquire lock to ensure this is the only process writing to a file
        lock.acquire()
        try:
            # Add the final Watts to the h5py file
            hf = h5py.File(hdf5_name, "r+")
            field_name = "ph_flux_time_itr_" + str(t_total_sec)
            try:
                hf.create_dataset(
                    field_name, data=Flux_final, compression="gzip", compression_opts=9
                )
            finally:
                hf.close()
            # Add the attribution dictionary to the pickled dictionary
            # Open the attribution dictionary
            with open(attr_name, "rb") as picklefile:
                Attribution_dict = pickle.load(picklefile)
            # Add the attributions for this step
            Attribution_dict[t_total_sec] = irradiance_association
            # Save the attribution dictionary
            with open(attr_name, "wb") as picklefile:
                pickle.dump(Attribution_dict, picklefile)
            print("Saving time step {} sucessful.".format(t_total_sec))
        except Exception as e:
            print("Saving time step {} failed because {}.".format(t_total_sec, e))
        finally:
            # Release the lock to ensure another process can save
            lock.release()

        return

    def grid_spacing_analysis(self, observer, prop_parameters={}, sim_name="filename"):
        """
        Determine grid spacing to appropriately model the field propagation
        with FFTs.

        Parameters
        ----------
        observer : dict
            Dictionary with observer properties defined with Optical System
            module.
        prop_parameters : dict, optional
            Dictionary with parameters of the propagation. The default is {}.
        sim_name : string, optional
            The simulation name to create a new folder. The default is 'filename'.

        Returns
        -------
        prop_parameters : dict
            Dictionary with parameters of the propagation.

        """

        # Ask for simulation parameters if they are not provided
        if "L" not in prop_parameters.keys():
            L = "L"
            while not isinstance(L, float) == True:
                L = input(
                    (
                        "Please enter the distance propagated through the"
                        " atmosphere (typically 5e4 to 10e4) [m]. Note this"
                        " is typically much less than the satellite "
                        "distance to the observer since attenuation"
                        " is concentrated in the final kilometers of"
                        " atmosphere.: "
                    )
                )
                try:
                    L = float(L)
                    break
                except:
                    print(
                        "Please enter a float for the distance propagated through the atmosphere."
                    )
            prop_parameters["L"] = L * u.m
            L = L * u.m
        else:
            L = prop_parameters["L"]
        R = L  # radius of wavefront
        if "Cn" not in prop_parameters.keys():
            Cn = "Cn"
            while not isinstance(Cn, float) == True:
                Cn = input(
                    "Please enter the number of the structure parameter (typically 10e-15 high turbulence to 10-18 low turbulence) [m^(-2/3)]: "
                )
                try:
                    Cn = float(Cn)
                    break
                except:
                    print("Please enter a float for the parameter.")
            prop_parameters["Cn"] = Cn * (u.m ** (-2 / 3))
            Cn = Cn * (u.m) ** (-2 / 3)
        else:
            Cn = prop_parameters["Cn"]
        if "c" not in prop_parameters.keys():
            c = "c"
            while not isinstance(c, int) == True:
                c = input(
                    "Please enter the parameter that scales the blurring (between 2 and 8): "
                )
                try:
                    c = int(c)
                    break
                except:
                    print("Please enter an integer for the parameter.")
            prop_parameters["c"] = c
        else:
            c = prop_parameters["c"]
        if "nscrns" not in prop_parameters.keys():
            nscrns = "nscrns"
            while not isinstance(nscrns, int) == True:
                nscrns = input(
                    "Please enter an initial guess for the number of phase screens (typically 10): "
                )
                try:
                    nscrns = int(nscrns)
                    break
                except:
                    print("Please enter an integer for the number of screens.")
            prop_parameters["nscrns"] = nscrns
        else:
            nscrns = prop_parameters["nscrns"]

        # Assume area of the central lobe of the incoming point source will be
        # 4 times the area of the optical system's aperature diameter
        # Usually this assumption is good enough to remove alising from the
        # Fourier transform from the region of interest, the optical system
        DROI = 4 * observer.aperature_diameter
        D1 = observer.wavelength * L / DROI

        # Determine the Source and Observation Grid Spacing to Avoid Aliasing
        # calculate coherence diameters for full propagation
        r0sw = self.fried_parameter_sw(Cn, L, observer.wavelength)  # spherical wave
        prop_parameters["r0sw"] = r0sw
        r0pw = self.fried_parameter(
            Cn, L, observer.wavelength
        )  # plane wave only needed if long propagation is used
        prop_parameters["r0pw"] = r0pw
        # calculate the Rytov parameter - log-amplitude variances
        p = np.linspace(0, L, 1000)  # a vector to calculate the integral in rytov
        rytov = (
            0.563
            * observer.wavenumber ** (7 / 6)
            * np.sum(
                np.multiply(Cn * (1 - p / L) ** (5 / 6), p ** (5 / 6) * (p[1] - p[0]))
            )
        )

        # Calculate the ro parameters to be used at each phase screen
        # Initial guess of number of required screens is set by the user
        A = np.zeros((2, nscrns + 2))
        alpha = np.arange(0, nscrns + 2, 1) / (nscrns + 2 - 1)
        A[0, :] = alpha ** (5 / 3)
        A[1, :] = np.multiply((1 - alpha) ** (5 / 6), alpha ** (5 / 6))
        b = [
            [r0sw.value ** (-5 / 3)],
            [rytov.value / 1.33 * (observer.wavenumber.value / L.value) ** (5 / 6)],
        ]  # values instead of units so that the optimization module works below.
        # Set initial guess based on the total coherence diameters
        x0 = ((nscrns + 2 / 3) * r0sw * np.ones((nscrns + 2, 1))) ** (-5 / 3)
        x0 = x0.flatten()
        # Constain the solution to only positive coherence diameters and an upper bound
        # that all ro values must have a contribution to Rytov (variance) of less than 0.1
        x1 = [0] * (nscrns + 2)
        # x1 = x1.flatten()
        rytov_max = 0.1
        x2 = rytov_max / 1.33 * (observer.wavenumber / L) ** (5 / 6) / A[1, :]
        x2[x2 == inf] = (
            50 ** (-5 / 3) * x2.unit
        )  # make large but not infinite to prevent errors
        # count = 0
        # for i in x2:
        #     if np.isinf(i):
        #         x2[count] = 50**(-5/3) # make large but not infinite to prevent errors
        #     count += 1
        x2_value = x2.value.tolist()
        bnds = list(zip(x1, x2_value))
        # x2 = x2.flatten()
        # bnds = list(1)
        # for i in np.arange(0,nscrns,1):
        #     bnds[t_step] = (x1[i],x2[i])
        # print(bnds)
        # bnds = tuple(bnds)
        # print(bnds)
        X = spo.minimize(
            self.ro_multi_fun, x0.value.flatten(), args=(A, b), bounds=bnds
        )
        X = X.x
        print("This is the output of the least squares minimization: {}.".format(X))
        r0scrn = (X * x0.unit) ** (-3 / 5)
        r0scrn[r0scrn == inf] = observer.wavelength
        prop_parameters["r0scrn"] = r0scrn

        # Check the new ro total and rytov, the least squares solution will not be able to satisfy everything
        bp = np.matmul(A, X)
        # print(bp)

        r0_new = bp[0] ** (-3 / 5) * u.m
        prop_parameters["r0_new"] = r0_new
        r0_diff = np.abs(r0_new - r0sw)
        rytov_new = (
            bp[1] * u.m ** (-5 / 3) * 1.33 * (L / observer.wavenumber) ** (5 / 6)
        )
        prop_parameters["rytov"] = rytov_new
        rytov_diff = np.abs(rytov_new - rytov)
        print("")
        print(
            "The new combined ro and rytov parameters are {} and {} respectively.".format(
                r0_new, rytov_new
            )
        )
        print(
            "The difference from the original values is {} and {} respectively.".format(
                r0_diff, rytov_diff
            )
        )

        # Determine the sampling needed to prevent aliasing
        # The c parameter can be set between 2 and 8, higher blurring assumed with larger number
        D1p = D1 + c * observer.wavelength * L / r0_new
        D2p = observer.aperature_diameter + c * observer.wavelength * L / r0_new

        # Simulate a set of possible delta1 and deltan values to capture the inequality constraints
        delta1_list = np.linspace(0, 1.1 * observer.wavelength * L / D2p, 100)
        # print(delta1_list)
        deltan_list = np.linspace(0, 1.1 * observer.wavelength * L / D1p, 100)
        # print(deltan_list)
        # Constraint 1
        deltan_max = (-D2p / D1p) * delta1_list + observer.wavelength * L / D1p
        # Constraint 3
        d2min3 = (1 + L / R) * delta1_list - observer.wavelength * L / D1p
        d2max3 = (1 + L / R) * delta1_list + observer.wavelength * L / D1p
        delta1_array, deltan_array = np.meshgrid(delta1_list, deltan_list)
        # Constraint 2
        N2 = (observer.wavelength * L + D1p * deltan_array + D2p * delta1_array) / (
            2 * np.outer(delta1_list, deltan_list)
        )
        N2_log = np.log2(N2)
        N2_log[N2_log == inf] = 0
        lvls = np.arange(0, int(np.max(N2_log)) + 2, 1)
        # print(N2_log)

        # Plot the range of possible values for delta1 and deltan
        fig, ax = plt.subplots(dpi=200)
        CS = ax.contourf(
            delta1_array, deltan_array, N2_log, levels=lvls, cmap="nipy_spectral"
        )
        cbar = fig.colorbar(CS)
        cbar.set_label("log2 N", rotation=270)
        ax.plot(delta1_list, deltan_max, "k")
        ax.plot(delta1_list, d2min3, "k")
        ax.plot(delta1_list, d2max3, "k")
        ax.axes.set_xticks(
            np.round(
                np.arange(0, np.round(np.max(delta1_list.value), decimals=1), 0.05),
                decimals=2,
            ).tolist(),
            minor=True,
        )
        ax.axes.set_yticks(
            np.round(
                np.arange(0, np.round(np.max(deltan_list.value), decimals=1), 0.01),
                decimals=2,
            ).tolist(),
            minor=True,
        )
        ax.set_title("Sampling of Grid and Source Bounds")
        ax.set_xlabel("Source Grid Spacing [" + str(delta1_list.unit) + "]")
        ax.set_xlim((0, delta1_list.value[-1]))
        ax.set_ylabel("Observation Grid Spacing [" + str(deltan_list.unit) + "]")
        ax.set_ylim((0, deltan_list.value[-1]))
        ax.grid(visible=True, which="major")  # major, minor, both
        plt.show()
        directory = os.getcwd()
        newdirectory = directory + "/" + sim_name + "/ProcessingImages/GridSampling"
        if not os.path.exists(newdirectory):
            os.makedirs(newdirectory)
        plot_name = "grid_sampling"
        plt.savefig(newdirectory + "/" + plot_name + ".pdf")
        plt.close()

        # Prompt the user to select a delta value based on the plot
        print("")
        print(
            "Based on the displayed limits on the graph, please select a source and observation grid spacing."
        )
        print(
            "Note when selecting the aperature is {} meters and it is best to have a least 10 sample points in each direction.".format(
                observer.aperature_diameter
            )
        )

        ncheck = False
        while ncheck == False:
            delta1 = "source grid"
            deltan = "observation grid"
            N = "total grid spaces"
            while not isinstance(delta1, float) == True:
                delta1 = input(
                    "Please select a value for the source grid spacing in meters: "
                )
                try:
                    delta1 = float(delta1)
                    break
                except:
                    print("Please enter a float or integer for the grid spacing value.")
            delta1 = delta1 * u.m
            while not isinstance(deltan, float) == True:
                deltan = input(
                    "Please select a value for the observation grid spacing in meters: "
                )
                try:
                    deltan = float(deltan)
                    break
                except:
                    print(
                        "Please enter an even number integer for the grid spacing value."
                    )
            deltan = deltan * u.m
            while not isinstance(N, float) == True:
                N_min = np.ceil(
                    (L * observer.wavelength * D1p)
                    / (
                        delta1
                        * (
                            observer.wavelength * L
                            - observer.aperature_diameter * delta1
                        )
                    )
                )
                N = input(
                    "Please select a value for the total number of grid spaces. Note the minimum number for the number of grid spaces required is {}. Choice of N: ".format(
                        N_min
                    )
                )
                try:
                    N = float(N)
                    if N % 2 == 0:
                        break
                    else:
                        print("Please pick an integer divisible by 2.")
                except:
                    print("Please enter a float or integer for the grid spacing value.")
            # Check the minimum number of propagations is not greater than the initially chosen value
            zideal = (delta1 * deltan * N) / observer.wavelength
            nmin = np.ceil(L / zideal) + 1
            if nscrns < nmin:
                print(
                    "The mininum number of propagations, {}, is greater than the initially selected. Please choose another combination of grid spacings and number of grid cells.".format(
                        nmin
                    )
                )
            else:
                ncheck = True
                print(
                    "The minimum number of planes is {}. The selected number {} is greater.".format(
                        nmin, nscrns
                    )
                )

        prop_parameters["delta1"] = delta1
        prop_parameters["deltan"] = deltan
        prop_parameters["N"] = N
        prop_parameters["nscrns"] = nscrns

        return prop_parameters

    def super_gaussian(self, x, y, Nx, Ny, dx, dy):
        """
        Attenuates the amplitude of the field at the edges to prevent wrapping
        during the Fourier tranform.

        Parameters
        ----------
        x : np.array
            Array that contains the x locations from the center axis.
        y : np.array
            Array that contains the y locations from the center axis.
        Nx : int
            Size of array in x direction
        Ny : int
            Size of array in y direction
        dx : float
            Step size between points in field in x direction [m].
        dy : float
            Step size between points in field in y direction [m].

        Returns
        -------
        sg : np.array
            Array with no units that attenuates the field amplitude along the edges.

        """
        sg = np.exp(-((x / (0.47 * Nx * dx)) ** 16)) * np.exp(
            -((y / (0.47 * Ny * dy)) ** 16)
        )
        return sg

    def angular_spec_multi_prop_vac(self, Uin, k, del1x, del1y, deltanx, deltany, z):
        """
        Function as described in Schmidt (2010) for vacuum propogation.
        No turbulence is accounted for in this simulation.

        Parameters
        ----------
        Uin : np.array
            Input complex value field array that describes a point source [sqrt(W)/m].
        k : float
            Central wavelength of the observed range wavenumber [1/m].
        del1x : float
            Step size between points in the source field in the x direction[m].
        del1y : float
            Step size between points in the source field in the y direction[m].
        deltanx : float
            Step size between point in the observed field at aperature entrance
            in the x direction [m].
        deltany : float
            Step size between point in the observed field at aperature entrance
            in the y direction[m].
        z : np.array
            Propagation distances of the phase screens starting at the source
            field and ending at the observed field. [m]

        Returns
        -------
        xn : np.array
            Array that describes the field locations in the x direction [m].
        yn : np.array
            Array that describes the field locations in the y direction [m].
        Uout : np.array
            Output complex value field that describes the recieved field [sqrt(W)/m].

        """
        Nx = np.size(Uin, 1)
        Ny = np.size(Uin, 0)
        x_vector = np.arange(-Nx / 2 + 0.5, Nx / 2 + 0.5, 1)
        y_vector = np.arange(-Ny / 2 + 0.5, Ny / 2 + 0.5, 1)
        xn, yn = np.meshgrid(x_vector, y_vector)

        # Create a Super Gaussian Absorbing Boundary
        sg = self.super_gaussian(xn, yn, Nx, Ny, 1, 1)

        # Create a vector with all the propagation plane distances
        # z = np.insert(z,0,0) 0 already accounted for
        n = len(z)
        # Calculate the changes in z between each plane
        delta_z = (z.value[1:n] - z.value[0 : n - 1]) * z.unit
        # Calculate the grid spacings at each propagation plane
        alpha = z / z[n - 1]  # ratio of propagation distance completed
        delta_xn = (np.ones((1, n)) - alpha) * del1x + alpha * deltanx
        delta_yn = (np.ones((1, n)) - alpha) * del1y + alpha * deltany
        # Calculate the scaling factor at each propagation plane
        # Note this does not need to be done in x and y. The scaling should be the same.
        m = delta_xn[0, 1:n] / delta_xn[0, 0 : n - 1]
        x1 = xn * delta_xn[0, 0]
        y1 = yn * delta_yn[0, 0]
        r1sq = x1**2 + y1**2

        # Multiply the source field by the first quadratic factor before starting Fourier Transform Loop
        try:
            Q1 = np.exp(1j * (k / 2) * ((1 - m[0]) / delta_z[0]) * r1sq)
        except:
            Q1 = np.exp(1j * (k / 2) * ((1 - m[0]) / delta_z[0]) * r1sq / u.rad)
        Uin = np.multiply(Uin, Q1)

        for i in np.arange(0, n - 2, 1):
            # Calculate the spatial frequencys of this propagation plane
            delta_f_x = 1 / (Nx * delta_xn[0, i])
            delta_f_y = 1 / (Ny * delta_yn[0, i])
            # Calculate the spatial frequencies of each grid point
            fX = xn * delta_f_x
            fY = yn * delta_f_y
            fsq = fX**2 + fY**2
            Z = delta_z[i]  # Propagation distance for this step

            # Calculate the Quadratic Phase Factor for this propogation step
            try:
                Qn = np.exp(-1j * np.pi**2 * 2 * (Z / m[i] / k) * fsq)
            except:
                Qn = np.exp(-1j * np.pi**2 * 2 * (Z / m[i] / k) * fsq * u.rad)

            # Use a Fourier and Inverse Fourier transform to propogate through this step
            Uin = np.multiply(
                sg,
                self.ift2(
                    np.multiply(
                        Qn, self.ft2(Uin / m[i], delta_xn[0, i], delta_yn[0, i], Nx, Ny)
                    ),
                    delta_f_x,
                    delta_f_y,
                    Nx,
                    Ny,
                ),
            )
            plt.contourf(x1, y1, Uin * np.conjugate(Uin))
            plt.show()

        # Calculate one more quadratic phase factor for the final plane
        xf = xn * delta_xn[0, n - 1]
        yf = yn * delta_yn[0, n - 1]
        rfsq = xf**2 + yf**2
        try:
            Qf = np.exp(1j * (k / 2) * ((m[n - 2] - 1) / (m[n - 2] * Z)) * rfsq)
        except:
            Qf = np.exp(1j * (k / 2) * ((m[n - 2] - 1) / (m[n - 2] * Z)) * rfsq / u.rad)
        Uout = np.multiply(Qf, Uin)

        return (xf, yf, Uout)

    def angular_spec_multi_prop(
        self,
        Uin,
        k,
        delta_n,
        delta_n_pscrn,
        scale,
        z,
        delta_z,
        sg,
        pscrn_file_name,
        psn_itr,
        x_y_prop_vec,
        x_y_pscrn_vec,
        x_prop,
        y_prop,
    ):
        """
        Function as described in Schmidt (2010) to apply phase screens as a field
        propagates towards the final observation plane.

        Parameters
        ----------
        Uin : np.array
            Input complex value field array that describes a point source energy [sqrt(photons)/m].
        k : float
            Optical wavevector [rad/m].
        delta_n : list
            Spacing of the field at each propagation step. [m]
        delta_n_pscrn : list
            Spacing of the larger phase screen at each propagation step. [m]
        scale : int
            Scale between the spacings at each propagation step.
        z : np.array
            Propagation distances of the phase screens starting at the source
            field and ending at the observed field. [m]
        delta_z : list
            Distance between each propgation step for the angular spectrum propagation. [m]
        sg : np.array
            Super Gaussian array which attenuates the field amplitude close to
            the edge of the field. This prevents wrap around during the Fourier
            Transform. Since array size is fixed to N, this array is the same
            for all steps.
        pscrn_file_name : string
            Name of the HDF5 file that contains all the phase screens.
        psn_itr : int
            There may be more than one set of phase screens for a simulation
            when the frozen flow assumption is violated for a long duration
            simulation. (>10 sec)
        x_y_prop_vec : list
            Center of the point source at each step projected into the image
            plane. [m]
        x_y_pscrn_vec : list
            Center of the point source at each step in the phase screen.
        x_prop : list
            Center of each propagation plane's phase screen, x direction.
        y_prop : list
            Center of each propagation plane's phase screen, y direction.

        Returns
        -------
        xn : np.array
            Array that describes the field locations in the x direction [m].
        yn : np.array
            Array that describes the field locations in the y direction [m].
        Uout : np.array
            Output complex value field that describes the recieved field [sqrt(photons)/m].

        """

        # N is constant for the propagation given the dimensions chosen by the user
        Nx = np.size(Uin, 1)
        Ny = np.size(Uin, 0)
        x_vector = np.arange(
            -Nx / 2 + 0.5, Nx / 2 + 0.5, 1
        )  # No units applied because it is used for multiple screen and frequency
        y_vector = np.arange(-Ny / 2 + 0.5, Ny / 2 + 0.5, 1)
        xn, yn = np.meshgrid(x_vector, y_vector)
        del x_vector
        del y_vector

        # Create a vector with all the propagation plane distances
        # z = np.insert(z,0,0) already added a 0 on the front
        n = len(z)

        # Calculate the scaling factor at each propagation plane
        # Note this does not need to be done in x and y. The scaling should be the same.
        m = delta_n[0, 1:n] / delta_n[0, 0 : n - 1]
        x1 = xn * delta_n[0, 0]
        y1 = yn * delta_n[0, 0]
        # x1 = (xn+(x_y_prop_vec[0][0]-x_prop[0][0]))*delta_n[0,0]
        # y1 = (yn+(x_y_prop_vec[0][1]-y_prop[0][0]))*delta_n[0,0]
        r1sq = x1**2 + y1**2
        del x1
        del y1

        # Multiply the source field by the first quadratic factor before starting Fourier Transform Loop
        # print((delta_z[0]*r1sq))
        try:
            Q1 = np.exp(1j * (k / 2) * ((1 - m[0]) / delta_z[0]) * r1sq)
        except:
            Q1 = np.exp(1j * (k / 2) * ((1 - m[0]) / delta_z[0]) * r1sq / u.rad)
        Uin = np.multiply(Uin, Q1)
        del r1sq
        del Q1

        # Load the HDF5 File containing the phase screens
        pscrns = h5py.File(pscrn_file_name, "r")
        # point_in_frame = True

        for i in np.arange(0, n - 1, 1):
            # print(i)
            # pdb.set_trace()
            # Load the phase screen for this plane
            screen_name = "time_itr_" + str(psn_itr) + "/screen_" + str(i + 1)
            try:
                pscrn = pscrns[screen_name]
                # Resample the phase screen for this propagation
                # Define the sample region grid points
                x_s = (
                    np.arange(-Nx / 2 + 0.5, Nx / 2 + 0.5, 1)
                    + (x_y_prop_vec[i + 1][0] - x_prop[0, i + 1])
                ) * delta_n[0, i + 1]
                y_s = (
                    np.arange(-Ny / 2 + 0.5, Ny / 2 + 0.5, 1)
                    + (x_y_prop_vec[i + 1][1] - x_prop[0, i + 1])
                ) * delta_n[0, i + 1]
                # Define the limits
                x_s_min = np.min(x_s)
                x_s_max = np.max(x_s)
                y_s_min = np.min(y_s)
                y_s_max = np.max(y_s)
                # Define the sample region grid points
                x_p = (
                    np.arange(
                        -np.shape(pscrn)[0] / 2 + 0.5, np.shape(pscrn)[0] / 2 + 0.5, 1
                    )
                    * delta_n_pscrn[0][i]
                )
                y_p = (
                    np.arange(
                        -np.shape(pscrn)[1] / 2 + 0.5, np.shape(pscrn)[1] / 2 + 0.5, 1
                    )
                    * delta_n_pscrn[0][i]
                )
                # Find the index closest to the sample limits on the phase screen
                # This reduces the computation time of the interpolation
                x_p_min_index = np.searchsorted(x_p, x_s_min) - 1
                x_p_max_index = np.searchsorted(x_p, x_s_max) + 1
                y_p_min_index = np.searchsorted(y_p, y_s_min) - 1
                y_p_max_index = np.searchsorted(y_p, y_s_max) + 1
                # Crop phase screen and grid points
                x_p = x_p[x_p_min_index:x_p_max_index]
                y_p = y_p[y_p_min_index:y_p_max_index]
                pscrn = pscrn[x_p_min_index:x_p_max_index, y_p_min_index:y_p_max_index]
                # Create meshgrids for interpolation
                x_s, y_s = np.meshgrid(x_s, y_s)
                x_p, y_p = np.meshgrid(x_p, y_p)
                # Interpolate the phase screen at the sample grid points
                pscrn = interpolate.griddata(
                    (x_p.ravel(), y_p.ravel()),
                    pscrn.ravel(),
                    (x_s, y_s),
                    method="cubic",
                    fill_value=0,
                )
            except:
                # pdb.set_trace()
                pscrn = np.zeros((Ny, Nx))
            # # Crop the phase screen for this propagation
            # # Account for the possible larger spatial step
            # N_ratio = Nx/scale[psn_itr][i]
            # # if unscaled (scale=1), just crop phase screen
            # if scale[psn_itr][i] == 1.0:
            #     pscrn = pscrn[int(x_y_pscrn_vec[i][0]-Nx/2):int(x_y_pscrn_vec[i][0]+Nx/2),int(x_y_pscrn_vec[i][1]-Ny/2):int(x_y_pscrn_vec[i][1]+Ny/2)]
            # # if the propagation step is entirely within four phase screen square
            # elif N_ratio <= 1:
            #     # take the one relevant square's value
            #     pscrn_value = pscrn[x_y_pscrn_vec[i][0],x_y_pscrn_vec[i][1]]
            #     # multiply it by the desired N
            #     pscrn = np.ones((Nx,Ny))*pscrn_value
            # else:
            #     # Find the nearest integer of N
            #     N_int_x = int(np.ceil(Nx/(2*scale[psn_itr][i])))
            #     N_int_y = int(np.ceil(Ny/(2*scale[psn_itr][i])))
            #     # Take slice of phase screen
            #     pscrn = pscrn[int(x_y_pscrn_vec[i][0]-N_int_x):int(x_y_pscrn_vec[i][0]+N_int_x),int(x_y_pscrn_vec[i][1]-N_int_y):int(x_y_pscrn_vec[i][1]+N_int_y)]
            #     # Resample to size of propagation
            #     # Calculate distance covered by phase screen subsection
            #     pscrn_dist = (N_int_x*2)*delta_n_pscrn[int(psn_itr)][i]
            #     # Calculate number of propagation dimensions steps fit within the screen
            #     N_prop_pscrn = int(np.ceil(pscrn_dist/delta_n[0,i+1]))
            #     if N_prop_pscrn %2 != 0:
            #         N_prop_pscrn += 1
            #     # Resize the array to the propagation step size
            #     try:
            #         pscrn = tr.resize(pscrn,(N_prop_pscrn,N_prop_pscrn))
            #     except:
            #         # # pdb.set_trace()
            #         point_in_frame = False
            #         break
            #     # Crop the pscrn to the desired size
            #     pscrn = pscrn[int(N_prop_pscrn/2-Nx/2):int(N_prop_pscrn/2+Nx/2),int(N_prop_pscrn/2-Ny/2):int(N_prop_pscrn/2+Ny/2)]

            # pscrn = (pscrn +np.pi)%(2*np.pi)-np.pi
            # pscrn = np.zeros(pscrn.shape)
            pscrn = np.multiply(sg, np.exp(1j * pscrn))

            # Calculate the spatial frequencys of this propagation plane (Note square for propagation frames)
            delta_f = 1 / (Nx * delta_n[0, i])
            # Calculate the spatial frequencies of each grid point
            fX = xn * delta_f
            fY = yn * delta_f
            # fX = (xn+(x_y_prop_vec[i+1][0]-x_prop[0][i+1]))*delta_f
            # fY = (yn+(x_y_prop_vec[i+1][1]-y_prop[0][i+1]))*delta_f
            fsq = fX**2 + fY**2
            Z = delta_z[i]  # Propagation distance for this step

            # Calculate the Quadratic Phase Factor for this propogation step
            try:
                Qn = np.exp(-1j * np.pi**2 * 2 * (Z / m[i] / k) * fsq)
            except:
                Qn = np.exp(-1j * np.pi**2 * 2 * (Z / m[i] / k) * fsq * u.rad)

            # Use a Fourier and Inverse Fourier transform to propogate through this step
            Uin = np.multiply(
                sg,
                np.multiply(
                    pscrn,
                    self.ift2(
                        np.multiply(
                            Qn,
                            self.ft2(Uin / m[i], delta_n[0, i], delta_n[0, i], Nx, Ny),
                        ),
                        delta_f,
                        delta_f,
                        Nx,
                        Ny,
                    ),
                ),
            )

        # Close the HDF5 File
        pscrns.close()

        # Calculate one more quadratic phase factor for the final plane
        xf = xn * delta_n[0, n - 1]
        yf = yn * delta_n[0, n - 1]
        # xf = (xn+(x_y_prop_vec[-1][0]-x_prop[0][-1]))*delta_n[0,n-1]
        # yf = (yn+(x_y_prop_vec[-1][1]-y_prop[0][-1]))*delta_n[0,n-1]
        rfsq = xf**2 + yf**2
        try:
            Qf = np.exp(1j * (k / 2) * ((m[n - 2] - 1) / (m[n - 2] * Z)) * rfsq)
        except:
            Qf = np.exp(1j * (k / 2) * ((m[n - 2] - 1) / (m[n - 2] * Z)) * rfsq / u.rad)
        Uout = np.multiply(Qf, Uin)
        # if point_in_frame == True:
        #     try:
        #         Qf = np.exp(1j*(k/2)*((m[n-2]-1)/(m[n-2]*Z))*rfsq)
        #     except:
        #         Qf = np.exp(1j*(k/2)*((m[n-2]-1)/(m[n-2]*Z))*rfsq/u.rad)
        #     Uout = np.multiply(Qf,Uin)
        # else:
        #     # Pass back a placeholder
        #     Uout = np.ones(np.shape(Uin))

        return xf, yf, Uout

    def central_point_source_sinc_func(
        self,
        x1,
        y1,
        r1,
        Dz,
        D2,
        wvl,
        k,
        P,
        time=0,
        source_name="source_name",
        sim_name="sim_name",
        plot_source=False,
    ):
        """
        Define a point source at the center of a field.

        Parameters
        ----------
        x1 : np.array
            Location of x coordinates in the array in meters.
        y1 : np.array
            Location of y coordinates in the array in meters.
        r1 : np.array
            Polar coordinates radius in the array in meters.
        Dz : float
            Propagation distance in meters.
        D2 : float
            Aperature size of the sensor's optic in meters.
        wvl : float
            Center wavelength of the observation in meters.
        k : float
            Optical wavevector of central wavelength [rad/meters].
        P : float
            Radiance value of the source being simulated in sqrt(Watts)/m.
        time : float, optional
            Simulation time to include in the field plot when requested. The default is 0.
        source_name : float, optional
            Numerical count of source to keep track of plots. The default is 0.
        sim_name : string, optional
            Simulation name to save the plots in an designed folder. The default is 'sim_name'.
        plot_source : float, optional
            Boolean to determine if the source should be plotted. The default is False.

        Returns
        -------
        Ur : np.array
            Complex array defining the square root amplitude and phase of an electric field. It is in units of [(W)^(1/2)/m]

        """
        # Use input to define needed parameters
        DROI = (
            4 * D2
        )  # diameter of observation-plane region of interest, multiplied by factor to ensure flat field wider than aperature [m]
        # arg = DROI/(wvl*Dz) # ensure U units are sqrt(W)/m [1/m]
        D1 = wvl * Dz / DROI  # width of central lobe, note D1 == 1/arg [m]
        R = Dz  # wavefront radius of curvature [m] is equivalent to the propagation distance

        # Define components of point source definition
        # Scale for power, Ensures final units are in [sqrt(W)/m]
        if P.unit == (u.ph/(u.s*u.m**2)):
            A = wvl * Dz * np.sqrt(P)
        if P.unit == (u.W / u.m**2):
            A = wvl * Dz * np.sqrt(P)
        elif P.unit == u.W:
            A = (
                wvl * Dz * np.sqrt(P) / u.m
            )  # Scaled to the squareroot of (power/meter^2), overall unit [sqrt(W)*m]
        else:
            A = wvl * Dz * np.sqrt(P)
            print(
                "Units for flux were unexpected. Power coefficient should have units of sqrt(W)/m. The flux units produced {} instead.".format(
                    A.unit
                )
            )
        # Limits field amplitude in x direction
        sinc_x = np.sinc((x1 / D1) * u.rad)  # Unitless
        # sinc_x_2 = np.sinc((x1*arg).value) #Unitless, equivalent to statement above
        # Limits field amplitude in y direction
        sinc_y = np.sinc((y1 / D1) * u.rad)  # Unitless
        # sinc_y_2 = np.sinc((y1*arg).value) #Unitless, equivalent to statement above
        #
        phase_dim = np.exp(-((r1 / (4 * D1)) ** 2))  # Unitless
        # phase_dim_2 = np.exp(-((r1*arg)/4)**2) # Unitless, equivalent to statement above
        # Spatial connection of phase as defined by a sinc function
        phase_spat = np.exp(((-1j * k) / (2 * R) * r1**2).value) / D1**2  # [1/m^2]
        # phase_spat_2 = np.exp(((-1j*k)/(2*R)*r1**2).value)*arg**2 # [1/m^2] Note arg^2 = DROI^2/(wvl*Dz)^2 , equivalent to statement above

        # Define the point source
        Ur = A * np.multiply(
            phase_spat, np.multiply(sinc_x, np.multiply(sinc_y, phase_dim))
        )

        # Plot the point sources if requested
        if plot_source == True:
            plane_title = "Initial Sources"
            self.plot_field(
                Ur, x1, y1, time, source_name, plane_title, sim_name, log_plot=False
            )
        return Ur

    def shifted_point_source_sinc_func(
        self,
        x1,
        y1,
        r1,
        rc,
        xc,
        yc,
        Dz,
        D2,
        wvl,
        k,
        A,
        time=0,
        source_name=0,
        sim_name="sim_name",
        plot_source=False,
    ):

        xc_mat = x1 - xc
        yc_mat = y1 - yc
        rc_mat, thetac_mat = self.cart2pol(xc_mat, yc_mat)
        arg = D2 / (wvl * Dz)
        Ur = A * np.multiply(
            np.exp((-1j * k) / (2 * Dz) * r1**2)
            * np.exp((1j * k) / (2 * Dz) * rc_mat**2),
            np.multiply(
                np.exp((-1j * k) / (Dz) * np.multiply(rc_mat, r1)),
                arg**2
                * np.multiply(
                    np.sinc(arg * (x1 - xc)),
                    np.multiply(
                        np.sinc(arg * (y1 - yc)), np.exp(-((arg / 4 * r1) ** 2))
                    ),
                ),
            ),
        )

        # Plot the point sources if requested
        if plot_source == True:
            # Only plot for the first time step otherwise too many images
            directory = os.getcwd()
            newdirectory = (
                directory + "/" + sim_name + "/ProcessingImages/PSFs/Initial_Sources"
            )
            if not os.path.exists(newdirectory):
                os.makedirs(newdirectory)
            try:
                fig, ax = plt.subplots(dpi=200)
                ax.contourf(xc_mat, yc_mat, np.real(Ur), cmap=plt.get_cmap("gray"))
                ax.axis("equal")
                ax.set_title("Source Complex Field")
                ax.set_xlabel("Source Plane [m]")
                ax.set_ylabel("Source Plane [m]")
                plot_name = (
                    "complex_field_source_" + str(source_name) + "_time_" + str(time)
                )
                plt.savefig(newdirectory + "/" + plot_name + ".pdf")
                plt.close()
            except:
                print("Contour Failed")

            try:
                PSF = np.real(np.multiply(Ur, np.conjugate(Ur)))
                if np.min(PSF) == 0.0:
                    PSF = PSF + 1e-50
                fig, ax = plt.subplots(dpi=200)
                pcm = ax.contourf(
                    xc_mat,
                    yc_mat,
                    PSF,
                    locator=ticker.AutoLocator(),
                    cmap=plt.get_cmap("gray"),
                )
                cbar = fig.colorbar(pcm)
                cbar.set_label("Power [W]", rotation=270)
                ax.axis("equal")
                ax.set_xlabel("Source Plane [m]")
                ax.set_ylabel("Source Plane [m]")
                ax.set_title("Source Real Field")
                plot_name = (
                    "real_field_source_" + str(source_name) + "_time_" + str(time)
                )
                plt.savefig(newdirectory + "/" + plot_name + ".pdf")
                plt.close()
                del PSF
            except:
                print("Contour Failed")
                PSF = 0
        return Ur

    def extended_phase_screen(
        self, observer, prop_parameters, tarray, sim_name, pscrn_file_name=0
    ):
        """
        Create and save the phase screens to be used in a propagation
        simulation.

        Parameters
        ----------
        observer : dict
            Dictionary with observer properties defined with Optical System
            module.
        prop_parameters : dict, optional
            Dictionary with parameters of the propagation.
        tarray : skylib.timelib.Time
            Time object that includes all the time steps in a simulation.
        sim_name : string
            Simulation name to separate different simulations.
        pscrn_file_name : string, optional
            Name of the HDF5 file that contains all the phase screens.
            The default is 0.

        Returns
        -------
        pscrn_file_name : string
            Name of the HDF5 file that contains all the phase screens.
        prop_parameters : dict
            Dictionary with parameters of the propagation.

        """
        # Take the results of the geometric analysis to develop phase screens
        # Unpack propagation parameters
        delta1 = prop_parameters["delta1"]
        delta2 = prop_parameters["deltan"]
        Dz = prop_parameters["L"]
        N = prop_parameters["N"]
        nscrns = int(prop_parameters["nscrns"])
        Cn = prop_parameters["Cn"]
        r0scrn = prop_parameters["r0scrn"]
        if "phase screen itr" not in prop_parameters.keys():
            if "timescale" not in prop_parameters.keys():
                timescale = "timescale"
                while isinstance(timescale, float) == False:
                    timescale = input(
                        "Please input how many seconds as a float before the phase screen should be updated: "
                    )
                    try:
                        timescale = float(timescale)
                        prop_parameters["timescale"] = timescale * u.second
                    except:
                        print(
                            "Please input a float for the number of seconds before a new phase screen should be created."
                        )
            else:
                timescale = prop_parameters["timescale"]
            timescale = timescale * u.s
            # Calculate the number of sets of phase screens that should be created
            t_total = tarray.utc[:, -1] - tarray.utc[:, 0]
            # Convert difference into seconds
            if t_total[1] != 0:
                # If there are months in the time object covert it to days
                t_total_days = self.month2day(t_total)
                psn_itr = np.ceil(
                    (
                        (
                            t_total[0] * u.year
                            + t_total_days
                            + t_total[3] * u.hour
                            + t_total[4] * u.minute
                            + t_total[5] * u.second
                        )
                        / timescale
                    ).decompose()
                )
            else:
                # Add the total time and divide by the timescale for each phase screen
                # to obtain the total number of iterations of phase screens needed
                psn_itr = np.ceil(
                    (
                        (
                            t_total[0] * u.year
                            + t_total[2] * u.day
                            + t_total[3] * u.hour
                            + t_total[4] * u.minute
                            + t_total[5] * u.second
                        )
                        / timescale
                    ).decompose()
                )
            prop_parameters["phase screen itr"] = psn_itr.value
        else:
            psn_itr = prop_parameters["phase screen itr"]
        if "zernike modes" not in prop_parameters.keys():
            j = "zernike modes"
            while isinstance(j, int) == False:
                j = input(
                    "Please input an integer for how many zernike modes the phase screens should have: "
                )
                try:
                    j = int(j)
                    prop_parameters["zernike modes"] = j
                except:
                    print("Please input an integer for the number of modes.")
        else:
            j = prop_parameters["zernike modes"]

        # Calcuate the Propagation Distance Between Each Plane
        # This will assist with the creation of a proper phase screen at each plane
        # Create a vector with all the propagation plane distances
        # pdb.set_trace()
        z = np.linspace(0, Dz, nscrns + 2)
        prop_parameters["z"] = z
        n = len(z)
        # Calculate the changes in z between each plane
        delta_z = z[1:n] - z[0 : n - 1]
        prop_parameters["delta_z"] = delta_z

        # Calculate Grid Spacings at Each Propagation Plane
        # This will assist with the creation of a proper phase screen at each plane
        alpha = z / z[n - 1]  # ratio of propagation distance completed
        delta_n = (np.ones((1, n)) - alpha) * delta1 + alpha * delta2
        prop_parameters["delta_n"] = delta_n

        # Calculate the size of the aperature diameter projected with the associated FOV
        # Via trigonometry
        D0 = 2 * (Dz + observer.focus_length) * np.tan(np.deg2rad(observer.AFOV) / 2)
        # Final plane size to prevent alising
        # DROI = 4*D2
        # Calculate the slope of the resulting parallelogram
        slope = Dz / ((D0 / 2) - (4 * observer.aperature_diameter / 2))
        # Use the slope to define the diameter at each propagation plane
        Dn = D0 - 2 * z / slope
        # Use the grid spacing in each propagation plane to determine the number of points
        # Add N (N/2 on each side) to each layer for point sources on the edge
        N_total_n = np.ceil(Dn / delta_n) + N
        for i in np.arange(0, nscrns + 2):
            if N_total_n[0, i] % 2 != 0:
                N_total_n[0, i] = N_total_n[0, i] + 1
        prop_parameters["N_total_n"] = N_total_n

        # Use these dimensions to create the phase screens
        # Save the phase screens in an HDF5 file
        if pscrn_file_name == 0:
            while isinstance(pscrn_file_name, str) == False:
                pscrn_file_name = input(
                    "Please input a file name for the file that will hold the phase screens: "
                )
        directory = os.getcwd()
        newdirectory = directory + "/" + sim_name + "/PhaseScreens/"
        if not os.path.exists(newdirectory):
            os.makedirs(newdirectory)
        full_name = newdirectory + pscrn_file_name + ".hdf5"
        hf = h5py.File(full_name, "w")
        hf.close()
        pscrn_file_name = full_name
        # Loop through phase screens
        for i in np.arange(0, int(psn_itr)):
            # Don't need a phase screen for the source and observation planes
            # Start iteration at 1 and end at number of phase screens
            for idscrn in np.arange(1, n - 1, 1):
                print("This is phase screen {} of iteration {}.".format(idscrn, int(i)))
                if idscrn == 1:
                    phscrn_prop_n, Z_p_th, Cj, S_mat, j_map = self.zernike_phase_screen(
                        j,
                        Cn,
                        delta_z[idscrn],
                        observer.aperature_diameter,
                        observer.wavelength,
                        int(N_total_n[0][idscrn]),
                        delta_n[0, idscrn],
                        prop_parameters,
                        int(i),
                        ro=r0scrn[idscrn],
                        plot_modes=False,
                        sim_name=sim_name,
                    )
                    del Z_p_th
                    del Cj
                    del S_mat
                elif idscrn == n - 2:
                    phscrn_prop_n, Z_p_th, Cj, S_mat, j_map = self.zernike_phase_screen(
                        j,
                        Cn,
                        delta_z[idscrn],
                        observer.aperature_diameter,
                        observer.wavelength,
                        int(N_total_n[0][idscrn]),
                        delta_n[0, idscrn],
                        prop_parameters,
                        int(i),
                        j_map=j_map,
                        ro=r0scrn[idscrn],
                        plot_modes=True,
                        sim_name=sim_name,
                    )
                else:
                    phscrn_prop_n, Z_p_th, Cj, S_mat, j_map = self.zernike_phase_screen(
                        j,
                        Cn,
                        delta_z[idscrn],
                        observer.aperature_diameter,
                        observer.wavelength,
                        int(N_total_n[0][idscrn]),
                        delta_n[0, idscrn],
                        prop_parameters,
                        int(i),
                        j_map=j_map,
                        ro=r0scrn[idscrn],
                    )
                screen_name = "time_itr_" + str(int(i)) + "/screen_" + str(idscrn)
                # Open HDF5 File to save
                hf = h5py.File(full_name, "r+")
                hf.create_dataset(
                    screen_name,
                    data=phscrn_prop_n,
                    compression="gzip",
                    compression_opts=9,
                )
                hf.close()
                del phscrn_prop_n

        return pscrn_file_name, prop_parameters

    def zernike_phase_screen(
        self,
        j,
        Cn,
        L,
        D2,
        lmbda,
        Nn,
        deln,
        prop_parameters,
        psn_itr,
        j_map=[],
        ro=0,
        Cj=0,
        S=0,
        Z_p_th=0,
        plot_modes=False,
        sim_name="filename",
    ):
        """
        Given all the inputs the function outputs a Kolmogorov phase screen built
        using Zernike modes. This will represent the satistical change in phase
        that occurs by transmitting through atmospheric turbulence.

        Parameters
        ----------
        j : int
            Number of Zernike modes in phase screen.
        Cn : float
            Cn is a parameter that describes the severity of turbulence abberations. [m^(-2/3)]
        L : float
            Length of path through turbluence [m]
        D2 : float
            Aperature size of the sensor's optic. [m]
        lmbda : float
            Wavelength of light in vacuum being observed [1/m]
        Nn : int
            Number of needed points in the phase screen array.
        deln : float
            Spacing size between points in the phase screen array.
        prop_parameters : dict
            Dictionary with parameters of the propagation.
        psn_itr : int
            Iteration of the phase screens to account for longer simulations
            where frozen flow cannot be assumed.
        j_map : dict, optional
            Organized mode numbers mapped to n and m parameters. The default is [].
        ro : float, optional
            Fried Parameter for this screen [m]. The default is 0.
        Cj : np.array, optional
            Output from the previously completed decomposition that describes
            the covariance of the Zernike modes. The default is 0.
        S : np.array, optional
            Output from the previously completed decomposition that describes
            the variance of mu. The default is 0.
        Z_p_th : np.array, optional
            Output of a previously created zernike modes, unmultiplied by the
            randomly drawn coefficient. The default is 0.
        plot_modes : bool, optional
            Boolean to determine if the individual Zernike modes should be
            plotted. The default is False.
        sim_name : string, optional
            Simulation name to save the phase screens. The default is 'filename'.

        Returns
        -------
        theta_atm : np.array
            Final phase screen for this screen size. [rad]
        Z_p_th : np.array
            Zernike mode array, unmultiplied by the randomly drawn coefficient.
            Output to generate new screens when the frozen flow assumption is
            passed in longer simulations.
        Cj : np.array
            Descomposition of the Zernike Covariance Matrix.
        S : np.array
            Variance vector S from the Cholesky Decomposition which can be used
            to define future mu vectors without recalculating the decomposition.
            It is only defined if the number of modes is greater than 10.
        j_map : dict
            Organized mode numbers mapped to n and m parameters.

        """
        # # Establish Size of Phase Screen
        scale = 1
        memory_constraints = False
        while memory_constraints == False:
            # reduce the number of steps for the computer's memory useage
            try:
                steps = int(Nn / scale)
                while steps > 10000:
                    scale = scale * 2
                    steps = int(Nn / scale)
                if steps % 2 != 0:
                    steps += 1
                xn = np.linspace(-Nn / 2 * deln, Nn / 2 * deln, steps)
                yn = np.linspace(-Nn / 2 * deln, Nn / 2 * deln, steps)
                x_mat, y_mat = np.meshgrid(xn, yn)
                # Test that the Zernike matrix can be formed
                Z_test = np.zeros((steps, steps, j - 1))
                r, theta = self.cart2pol(x_mat, y_mat)
                memory_constraints = True
                # Save space in memory
                del x_mat
                del y_mat
                del Z_test
            except:
                scale = scale * 2

        if "scale_pscrn" not in list(prop_parameters.keys()):
            prop_parameters["scale_pscrn"] = {}
            prop_parameters["scale_pscrn"][psn_itr] = []
        elif psn_itr not in list(prop_parameters["scale_pscrn"].keys()):
            prop_parameters["scale_pscrn"][psn_itr] = []
        if "deltan_pscrn" not in list(prop_parameters.keys()):
            prop_parameters["deltan_pscrn"] = {}
            prop_parameters["deltan_pscrn"][psn_itr] = []
        elif psn_itr not in list(prop_parameters["deltan_pscrn"].keys()):
            prop_parameters["deltan_pscrn"][psn_itr] = []
        if "N_total_pscrn" not in list(prop_parameters.keys()):
            prop_parameters["N_total_pscrn"] = {}
            prop_parameters["N_total_pscrn"][psn_itr] = []
        elif psn_itr not in list(prop_parameters["N_total_pscrn"].keys()):
            prop_parameters["N_total_pscrn"][psn_itr] = []
        prop_parameters["scale_pscrn"][psn_itr].append(scale)
        del_pscrn = xn[1] - xn[0]
        prop_parameters["deltan_pscrn"][psn_itr].append(del_pscrn)
        prop_parameters["N_total_pscrn"][psn_itr].append(steps)

        # Calculate D for the coefficient determination to be the diameter of the phase screen
        D_phscrn = ((Nn * deln) ** 2 + (Nn * deln) ** 2) ** (1 / 2)

        # Create the map between j modes of polynommials and n & m parameters
        if j_map == []:
            # Only need to run once
            j_map = self.j_map(j)
            try:
                check = j_map[j + 1]
                print("Check is true: {}".format(check))
                j += 1
                print("j is now {}.".format(j))
            except KeyError:
                j = j

            if j > 10:
                # Reorganize the mapping to create a decomposable matrix
                j_map = self.reorder_j_map(j, j_map)

        # Create the Zernike Polynomial Matrix for these dimensions and number of modes, if not already given
        if np.size(Z_p_th) == 1:
            roh = r / (D_phscrn / 2)  # Percentage of Radius
            # print('The percentage of radius, roh is: {}'.format(roh))
            Z_p_th = self.zernike_poly_matrix(j, j_map, steps, steps, roh, theta)
            # Z_p_th = (Z_p_th +np.pi)%(2*np.pi)-np.pi
            # Save space in memory
            del r
            del theta
            del roh

        # Calculate Fried Parameter
        if ro == 0:
            ro = self.fried_parameter(Cn, L, lmbda)
        # ro = .01
        # print('{} is the Fried parameter for this simulation'.format(ro))

        # Calculate the a_j Coefficients
        # print('{} is the phase screen diameter'.format(D_phscrn))

        if np.size(Cj) == 1:
            a_j, Cj, S = self.a_coeff(j, j_map, ro, D2)
        else:
            a_j = self.a_coeff_Cj_input(j, j_map, ro, D2, Cj, S)

        # print('This is the aj coefficient matrix of size {}: {}.'.format(np.shape(a_j),a_j))
        # print('{} is the covariance matrix'.format(Cj))
        # print((D_phscrn/ro)**(5/6))

        # Use matrix math to apply the a_j weighting and sum the Zernike modes
        theta_atm = np.einsum("ijk,k->ij", Z_p_th, a_j) * (D_phscrn / ro) ** (5 / 6)
        theta_atm = ((theta_atm + np.pi) % (2 * np.pi) - np.pi) * u.rad

        # print('This is the phase screen of size {}: {}'.format(np.shape(theta_atm),theta_atm))

        # Plot First Few Zernike Modes
        if plot_modes == True and j > 10:
            # Only wrap if plotting
            Z_p_th = (Z_p_th + np.pi) % (2 * np.pi) - np.pi
            # Recreate x and y matrices
            x_mat, y_mat = np.meshgrid(xn, yn)
            # Set up subplots of Zernike modes
            fig, axs = plt.subplots(
                2, 3, sharey=True, sharex=True, constrained_layout=True, dpi=200
            )
            axs[0, 0].contourf(
                x_mat,
                y_mat,
                Z_p_th[:, :, 1],
                cmap=plt.get_cmap("gray"),
                vmin=-np.pi,
                vmax=np.pi,
            )
            # axs[0,0].set_xlabel('[m]')
            # axs[0,0].set_ylabel('[m]')
            axs[0, 0].set_title("Tilt Z3")
            axs[0, 0].axes.xaxis.set_visible(False)
            axs[0, 1].contourf(
                x_mat,
                y_mat,
                Z_p_th[:, :, 3],
                cmap=plt.get_cmap("gray"),
                vmin=-np.pi,
                vmax=np.pi,
            )
            # axs[0,1].set_xlabel('[m]')
            # axs[0,1].set_ylabel('[m]')
            axs[0, 1].set_title("Astigmatism Z5")
            axs[0, 1].axes.xaxis.set_visible(False)
            axs[0, 2].contourf(
                x_mat,
                y_mat,
                Z_p_th[:, :, 2],
                cmap=plt.get_cmap("gray"),
                vmin=-np.pi,
                vmax=np.pi,
            )
            # axs[0,2].set_xlabel('[m]')
            # axs[0,2].set_ylabel('[m]')
            axs[0, 2].set_title("Defocus Z4")
            axs[0, 2].axes.xaxis.set_visible(False)
            axs[1, 0].contourf(
                x_mat,
                y_mat,
                Z_p_th[:, :, 6],
                cmap=plt.get_cmap("gray"),
                vmin=-np.pi,
                vmax=np.pi,
            )
            # axs[1,0].set_xlabel('[m]')
            # axs[1,0].set_ylabel('[m]')
            axs[1, 0].set_title("Coma Z8")
            axs[1, 1].contourf(
                x_mat,
                y_mat,
                Z_p_th[:, :, 7],
                cmap=plt.get_cmap("gray"),
                vmin=-np.pi,
                vmax=np.pi,
            )
            # axs[1,1].set_xlabel('[m]')
            # axs[1,1].set_ylabel('[m]')
            axs[1, 1].set_title("Trefoil Z9")
            pcm = axs[1, 2].contourf(
                x_mat,
                y_mat,
                theta_atm.value,
                cmap=plt.get_cmap("gray"),
                vmin=-np.pi,
                vmax=np.pi,
            )
            # axs[1,2].set_xlabel('[m]')
            # axs[1,2].set_ylabel('[m]')
            axs[1, 2].set_title("Final Screen")
            # plt.xlabel('X Pixels',loc='center')
            # plt.ylabel('Y Pixels',loc='center')
            cbar = fig.colorbar(pcm, ax=axs[:, 2], location="right", shrink=0.6)
            cbar.set_label("Phase Shift [rad]", rotation=270, labelpad=10)
            # fig.supxlabel('phase screen distance [m]')
            # fig.supylabel('phase screen distance [m]')
            directory = os.getcwd()
            newdirectory = directory + "/" + sim_name + "/ProcessingImages/ZernikeModes"
            if not os.path.exists(newdirectory):
                os.makedirs(newdirectory)
            plot_name = sim_name + "_zernike_mode"
            plt.savefig(newdirectory + "/" + plot_name + ".pdf")
            plt.close()
            del x_mat
            del y_mat

        return (
            theta_atm,
            Z_p_th,
            Cj,
            S,
            j_map,
        )  # Last 3 Outputs Will Assist Creation of Subsequent Screens With These Dimensions

    def zernike_poly_matrix(self, j, j_map, Nx, Ny, r, theta):
        """
        Calculate the Radial Function given a radial degree, azimuthal frequency,
        and polar corrdinate in the plane.
        This Zernike Polynomial Matrix will be the size of the screen being created
        by the number of modes being simulated. (num_y x num_x x num_modes)

        Parameters
        ----------
        j : int
            Number of Zernike modes.
        j_map : dict
            Organized mode numbers mapped to n and m parameters.
        r : np.array
            Radial values of polar coordinates as percentage of full screen.
            This should be dimensionless with values between 0 and 1.
        theta : np.array
            Angular values of polar coordinates.

        Returns
        -------
        Z_part : np.array
            Array the size of the number of zernike modes.

        """

        j_count = 1
        Z_matrix = np.zeros((Ny, Nx, j - 1), dtype=np.float32)
        # Z_matrix = np.ones((j-1,1))

        # print('The radius is {}.'.format(r))
        # print('The theta is {}.'.format(theta))

        while j_count < j:
            index = j_map[j_count + 1]
            # print(j_map[j_count+1])
            s = 0
            n = index[0]
            m = np.abs(index[1])
            s_weight = index[3]
            # print('This is the weighting s matrix of size {}: {}'.format(np.shape(s_weight),s_weight))
            if n == m:
                r_array = r ** (n - 2 * s)
                Rnm = r_array * s_weight
            else:
                r_array = np.zeros((Ny, Nx, int((n - m) / 2 + 1)))
                while s <= (n - m) / 2:
                    r_array[:, :, s] = r ** (n - 2 * s)
                    s += 1

                Rnm = np.einsum("ijk,k->ij", r_array, s_weight)
            # print('This is the r array of size {}: {}'.format(np.shape(r_array),r_array))
            # print('This is the Rnm matrix of the {}th mode with size {}: {}'.format(index[2],np.shape(Rnm),Rnm))

            # R_sum = 0
            # indx = (n-m)/2
            # print('The index will go to {}.'.format(indx))
            # while s <= (n-m)/2:
            #     # print('n is {}.'.format(n))
            #     # print('m is {}.'.format(m))
            #     # print('s is {}.'.format(s))
            #     # R_sum = R_sum + ((-1)**s*sps.factorial(n-s))/(sps.factorial(s)*sps.factorial((n+m)/2-s)*sps.factorial((n-m)/2-s))*r**(n-2*s)

            #     s += 1

            # print('The R_sum is {}.'.format(R_sum))

            # cosine_term = np.cos(m*theta)
            # print('The cosine term is {}.'.format(cosine_term))
            # sine_term = np.sin(m*theta)
            # print('The sine term is {}.'.format(sine_term))

            if m == 0:
                Z_matrix[:, :, j_count - 1] = (n + 1) ** (0.5) * Rnm
                # print('Azimuthal frequency is 0. m = 0.')
            elif index[1] > 0:
                Z_matrix[:, :, j_count - 1] = np.multiply(
                    (n + 1) ** (0.5) * (2 ** (0.5)) * Rnm, np.cos(m * theta)
                )
                # print('This is an even mode.')
            else:
                Z_matrix[:, :, j_count - 1] = np.multiply(
                    (n + 1) ** (0.5) * (2 ** (0.5)) * Rnm, np.sin(m * theta)
                )
                # print('This is an odd mode.')

            # if m == 0:
            #     Z_part = (n+1)**(.5)*R_sum
            #     #print('Azimuthal frequency is 0. m = 0.')
            # elif index[1] > 0:
            #     Z_part = (n+1)**(.5)*R_sum*2**(.5)*np.cos(m*theta)
            #     #print('This is an even mode.')
            # else:
            #     Z_part = (n+1)**(.5)*R_sum*2**(.5)*np.sin(m*theta)
            #     #print('This is an odd mode.')

            # print('Z_part is {}.'.format(Z_part))

            # Z_matrix[j_count-1] = Z_part
            j_count += 1

        # print('This is the Zernike polynomial matrix of size {}: {}'.format(np.shape(Z_matrix),Z_matrix))

        return Z_matrix

    def fried_parameter(self, Cn, L, lmbda):
        """
        Calculate Fried's Parameter for Creation of Zernike Modes and for a Plane Wave

        Parameters
        ----------
        Cn : float
            Cn is a parameter that describes the severity of turbulence abberations. [m^(-2/3)]
        L : float
            Length of path through turbluence [m]
        lmbda : float
            Wavelength of light in vacuum being observed [1/m]

        Returns
        -------
        ro : float
            Fried Parameter [m]

        """
        k = 2 * np.pi / lmbda
        # The Fried parameter simplifies to this expression when Cn is not a function of the propogation distance
        # an integral is required otherwise.
        ro = (0.423 * (Cn) * L * (k**2)) ** (-3 / 5)
        # Integral Version
        # ro = (0.423*k**2*np.trapz(Cn(L),range(0,L,step)))**(-3/5)
        return ro

    def fried_parameter_sw(self, Cn, L, lmbda):
        """
        Calculate Fried's Parameter for a Spherical Wave

        Parameters
        ----------
        Cn : float
            Cn is a parameter that describes the severity of turbulence abberations. [m^(-2/3)]
        L : float
            Length of path through turbluence [m]
        lmbda : float
            Wavelength of light in vacuum being observed [m]

        Returns
        -------
        ro : float
            Fried Parameter [m]

        """
        k = 2 * np.pi / lmbda
        # The Fried parameter simplifies to this expression when Cn is not a function of the propogation distance
        # an integral is required otherwise.
        ro = (0.423 * (Cn) * L * (k**2) * (3 / 8)) ** (-3 / 5)
        return ro

    def a_coeff_Cj_input(self, j, j_map, ro, D, Phi, S=0):
        """
        Randomly choose the weighting of the Zernike modes to build the phase screen.
        This simplified version is for iterations after the Zernike covariance
        matrix has already been created and does not need to be recalculated.

        Parameters
        ----------
        j : int
            Number of Zernike modes.
        j_map : dict
            Mapping of Zernike modes to the n and m parameters.
        ro : float
            Fried parameter which scales the modes by the severity of the
            turbulance and number of screens.
        D : float
            Maximum diameter of the recieving screen.
        Phi : np.array
            Output from the previously completed decomposition that describes
            the covariance of the Zernike modes.
        S : TYPE, optional
            Output from the previously completed decomposition that describes
            the variance of mu.
            The default is 0 which is the case for non-Cholesky Decompositions
            which is possible when the number of modes is less than 10.

        Returns
        -------
        a_matrix : np.array
            Coefficients for the Zernike modes which will be multipled by the modes to create one overall screen.

        """
        # Choose new mu for this phase screen
        if j < 10:
            mu = -1 * np.ones((j - 1, 1)) + 2 * np.random.rand(j - 1, 1)
            # print(mu)
        else:
            mu = []
            for i in S:
                mu_i = np.random.normal(0, i)
                # print(mu_i)
                mu.append(mu_i)

        # Multiply the decomposition by the mu
        a_matrix = np.matmul(Phi, mu)
        # print('{} is the aj matrix.'.format(a_matrix))

        return a_matrix

    def a_coeff(self, j, j_map, ro, D):
        """
        Randomly choose the weighting of the Zernike modes to build the phase screen.
        This complex version is for the initial iteration of generating the coefficients.

        Parameters
        ----------
        j : int
            Number of Zernike modes.
        j_map : dict
            Mapping of Zernike modes to the n and m parameters.
        ro : float
            Fried parameter which scales the modes by the severity of the
            turbulance and number of screens.
        D : float
            Maximum diameter of the recieving screen.

        Returns
        -------
        a_matrix : np.array
            Coefficients for the Zernike modes which will be multipled by the
            modes to create one overall screen.
        Phi : np.array
            Descomposition of the Zernike Covariance Matrix.
        S : np.array
            Variance vector S from the Cholesky Decomposition which can be used
            to define future mu vectors without recalculating the decomposition.
            It is only defined if the number of modes is greater than 10.

        """
        # Generate Covariance Matrix
        Cj = self.zernike_cov(j, j_map, ro, D)
        # print('{} is the covariance matrix'.format(Cj))

        if j < 10:
            # Cholesky Decomposition
            Phi = np.linalg.cholesky(Cj)
            S = 0
            # Generate Unit Variance Random Vector
            mu = -1 * np.ones((j - 1, 1)) + 2 * np.random.rand(j - 1, 1)
            # print(mu)
        else:
            # SVD Decomposition
            Phi, S, V = np.linalg.svd(Cj)
            # Generate Gaussian Random Vector with Variance from S
            # print(S)
            mu = []
            for i in S:
                mu_i = np.random.normal(loc=0, scale=i)
                # print(mu_i)
                mu.append(mu_i)

        # print('mu is {}'.format(mu))
        # Multiply the decomposition by the mu
        a_matrix = np.matmul(Phi, mu)
        # print('{} is the aj matrix.'.format(a_matrix))

        return a_matrix, Phi, S

    def zernike_cov(self, j, j_map, ro, D):
        """
        Calculates the Zernike covariance matrix that will subsequently be
        decomposed and multiplied by a random normalized vector to scale the
        Zernike phase screens.

        Parameters
        ----------
        j : int
            Number of Zernike modes.
        j_map : dict
            Mapping of Zernike modes to the n and m parameters.
        ro : float
            Fried parameter which scales the modes by the severity of the
            turbulance and number of screens.
        D : float
            Maximum diameter of the recieving screen.

        Returns
        -------
        Z_cov : np.array
            Zernike covariance matrix describing the covariance of each mode.
            Note that the j_map helps build the matrix appropriately, so that
            it can be decomposed. Decomposition requires a Cholesky decomposition
            and thus a specific ordering when there are more than 10 modes.

        """
        # The covariance matrix will be the size of jxj
        # A nested for loop will accomplish this
        Z_cov = np.ones((j - 1, j - 1))
        j_count = 2

        # print('This is the mapping of n and m to j values. {}'.format(j_map))

        while j_count <= j:
            index = j_map[j_count]
            n = index[0]
            m = index[1]
            j_index = index[2]

            jprime_count = 2
            while jprime_count <= j:
                # print(j_count)
                # print(jprime_count)
                indexprime = j_map[jprime_count]
                nprime = indexprime[0]
                mprime = indexprime[1]
                jprime_index = indexprime[2]

                delta = self.kronecker_del(
                    np.abs(m), np.abs(mprime), j_index, jprime_index
                )

                # print('{} is the kronecker output.' .format(delta))
                if delta == 0:
                    Z_cov[j_count - 2, jprime_count - 2] = 0
                else:
                    # print('{} is n'.format(n))
                    # print('{} is n prime'.format(nprime))
                    # print('{} is m'.format(m))
                    # print('{} is m prime'.format(mprime))
                    Kzz = self.k_coeff(n, nprime, np.abs(m))
                    # print('{} is the Kzz coefficient.' .format(Kzz))
                    # print('')
                    # comp1 = sps.gamma((n+nprime-(5/3))/2)
                    # print('{} is the output of the first gamma function in the numerator'.format(comp1))
                    # comp2 = (D/ro)**(5/3)
                    # print('{} is the output of the D over ro term'.format(comp2))
                    # comp3 = sps.gamma((n-nprime-(17/3))/2)
                    # print('{} is the output of the first gamma function in the denominator'.format(comp3))
                    # comp4 = sps.gamma((nprime-n-(17/3))/2)
                    # print('{} is the output of the second gamma function in the denominator'.format(comp4))
                    # comp5 = sps.gamma((n+nprime+(23/3))/2)
                    # print('{} is the output of the third gamma function in the denominator'.format(comp5))
                    Z_cov[j_count - 2, jprime_count - 2] = (
                        delta
                        * (
                            Kzz
                            * sps.gamma((n + nprime - (5 / 3)) / 2)
                            * (D / ro) ** (5 / 3)
                        )
                        / (
                            sps.gamma((n - nprime + (17 / 3)) / 2)
                            * sps.gamma((nprime - n + (17 / 3)) / 2)
                            * sps.gamma((n + nprime + (23 / 3)) / 2)
                        )
                    )
                    # Z_cov[j_count-2,jprime_count-2] = delta*(Kzz*sps.gamma((n+nprime-(5/3))/2))/(sps.gamma((n-nprime+(17/3))/2)*sps.gamma((nprime-n+(17/3))/2)*sps.gamma((n+nprime+(23/3))/2))

                jprime_count += 1
            j_count += 1

        Z_cov = np.real(Z_cov)

        return Z_cov

    def k_coeff(self, n, nprime, m):
        """
        Calculate the covariance coefficient between two Zernike modes.

        Parameters
        ----------
        n : int
            Radial degree of the current Zernike mode.
        nprime : int
            Radial degree of the compared Zernike mode.
        m : int
            Azimuthal frequency of the current Zernike mode.

        Returns
        -------
        Kzzprime : float
            Kzz' coefficient that scales the defined covariance between two
            modes.

        """
        k_const = (
            sps.gamma(14 / 3)
            * ((24 / 5) * sps.gamma(6 / 5)) ** (5 / 6)
            * (sps.gamma(11 / 6)) ** 2
            / (2 * np.pi**2)
        )
        # print('{} is the magnitude of the constant part of Kzz'.format(k_const))
        Kzzprime = k_const * (
            (-1) ** ((n + nprime - 2 * m) / 2) * ((n + 1) * (nprime + 1)) ** (0.5)
        )

        return Kzzprime

    def kronecker_del(self, m, mprime, j, jprime):
        """
        Calculate the kronecker delta for a particular pair of Zernike modes.
        This binary description makes sure the covariance is only defined for
        real and related modes. This requires the azimuthal frequency (m) to be
        equivalent and the mode number to be either equivalent or both modes be
        either even or odd.

        Parameters
        ----------
        m : int
            Azimuthal frequency of the current mode.
        mprime : int
            Azimuthal frequency of the compared mode.
        j : int
            Mode number of the current mode.
        jprime : int
            Mode number of the compared mode.

        Returns
        -------
        delta : int
            Value of 0 or 1 to describe if the two modes have a covariance
            between them.

        """
        # print('{} is m'.format(m))
        # print('{} is m prime'.format(mprime))
        # print('{} is j'.format(j))
        # print('{} is j prime'.format(jprime))
        if m == mprime:
            # print('m is equal to mprime')
            if m == 0:
                delta = 1
                # print('Mode is in the first column.')
            elif j % 2 == jprime % 2:
                delta = 1
                # print('The parity is equal.')
            else:
                delta = 0
                # print('Modes are in the same column, but modes dont have the same parity.')
        else:
            delta = 0
            # print('The modes are not in the same column.')

        # print('{} is the k delta.'.format(delta))
        # print('')
        return delta

    def j_map(self, j):
        """
        Given the desired number of included Zernike modes in the phase screen
        determine the mapping between j and radial degree, n, and the azimuthal
        frequency, m, values.

        Parameters
        ----------
        j : int
            Number of Zernike modes desired

        Returns
        -------
        j_map : dict
            Mapping of Zernike modes to assign appropriate weighting to the
            modes for subsequent phase screen generation.

        """
        n = 0  # Radial degree of mode
        m = 0  # Azimuthal frequency of mode
        j_count = 1
        j_map = {}
        while j_count <= j:

            Rnm = self.r_mn_array(n, np.abs(m))

            if n % 2 == 0:
                if n == 0 and m == 0:
                    j_count += 1
                elif m == 0:
                    j_map[j_count] = [n, m, j_count, Rnm]
                    j_count += 1
                else:
                    if j_count % 2 == 0:
                        j_map[j_count] = [n, m, j_count, Rnm]
                        j_count += 1
                        j_map[j_count] = [n, -m, j_count, Rnm]
                        j_count += 1
                    else:
                        j_map[j_count] = [n, -m, j_count, Rnm]
                        j_count += 1
                        j_map[j_count] = [n, m, j_count, Rnm]
                        j_count += 1

                if n == m:
                    n += 1
                    m = 0
                else:
                    m += 2

            elif n % 2 == 1:
                if m == 0:
                    j_count = j_count
                else:
                    if j_count % 2 == 0:
                        j_map[j_count] = [n, m, j_count, Rnm]
                        j_count += 1
                        j_map[j_count] = [n, -m, j_count, Rnm]
                        j_count += 1
                    else:
                        j_map[j_count] = [n, -m, j_count, Rnm]
                        j_count += 1
                        j_map[j_count] = [n, m, j_count, Rnm]
                        j_count += 1

                if n == m:
                    n += 1
                    m = 0
                elif m == 0:
                    m += 1
                else:
                    m += 2
            else:
                print("Mod operator error.")

        return j_map

    def reorder_j_map(self, j, j_map):
        """
        If the number of modes is greater than 10, the j_map must be
        reorganized. This ensures the decomposition of the covariance matrix
        is possible.

        Parameters
        ----------
        j : int
            Number of Zernike modes desired
        j_map : dict
            Mapping of Zernike modes to assign appropriate weighting to the
            modes for subsequent phase screen generation.

        Returns
        -------
        j_map_new : dict
            Reorganized mapping of the Zernike modes.

        """
        j_map_new = {}
        # The block diagonal version of the covariance matrix requires the modes
        # with the same azimuthal frequency to be aligned
        modes = np.arange(2, j + 1, 1)
        j_count = 2
        while j_count <= j:
            index = j_map[j_count]
            m = index[1]
            used_modes = []
            count = 0
            for i in modes:
                # print(i)
                index2 = j_map[i]
                # print(index2)
                nprime = index2[0]
                mprime = index2[1]
                Rmnprime = index2[3]
                if m == mprime:
                    j_map_new[len(j_map_new) + 2] = [
                        nprime,
                        mprime,
                        i,
                        Rmnprime,
                    ]  # Adding the index on the end to keep track of which mode is associated with each key
                    used_modes.append(count)
                if modes == []:
                    break
                count += 1

            # print(j_map_new)
            # print(used_modes)
            modes = np.delete(modes, used_modes)
            # print(modes)
            j_count += 1

        return j_map_new

    def r_mn_array(self, n, m):
        """
        Defines each Zernike mode's Rnm value as defined by the radial
        degree, n, and azimuthal frequency, m.


        Parameters
        ----------
        n : int
            Radial degree of the Zernike mode.
        m : int
            Azimuthal frequency of the Zernike mode.

        Returns
        -------
        Rnm : float
            Rnm value.

        """
        s = 0
        if n == m:
            Rnm = ((-1) ** s * sps.factorial(n - s)) / (
                sps.factorial(s)
                * sps.factorial((n + m) / 2 - s)
                * sps.factorial((n - m) / 2 - s)
            )
        else:
            Rnm = np.arange(0, (n - m) / 2 + 1, 1)
            while s <= (n - m) / 2:
                Rnm[s] = ((-1) ** s * sps.factorial(n - s)) / (
                    sps.factorial(s)
                    * sps.factorial((n + m) / 2 - s)
                    * sps.factorial((n - m) / 2 - s)
                )
                s += 1
        return Rnm

    def cart2pol(self, x, y):
        """
        With the definition of x and y locations on a grid, convert to polar coordinates.

        Parameters
        ----------
        x : np.array
            X locations on grid.
        y : np.array
            Y locations on grid.

        Returns
        -------
        r : np.array
            Radius of polar coordinates on grid in units of x and y.
        phi : TYPE
            Angle of polar coordinates on grid in radians.

        """
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (r, phi)

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def circ(self, x, y, D):
        """
        Create a circular mask with the x and y grids with the origin at the center and the mask diameter.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        D : TYPE
            DESCRIPTION.

        Returns
        -------
        z : TYPE
            DESCRIPTION.

        """
        r = np.sqrt(x**2 + y**2)
        dim = np.shape(r)
        z = np.ones(dim, dtype=np.float16)
        z[r > D / 2] = 0
        z[r == D / 2] = 0.5
        return z

    def ro_multi_fun(self, X, A, b):
        total_diff = np.sum((np.matmul(A, X) - np.transpose(b)) ** 2)
        return total_diff

    def corr2_ft(self, u1, u2, mask, N_total, delta_n):
        # Pull off size of the array to calculate the correlation
        Nx = np.size(u1, 1)
        Ny = np.size(u1, 0)
        # Perform a 2-D discrete correlation to capture effects of the aperature mask
        # frequency grid spacing
        delta_f = 1 / (N_total * delta_n)
        # discrete fourier transform of the signals
        U1 = self.ft2(np.matmul(u1, mask), delta_n, delta_n, Nx, Ny)
        U2 = self.ft2(np.matmul(u2, mask), delta_n, delta_n, Nx, Ny)
        U12corr = self.ift2(np.matmul(np.conjugate(U1), U2), delta_f, delta_f, Nx, Ny)
        maskcorr = (
            self.ift2(
                np.abs(self.ft2(mask, delta_n, delta_n, Nx, Ny)) ** 2,
                delta_f,
                delta_f,
                Nx,
                Ny,
            )
            * delta_n**2
        )
        # Some logical term I am missing that will limit the output??
        corr = np.divide(U12corr, np.matmul(maskcorr, mask))
        return corr

    def pupil_plane_corrd(
        self,
        pixel_size_x,
        pixel_size_y,
        num_pixel_x,
        num_pixel_y,
        wvl,
        focal_length,
        D2,
    ):
        """
        Define the pupil plane coordinate requirements for defining the
        impulse response function.

        Parameters
        ----------
        pixel_size_x : float
            Pixel pitch in the dimension with the largest number of pixels in x direction.
        pixel_size_y : float
            Pixel pitch in the dimension with the largest number of pixels in y direction.
        num_pixel_x : int
            Number of pixels in the x dimension of the focal plane.
        num_pixel_y : int
            Number of pixels in the y dimension of the focal plane.
        wvl : float
            Center wavelength of the measured light.
        focal_length : float
            Focal length of the optic.
        D2 : float
            Diameter of the optic aperature

        Returns
        -------
        delta_p : float
            Spacing between points in the pupil plane.
        N : int
            Number of grid points required to simulate the pupil plane.

        """
        # Define the pupil plane coordinates
        delta_p_x = (wvl * focal_length) / (num_pixel_x * pixel_size_x)
        # Define the pupil plane coordinates
        delta_p_y = (wvl * focal_length) / (num_pixel_y * pixel_size_y)
        # Determine how many grid points are required
        # Determine the limiting number of pixels (the maximum number in any direction on the array)
        num_pixel = np.max([num_pixel_x, num_pixel_y])
        # Calculate the minimum grid points required in each direction
        N_min_x = np.ceil(D2 * 1 / delta_p_x)
        N_min_y = np.ceil(D2 * 1 / delta_p_y)
        # Determine the limiting minimum for a square array (the largest)
        N_min = np.max([N_min_x, N_min_y])
        # Only overwrite the number of pixels if more are greater than the largest pixel dimension
        if N_min > num_pixel:
            N = N_min
        else:
            N = num_pixel
        # Ensure the number of pixels returned is an even number
        if N % 2 != 0:
            N += 1
        if N_min == N_min_x:
            delta_p = delta_p_x
        else:
            delta_p = delta_p_y
        return delta_p, N

    def magnify_field(self, Uin, xin, yin, z1, z2):
        """
        Convert objective plane to the image plane coordinates

        Parameters
        ----------
        Uin : np.array
            Objective plane field.
        xin : np.array
            Original coordinates meshgrid.
        yin : np.array
            Original coordinates meshgrid.
        z1 : float
            Distance between objective plane and image plane.
        z2 : float
            Distance between lens and image plane.

        Returns
        -------
        Uout : np.array
            Objective plane field in image plane coordinates.
        xout : np.array
            Scaled coordinate meshgrid.
        yout : np.array
            Scaled coordinate meshgrid.

        """
        # Calculate the magnification based on the ratio of the objective plane to the image plane distances from the lens
        M = -z2 / z1
        # Scale the field using the similarity theorem
        Uout = 1 / np.abs(M) * Uin
        xout = M * xin
        yout = M * yin

        return Uout, xout, yout

    def downscale_field(self, Uin, subpix_per_pix_x, subpix_per_pix_y):
        """
        The input field is subsampled of the true sampling, such as the image plane. The
        scaling factor must evenly divide the samples in the input field. This function
        sums the pixels within each subpix_per_pix by subpix_per_pix region.

        Parameters
        ----------
        Uin : np.array
            Subsampled input array, this array should be in W/m**2 to ensure the complex
            components are incorporated.
        subpix_per_pix_x : int
            Number of subpixels in the x direction per final output pixel.
        subpix_per_pix_y : int
            Number of subpixels in the y direction per final output pixel.

        Returns
        -------
        Uout : np.array
            Downsampled array with units of W/m**2. The number of pixels is equivalent to
            the size of the input array divided by the subpixel_per_pixel parameter.

        """
        # Extract shape of input array to define the final output array size
        Na = np.shape(Uin)
        Nx = Na[1]
        Ny = Na[0]
        # Check if the original array is dividible by the subpix_per_pix parameters
        # If not prompt the user for a different subpix parameter
        while Nx % subpix_per_pix_x != 0:
            subpix_per_pix_x = "placeholder"
            while not isinstance(subpix_per_pix_x, int) == True:
                subpix_per_pix_x = input(
                    "Subpixel per pixel parameter does not evenly subdivide the number of pixels in the x direction, Nx = {}. Please enter an integer that subdivides Nx: ".format(
                        Nx
                    )
                )
                try:
                    subpix_per_pix_x = int(subpix_per_pix_x)
                    break
                except:
                    print("Number entered is not an integer.")
        num_pix_x = int(Nx / subpix_per_pix_x)
        while Ny % subpix_per_pix_y != 0:
            subpix_per_pix_y = "placeholder"
            while not isinstance(subpix_per_pix_y, int) == True:
                subpix_per_pix_y = input(
                    "Subpixel per pixel parameter does not evenly subdivide the number of pixels in the x direction, Ny = {}. Please enter an integer that subdivides Ny: ".format(
                        Ny
                    )
                )
                try:
                    subpix_per_pix_y = int(subpix_per_pix_y)
                    break
                except:
                    print("Number entered is not an integer.")
        num_pix_y = int(Ny / subpix_per_pix_y)
        # Define the output array as an array of zeros
        Uout = np.zeros([num_pix_y, num_pix_x])
        # Loop through each pixel in the new array and sum the corresponding pixels of
        # the original array
        for i in np.arange(0, num_pix_x, 1):
            for j in np.arange(0, num_pix_y, 1):
                min_x = int(i * subpix_per_pix_x)
                max_x = int((i + 1) * subpix_per_pix_x)
                min_y = int(j * subpix_per_pix_y)
                max_y = int((j + 1) * subpix_per_pix_y)
                Uout[j, i] = np.sum(Uin[min_y:max_y, min_x:max_x])

        return Uout

    def impulse_response(self, pupil_mask, wvl, z2, dx, dy, paddingx, paddingy):
        """
        Define the impulse response in the spatial domain.

        Parameters
        ----------
        pupil_mask : np.array
            Array that describes the pupil mask. 1 for transmission and 0 for occlusion.
        wvl : float
            Center wavelength of the measured light in meters.
        z2 : float
            Distance from pupil plane to image plane in meters.
        dx : float
            Distance between array spaces in x direction on pupil plane in meters.
        dy : TYPE
            Distance between array spaces in y direction on pupil plane in meters.
        paddingx : int
            Size of the final array with padding added in x direction.
        paddingy : int
            Size of the final array with padding added in y direction.

        Returns
        -------
        h_tilde : np.array
            Array that describes the impulse response function in the image plane coordinates. PSF == h^2

        """
        d_x_tilde = dx / (wvl * z2)
        d_y_tilde = dy / (wvl * z2)
        h_tilde = self.ft2(pupil_mask, d_x_tilde, d_y_tilde, paddingx, paddingy)
        return h_tilde

    def resample_field(self, Uin, xin, yin, xout, yout):
        """
        Resample a field at a given set of points.

        Parameters
        ----------
        Uin : np.array
            Original complex field describing amplitude and phase. [sqrt(W)/m]
        xin : np.array
            Original coordinates meshgrid. [m]
        yin : np.array
            Original coordinates meshgrid. [m]
        xout : np.array
            Final coordinate meshgrid. [m]
        yout : np.array
            Final coordinate meshgrid. [m]

        Returns
        -------
        Uout : np.array
            Complex array describing the field at the newly sampled locations.

        """
        # Crop the image to the minimum number of cells required to cover the
        # coordinates of the interpolated grid
        # Define the limits
        x_min = np.min(xout)
        x_max = np.max(xout)
        y_min = np.min(yout)
        y_max = np.max(yout)
        # Find the index closest to the sample limits on the phase screen
        # This reduces the computation time of the interpolation
        x_min_index = np.searchsorted(xin[0, :], x_min) - 1
        x_max_index = np.searchsorted(xin[0, :], x_max) + 1
        y_min_index = np.searchsorted(yin[:, 0], y_min) - 1
        y_max_index = np.searchsorted(yin[:, 0], y_max) + 1
        # Crop phase screen and grid points
        xin = xin[y_min_index:y_max_index, x_min_index:x_max_index]
        yin = yin[y_min_index:y_max_index, x_min_index:x_max_index]
        Uin = Uin[y_min_index:y_max_index, x_min_index:x_max_index]
        # Interpolate the remaining screen
        Uout = interpolate.griddata(
            (xin.ravel(), yin.ravel()),
            Uin.ravel(),
            (xout, yout),
            method="linear",
            fill_value=0,
        )
        # Units are lost in interpolation
        Uout = Uout * Uin.unit

        return Uout

    def field_mask(self, Uin, xin, yin, D):
        """
        Apply a mask to a field at the size of an aperature.

        Parameters
        ----------
        Uin : np.array
            Complex array describing the field incident on an optic.
        xin : np.array
            Pixel spacing in the x direction at the aperature/objective plane.
        yin : np.array
            Pixel spacing in the y direction at the aperature/objective plane.
        D : float
            Diameter of the aperature.

        Returns
        -------
        Uout : np.array
            Complex array describing the field incident on an optic multiplied
            by the aperature mask

        """
        aperature_mask = self.circ(xin, yin, D)
        Uout = Uin * aperature_mask
        return Uout

    def intensity_calc(self, Uin, dx, dy):
        """
        Determine the intensity of a field.

        Parameters
        ----------
        Uin : np.array
            Complex array describing a field.

        Returns
        -------
        I : float
            Final complex value of the summed amplitudes.

        """
        # Multiply the array by its conjugate to get the irradiance [W/m**2]
        Irradiance = np.multiply(Uin, np.conjugate(Uin))
        # Determine the area of each cell in the area
        area = dx * dy
        # Sum all the amplitude values
        I = np.sum(Irradiance) * area
        return I

    def scale_field(
        self, Uin, dx, dy, power_aperature, power_image, transmission_factor=1
    ):
        """
        Scale the irradiance to make it match the energy recieved on the focal plane.
        Note an transmission factor can be included to scale the recieved power due to
        loss in the optical train.

        Parameters
        ----------
        Uin : np.array
            Complex array describing the irradiance on the image plane [sqrt(W)/m].
        dx : float
            Size of pixel in x direction.
        dy : float
            Size of pixel in y direction.
        power_aperature : float
            Power recieved by the camera's aperature [W].
        power_image : float
            Power propagated to the image plane [W].
        transmission_factor : float
            Energy percentage passed through the optic.
            The value should be between 0 and 1.

        Returns
        -------
        Scaled_field : TYPE
            DESCRIPTION.
        image_intensity : TYPE
            DESCRIPTION.

        """

        # Calculate the scaling factor for irradiance
        Irradiance_scale = (power_aperature * transmission_factor) / (power_image)
        # Use the scale's square root to scale the field
        Field_scale = Irradiance_scale**0.5
        # Scale the irradiance
        Scaled_field = Uin * Field_scale

        # Check the intensity value
        # First calculate the irradiance of the scaled field
        Scaled_irradiance = np.multiply(Scaled_field, np.conjugate(Scaled_field))
        area = dx * dy
        image_intensity = np.sum(Scaled_irradiance) * area

        return Scaled_field, image_intensity

    def pixel_attribution(self, Uin, name_dict, dx, dy, power_ref, source_name):
        """
        Use the power collected by the limiting magnitude of the optic as a common
        reference to attribute power on a pixel to particular sources. This will be used
        to attribute data to particular sources during event generation.

        Parameters
        ----------
        Uin : np.array
            Complex array describing the irradiance on the image plane [sqrt(W)/m].
        name_dict : dictionary
            Dictionary with pixel locations as keys that contains lists of the source
            names over a particular threshold of influence on that key.
        dx : float
            Size of pixel in x direction.
        dy : float
            Size of pixel in y direction.
        power_ref : TYPE
            DESCRIPTION.
        source_name : TYPE
            DESCRIPTION.

        Returns
        -------
        name_dict : list
            Updated list of lists the same size as the image plane that contains the
            source names on the pixels they influence over a particular threshold.

        """
        # Calculate the watts on each pixel
        Image_Watts = (Uin * np.conjugate(Uin)) * dx * dy

        # Calculate the boolean array that identifies which pixels are influenced enough
        # by this particular point source. Enough is set by the limiting magnitude of
        # the system; the irradiance from such a point source is multiplied by the
        # transmission factor. The threshold is set at 10% of this value due to power
        # being subdivided into multiple pixels.
        Image_Watts_Percentage = Image_Watts / power_ref
        Image_Watts_Boolean = Image_Watts_Percentage >= 0.1

        # Use where the boolean array is true to modify the array of source names
        true_idx = np.where(Image_Watts_Boolean)
        for i, yloc in enumerate(true_idx[0]):
            pixel_key = (true_idx[1][i], yloc)
            source_percentage_tuple = (
                str(source_name),
                float(np.real(Image_Watts_Percentage[yloc, true_idx[1][i]])),
            )
            if pixel_key not in list(name_dict.keys()):
                name_dict[pixel_key] = [source_percentage_tuple]
            else:
                if source_percentage_tuple not in name_dict[pixel_key]:
                    name_dict[pixel_key].append(source_percentage_tuple)

        return name_dict

    def image_shift(self, Ui, ptx, pty):
        """
        Shifts image by the displacement of the center by cropping the original
        and adding padding.

        Parameters
        ----------
        Ui : np.array
            Complex array describing the focused image of the point source.
        ptx : int
            If the center of the field is (0,0) this value is the
            displacement in the x direction from (0,0).
        pty : int
            If the center of the field is (0,0) this value is the
            displacement in the y direction from (0,0).

        Returns
        -------
        Uout : np.array
            Complex array describing the focused image of the point source with
            a translation of the (0,0) center of the field to a point (ptx,pty)

        """
        # Extract dimensions of the final image
        Udim = np.shape(Ui)
        # Define the center location
        u_center_x = Udim[1] / 2
        u_center_y = Udim[0] / 2
        # Determine the amount of padding and to which side the padding should
        # be added. Each direction should be checked and padding is added where
        # the pt is not the center.
        # Maintaining the field's shape is unimportant as it will be
        # subsequently cropped to the image dimensions
        x_diff = np.abs(ptx - u_center_x)
        y_diff = np.abs(pty - u_center_y)
        if ptx > u_center_x:
            # Add min padding
            x_max_pad = 0
            x_min_pad = x_diff * 2
        elif ptx < u_center_x:
            # Add max padding
            x_max_pad = x_diff * 2
            x_min_pad = 0
        else:
            x_min_pad = 0
            x_max_pad = 0
        if pty > u_center_y:
            # Add min padding
            y_max_pad = 0
            y_min_pad = y_diff * 2
        elif pty < u_center_y:
            # Add max padding
            y_max_pad = y_diff * 2
            y_min_pad = 0
        else:
            y_max_pad = 0
            y_min_pad = 0
        # Add adding
        Uout = np.pad(
            Ui,
            [(int(y_min_pad), int(y_max_pad)), (int(x_min_pad), int(x_max_pad))],
            "constant",
        )

        return Uout

    def image_crop(self, Ui, Ix, Iy):
        """
        Crops image to the final dimensions as measured from the center.

        Parameters
        ----------
        Ui : np.array
            Complex field describing the focused image on the array.
        Ix : int
            Pixels in x direction of array.
        Iy : TYPE
            Pixels in y direction of array.

        Returns
        -------
        Uout : np.array
            Final image with dimensions Ix by Iy.

        """
        # Extract dimensions of the final image
        Udim = np.shape(Ui)
        # Find the difference in size
        x_diff = Udim[1] - Ix
        y_diff = Udim[0] - Iy

        if x_diff != 0 and y_diff != 0:
            # Crop both directions
            # Take slice to define output
            Uout = Ui[
                int(y_diff / 2) : int(Udim[0] - y_diff / 2),
                int(x_diff / 2) : int(Udim[1] - x_diff / 2),
            ]
        elif x_diff != 0:
            # Crop only x direction
            # Take slice to define output
            Uout = Ui[:, int(x_diff / 2) : int(Udim[1] - x_diff / 2)]
        elif y_diff != 0:
            # Crop only y direction
            # Take slice to define output
            Uout = Ui[int(y_diff / 2) : int(Udim[0] - y_diff / 2), :]
        else:
            Uout = Ui

        return Uout

    def plot_field(
        self, Ui, xi, yi, time, source_name, plane_title, sim_name, log_plot=False
    ):
        """
        Plot the corresponding field and label it appropriately

        Parameters
        ----------
        Ui : np.array
            Array containing amplitude and phase information encoded as a complex number.
        xi : np.array
            Array with x distance values from the center of the array for each grid space.
        yi : np.array
            Array with y distance values from the center of the array for each grid space.
        time : float
            Simulation time step to annotate produced images.
        source_name: string
            Name assigned to source to match up sources.
        plane_title : string
            Description of the plane which is being depicted.
        sim_name : string
            Simulation name to separate different simulations.
        log_plot : bool
            Boolean to assign a logarithmic scale to the irradiance plots.

        Returns
        -------
        None.

        """
        # Ensure the directory exists to save the images
        directory = os.getcwd()
        newdirectory = (
            directory + "/" + sim_name + "/ProcessingImages/FieldImages/" + plane_title
        )
        if not os.path.exists(newdirectory):
            os.makedirs(newdirectory)
        # Ensure the source name doesn't have any unwanted
        if "*" in source_name:
            source_name = source_name.replace("*", "_")
        if " " in source_name:
            source_name = source_name.replace(" ", "_")

        # Plot the intensity on this plane
        # Intensity or irradiance is U^2 [W/m^2]
        I = np.real(Ui * np.conjugate(Ui))
        I_unit = I.unit
        fig, ax = plt.subplots(dpi=200)
        if log_plot == True:
            I = np.ma.array(I.value, mask=I.value <= 0)
            pcm = ax.contourf(
                xi, yi, I, locator=ticker.LogLocator(), cmap=plt.get_cmap("gray")
            )
        else:
            pcm = ax.contourf(xi, yi, I, cmap=plt.get_cmap("gray"))
        cbar = fig.colorbar(pcm)
        color_bar_label = "Flux Density [" + str(I_unit) + "]"
        cbar.set_label(color_bar_label)
        ax.axis("equal")
        ax.set_title("Flux Density in " + plane_title)
        ax.set_xlabel(plane_title + " [" + str(xi.unit) + "]")
        ax.set_ylabel(plane_title + " [" + str(yi.unit) + "]")
        plot_name = "Flux_Density_" + str(source_name) + "_time_" + str(time)
        plt.savefig(newdirectory + "/" + plot_name + ".pdf")
        plt.close()

        # Plot the phase component of the field
        U_phase = np.arctan2(-np.imag(Ui), np.real(Ui))
        fig, ax = plt.subplots(dpi=200)
        pcm = ax.contourf(xi, yi, U_phase, cmap=plt.get_cmap("gray"))
        cbar = fig.colorbar(pcm)
        color_bar_label = "Phase [rad]"
        cbar.set_label(color_bar_label)
        ax.axis("equal")
        ax.set_title("Phase in " + plane_title)
        ax.set_xlabel(plane_title + " [" + str(xi.unit) + "]")
        ax.set_ylabel(plane_title + " [" + str(yi.unit) + "]")
        plot_name = "Phase_" + str(source_name) + "_time_" + str(time)
        plt.savefig(newdirectory + "/" + plot_name + ".pdf")
        plt.close()

        return

    def month2day(self, date_array):
        """
        Convert months in a date array to total days

        Parameters
        ----------
        date_array : CalendarArray
            skyfield.timelib.CalendarArray with 6 fields for year, months,
            days, hours, minutes, and seconds

        Returns
        -------
        total_days : int
            Total number of days in the simulation including those in a month.

        """
        # Determine if the year is a leap year
        if date_array[0] % 4 == 0:
            # A leap year
            month_2_days = {
                1: 31,
                2: 60,
                3: 91,
                4: 121,
                5: 152,
                6: 182,
                7: 213,
                8: 244,
                9: 274,
                10: 305,
                11: 335,
                12: 366,
            }
        else:
            # Not a leap year
            month_2_days = {
                1: 31,
                2: 59,
                3: 90,
                4: 120,
                5: 151,
                6: 181,
                7: 212,
                8: 243,
                9: 273,
                10: 304,
                11: 334,
                12: 365,
            }
        # Get days from the whole number component of the month component
        days_from_whole_month = month_2_days[int(date_array[1])]
        # Add to the days not in a whole month
        total_days = days_from_whole_month + int(date_array[2])
        return total_days

    def conv2(self, A, B, delta_a, delta_b, delta_c, padding_x, padding_y):
        """
        Convolve two arrays currently in the spatial domain by taking their fourier transforms and multiplying them.
        Returns one array back in the spatial domain.

        Parameters
        ----------
        A : np.array
            Object 1 as a numpy array.
        B : np.array
            Object 2 as a numpy array.
        delta_a : float
            Spatial separation between values in numpy arrays which must put each object in the same coordinate frame.
        delta_b : float
            Spatial separation between values in numpy arrays which must put each object in the same coordinate frame.
        delta_c : float
            Spatial separation between values in numpy arrays in the final coordinate frame.
        padding_x : int
            Size of the final array with padding added in x direction.
        padding_y : int
            Size of the final array with padding added in y direction.


        Returns
        -------
        C : np.array
            Convolution of objects 1 and 2 in the spatial domain.

        """
        # Take fourier transform of each array to allow for convolution via multiplication in fourier domain
        A_ft = self.ft2(A, delta_a, delta_a, padding_x, padding_y)
        B_ft = self.ft2(B, delta_b, delta_b, padding_x, padding_y)
        # Multiply the arrays in the fourier domain to convolve them
        A_conv_B = np.multiply(A_ft, B_ft)
        # Take the inverse fourier transform to return the convolved array to the spatial temporal domain
        N = np.ones(A_conv_B.shape, dtype=int)
        f_delta = 1 / (N * delta_c)
        C = self.ift2(A_conv_B, f_delta, f_delta, padding_x, padding_y)
        return C

    def ft2(self, Uin, deltax, deltay, paddingx, paddingy):
        """
        Two dimenstional DFT as described in Schmidt 2010.

        Parameters
        ----------
        Uin : np.array
            Complex scalar field respresenting the amplitude and phase of the wave.
        deltax : float
            Spacing between x coordinates.
        deltay : float
            Spacing between y coordinates.
        paddingx : int
            Size of the final array with padding added in x direction.
        paddingy : int
            Size of the final array with padding added in y direction.

        Returns
        -------
        Uout : np.array
            Complex scalar field in the frequency domain.

        """
        # Correct for any NAN values before DFT
        nan_test = np.isnan(Uin)
        if nan_test.any():
            x = np.arange(0, Uin.shape[1])
            y = np.arange(0, Uin.shape[0])
            # mask invalid values
            Uin = np.ma.masked_invalid(Uin)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~Uin.mask]
            y1 = yy[~Uin.mask]
            newarr = Uin[~Uin.mask]

            Uin = interpolate.griddata(
                (x1, y1), newarr.ravel(), (xx, yy), method="cubic", fill_value=0
            )

        # print('Inside DFT calculations:')
        # print('FT shift: ')
        Uin_shifted = np.fft.fftshift(Uin)
        nan_test = np.isnan(Uin_shifted)
        if nan_test.any():
            x = np.arange(0, Uin_shifted.shape[1])
            y = np.arange(0, Uin_shifted.shape[0])
            # mask invalid values
            Uin_shifted = np.ma.masked_invalid(Uin_shifted)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~Uin_shifted.mask]
            y1 = yy[~Uin_shifted.mask]
            newarr = Uin_shifted[~Uin_shifted.mask]

            Uin_shifted = interpolate.griddata(
                (x1, y1), newarr.ravel(), (xx, yy), method="cubic", fill_value=0
            )
        # print(Uin_shifted)
        # print('Fourier Transform: ')
        # print(np.fft.fft2(np.fft.fftshift(Uin)))
        # print('FT shift: ')
        # print(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Uin))))
        # print('Multiply Dimensions: ')
        if np.shape(Uin)[0] > 10000 or np.shape(Uin)[1] > 10000:
            # Delete unused array
            del nan_test
            # Use Scipy fft to ensure complex 128 is not used
            Uout = (
                np.fft.fftshift(fft.fft2(Uin_shifted, s=[paddingy, paddingx]))
                * deltax
                * deltay
            )
        else:
            Uout = (
                np.fft.fftshift(np.fft.fft2(Uin_shifted, s=[paddingy, paddingx]))
                * deltax
                * deltay
            )
        # print(Uout)
        return Uout

    def ift2(self, Uin, deltafx, deltafy, paddingx, paddingy):
        """
        Two dimenstional DIFT as described in Schmidt 2010.

        Parameters
        ----------
        Uin : TYPE
            DESCRIPTION.
        deltafx : TYPE
            DESCRIPTION.
        deltafy : TYPE
            DESCRIPTION.
        paddingx : int
            Size of the final array with padding added in x direction.
        paddingy : int
            Size of the final array with padding added in y direction.

        Returns
        -------
        Uout : TYPE
            DESCRIPTION.

        """
        dim = np.shape(Uin)
        Nx = dim[1]
        Ny = dim[0]

        # Correct for any NAN values before DIFT
        nan_test = np.isnan(Uin)
        if nan_test.any():
            x = np.arange(0, Uin.shape[1])
            y = np.arange(0, Uin.shape[0])
            # mask invalid values
            Uin = np.ma.masked_invalid(Uin)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~Uin.mask]
            y1 = yy[~Uin.mask]
            newarr = Uin[~Uin.mask]

            Uin = interpolate.griddata(
                (x1, y1), newarr.ravel(), (xx, yy), method="cubic", fill_value=0
            )

        # print('Inside DIFT calculations:')
        # print('FT shift: ')
        Uin_shifted = np.fft.ifftshift(Uin)
        nan_test = np.isnan(Uin_shifted)
        if nan_test.any():
            x = np.arange(0, Uin_shifted.shape[1])
            y = np.arange(0, Uin_shifted.shape[0])
            # mask invalid values
            Uin_shifted = np.ma.masked_invalid(Uin_shifted)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~Uin_shifted.mask]
            y1 = yy[~Uin_shifted.mask]
            newarr = Uin_shifted[~Uin_shifted.mask]

            Uin_shifted = interpolate.griddata(
                (x1, y1), newarr.ravel(), (xx, yy), method="cubic", fill_value=0
            )
        # print(Uin_shifted)
        # print('Inverse Fourier Transform: ')
        # print(np.fft.ifft2(np.fft.ifftshift(Uin)))
        # print('IFT shift: ')
        # print(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Uin))))
        # print('Multiply Dimensions: ')
        if Nx > 10000 or Ny > 10000:
            # Delete unused array
            del nan_test
            # Use Scipy fft to ensure complex 128 is not used
            Uout = (
                np.fft.ifftshift(fft.ifft2(Uin_shifted, s=[paddingy, paddingx]))
                * Nx
                * Ny
                * deltafx
                * deltafy
            )
        else:
            Uout = (
                np.fft.ifftshift(np.fft.ifft2(Uin_shifted, s=[paddingy, paddingx]))
                * Nx
                * Ny
                * deltafx
                * deltafy
            )
        # print(Uout)
        return Uout

    def saft(self, Uin, mx, my, nx2, ny2):
        """
        This function calculates the semi-analytical Fourier Transform by evaluating the
        integral instead of using the FFT method. This function is implemented as
        described in "Fast computation of Lyot-style coronagraph propagation" by Soummer,
        Pueyo, Sivaramakrishnan, and Vanderbei (2007). It is used in this code to
        compute the airy disk on the image plane.

        Parameters
        ----------
        Uin : np.array
            Complex field at the first propagation plane.
        mx : int
            Describes the size of the subsection of the image plane sampled in
            resolution units (wvl/D). nx2 = mx * gamma
            (gamma is number of pixels or subpixels in airy disk)
        my : int
            Describes the size of the subsection of the image plane sampled in
            resolution units (wvl/D). ny2 = my * gamma
        nx2 : int
            Number of points in the x direction in the discretized impulse response.
        ny2 : int
            Number of points in the y direction in the discretized impulse response.

        Returns
        -------
        Uout : np.array
            Complex field at the second propagation plane.
        X2 : np.array
            List of the resolution units (wvl/D) in the second propgation plane.
        dx2 : float
            Spacing of the resolution units (wvl/D) in the second propgation plane.

        """
        # First pull the number of grid points in the first propagation plane from the
        # size of the field
        Na = np.shape(Uin)
        nx1 = Na[1]
        ny1 = Na[0]

        # Calculate the spacing of each propagation plane based on the total distance
        # and the number of grid points
        dx1 = 1 / nx1
        dy1 = 1 / ny1
        dx2 = mx / nx2
        dy2 = my / ny2

        # To calculate the matrix Fourier transform the intergral is evaluated at each
        # spatial component of the first plane and each resolution unit of the second
        # plane which can be accomplished by a matrix multiplication
        # Using the spacings calculated above, an array is defined to define each of
        # these spatial and resolution unit values the Fourier transform will be
        # calculated at.
        # Unitless First Plane
        X1 = np.arange(-nx1 / 2 + 0.5, nx1 / 2 + 0.5, 1) * dx1
        Y1 = np.arange(-ny1 / 2 + 0.5, ny1 / 2 + 0.5, 1) * dy1
        # Resolution Units Second Plane
        X2 = np.arange(-nx2 / 2 + 0.5, nx2 / 2 + 0.5, 1) * dx2
        Y2 = np.arange(-ny2 / 2 + 0.5, ny2 / 2 + 0.5, 1) * dy2
        # The 2D evaluation multiplies the X2*X1' and Y1*Y2'
        X2_X1 = np.outer(X2, X1)
        Y1_Y2 = np.outer(Y1, Y2)

        # Calculating the final two dimensional FT uses the two matrix products and a
        # normalizing coefficient m/(Na*Nb) where Na and Nb are the number of grid
        # points in one dimension (assuming a square grid) on each plane
        Uout = (mx / (nx1 * nx2)) * np.matmul(
            np.exp((-2 * np.pi * 1j * X2_X1)),
            np.matmul(Uin, np.exp((-2 * np.pi * 1j * Y1_Y2))),
        )

        return Uout
