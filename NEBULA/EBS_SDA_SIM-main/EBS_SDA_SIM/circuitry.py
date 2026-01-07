# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:37:50 2021

@author: rache
Deleted for loop versions of processing and unit dependance for speed
"""

import numpy as np
import pandas as pd
import h5py
import os
import pdb
import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import interpolate
from astropy import units as u
from astropy.units import cds


class ebCircuitSim:

    def __init__(self, **kwargs):

        return

    def __str__(self):
        """
        This class contains the circuitry model for an event-based sensor.
        """
        return

    def generate_events(
        self,
        observer,
        t_array,
        frames_file_name,
        att_dict_file_name,
        sim_name,
        circuit_para,
        time_window=2 * u.s,
        seed=None,
        shot_noise=True,
        high_freq_noise=True,
        junction_leak=False,
        parasitic_leak=False,
        plot=False,
        plot_freq=100,
    ):

        shape = np.ones((observer.num_y_pix, observer.num_x_pix))

        # Start with a fixed seed for the random generator for reproduceability
        # If seed for sampling is not provided, generate a random seed.
        if not isinstance(seed, int):
            seed_seq = np.random.SeedSequence()
            seed = seed_seq.generate_state(1)
        rng = np.random.default_rng(seed)

        # Initialize I_1, I_mem, Ne, Ne_time, T_hold at zero
        # FIXME : Should I maintain this old style of dark current adjustment? Commenting out for now
        # I_dark_constant = self.calc_dark_constant(
        #     circuit_para.Ea_dark, circuit_para.I_dark_ref, circuit_para.T_dark_ref
        # )
        # I_dark_scaled = self.calc_temp_adjust_dark_current(
        #     circuit_para.Ea_dark, I_dark_constant, circuit_para.T
        # )
        # I_dark_scaled_log = np.log(I_dark_scaled.value) * I_dark_scaled.unit

        # Calculate the step size and timestep for this iteration
        step_size = self.time_diff(t_array.utc[:, 1] - t_array.utc[:, 0])
        num_steps_per_second = int(np.round(1 / step_size.to(u.s).value))

        # Initialize the array of Poisson pulls for the dark current
        electron_dark_rate_sub = (
            circuit_para.I_dark.to(cds.e / u.s) / num_steps_per_second
        )
        electron_dark_rate_matrix = (
            rng.poisson(
                np.ones((num_steps_per_second, observer.num_y_pix, observer.num_x_pix))
                * electron_dark_rate_sub.value
            )
            * electron_dark_rate_sub.unit
        )
        electron_dark_rate_sum = np.sum(electron_dark_rate_matrix, axis=0)
        electron_dark_rate_array = (
            np.ones((1, observer.num_y_pix, observer.num_x_pix))
            * electron_dark_rate_sub
        )
        I_dark_log_value = np.log(circuit_para.I_dark.value)
        slice = 0

        # Load HDF5 File with PSFs
        hf = h5py.File(frames_file_name, "r")
        # Load Attribution Dictionary
        with open(att_dict_file_name, "rb") as picklefile:
            attribution_dict = pickle.load(picklefile)
        # Load file names
        frame_names = list(hf.keys())
        # remove any non frame names that do not include "ph_flux_time_itr" in their name
        frame_names = [name for name in frame_names if "ph_flux_time_itr" in name]

        # Initialize Memorized Current at Dark Current with a Normalized Distribution
        # Around the Dark Current, Note that the input is not in log space but the memorized output is in log space
        # We are assuming the memorized current starts at the dark current level
        I_mem_current = self.initialize_mem_current(circuit_para.I_dark, shape, rng)
        I_mem_last = I_mem_current
        I_ref = I_mem_last

        # Initialize Event Arrays to Track Events
        N_event_total = np.zeros(shape.shape)
        N_event_time = np.zeros(shape.shape)
        N_event_polarity = np.zeros(shape.shape)
        # T_hold = np.zeros(shape.shape)
        # T_hold_time = np.zeros(shape.shape)
        t_total_sec = None
        I_filter_current = None

        # Create structure to hold events
        # events = pd.DataFrame(list(),columns = ['t','x','y','p','attr'])
        events = []
        events_prev = (
            []
        )  # Events from previous step to send into the next step (simplier than taking a window of events, but that is also possible)
        events_attr_list = []
        events_attr_list_prev = []

        # Initialize Compariative Theta Values
        Theta_on = rng.normal(
            circuit_para.theta_on, circuit_para.theta_on_var, shape.shape
        )
        Theta_off = rng.normal(
            circuit_para.theta_off, circuit_para.theta_off_var, shape.shape
        )

        # Initialize Arbiter
        last_row = 0
        last_column = 0
        if not circuit_para.T_refr_sigma.value:
            T_refr = np.ones(shape.shape) * circuit_para.T_refr
        else:
            T_refr = self.log_normal_noise_add(
                np.zeros(shape.shape) * u.s,
                circuit_para.T_refr.to(u.s),
                circuit_para.T_refr_sigma.to(u.s),
                rng,
            )

        # Initialize Bandpass Filter
        if not circuit_para.photodiode_3db_freq_sigma.value:
            pd_3db = np.ones(shape.shape) * circuit_para.photodiode_3db_freq
        else:
            pd_3db = self.log_normal_noise_add(
                np.zeros(shape.shape) * u.Hz,
                circuit_para.photodiode_3db_freq,
                circuit_para.photodiode_3db_freq_sigma,
                rng,
            )
        if not circuit_para.source_follower_3db_freq_sigma.value:
            sf_3db = np.ones(shape.shape) * circuit_para.source_follower_3db_freq
        else:
            sf_3db = self.log_normal_noise_add(
                np.zeros(shape.shape) * u.Hz,
                circuit_para.source_follower_3db_freq,
                circuit_para.source_follower_3db_freq_sigma,
                rng,
            )

        # Pull Directory for Plotting
        directory = os.getcwd()
        # Create new directory for plots
        newdirectory = directory + "/" + sim_name + "/ProcessingImages/Circuit/"
        if not os.path.exists(newdirectory):
            os.makedirs(newdirectory)
        # Create Pixel Arrays for Plotting
        x_vector = np.arange(1, observer.num_x_pix + 1, 1)
        y_vector = np.arange(1, observer.num_y_pix + 1, 1)
        x_mesh, y_mesh = np.meshgrid(x_vector, y_vector, copy=False)
        del x_vector
        del y_vector

        # Run through each timestep
        for i in np.arange(0, len(t_array.utc[0, :]), 1):

            # If this is not the first simulation timestep, store the previous step's
            # information prior to overwriting it
            if i != 0:
                t_total_sec_last = t_total_sec
                I_filter_last = I_filter_current
                I_mem_last = I_mem_current

            # Solve the time step for this iteration
            t_total_sec = self.time_diff(t_array.utc[:, i] - t_array.utc[:, 0])

            # Load Incident Energy from HDF5 File
            # Note on first time step load both the current and next step
            # Due to the non-deterministic noise being added, subsequent steps do not
            # recalculate the current step, but only calculate the next current for
            # interpolation purposes
            incident_photons_rate = hf[frame_names[i]]
            # Convert the incident energy to photocurrent, Note white noise is added here for the conversion to Amps
            I_p_current = self.photo_current_photons_per_sec(
                incident_photons_rate, circuit_para.eta, rng
            )
            if i % plot_freq == 0 and plot:
                # Calculate RA and DEC for the pixels at this timestep
                RA, DEC = self.frame_RA_DEC(observer, i, x_mesh, y_mesh)
                self.plot_frame(
                    I_p_current,
                    RA,
                    DEC,
                    t_total_sec,
                    "Induced_Photocurrent",
                    sim_name,
                    newdirectory,
                )

            # Adding Dark Noise and High Frequency Noise from other components to the simulation
            # If there is high frequency noise, calculate the standard deviation of the noise based on the induced photocurrent without the dark current added
            if high_freq_noise:
                std_extra_noise = self.high_freq_noise_std_calc(
                    I_p_current, circuit_para.high_freq_std_fit
                )  # Plot current photocurrent
            # If shot noise apply the rolling poisson distribution and then add the incident energy and dark current
            if shot_noise:
                electron_dark_rate_matrix, electron_dark_rate_sum, slice, I_dark_i = (
                    self.rolling_poisson_dark_current(
                        electron_dark_rate_matrix,
                        electron_dark_rate_array,
                        electron_dark_rate_sum,
                        slice,
                        num_steps_per_second,
                        rng,
                    )
                )
                I_p_current = I_p_current + I_dark_i
            # If high frequency noise is requested, scale the noise by a quadratic factor of the induced photocurrent after the shot noise is added
            if high_freq_noise:
                I_p_current = (
                    rng.normal(loc=I_p_current.value, scale=std_extra_noise) * u.A
                )

            # Plot total current
            if i % plot_freq == 0 and plot:  # Only Plot Every 5000 time steps
                self.plot_frame(
                    I_p_current,
                    RA,
                    DEC,
                    t_total_sec,
                    "Total_Current",
                    sim_name,
                    newdirectory,
                )

            # Logarithmically scale the photocurrent
            I_log_current = self.log_Intensity(I_p_current)
            # Plot logarithmic current
            if i % plot_freq == 0 and plot:
                self.plot_frame(
                    I_p_current,
                    RA,
                    DEC,
                    t_total_sec,
                    "Log_Current",
                    sim_name,
                    newdirectory,
                )

            # Apply the low pass filter for either the photodiode and/or the source follower
            if circuit_para.photodiode_3db_freq_sigma.value > 0:
                I_filter_current, I_ref = self.low_pass_filter(
                    I_log_current, I_ref, pd_3db, step_size
                )
            elif circuit_para.source_follower_3db_freq_sigma.value > 0:
                I_filter_current, I_ref = self.low_pass_filter(
                    I_log_current, I_ref, sf_3db, step_size
                )
            else:
                print(
                    "Warning: No Low Pass Filter Applied. Circuit Parameters are not set correctly."
                )
            # Plot low passed current
            if i % plot_freq == 0 and plot:
                self.plot_frame(
                    I_filter_current,
                    RA,
                    DEC,
                    t_total_sec,
                    "Low_Pass_Filtered_Current",
                    sim_name,
                    newdirectory,
                )

            # Account for parasitic, junction leak, and/or shot noise prior to checking the comparator
            if junction_leak:
                I_mem_current = self.junction_leak(
                    step_size.value,
                    circuit_para.R_leak.value,
                    Theta_on,
                    I_mem_current,
                    I_dark_log_value,
                )
            if parasitic_leak:
                I_mem_current = self.parasitic_leak(
                    incident_photons_rate.value,
                    circuit_para.P_leak.value,
                    step_size,
                    Theta_on,
                    I_mem_current,
                    I_dark_log_value,
                )

            # Plot the memorized current
            if i % plot_freq == 0 and plot:
                self.plot_frame(
                    I_mem_current,
                    RA,
                    DEC,
                    t_total_sec,
                    "Memorized_Current",
                    sim_name,
                    newdirectory,
                )
            # If this is the first timestep, we do not have prior information to compare
            # to, so continue to the next timestep
            if i == 0:
                continue
            # Check the comparator for a threshold change
            del_I_filter, del_I_mem, N_event_total, N_event_time, N_event_polarity = (
                self.threshold_check(
                    t_total_sec_last.value,
                    t_total_sec.value,
                    I_mem_last,
                    I_mem_current,
                    I_filter_last,
                    I_filter_current,
                    Theta_on,
                    Theta_off,
                    N_event_total,
                    N_event_time,
                    N_event_polarity,
                )
            )

            # Run the event total through the arbiter
            # attribution_dict = {}
            # attribution_dict[t_total_sec] = {}
            # Send in only previous set of events for attribution
            previous_list_len = len(events_prev)
            (
                events_prev,
                events_attr_list_prev,
                N_event_time,
                N_event_polarity,
                I_mem_current,
                last_row,
                last_column,
                curr_time,
            ) = self.arbiter(
                events_prev,
                events_attr_list_prev,
                N_event_time,
                N_event_polarity,
                t_total_sec_last.value,
                t_total_sec.value,
                time_window,
                I_filter_last,
                I_filter_current,
                del_I_filter,
                I_mem_current,
                del_I_mem,
                Theta_on,
                Theta_off,
                last_row,
                last_column,
                observer.num_x_pix,
                T_refr.value,
                circuit_para.recording_freq,
                attribution_dict,
                rng,
            )
            # reduce updated lists to the new information
            events_prev = events_prev[previous_list_len:]
            events_attr_list_prev = events_attr_list_prev[previous_list_len:]
            events.extend(events_prev)
            events_attr_list.extend(events_attr_list_prev)
        # Close HDF5 file
        hf.close()
        # make the event output a dataframe
        if not events:
            events = pd.DataFrame(list(), columns=["t", "x", "y", "p", "attr"])
        else:
            events = pd.DataFrame(np.array(events), columns=["t", "x", "y", "p"])
            events["attr"] = events_attr_list
        return events

    def time_diff(self, t_total):
        """Convert the time difference input into seconds."""
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
        return t_total_sec

    def photo_current_watts(self, Incident_energy, wvl, eta, R_lambda=0):
        """
        Determine the induced photocurrent from the incident energy (W)

        Parameters
        ----------
        Incident_energy : H5py._hl.dataset module
            H5py contains the numpy array with this time step's incident energy.
        wvl : float
            Effective wavelength of the observed spectrum in meters.
        eta : float
            Quantum efficiency at that wavelength.
        R_lambda : float, optional
            Responsivity of the array in (A/W). The default is 0.

        Returns
        -------
        I_p : numpy array
            Induced photocurrent in Amps.

        """

        # Responsivity, R, (A/W) is a function of quantum efficiency, eta
        # eta = R/lambda*hc/e
        # h = planck's constant [Js]
        h = 6.62607015e-34 * u.J * u.s
        # c = speed of light [m/s]
        c = 299792458 * u.m / u.s
        # e = elementary charge [C] = [A*s]
        e = 1.602176634e-19 * u.A * u.s

        if R_lambda == 0:
            R_lambda = eta * wvl * e / (h * c)

        I_p = R_lambda * np.real(Incident_energy) * u.W
        I_p = I_p.decompose()
        return I_p

    def photo_current_photons_per_sec(
        self, Incident_energy, eta, rng, white_noise=True
    ):
        """
        Determine the induced photocurrent from the incident energy (photons/sec)

        Parameters
        ----------
        Incident_energy : H5py._hl.dataset module
            H5py contains the numpy array with this time step's incident energy.
        eta : float
            Quantum efficiency at that the effective wavelength.

        Returns
        -------
        I_p : numpy array
            Induced photocurrent in Amps.

        """
        # Caclulate the
        # Check if Incident_energy has units and is in photons/sec
        if hasattr(Incident_energy, "unit") and Incident_energy.unit == (u.ph / u.s):
            # e = elementary charge [C] = [A*s]
            e = 1.602176634e-19 * u.A * u.s
            eta = eta * (1 / u.ph)
        else:
            # If no units, treat Incident_energy as a plain number and use e as a float
            e = 1.602176634e-19
        I_p = (
            eta * e * np.array(Incident_energy).real
        ) * u.A  # photons in the numerator are canceled out by Quantum efficiency

        # This will not be a perfect conversion. Apply a gaussian noise to the current
        if white_noise:
            I_p = self.white_noise_add(I_p, rng, double_noise=True)
        return I_p

    def calc_dark_constant(self, E_a, I_dark_ref, T_ref):
        """
        Calculate the dark contant given the sensor parameters.

        Parameters
        ----------
        E_a : float
            Activation energy of the photodiode.
        I_dark_ref : float
            Reference dark current for this sensor.
        T_ref : float
            Reference operational temperature where the reference dark current
            is on the energy activation slope.

        Returns
        -------
        dark_constant : float
            Constant used to calculate the dark current given a specific
            operating temperature.

        """
        # Boltzman Constant
        k = 8.617333262e-5 * u.eV / u.K
        dark_constant = np.log10(I_dark_ref.value) + (E_a) / (k * T_ref)
        return dark_constant

    def calc_temp_adjust_dark_current(self, E_a, dark_constant, T):
        """
        Calculate the dark current for the sensor given an operating temperature.

        Parameters
        ----------
        E_a : float
            Activation energy of the photodiode.
        dark_constant : float
            Constant set by reference temperature and resulting dark current.
        T : float
            Operation temperature of the sensor.

        Returns
        -------
        I_dark : float
            Dark current for this sensor.

        """
        # Boltzman Constant
        k = 8.617333262e-5 * u.eV / u.K
        # Calculate the nominal dark current with the thermal voltage
        # eV/kT is the thermal voltage
        I_dark = 10 ** (dark_constant - (E_a) / (k * T)) * u.A
        return I_dark

    def initialize_mem_current(self, I_dark, shape, rng):
        """
        Initialize the memorized current to not just be the dark current, but a gaussian
        around that dark current because if shot noise has caused an event to occur on
        any pixel, the memorized current value can be anywhere within the gaussian curve
        defined by variance of 2QI.

        Parameters
        ----------
        I_dark : float
            Dark current for this sensor.
        shape : tuple or np.array
            Tuple with number of pixels in the y direction and x diection. (y pix, x pix)
            Or array with desired shape of the final matrix
        rng: np.rng
            Numpy random number generator to make make multiple runs with the same seed
            possible

        Returns
        -------
        I_mem_matrix : np.array
            Initialized matrix around the expected dark current value in units of log(A).

        """
        # Change the value into electrons per second
        electron_dark_rate = I_dark.to(cds.e / u.s)
        # Take a normal distribution around that rate
        if isinstance(shape, tuple):
            I_mem_matrix = (
                rng.normal(
                    loc=electron_dark_rate.value,
                    scale=np.sqrt(2 * electron_dark_rate.value),
                    size=shape,
                )
                * electron_dark_rate.unit
            )
        else:
            I_mem_matrix = (
                rng.normal(
                    loc=electron_dark_rate.value,
                    scale=np.sqrt(2 * electron_dark_rate.value),
                    size=np.shape(shape),
                )
                * electron_dark_rate.unit
            )
        # Convert back to Amps
        I_mem_matrix = I_mem_matrix.to(u.A)
        # Make sure no values are negative
        I_mem_matrix[I_mem_matrix < 0] = 0
        # Take log value of the current
        I_mem_matrix = np.log(I_mem_matrix.value)
        return I_mem_matrix

    def initialize_dark_current(
        self, I_dark, num_steps_per_second, num_pix_x, num_pix_y, rng
    ):
        """
        Pull a number of poisson draws equivalent to the number of simulation steps per
        second to have a value of electrons per second.

        Parameters
        ----------
        I_dark : float
            Float with astropy units of A describing the dark current.
        num_steps_per_second : int
            Number of steps per second in the simulation
        num_pix_x : int
            Number of pixels in x-direction
        num_pix_y : int
            Number of pixels in y-direction
        rng: np.rng
            Numpy random number generator to make make multiple runs with the same seed
            possible

        Returns
        -------
        electron_dark_rate_matrix : np.array
            Array of possion draws to initialize the dark current noise.
        electron_dark_rate_array : np.array
            Array with dark current in subdivided rate for update pulls
        electron_dark_rate_sum : np.array
            Array with summed poisson draws not converted to A to stay in electrons
            per second.
        I_dark_0 : np.array
            Array with the summed dark current in units of A for the first time step.
        """
        electron_dark_rate_sub = I_dark.to(cds.e / u.s) / num_steps_per_second
        electron_dark_rate_matrix = (
            rng.poisson(
                np.ones((num_steps_per_second, num_pix_y, num_pix_x))
                * electron_dark_rate_sub.value
            )
            * electron_dark_rate_sub.unit
        )
        electron_dark_rate_sum = np.sum(electron_dark_rate_matrix, axis=0)
        I_dark_0 = electron_dark_rate_sum.to(u.A)
        electron_dark_rate_array = (
            np.ones((1, num_pix_y, num_pix_x)) * electron_dark_rate_sub
        )
        return (
            electron_dark_rate_matrix,
            electron_dark_rate_array,
            electron_dark_rate_sum,
            I_dark_0,
        )

    def rolling_poisson_dark_current(
        self,
        electron_dark_rate_matrix,
        electron_dark_rate_array,
        electron_dark_rate_sum,
        i,
        num_steps_per_second,
        rng,
    ):
        """
        Update the matrix that contains the poisson pulls.

        Parameters
        ----------
        electron_dark_rate_matrix : TYPE
            DESCRIPTION.
        electron_dark_rate_array : TYPE
            DESCRIPTION.
        electron_dark_rate_sum : TYPE
            DESCRIPTION.
        i : int
            The current simulation time step is what portion of the larger second rate
            Iterates by one each timestep.
        num_steps_per_second : TYPE
            DESCRIPTION.
        rng : TYPE
            DESCRIPTION.

        Returns
        -------
        electron_dark_rate_matrix : TYPE
            DESCRIPTION.
        electron_dark_rate_sum : TYPE
            DESCRIPTION.
        i : TYPE
            DESCRIPTION.
        I_dark_i : TYPE
            DESCRIPTION.

        """
        # Poisson draw on dark current
        dark_rate_pull = (
            rng.poisson(electron_dark_rate_array.value) * electron_dark_rate_array.unit
        )
        # Calculate the dark rate sum
        electron_dark_rate_sum = (
            electron_dark_rate_sum
            - electron_dark_rate_matrix[i, :, :]
            + dark_rate_pull[0, :, :]
        )
        # Update the pull array
        electron_dark_rate_matrix[i, :, :] = dark_rate_pull
        # Convert dark electrons/second rate to Amperes
        I_dark_i = electron_dark_rate_sum.to(u.A)
        # Update the variable keeping track of which value to update in the electron dark
        # rate matrix
        i += 1
        if i == num_steps_per_second:
            i = 0
        return electron_dark_rate_matrix, electron_dark_rate_sum, i, I_dark_i

    def dark_current_add(self, I_p, E_a, T, I_dark=0, dark_constant=0):
        """
        Apply dark current to the induced photocurrent.

        Parameters
        ----------
        I_p : np.array
            Induced photocurrent.
        E_a : float
            Activation energy of the photodiode.
        T : float
            Operation temperature of the sensor.
        I_dark : float
            Precomputed dark current for this sensor.
        dark_constant : float
            Constant set by reference temperature and resulting dark current.

        Returns
        -------
        I_total : np.array
            Current updated with the dark current.

        """
        if I_dark == 0 and dark_constant != 0:
            # Calculate the nominal dark current with the thermal voltage
            I_dark = self.calc_temp_adjust_dark_current(E_a, dark_constant, T)
        elif I_dark == 0 and dark_constant == 0:
            # Insuffient data is provided
            print(
                "Neither the constant reference value or the dark current was provided. The output is not modified from the input."
            )
            I_dark = 0
        I_total = I_p + I_dark
        return I_total

    def temp_dependant_dark_current(
        self,
        temp,
        m_temp,
        m_temp_sigma,
        inter_temp,
        inter_temp_sigma,
        m_dark,
        m_dark_sigma,
        inter_dark,
        inter_dark_sigma,
        rng,
    ):
        """
        Using the linear relationship between temperature and dark current to produce an
        expected event/per pixel/per second noise rate, to go from temperature of the
        observation to the estimated dark current leaking through the photoreceptor.

        Parameters
        ----------
        temp : float
            Temperature at time of observation either ambient or that of the temperature
            controlled sensor.
        m_temp : float
            Slope of the relationship between temperature and noise event rate in the
            logarithmic scale.
        m_temp_sigma : float
            Standard deviation of the slope of the relationship between temperature and
            noise event rate in the logarithmic scale.
        inter_temp : float
            Intercept of the relationship between temperature and noise event rate in
            the logarithmic scale.
        inter_temp_sigma : float
            Standard deviation of the intercept of the relationship between temperature
            and noise event rate in the logarithmic scale.
        m_dark : float
            Slope of the relationship between dark current and noise event rate in the
            logarithmic scale.
        m_dark_sigma : float
            Standard deviation of the slope of the relationship between dark current and
            noise event rate in the logarithmic scale.
        inter_dark : float
            Intercept of the relationship between dark current and noise event rate in
            the logarithmic scale.
        inter_dark_sigma : float
            Standard deviation of the intercept of the relationship between dark current
            and noise event rate in the logarithmic scale.
        rng : random noise generator
            Simulation random noise generator to make the result reproduceable.

        Returns
        -------
        dark : float
            Dark current of the camera at this temperature. u.A

        """

        dark = (
            (
                rng.normal(loc=m_temp, scale=m_temp_sigma) * temp
                + rng.normal(loc=inter_temp, scale=inter_temp_sigma)
                - rng.normal(loc=inter_dark, scale=inter_dark_sigma)
            )
            / rng.normal(loc=m_dark, scale=m_dark_sigma)
            * u.A
        )

        return dark

    def log_Intensity(self, I_in):
        """
        Convert the linear photocurrent to a logarithmic scale

        Parameters
        ----------
        I_in : numpy array
            Induced photocurrent.

        Returns
        -------
        I_p : TYPE
            DESCRIPTION.

        """

        # Without for loop
        I_p = np.log(I_in.value, where=(I_in > 0))

        return I_p

    def white_noise_add(
        self, I_in, rng, white_noise_threshold=0 * u.A, double_noise=False
    ):
        """
        Add white noise to the current values.

        Parameters
        ----------
        I_in : np.array
            Induced photocurrent in linear scale.
        rng : generator object
            Generator object from numpy random generator module
        white_noise_threshold : float, optional
            For simulations that switch between shot and white noise at a certain Amp
            value. The default is 0*u.A.

        Returns
        -------
        None.

        """
        # White noise centered at the current value
        # Standard deviation scaled as a square root of the electrons per second
        # First convert to electrons per second
        electron_rate = I_in.to(u.cds.e / u.s)
        # Apply the white noise
        if double_noise:
            electron_rate_new = (
                rng.normal(
                    loc=electron_rate.value,
                    scale=np.sqrt(4 * electron_rate.value),
                    size=np.shape(I_in),
                )
                * electron_rate.unit
            )
        else:
            electron_rate_new = (
                rng.normal(
                    loc=electron_rate.value,
                    scale=np.sqrt(2 * electron_rate.value),
                    size=np.shape(I_in),
                )
                * electron_rate.unit
            )
        # Ensure all values are greater than 0 before log conversion
        electron_rate_new[electron_rate_new <= 0] = 1 * u.cds.e / u.s

        # Convert back to Amperes
        I_out = electron_rate_new.to(u.A)

        # Only apply white noise to pixels with a current value over a given threshold
        # if white_noise_threshold.value > 0:
        #     I_out = I_in * (I_in <= white_noise_threshold) + I_noise * (I_in > white_noise_threshold)
        # else:
        #     I_out = I_noise

        # Return all negative array values to zero
        # I_out[I_out<0] = 0

        return I_out

    def high_freq_noise_std_calc(self, I_p_current, std_fit_params):
        I_in_i_log = self.log_Intensity(I_p_current)
        std_extra_noise = np.exp(
            std_fit_params[0] * I_in_i_log**2
            + std_fit_params[1] * I_in_i_log
            + std_fit_params[2]
        )
        return std_extra_noise

    def log_normal_noise_add(self, Array, mean, sigma, rng):
        """
        Add to or define an array from a log normal distribution

        Parameters
        ----------
        Array : np.array
            Input array.
        mean : float
            Mean of the log normal distribution.
        sigma : float
            1 standard deviation of the distribution.
        rng : generator object
            Generator object from numpy random generator module

        Returns
        -------
        Array : np.array
            Array drawn from the log normal distribution.

        """
        Array = (
            Array
            + np.log(
                rng.lognormal(mean=mean.value, sigma=sigma.value, size=np.shape(Array))
            )
            * mean.unit
        )
        return Array

    def shot_noise_add(self, current, rng):
        """
        Add shot noise to the photodiode energy in Amperes. This is applied to the
        summation of the induced photocurrent and the dark current.round out the research as written in our original abstract

        Parameters
        ----------
        current : np.array
            Array describing the energy in the circuit in Amperes. This should be a
            summation of dark current and the incident energy.
        rng : generator object
            Generator object from numpy random generator module

        Returns
        -------
        current_new : np.array
            Array describing the poisson drawn current in the circuit in Amperes.

        """
        # Convert amperes to electron rate for poisson pull
        electron_rate = current.to(u.cds.e / u.s)
        # Draw from the poisson distribution
        electron_rate_new = rng.poisson(electron_rate.value) * electron_rate.unit
        # Convert back to amperes
        current_new = electron_rate_new.to(u.A)
        return current_new

    def update_cutoff_freq(self, I_in, I_dark, cutoff_freq):
        """
        The cutoff frequency is a function of the ratio between the current value and
        the dark current (I_source+I_dark)/I_dark. The cutoff is maximized at 3000Hz.

        Parameters
        ----------
        I_in : np.array
            Incident current + dark current before logarithmic conversion. units of A
        I_dark : np.array or float
            nominal dark current. units of A
        cutoff_freq : float
            nominal cutoff frequency at the dark current rate. units of Hz

        Returns
        -------
        cutoff_freq_matrix : np.array
            Cutoff frequency across the array at this timestep due to the current
            intensity. units of Hz

        """
        # If the frequency cutoff is being modeled as a function of the current level
        # a matrix containing the timestep's frequency cutoff is necessary
        # cutoff_freq_matrix = (I_in / I_dark) * cutoff_freq.value # Brian's Paper
        cutoff_freq_matrix = (I_in / I_dark) ** 0.5 * cutoff_freq.value
        # The frequency cutoff maximizes at 3000Hz
        cutoff_freq_matrix[cutoff_freq_matrix > 3000] = 3000
        # Apply the units
        cutoff_freq_matrix = cutoff_freq_matrix * cutoff_freq.unit
        return cutoff_freq_matrix

    def low_pass_filter(self, I_in, I_ref, cutoff_freq, deltat):
        """
        Low pass filter that integrates the current value to a reference and then
        applies the low pass filter to determine the compared current value.

        Parameters
        ----------
        I_in : np.array
            Current induced current in Amps.
        I_ref : np.array
            Previous current after low pass filter in Amps.
        cutoff_freq : float
            3db cutoff frequency for the low pass filter transfer function in Hz.
        deltat : float
            Timestep duration in seconds.

        Returns
        -------
        I_1p : np.array
            Current value of system after the low pass filter in Amps.
        I_ref : np.array
            Internal value of the filter in Amps.

        """

        # Discrete time first-order response
        alpha = deltat / ((1 / (2 * np.pi * cutoff_freq)) + deltat)

        # Multiple steps in the low pass filter, the first step is classical low pass filter
        I_ref = np.multiply((1 - alpha.value), I_ref) + np.multiply(alpha.value, I_in)
        # The second step is another low pass filter using the internal setting as the new set point
        # I_1p = (np.multiply((1-alpha),I_ref.value) + np.multiply(alpha,I_ref.value))*I_in.unit

        return I_ref, I_ref

    def threshold_check(
        self,
        last_sim_time,
        curr_sim_time,
        last_I_mem,
        curr_I_mem,
        last_I_lc,
        curr_I_lc,
        Theta_on,
        Theta_off,
        N_event_total,
        N_event_time,
        N_event_polarity,
    ):
        """
        Check the difference between the induced and memorized current to determine how
        many events could possibly occur in this timestep and keep track of the time the
        next event is triggered on each pixel.

        Parameters
        ----------
        last_sim_time : float
            Time that the last simulation step ended [s].
        curr_sim_time : np.array
            Time that the current simulation step ends [s].
        last_I_mem : np.array
            Memorized current at the end of the last simulation step [log A].
        curr_I_mem : np.array
            Memorized current at the end of the current simulation step [log A].
        last_I_lc : np.array
            Induced current at the end of the last simulation step [log A].
        curr_I_lc : np.array
            Induced current at the end of the current simulation step [log A].
        Theta_on : np.array
            Positive threshold difference between the induced and memorized current to
            produce an event.
        Theta_off : np.array
            Negative threshold difference between the induced and memorized current to
            produce an event.
        N_event_total : np.array
            Array that keeps track of the total number of events that a pixel could
            experience during a simulation. This helps quantify the number of lost
            events of the course of a simulation.
        N_event_time : np.array
            Array that keeps track of the most recent time a pixel surpassed a
            threshold. Additionally, pixels with time of 0 are no longer being held
            waiting for arbitration, so they can produce new events.
        N_event_polarity : np.array
            Array that keeps track of the polarity of the most recent triggered event
            for arbitration.

        Returns
        -------
        del_I_lc : np.array
            Slope of the induced current in this timestep [log A/s].
        del_I_mem : np.array
            Slope of the memorized current in this timestep [log A/s].
        N_event_total : np.array
            Updated array that keeps track of the total number of events that a pixel
            could experience during a simulation. This helps quantify the number of lost
            events of the course of a simulation.
        N_event_time : np.array
            Updated array that keeps track of the most recent time a pixel surpassed a
            threshold. Additionally, pixels with time of 0 are no longer being held
            waiting for arbitration, so they can produce new events.
        N_event_polarity : np.array
            Updated array that keeps track of the polarity of the most recent triggered
            event for arbitration.

        """
        shape = np.shape(N_event_polarity)
        # Calculate the slope of the low passed logarithmic current between the last
        # time and the current time step
        del_I_lc = (curr_I_lc - last_I_lc) / (curr_sim_time - last_sim_time)

        # Calculate the slope of the memorized current between the last time and the
        # current time step
        del_I_mem = (curr_I_mem - last_I_mem) / (curr_sim_time - last_sim_time)

        # Calculate the total number of events that could occur from the change over
        # this time frame, add it to the running total to keep track of all the
        # potential events that could happen over the course of the simulation
        # without the refractory period or arbiter getting in the way
        N_event_total = (
            N_event_total
            + np.floor(((curr_I_lc - curr_I_mem) * (curr_I_lc > curr_I_mem)) / Theta_on)
            + np.floor(
                (np.abs(curr_I_lc - curr_I_mem) * (curr_I_lc < curr_I_mem)) / Theta_off
            )
        )

        # Calculate the time that the next event will occur if a pixel is currently not
        # waiting for arbitration
        pos_t, neg_t = self.time_next_event(
            last_sim_time,
            curr_sim_time,
            last_I_mem,
            curr_I_mem,
            del_I_mem,
            last_I_lc,
            curr_I_lc,
            del_I_lc,
            Theta_on,
            Theta_off,
            N_event_time,
        )
        N_event_time = N_event_time + np.abs(pos_t) + np.abs(neg_t)

        # Record the polarity of the events that have been added to the queue
        N_event_polarity = (
            N_event_polarity
            + np.ones(shape) * (pos_t != 0)
            + -1 * np.ones(shape) * (neg_t != 0)
        )

        return del_I_lc, del_I_mem, N_event_total, N_event_time, N_event_polarity

    def time_next_event(
        self,
        last_sim_time,
        curr_sim_time,
        last_I_mem,
        curr_I_mem,
        del_I_mem,
        last_I_lc,
        curr_I_lc,
        del_I_lc,
        Theta_on,
        Theta_off,
        N_event_time,
    ):
        """
        Generates the time of the next event given it is before the end of the current
        timestep of the simulation.

        Parameters
        ----------
        last_sim_time : float
            Time that the last simulation step ended [s].
        curr_sim_time : np.array
            Time that the current simulation step ends [s].
        last_I_mem : np.array
            Memorized current at the end of the last simulation step [log A].
        curr_I_mem : np.array
            Memorized current at the end of the current simulation step [log A].
        del_I_mem : np.array
            Slope of the memorized current in this timestep [log A/s].
        last_I_lc : np.array
            Induced current at the end of the last simulation step [log A].
        curr_I_lc : np.array
            Induced current at the end of the current simulation step [log A].
        del_I_lc : np.array
            Slope of the induced current in this timestep [log A/s].
        Theta_on : np.array
            Positive threshold difference between the induced and memorized current to
            produce an event.
        Theta_off : np.array
            Negative threshold difference between the induced and memorized current to
            produce an event.
        N_event_time : np.array
            Array that keeps track of the most recent time a pixel surpassed a
            threshold. Additionally, pixels with time of 0 are no longer being held
            waiting for arbitration, so they can produce new events.

        Returns
        -------
        positive_event_times : np.array
            Array with times that a positive event is triggered [s].
        negative_event_times : np.array
            Array with times that a negative event is triggered [s].

        """
        # Calculate the time where the positive or negative threshold is reached
        # Assumes a linear progression between current values
        # diff_del_1 = (del_I_lc-del_I_mem)
        # diff_del_2 = (del_I_mem-del_I_lc)
        # if 0.0 in diff_del_1:
        #     pdb.set_trace()
        # if 0.0 in diff_del_2:
        #     pdb.set_trace()
        positive_event_times = (
            (N_event_time == 0)
            * (curr_I_lc > curr_I_mem)
            * (
                (Theta_on - last_I_lc + last_I_mem) / (del_I_lc - del_I_mem)
                + last_sim_time
            )
        )
        negative_event_times = (
            (N_event_time == 0)
            * (curr_I_lc < curr_I_mem)
            * (
                (Theta_off + last_I_lc - last_I_mem) / (del_I_mem - del_I_lc)
                + last_sim_time
            )
        )

        # Only return times in the current simulation time
        positive_event_times = np.zeros(
            np.shape(positive_event_times)
        ) + positive_event_times * (last_sim_time < positive_event_times) * (
            positive_event_times < curr_sim_time
        )
        negative_event_times = np.zeros(
            np.shape(negative_event_times)
        ) + negative_event_times * (last_sim_time < negative_event_times) * (
            negative_event_times < curr_sim_time
        )

        # if np.max(positive_event_times) != 0 or np.max(negative_event_times) != 0:
        #     pdb.set_trace()

        return positive_event_times, negative_event_times

    def interpolate_current(self, t0, t1, I_0, del_I):
        """
        Interpolate the current at a given intermediary time.

        Parameters
        ----------
        t0 : float
            Time that the last simulation step ended [s].
        t1 : float
            Time that the current will be calculated [s].
        I_0 : float or np.array
            Previous current value [log A].
        del_I : float or np.array
            Slope of current during this time step [log A/s].

        Returns
        -------
        I_1 : float or np.array
            Current value at the intermediary time [log A].

        """
        I_1 = I_0 + del_I * (t1 - t0)

        return I_1

    def threshold_check_old(
        self, sim_time, I_1p, I_mem, Theta_on, Theta_off, Ne, Ne_times, T_hold
    ):
        """
        Checks the current change from the memorized current and keeps track of pixels
        that are waiting for arbitration.

        Parameters
        ----------
        sim_time : float
            Simulation time in seconds.
        I_1p : np.array
            Array with the current values after the low pass filter in Amps.
        I_mem : np.array
            Memorized current values in Amps.
        Theta_on : np.array
            Threshold values for a positive polarity event unique to each pixel.
        Theta_off : np.array
            Threshold values for a negative polarity event unique to each pixel.
        Ne : np.array
            Array tracking which pixels are waiting to record an event and the polarity.
        Ne_times : np.array
            Array tracking the time each pixel was last added to the arbitration queue.
        T_hold : np.array
            Array tracking if a pixel is currently in the refractory period.

        Returns
        -------
        Ne : np.array
            Array tracking which pixels are waiting to record an event and the polarity.
        Ne_loss : np.array
            Array tracking how many events were not recorded during the arbitration process.
        Ne_times : np.array
            Array tracking the time each pixel was last added to the arbitration queue.

        """

        shape = np.shape(I_1p)

        # Calculate the log change
        Delta_I = I_1p - I_mem
        # Based on the change in current how many total events would this pixel create in this timestep
        Ne_total = np.floor(Delta_I / Theta_on) * (Delta_I >= 0) + np.ceil(
            Delta_I / Theta_off
        ) * (Delta_I < 0)
        # Add events to queue
        Ne = Ne + (
            np.ones(shape) * (Ne_total >= 1) + np.ones(shape) * -1 * (Ne_total <= -1)
        ) * ((Ne == 0) * (T_hold == 0))
        Ne_times = Ne_times + sim_time * (
            np.ones(shape) * (Ne_total >= 1) + np.ones(shape) * (Ne_total <= -1)
        ) * ((Ne == 0) * (T_hold == 0))

        return I_mem, Ne, Ne_times

    def junction_leak(self, delta_t, R_leak, Theta_on, I_mem, I_dark_log):
        """
        Apply the junction leak to the memorized current over the course of this timestep.

        Parameters
        ----------
        delta_t : float
            Duration of this timestep in seconds.
        R_leak : float
            Rate at which an event will occur due to the leak in the reset circuit in
            Hz.
        Theta_on : np.array
            Threshold change values required for each pixel.
        I_mem : np.array
            Previously memorized current for each pixel.
        I_dark_log : float
            Minimum memorized current will be equivalent to the dark current of the
            circuit. Any values dropping below this in the log scale will be reset to
            this value.

        Returns
        -------
        I_mem : np.array
            Updated array of memorized currents at the end of the timestep in Amps.

        """
        shape = np.shape(I_mem)

        I_mem = (
            I_mem - delta_t * R_leak * Theta_on
        )  # FIXME : Tell Brian about logic to get .2 of change in log scale over the desired timescale. .2 change needed for event * number of events per second = amount of change needed per second to achieve rate * timestep = amount of change for this timestep to achieve rate, Ask brian about 5Hz rate based on the 2fA line and 10e-3 background
        I_mem = I_mem * (I_mem > I_dark_log) + np.ones(shape) * I_dark_log * (
            I_mem <= I_dark_log
        )

        return I_mem

    def parasitic_leak(
        self, Incident_energy, parasitic_leak_rate, delta_t, Theta_on, I_mem, I_dark
    ):
        """
        Apply a parasitic leak to the memorized current over the course of this
        timestep.

        Parameters
        ----------
        Incident_energy : np.array
            Irradiance on each pixel.
        parasitic_leak_rate : float
            Rate at which an event will occur due to stray photons reaching the reset
            circuit in Hz/W.
        delta_t : float
            Duration of this timestep in seconds.
        Theta_on : np.array
            Threshold change values required for each pixel.
        I_mem : np.array
            Previously memorized current for each pixel.
        I_dark_log : float
            Minimum memorized current will be equivalent to the dark current of the
            circuit. Any values dropping below this in the log scale will be reset to
            this value.

        Returns
        -------
        I_mem : np.array
            Updated array of memorized currents at the end of the timestep in Amps.

        """

        # Unpack pixel dimensions
        # For DAVIS 346
        # fperlux = 2.7e-3 #Hz/lux, estimate from online since it is source specific
        # fperWperm**2 =  0.018441054 #Hz/(W/m^2), using 638.002 lumens/W
        # fperW = 81960240 #Hz/W, using 638.002 lumens/W and 1.5e-5 m for the pixel pitch in both directions

        # First calculate the event rate unique to each pixel
        R_para = np.real(Incident_energy) * u.W * parasitic_leak_rate

        shape = np.shape(I_mem)

        I_mem = I_mem - delta_t * R_para * Theta_on
        I_mem = I_mem * (I_mem > I_dark) + np.ones(shape) * I_dark * (I_mem <= I_dark)

        return I_mem

    def shot_noise_after_low_pass(
        self,
        last_sim_time,
        curr_sim_time,
        R_n,
        N_event_total,
        N_event_time,
        N_event_polarity,
        rng,
    ):  # (self,delta_t,last_sim_time,R_n,I_p,I_mem,N_event_total,N_event_time,N_event_polarity,Theta_on,Theta_off,rng,F = 0.25):
        """
        Determine if which pixels are triggered by shot noise.

        Parameters
        ----------
        delta_t : float
            Timestep in simulation.
        R_n : float
            Observed noise rate on dark pixels in Hz.
        I_p : np.array
            Array of incident energy on .
        I_mem : TYPE
            DESCRIPTION.
        Ne : TYPE
            DESCRIPTION.
        Theta_on : TYPE
            DESCRIPTION.
        Theta_off : TYPE
            DESCRIPTION.
        T_hold : TYPE
            DESCRIPTION.
        rng : generator object
            Generator object from numpy random generator module
        F : TYPE, optional
            DESCRIPTION. The default is 0.25.

        Returns
        -------
        I_mem : TYPE
            DESCRIPTION.
        Ne_false : TYPE
            DESCRIPTION.
        Ne : TYPE
            DESCRIPTION.

        """
        # Poisson distribution
        lambda_r = R_n * (
            curr_sim_time - last_sim_time
        )  # rate at a given average millilux background for this date

        shape = np.shape(N_event_total)

        # Sample shot events from a poisson distribution
        N_shot_events = rng.poisson(lambda_r, size=shape) * (N_event_time == 0)
        # Make all values equal to one event since the refractory period cannot be accounted for
        N_shot_events = np.zeros(shape=shape) + np.ones(shape=shape) * (
            N_shot_events > 0
        )
        # Add to the event total
        N_event_total = N_event_total + N_shot_events
        # Assign the polarity randomly
        N_polarity_determination = rng.uniform(-1, 1, size=shape)
        N_polarity_determination = np.ones(shape=shape) * 1 * (
            N_polarity_determination >= 0
        ) + np.ones(shape=shape) * -1 * (N_polarity_determination < 0)
        N_event_polarity = N_event_polarity + N_shot_events * (N_polarity_determination)
        # Give the triggered shot noise events a random time stamp during the duration of
        # this simulation step
        N_event_time = N_event_time + (
            last_sim_time + rng.uniform(0, 1, size=shape)
        ) * (N_shot_events == 1)

        return N_event_total, N_event_time, N_event_polarity

    def refractory_period_check(self, T_refr, delta_t, T_hold, T_hold_time, Ne):

        shape = np.shape(Ne)

        # Add new pixels with events to the hold matrix
        T_hold = T_hold + np.ones(shape) * ((Ne == 1 + Ne == -1) * (T_hold == 0))
        # Update the hold matrix to the value 2 for pixels past the refractory period
        T_hold[T_hold_time >= T_refr] = 2
        # Update the hold time of the pixels past the refractory period after they are arbitrated back to 0
        # Update the hold time of the no zero hold pixels
        T_hold_time = T_hold_time + np.ones(shape) * delta_t * (T_hold_time > 0)

        return T_hold, T_hold_time

    def arbiter(
        self,
        recorded_events,
        recorded_events_attr_list,
        N_event_time,
        N_event_polarity,
        last_sim_time,
        curr_sim_time,
        win_time,
        last_I_lc,
        curr_I_lc,
        del_I_lc,
        curr_I_mem,
        del_I_mem,
        Theta_on,
        Theta_off,
        last_row,
        last_column,
        max_col,
        T_ref,
        recording_freq,
        attribution_dict,
        rng,
        R_n=0,
        maximum_events=0,
        added_shot_noise=False,
    ):
        # Generate queue of events
        event_queue = self.gen_event_queue(N_event_time, N_event_polarity)

        # Calculate the refractory period time for each potential event
        N_event_ref_time = self.create_n_event_ref(N_event_time, T_ref)

        # Start the arbiter at the right time value
        time_step = (1 / recording_freq).to(
            u.s
        )  # maximum recording frequency of arbiter (~1GHz)
        curr_time = last_sim_time + time_step.value

        # Select the first event and keep track of the last row and column in the list
        # of events to be processed
        # pdb.set_trace()
        event_index, event_avail = self.get_next_event(
            event_queue, last_row, last_column, max_col, curr_time, curr_sim_time
        )

        # While there are events left to process and the maximum time of the simulation
        # step has not been reached
        while curr_time <= curr_sim_time and event_avail == True:
            # Loop through events in the event index
            for i in np.arange(event_index.size):
                # Process the events
                recorded_events, event_queue, last_column, last_row, curr_time = (
                    self.process_event(
                        recorded_events,
                        recorded_events_attr_list,
                        event_queue,
                        event_index[i],
                        curr_time,
                        win_time.value,
                        curr_sim_time,
                        attribution_dict,
                    )
                )
                # Reset the arrays tracking the polarity
                N_event_polarity[last_row, last_column] = 0
                # Check if new event is triggered &
                # Update arrays tracking event time and refractory circuit release time
                N_event_time, New_event_possible = self.n_event_time_update(
                    N_event_time, N_event_ref_time, N_event_polarity, curr_time
                )
                # If a new event is possible, the circuit is reset
                # Therefore update the memorized current
                # & check if another event will happen in this period
                if np.any(New_event_possible == 1):
                    # Update the memorized current to the current at the current time
                    curr_I_mem, last_I_lc_i, curr_I_mem_i = (
                        self.update_memorized_current(
                            New_event_possible,
                            last_sim_time,
                            curr_sim_time,
                            curr_time,
                            last_I_lc,
                            del_I_lc,
                            curr_I_mem,
                            del_I_mem,
                        )
                    )
                    # Check if another event will be triggered due to the current changing
                    # before the end of the time step
                    event_queue, N_event_time, N_event_ref_time, N_event_polarity = (
                        self.check_for_new_event(
                            event_queue,
                            New_event_possible,
                            curr_time,
                            curr_sim_time,
                            last_I_lc_i,
                            curr_I_mem_i,
                            del_I_mem,
                            curr_I_lc,
                            del_I_lc,
                            Theta_on,
                            Theta_off,
                            N_event_time,
                            N_event_ref_time,
                            N_event_polarity,
                            T_ref,
                        )
                    )
                    # positive_event_time, negative_event_time = self.time_next_event(curr_time, curr_sim_time, last_I_lc_i, curr_I_mem_i, del_I_mem[New_event_possible], last_I_lc_i, curr_I_lc[New_event_possible], del_I_lc[last_row,last_column], Theta_on[last_row,last_column], Theta_off[last_row,last_column], N_event_time=0)
                    # if positive_event_time != 0 or negative_event_time != 0:
                    #     if positive_event_time != 0:
                    #         new_event_time = float(positive_event_time)
                    #         new_event_polarity = 1
                    #     else:
                    #         new_event_time = float(negative_event_time)
                    #         new_event_polarity = -1
                    #     # add to time and polarity arrays
                    #     N_event_time[last_row,last_column] = new_event_time
                    #     N_event_polarity[last_row,last_column] = new_event_polarity
                    #     # Update event queue
                    #     if not isinstance(T_ref,float):
                    #         event_queue = self.add_event_queue(event_queue, last_column, last_row, new_event_time, new_event_polarity, T_ref[last_row,last_column])
                    #     else:
                    #         event_queue = self.add_event_queue(event_queue, last_column, last_row, new_event_time, new_event_polarity, T_ref)
                # if new_event == False and added_shot_noise == True:
                #     # If another event due to change is not triggered and shot noise is
                #     # artificially added check if a shot noise event is triggered
                #     new_event_count, new_event_time, new_event_polarity = self.shot_noise(curr_time, curr_sim_time, R_n, np.zeros(np.shape((1,))), np.zeros(np.shape((1,))), np.zeros(np.shape((1,))), rng)
                #     if int(new_event_count) != 0:
                #         new_event_time = float(new_event_time)
                #         new_event_polarity = int(new_event_polarity)
                # Add one microsecond to the current recording time
                curr_time += time_step.value
                # If the current recording time exceeds the simulation time, break the loop
                if curr_time > curr_sim_time:
                    break
                if maximum_events > 0 and len(recorded_events) > maximum_events:
                    break
            # Break if maximum events generated
            if maximum_events > 0 and len(recorded_events) > maximum_events:
                break
            # Get next event / events, only if the current time is <= simulation time
            if curr_time <= curr_sim_time:
                event_index, event_avail = self.get_next_event(
                    event_queue,
                    last_row,
                    last_column,
                    max_col,
                    curr_time,
                    curr_sim_time,
                )
        # Update arrays tracking event time and refractory circuit release time one last time before next time step
        N_event_time, New_event_possible = self.n_event_time_update(
            N_event_time, N_event_ref_time, N_event_polarity, curr_time
        )
        # If a new event is possible, the circuit memorized current is reset
        if np.any(New_event_possible == 1):
            # Update the memorized current to the current at the current time
            curr_I_mem, last_I_lc_i, curr_I_mem_i = self.update_memorized_current(
                New_event_possible,
                last_sim_time,
                curr_sim_time,
                curr_time,
                last_I_lc,
                del_I_lc,
                curr_I_mem,
                del_I_mem,
            )
        return (
            recorded_events,
            recorded_events_attr_list,
            N_event_time,
            N_event_polarity,
            curr_I_mem,
            last_row,
            last_column,
            curr_time,
        )

    def gen_event_queue(self, N_event_time, N_event_polarity):
        """
        Take the array of time values indicating when event thresholds were reached
        flatten it into a queue of events to be processed in this simulation timestep.

        Parameters
        ----------
        N_event_time : np.array
            Array that keeps track of the most recent time a pixel surpassed a
            threshold. Additionally, pixels with time of 0 are no longer being held
            waiting for arbitration, so they can produce new events.
        N_event_polarity : np.array
            Array that keeps track of the polarity of the most recent triggered event
            for arbitration.
        T_ref : float or np.array
            Float or array that captures the refractory period time [s].

        Returns
        -------
        event_queue : np.array
            Contains the unique x and y location of each event, time the event passes
            the refractory period, polarity of the event, and a boolean that tracks
            whether the event has been processed or not.

        """
        # Determine the locations of the non-zero time values
        event_locations = np.where(((N_event_time * np.abs(N_event_polarity)) > 0))

        # Flatten the array with times of the events
        event_times = N_event_time[event_locations]
        # Flatten the array with polarity of the events
        event_polarities = N_event_polarity[event_locations]

        # Add the refractory period to the event times (Processing of event can happen instantly after event occurs, circuit simply does not reset until refractory period is over)
        # if not isinstance(T_ref, float):
        #     # Flatten the array with the refractory periods
        #     T_ref = T_ref[event_locations]
        # event_times = event_times + T_ref

        # Create and append a variable that tracks which events have been processed
        events_processed = np.ones((event_times.shape[0]), dtype=bool)
        event_queue = np.column_stack(
            (
                event_times,
                event_locations[1],
                event_locations[0],
                event_polarities,
                events_processed,
            )
        )

        # Organize the array in ascending time order
        event_queue = event_queue[
            event_queue[:, 0].argsort()
        ]  ##FIXME: Might not need this arg sort since it has been added to get next event based on group of events being taken

        return event_queue

    def get_next_event(
        self, event_queue, last_row, last_column, max_col, curr_time, curr_sim_time
    ):
        """
        Determine the next event and/or events to be processed.

        Parameters
        ----------
        event_queue : np.array
            Contains the unique x and y location of each event, time the event passes
            the refractory period, polarity of the event, and a boolean that tracks
            whether the event has been processed or not.
        last_row : int
            Last processed row of the arbiter.
        last_column : int
            Last processed column of the arbiter.
        max_col : int
            Number of columns in each row to enable assigning a unique identifier to
            each pixel.
        curr_time : float
            Current time of the simulation at the microsecond level of precision [s].
        curr_sim_time: float
            Simulation time at the chosen macro discretization. After this time, the
            next frame of the physics simulation output is loaded. [s]

        Returns
        -------
        next_event_index : np.array
            Rows in the event queue that can be processed without updating the event
            queue and selecting the next pixel.
        """
        # Track indicies of events to be processed
        next_event_index = []
        # Track whether an event is available to process
        event_available = False

        # First check if there are any events left to process
        if np.any(event_queue[:, 4] == 1):
            # If there are no events currently waiting to process, the next unprocessed
            # one in time is the next event
            if np.all((event_queue[event_queue[:, 4] == 1][:, 0] >= curr_time)):
                if np.any(event_queue[event_queue[:, 4] == 1][:, 0] <= curr_sim_time):
                    # Make sure events are sorted by the time they occurred
                    event_queue = event_queue[event_queue[:, 0].argsort()]
                    ##FIXME: Might need to consider the readout rate versus the
                    # The events are ordered by the time of their refractory period ending
                    next_event_index = np.where(
                        (
                            (event_queue[:, 4] == 1)
                            * (event_queue[:, 0] <= curr_sim_time)
                        )
                    )[0]
                    event_available = True
            # Otherwise, choose the closest event to the arbiter's location based on the
            # last row and column. The arbiter scans each row, then each column in the
            # row sequentially. If there is a large group of events back to back in this
            # order, no new events will be added to this block. Output the whole block
            # for processing.
            else:
                # Only consider events with timestamps before the current time
                short_event_list = event_queue[
                    ((event_queue[:, 4] == 1) * (event_queue[:, 0] <= curr_time))
                ]
                if (
                    np.shape(short_event_list)[0] != 0
                    and np.shape(short_event_list)[0] != 1
                ):
                    # Ensure the numpy array of pixels is ordered by row then column
                    ind = np.lexsort((short_event_list[:, 1], short_event_list[:, 2]))
                    short_event_list = short_event_list[ind]
                    unique_loc_ids = (
                        short_event_list[:, 2] * max_col + short_event_list[:, 1]
                    )
                    # Find the next pixel in the arbitration process
                    ind = np.searchsorted(
                        unique_loc_ids, last_row * max_col + last_column
                    )
                    # Reorder the events with the next pixel as the first row
                    short_event_list = short_event_list.take(
                        range(ind, len(short_event_list) + ind), axis=0, mode="wrap"
                    )
                    # # Reorder the local ids (No longer needed because noise is added on front end of circuitry simulation)
                    # unique_loc_ids = unique_loc_ids.take(range(ind,len(short_event_list)+ind), axis=0, mode='wrap')
                    # # Determine if there is a block of events to select by finding the
                    # # first non-one value in the change of unique ids
                    # unique_loc_gap_id = np.where(((unique_loc_ids[1:]-unique_loc_ids[0:-1])>1))[0]
                    # # If no gaps are greater than one, the whole short list is processed
                    # if unique_loc_gap_id.size == 0:
                    #     event_block = short_event_list
                    # else:
                    #     # Slice from the beginning the uninterputed event list to the
                    #     # first identified break
                    #     event_block = short_event_list[0:unique_loc_gap_id+1,:]
                    # Loop through the events to preserve the order
                    for i in np.arange(len(short_event_list)):
                        next_event_index.append(
                            np.where(
                                (event_queue == short_event_list[i, :]).all(axis=1)
                            )[0][0]
                        )
                    next_event_index = np.array(next_event_index)
                    event_available = True
                elif np.shape(short_event_list)[0] == 1:
                    # Only one event is available
                    next_event_index = np.where(
                        (event_queue == short_event_list).all(axis=1)
                    )[0][0]

        return next_event_index, event_available

    def process_event(
        self,
        recorded_events,
        recorded_events_attr,
        event_queue,
        event_index,
        curr_time,
        win_time,
        curr_sim_time,
        attribution_dict,
    ):

        # Add [t,x,y,polarity,attribution] to the recorded event list
        event_2_process = list(event_queue[event_index, :])
        # Remove the processing term
        event_2_process.pop()
        # If the current simulation time is later than the time after the refractory
        # period, set that as the processing time
        # Else, set the current time to the time the refactory period ends
        if event_2_process[0] < curr_time:
            event_2_process[0] = np.around(curr_time, decimals=6)
        else:
            event_2_process[0] = np.around(event_2_process[0], decimals=6)
            curr_time = event_2_process[0]

        # Check if first event
        if not recorded_events_attr:
            first_event = True
        else:
            first_event = False
        # If this pixel is attributed in the current timestep, keep that attribution
        if hasattr(curr_sim_time, "unit"):
            attr_keys = attribution_dict[curr_sim_time].keys()
            curr_sim_time_id = curr_sim_time
        else:
            attr_keys = attribution_dict[curr_sim_time * u.s].keys()
            curr_sim_time_id = curr_sim_time * u.s
        if (event_2_process[1], event_2_process[2]) in list(attr_keys):
            # attribution_dict[curr_sim_time][(event_2_process[1],event_2_process[2])]
            recorded_events_attr.append(
                attribution_dict[curr_sim_time_id][
                    (event_2_process[1], event_2_process[2])
                ]
            )
            # for j in attribution_dict[curr_sim_time][(event_2_process[1],event_2_process[2])]:
            #     pdb.set_trace()
            #     recorded_events_attr =
            # event_2_process.append(j[0])
            # event_2_process.append(j[1])
        else:
            # Check if previous attributions
            previous_att = False
            # If within a certain time threshold from the current event and on the same pixel
            # if len(recorded_events) == 0:
            if first_event:
                previous_att = False
            else:
                recorded_events_check = np.array(recorded_events, dtype=np.int16)
                # if len(recorded_events_check.shape) == 1:
                #     prior_rel_event_mask = (recorded_events_check[0]>=((curr_time-win_time)))*(recorded_events_check[1]==event_2_process[1])*(recorded_events_check[2]==event_2_process[2])
                # else:
                #     prior_rel_event_mask = (recorded_events_check[:,0]>=((curr_time-win_time)))*(recorded_events_check[:,1]==event_2_process[1])*(recorded_events_check[:,2]==event_2_process[2])
                prior_rel_event_mask = (
                    (recorded_events_check[:, 0] >= ((curr_time - win_time)))
                    * (recorded_events_check[:, 1] == event_2_process[1])
                    * (recorded_events_check[:, 2] == event_2_process[2])
                )
                # if np.any((recorded_events['t']>=((curr_time-win_time)))*(recorded_events['x']==event_2_process[1])*(recorded_events['y']==event_2_process[2])):
                if np.any(prior_rel_event_mask):
                    # Attribute the event to the same source
                    previous_att = True
                    # Find associated events
                    last_prev_events = np.where(prior_rel_event_mask)[0][-1]
                    # Assign to previous association
                    recorded_events_attr.append(recorded_events_attr[last_prev_events])

            # elif recorded_events.size == 5:
            #     pdb.set_trace()
            #     if np.any(((recorded_events[0].astype(float)>=((curr_time-win_time).value))*(recorded_events[1].astype(float)==event_2_process[1])*(recorded_events[2].astype(float)==event_2_process[2]))):
            #         # Attribute the event to the same source
            #         previous_att = True
            #         # Assign to previous association
            #         event_2_process.append(recorded_events[4])
            # elif np.any(((recorded_events[:,0].astype(float)>=((curr_time-win_time).value))*(recorded_events[:,1].astype(float)==event_2_process[1])*(recorded_events[:,2].astype(float)==event_2_process[2]))):
            #     # Attribute the event to the same source
            #     previous_att = True
            #     # Find the association of the last event event that fits this description
            #     event_locations = np.where(((recorded_events[:,0].astype(float)>=((curr_time-win_time).value))*(recorded_events[:,1].astype(float)==event_2_process[1])*(recorded_events[:,2].astype(float)==event_2_process[2])))
            #     # All the associations within this window should be the same, assign to most recent
            #     event_2_process.append(recorded_events[event_locations[0][-1],4])
            if previous_att == False:
                # Attribute the event to noise
                recorded_events_attr.append("noise")
                # event_2_process.append('noise')

        # Create a new dataframe and merge it with the old one
        # new_event = pd.DataFrame({'t':event_2_process[0],'x':event_2_process[1],'y':event_2_process[2],'p':event_2_process[3],'attr':event_2_process[4]},index=[len(recorded_events)])
        # recorded_events = pd.concat([recorded_events,new_event])

        # Dataframes are too cumbersome, keep as list and convert to dataframe on output
        # if first_event:
        #     recorded_events = event_2_process
        # else:
        #     recorded_events = np.row_stack((recorded_events,np.array(event_2_process)))
        recorded_events.append(event_2_process)

        # if recorded_events.size == 0:
        #     recorded_events = np.array(event_2_process)
        # elif recorded_events.size == 5:
        #     recorded_events = np.vstack((recorded_events.reshape(-1),event_2_process))
        # else:
        #     recorded_events = np.vstack((recorded_events,event_2_process))
        # Update the event queue to 0 in the final column to identify the event as processed
        event_queue[event_index, 4] = 0

        return (
            recorded_events,
            event_queue,
            int(event_2_process[1]),
            int(event_2_process[2]),
            curr_time,
        )

    def create_n_event_ref(self, N_event_time, T_ref):
        N_event_ref_time = np.zeros(N_event_time.shape)
        N_event_time_mask = N_event_time > 0
        if not isinstance(T_ref, float):
            N_event_ref_time[N_event_time_mask] = (
                N_event_time[N_event_time_mask] + T_ref[N_event_time_mask]
            )
        else:
            N_event_ref_time[N_event_time_mask] = (
                N_event_time[N_event_time_mask] + T_ref
            )
        return N_event_ref_time

    def n_event_time_update(
        self, N_event_time, N_event_ref_time, N_event_polarity, curr_time
    ):
        # Allow N_event_polarity to be updated to 0 once event is recorded, but do not
        # Update N_event_time to 0 until both the event is record (polarity = 0) and
        # triggered time + refractory period has passed
        New_event_possible = (
            (N_event_ref_time > 0)
            * (N_event_ref_time <= curr_time)
            * (N_event_polarity == 0)
        )
        if np.any(New_event_possible):
            # If they have passed their refractory period and the polarity has been
            # reset, reset the event time to allow for new events
            N_event_time[New_event_possible] = 0
        return N_event_time, New_event_possible

    def update_memorized_current(
        self,
        New_event_possible,
        last_sim_time,
        curr_sim_time,
        curr_time,
        last_I_lc,
        del_I_lc,
        curr_I_mem,
        del_I_mem,
    ):
        ##FIXME: I may be missing the proper update time when I skip time during the event recording, I will need to think how to take the refractory time over the current sim time in that one situation
        # Find the current at the current time for all the
        last_I_lc_i = self.interpolate_current(
            last_sim_time,
            curr_time,
            last_I_lc[New_event_possible],
            del_I_lc[New_event_possible],
        )
        # Update the final memorized current value
        curr_I_mem_i = self.interpolate_current(
            curr_time, curr_sim_time, last_I_lc_i, del_I_mem[New_event_possible]
        )
        curr_I_mem[New_event_possible] = curr_I_mem_i
        return curr_I_mem, last_I_lc_i, curr_I_mem_i

    def check_for_new_event(
        self,
        event_queue,
        New_event_possible,
        curr_time,
        curr_sim_time,
        last_I_lc_i,
        curr_I_mem_i,
        del_I_mem,
        curr_I_lc,
        del_I_lc,
        Theta_on,
        Theta_off,
        N_event_time,
        N_event_ref_time,
        N_event_polarity,
        T_ref,
    ):
        positive_event_time, negative_event_time = self.time_next_event(
            curr_time,
            curr_sim_time,
            last_I_lc_i,
            curr_I_mem_i,
            del_I_mem[New_event_possible],
            last_I_lc_i,
            curr_I_lc[New_event_possible],
            del_I_lc[New_event_possible],
            Theta_on[New_event_possible],
            Theta_off[New_event_possible],
            N_event_time[New_event_possible],
        )
        if np.any((positive_event_time > 0) | (negative_event_time > 0)):
            N_event_time[New_event_possible] = (
                N_event_time[New_event_possible]
                + np.abs(positive_event_time)
                + np.abs(negative_event_time)
            )
            # Record the polarity of the events that have been added to the queue
            N_event_polarity[New_event_possible] = (
                N_event_polarity[New_event_possible]
                + np.ones(positive_event_time.shape) * (positive_event_time != 0)
                + -1 * np.ones(negative_event_time.shape) * (negative_event_time != 0)
            )
            # Update event queue
            # Determine where new events are located
            New_event_mask = (N_event_time * New_event_possible) > 0
            event_queue = self.add_event_queue(
                event_queue, New_event_mask, N_event_time, N_event_polarity
            )
            # Update refractory period
            N_event_ref_time = self.create_n_event_ref(N_event_time, T_ref)
        return event_queue, N_event_time, N_event_ref_time, N_event_polarity

    def add_event_queue(
        self, event_queue, New_event_mask, N_event_time, N_event_polarity
    ):
        """
        Add events to the event queue

        Parameters
        ----------
        event_queue : np.array
            Contains the unique x and y location of each event, time the event passes
            the refractory period, polarity of the event, and a boolean that tracks
            whether the event has been processed or not.
        pixel_x_index : np.array
            x locations on the array that have new events.
        pixel_y_index : np.array
            y locations on the array that have new events.
        event_time : np.array
            Times that the new events are triggered.
        event_polarity : np.array
            Polarities of the new events that are triggered.
        T_ref : float or np.array
            Float or array that captures the refractory period time [s].

        Returns
        -------
        event_queue : np.array
            Updated array that contains the unique x and y location of each event,
            time the event passes the refractory period, polarity of the event, and a
            boolean that tracks whether the event has been processed or not.

        """
        y_locations, x_locations = np.where(New_event_mask == 1)

        # Flatten the array with times of the events
        event_times = N_event_time[New_event_mask]
        # Flatten the array with polarity of the events
        event_polarities = N_event_polarity[New_event_mask]

        # Add the refractory period to the event times (Processing of event can happen instantly after event occurs, circuit simply does not reset until refractory period is over)
        # if not isinstance(T_ref, float):
        #     # Flatten the array with the refractory periods
        #     T_ref = T_ref[event_locations]
        # event_times = event_times + T_ref

        # Create and append a variable that tracks which events have been processed
        events_processed = np.ones((event_times.shape[0]), dtype=bool)
        new_event_info = np.column_stack(
            (event_times, x_locations, y_locations, event_polarities, events_processed)
        )

        # Organize the array in ascending time order
        new_event_info = new_event_info[new_event_info[:, 0].argsort()]
        # Add new event or events to the event queue
        # event_processed = 1
        # Add the refractory period to the event times
        # if not isinstance(T_ref,float):
        #     # Flatten the array with the refractory periods
        #     T_ref = T_ref[(pixel_y_index,pixel_x_index)]
        # event_time = event_time + T_ref
        # Add new event or events to the event queue
        # new_event_info = np.column_stack((event_time,pixel_x_index,pixel_y_index,event_polarity,event_processed))
        event_queue = np.row_stack((event_queue, new_event_info))

        return event_queue

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

    def plot_frame(
        self,
        data,
        RA,
        DEC,
        timestep,
        plot_name,
        sim_name,
        directory,
        plot_type="contour",
        color_max=0,
        color_min=0,
    ):
        try:
            fig, ax = plt.subplots(dpi=200)
            if plot_type == "contour":
                pcm = ax.contourf(
                    RA,
                    DEC,
                    data.value,
                    locator=ticker.LogLocator(),
                    cmap=plt.get_cmap("gray"),
                )
                cbar = fig.colorbar(pcm)
                cbar.set_label("[" + str(data.unit) + "]", rotation=270, labelpad=15)
            if plot_type == "pixel color":
                pcm = ax.pcolor(
                    RA,
                    DEC,
                    data.value,
                    cmap=plt.get_cmap("gray"),
                    vmin=color_min,
                    vmax=color_max,
                    shading="auto",
                )
            ax.axis("equal")
            ax.set_xlabel("FOV [deg]")
            ax.set_ylabel("FOV [deg]")
            ax.set_title(
                plot_name.replace("_", " ")
                + " at "
                + str(timestep.value)
                + str(timestep.unit)
            )
            plt.savefig(directory + plot_name + ".pdf")
        except:
            print("plot failed")

    def frame_RA_DEC(self, observer, step, x_mesh, y_mesh):
        """
        Convert the pixel x & y locations at a given time to Barycentric coordinates

        Parameters
        ----------
        observer : object
            Object containing all the observer data including the pointing dictionary.
        step : int
            Step in simulation to pull out the right pointing object.
        x_mesh : np.array
            Mesh that describes the x pixel number for each array point.
        y_mesh : TYPE
            Mesh that describes the y pixel number for each array point.

        Returns
        -------
        RA : np.array
            Mesh that describes the right ascension for each array point.
        DEC : TYPE
            Mesh that describes the declination for each array point.

        """
        # Convert pixel locations matricies to (Ra,Dec)
        # Pointing for timestep
        wcs_proj = observer.optics_wcs_dict[step]
        # Conversion to (Ra,Dec)
        RA, DEC = np.transpose(
            wcs_proj.wcs_pix2world(
                np.transpose([x_mesh.flatten(), y_mesh.flatten()]), 0
            )
        )
        RA = RA.reshape((observer.num_y_pix, observer.num_x_pix))
        DEC = DEC.reshape((observer.num_y_pix, observer.num_x_pix))
        return RA, DEC
