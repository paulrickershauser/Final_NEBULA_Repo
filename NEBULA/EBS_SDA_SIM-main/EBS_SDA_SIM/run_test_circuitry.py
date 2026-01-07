# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:28:01 2023
Circuitry Test Code

@author: Rachel Oliver
Removed unit dependance to speed up calculations
"""

from circuitry import ebCircuitSim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time
import pdb
import os
import scipy.optimize as opt
from astropy import units as u
from astropy.units import cds


class circuitTest(ebCircuitSim):
    """
    This class contains differnent methods to easily set up and run tests of the EBS circuit model
    """

    def __init__(self):
        directory = os.getcwd()
        self.file_directory = directory + "/test_circuitry/"
        # # Load simulation parameters
        # with open(self.file_directory+'observer.pickle') as picklefile:
        #     self.observer = pickle.load(picklefile)
        # with open(self.file_directory+'ebs_circuit_parameters.pickle') as picklefile:
        #     self.circuit_para = pickle.load(picklefile)
        self.frames_file_name = "C:/Users/User/Documents/GitHub/EBS_SDA_SIM/EBS_SDA_SIM/test_circuitry/FinalFields.hdf5"

    def __str__(self):
        print("This object contains standard methods for running EBS SDA.")
        return

    def transfer_function_test(self, signal="step"):
        """
        Test the integration of the instantaneous current and low pass filter when a
        square wave is applied.

        Inputs
        -------
        signal : string
            String which indicates the type of modeled signal. Valid options include
            'step', 'square wave', 'sinusoid'

        Returns
        -------
        Ip : np.array
            Input square wave for 1000 steps. [A]
        Iset : np.array
            Set point array that should accumulate values in Ip. [A]
        Ifinal : np.array
            Final current array that should low pass filter the set point because there
            is a time delayed response. [A]

        """
        # Load realistic sensor parameters
        with open(
            "C:/Users/User/Documents/GitHub/EBS_SDA_SIM/EBS_SDA_SIM/test_circuitry/ebs_circuit_parameters.pickle",
            "rb",
        ) as picklefile:
            circuit_para = pickle.load(picklefile)

        # Simulate noise with no addtional signal except the dark current
        # Determine the photon per second equivalent of the dark current
        I_dark_constant = self.calc_dark_constant(
            circuit_para.Ea_dark, circuit_para.I_dark_ref, circuit_para.T_dark_ref
        )
        I_dark_scaled = self.calc_temp_adjust_dark_current(
            circuit_para.Ea_dark, I_dark_constant, circuit_para.T
        )
        I_ref_0 = self.log_Intensity(I_dark_scaled)
        I_signal = 1e-14 * u.A

        # 1) Establish the instantaneous current input as the signal type indicated.
        # The step size of the impulse may effect the result, discretizing to half of
        # the frequency cutoff should be sufficient, but multiple times will be tested
        # 2) Run through the log conversion since we are working with very small signal
        # values
        # 3) Run through the low pass filter, at different cutoff frequencies
        # 4) Plot the responses of different sampling plots with same low pass frequency cutoffs
        # 5) Plot the responses of plots with same sampling but varying low pass frequency cutoffs

        # Set Simulation Duration and Sampling Parameters
        sim_length = 2 * u.s
        sim_freq = [10, 20, 100, 500, 1000] * u.Hz
        sim_step_total = ((sim_length * sim_freq).decompose()).astype(int)
        num_steps_max = np.max(sim_step_total)
        cutoff_freq = [1 / 10, 1, 10, 100, 1000] * u.Hz

        # Step Response = Consistent star irradiance is reaching the pixel
        step_responses = self.generate_step(
            I_signal, I_dark_scaled, sim_length, sim_step_total, 10 * u.s
        )
        step_responses_log = self.apply_log_conversion(step_responses)
        step_responses_low_pass = self.apply_low_pass_filter(
            step_responses_log, sim_length, I_ref_0, cutoff_freq
        )
        self.plot_low_pass_filter(
            step_responses_low_pass,
            step_responses_log,
            sim_length,
            "Step",
            num_steps_max,
        )

        # Square Wave = Consistent star irradiances are entering and exiting the pixel
        square_responses = self.generate_square_wave(
            I_signal, I_dark_scaled, sim_length, sim_step_total, 8 * u.s, 12 * u.s
        )
        square_responses_log = self.apply_log_conversion(square_responses)
        square_responses_low_pass = self.apply_low_pass_filter(
            square_responses_log, sim_length, I_ref_0, cutoff_freq
        )
        self.plot_low_pass_filter(
            square_responses_low_pass,
            square_responses_log,
            sim_length,
            "Square",
            num_steps_max,
        )

        # Sinusoid = Constantly changing irradiances indicative of disturbances
        # Note the sinusoid is shifted so all current values are positive
        sine_responses = self.generate_sinusoid(
            I_dark_scaled, sim_length, sim_step_total, 4 * u.s
        )
        sine_responses_log = self.apply_log_conversion(sine_responses)
        sine_responses_low_pass = self.apply_low_pass_filter(
            sine_responses_log, sim_length, I_ref_0, cutoff_freq
        )
        self.plot_low_pass_filter(
            sine_responses_low_pass,
            sine_responses_log,
            sim_length,
            "Sinusoid",
            num_steps_max,
        )

        # Impulse = Step Response that only lasts one time step indicative of disturbances
        impulse_responses = self.generate_impulse(
            I_dark_scaled, sim_length, sim_step_total, 1 * u.s
        )
        impulse_responses_log = self.apply_log_conversion(impulse_responses)
        impulse_responses_low_pass = self.apply_low_pass_filter(
            impulse_responses_log, sim_length, I_ref_0, cutoff_freq
        )
        self.plot_low_pass_filter(
            impulse_responses_low_pass,
            impulse_responses_log,
            sim_length,
            "Impulse",
            num_steps_max,
            1,
        )

        return

    def generate_step(
        self, I_signal, I_dark_scaled, sim_length, sim_step_total, step_on
    ):
        signal = {}
        for i, num_steps in enumerate(sim_step_total):
            x_vec = np.linspace(0, sim_length.value, num_steps) * sim_length.unit
            signal_i = np.ones(x_vec.shape) * I_dark_scaled * (
                x_vec < step_on
            ) + np.ones(x_vec.shape) * I_signal * (x_vec >= step_on)
            signal[num_steps] = signal_i
        return signal

    def generate_square_wave(
        self,
        I_signal,
        I_dark_scaled,
        sim_length,
        sim_step_total,
        square_start,
        square_stop,
    ):
        signal = {}
        for i, num_steps in enumerate(sim_step_total):
            x_vec = np.linspace(0, sim_length.value, num_steps) * sim_length.unit
            signal_i = np.ones(x_vec.shape) * I_dark_scaled * (
                (x_vec < square_start) + (x_vec > square_stop)
            ) + np.ones(x_vec.shape) * I_signal * (
                (square_start <= x_vec) * (x_vec <= square_stop)
            )
            signal[num_steps] = signal_i
        return signal

    def generate_sinusoid(self, I_dark_scaled, sim_length, sim_step_total, period):
        I_signal_diff = I_dark_scaled * 0.2
        signal = {}
        for i, num_steps in enumerate(sim_step_total):
            x_vec = np.linspace(0, sim_length.value, num_steps) * sim_length.unit
            signal_i = (I_signal_diff / 2) * np.sin(
                (x_vec * (2 * np.pi / period)).value
            ) + (I_dark_scaled)
            signal[num_steps] = signal_i
        return signal

    def generate_impulse(self, I_dark_scaled, sim_length, sim_step_total, impulse_time):
        I_signal_max = I_dark_scaled * 1.2
        signal = {}
        for i, num_steps in enumerate(sim_step_total):
            x_vec = np.linspace(0, sim_length.value, num_steps) * sim_length.unit
            signal_i = np.ones(x_vec.shape) * I_dark_scaled * (x_vec != impulse_time)
            signal_i[int((num_steps * (impulse_time / sim_length)).value)] = (
                I_signal_max
            )
            signal[num_steps] = signal_i
        return signal

    def apply_log_conversion(self, signal):
        log_signal = {}
        for i, steps in enumerate(signal.keys()):
            log_signal_i = self.log_Intensity(signal[steps])
            log_signal[steps] = log_signal_i

        return log_signal

    def apply_low_pass_filter(self, signal, sim_length, I_ref, filter_freq):
        low_pass_signal = {}
        for i, num_steps in enumerate(signal.keys()):
            low_pass_signal[num_steps] = {}
            time_step = sim_length / num_steps
            for j, freq in enumerate(filter_freq):
                if (1 / time_step) < 2 * freq:
                    continue
                low_pass_signal_i = np.zeros(signal[num_steps].size)  # * u.A
                I_ref_i = I_ref
                for k in np.arange(num_steps):
                    I_p_i, I_ref_i = self.low_pass_filter(
                        signal[num_steps][k], I_ref_i, freq, time_step
                    )
                    # pdb.set_trace()
                    low_pass_signal_i[k] = I_p_i
                low_pass_signal[num_steps][freq] = low_pass_signal_i
                print(
                    "Low Pass Filter on {} Steps with {} Frequency is complete".format(
                        num_steps, str(freq)
                    )
                )
        return low_pass_signal

    def shot_noise_histogram_generation(self, seed=None):
        # Load realistic sensor parameters
        with open(
            "C:/Users/User/Documents/GitHub/EBS_SDA_SIM/EBS_SDA_SIM/test_circuitry/ebs_circuit_parameters.pickle",
            "rb",
        ) as picklefile:
            circuit_para = pickle.load(picklefile)

        # Set the shape of the array to be only one pixel
        shape = np.ones((1, 1))
        pdb.set_trace()
        # Simulate noise with no addtional signal except the dark current
        # Determine the photon per second equivalent of the dark current
        I_dark_constant = self.calc_dark_constant(
            circuit_para.Ea_dark, circuit_para.I_dark_ref, circuit_para.T_dark_ref
        )
        I_dark_scaled = self.calc_temp_adjust_dark_current(
            circuit_para.Ea_dark, I_dark_constant, circuit_para.T
        )

        # Simulate over 10000 time steps with each time step being 1/500th of a second
        step_size = 1 * u.s
        num_steps = 10000
        time = 0 * u.s
        time_initial = 0 * u.s
        time_final = time + step_size * num_steps
        events = []

        # Keep a time history of the induced current
        I_dark = []

        # Set the random number generator if a seed is not provided
        if not isinstance(seed, int):
            seed_seq = np.random.SeedSequence()
            seed = seed_seq.generate_state(1)

        # self.run_shot_noise_sim(I_dark, num_pix_x, num_pix_y, num_steps_per_second, number_of_pulls, cutoff_freq, pos_threshold, neg_threshold, seed)

        # Calculate event pairs
        events, onon, offoff, onoff, offon = self.calculate_event_pairs(events)

        # Plot histograms of each of the event pair groupings
        self.plot_histogram(
            onon, "Time Difference Between On On Pairs", "Time [s]", "Counts"
        )
        self.plot_histogram(
            offoff, "Time Difference Between Off Off Pairs", "Time [s]", "Counts"
        )
        self.plot_histogram(
            onoff, "Time Difference Between On Off Pairs", "Time [s]", "Counts"
        )
        self.plot_histogram(
            offon, "Time Difference Between Off On Pairs", "Time [s]", "Counts"
        )

        return events, I_dark, onon, offoff, onoff, offon, seed

    def gaussian_noise_test(self, seed=None):
        # Load realistic sensor parameters
        with open(
            "C:/Users/User/Documents/GitHub/EBS_SDA_SIM/EBS_SDA_SIM/test_circuitry/ebs_circuit_parameters.pickle",
            "rb",
        ) as picklefile:
            circuit_para = pickle.load(picklefile)

        # Set the shape of the array to be only one pixel
        shape = np.ones((1, 1))
        # pdb.set_trace()
        # Simulate noise with no addtional signal except the dark current
        # Determine the photon per second equivalent of the dark current
        I_dark_constant = self.calc_dark_constant(
            circuit_para.Ea_dark, circuit_para.I_dark_ref, circuit_para.T_dark_ref
        )
        I_dark_scaled = self.calc_temp_adjust_dark_current(
            circuit_para.Ea_dark, I_dark_constant, circuit_para.T
        )

        # Simulate over 10000 time steps with each time step being 1/500th of a second
        step_size = 1 * u.s
        num_steps = 10000
        time = 0 * u.s
        time_initial = 0 * u.s
        time_final = time + step_size * num_steps
        events = []

        # Initialize a few parameters for the simulation, do not add additional variance
        I_ref = np.log(I_dark_scaled.value) * I_dark_scaled.unit
        I_mem_current = np.log(I_dark_scaled.value) * I_dark_scaled.unit
        N_event_total = np.zeros(shape.shape)
        N_event_time = np.zeros(shape.shape)
        N_event_polarity = np.zeros(shape.shape)
        last_row = 0
        last_column = 0
        I_filter_current = None
        I_filter_last = None
        I_mem_last = None
        time_last = None

        # Keep a time history of the induced current
        I_dark = []

        # Set the random number generator if a seed is not provided
        if not isinstance(seed, int):
            seed_seq = np.random.SeedSequence()
            seed = seed_seq.generate_state(1)
        rng = np.random.default_rng(seed)

        while time <= time_final:
            # Calculate the shot noise on the dark current for this timestep
            dark_rate, time_scale = self.white_noise_add(I_dark_scaled, rng)
            # Append the current to the tracking list
            I_dark.append(dark_rate)
            # Convert the current into the log scale
            I_log_current = self.log_Intensity(dark_rate)
            # Low pass the converted current
            I_filter_current, I_ref = self.low_pass_filter(
                I_log_current, I_ref, circuit_para.photodiode_3db_freq, step_size
            )
            # Only compare to the previous time, if past the initial time step
            if time != time_initial:
                (
                    del_I_filter,
                    del_I_mem,
                    N_event_total,
                    N_event_time,
                    N_event_polarity,
                ) = self.threshold_check(
                    time_last,
                    time,
                    I_mem_last,
                    I_mem_current,
                    I_filter_last,
                    I_filter_current,
                    circuit_para.theta_on,
                    circuit_para.theta_off,
                    N_event_total,
                    N_event_time,
                    N_event_polarity,
                )
                attribution_dict = {}
                attribution_dict[time] = {}
                (
                    events,
                    N_event_time,
                    N_event_polarity,
                    I_mem_current,
                    last_row,
                    last_column,
                ) = self.arbiter(
                    events,
                    N_event_time,
                    N_event_polarity,
                    time_last,
                    time,
                    2 * u.s,
                    I_filter_last,
                    I_filter_current,
                    del_I_filter,
                    I_mem_current,
                    del_I_mem,
                    circuit_para.theta_on,
                    circuit_para.theta_off,
                    last_row,
                    last_column,
                    1,
                    circuit_para.T_refr,
                    circuit_para.recording_freq,
                    attribution_dict,
                    rng,
                )
            # Increment time
            time_last = time.value * time.unit
            time += step_size
            I_filter_last = I_filter_current
            I_mem_last = I_mem_current

        # Calculate event pairs
        events, onon, offoff, onoff, offon = self.calculate_event_pairs(events)

        # Plot histograms of each of the event pair groupings
        self.plot_histogram(
            onon, "Time Difference Between On On Pairs", "Time [s]", "Counts"
        )
        self.plot_histogram(
            offoff, "Time Difference Between Off Off Pairs", "Time [s]", "Counts"
        )
        self.plot_histogram(
            onoff, "Time Difference Between On Off Pairs", "Time [s]", "Counts"
        )
        self.plot_histogram(
            offon, "Time Difference Between Off On Pairs", "Time [s]", "Counts"
        )

        return events, I_dark, onon, offoff, onoff, offon, seed

    def poisson_scaling_test(self):
        """
        Demonstrate the effect of using the time step to scale the photon/sec count

        Returns
        -------
        output_photon_rates : list
            List of produced photon per second counts.

        """
        # Produce a random number generator that can be reset after each set of poisson
        # distribution pulls
        seed_seq = np.random.SeedSequence()
        seed = seed_seq.generate_state(1)

        # Establish a set number of seconds to simulate
        n = 20000 * u.s
        # Set a number of photons/sec to simulate, make size of number of pulls
        photon_rate = 20000 * u.ph / u.s
        # Simulate different simulation time scales
        time_steps = np.array([1, 1 / 10, 1 / 100, 1 / 1000, 1 / 10000]) * u.s
        # Make input array into shot noise generator the size of number of pulls to make
        number_of_pulls = n / time_steps

        # Loop through the time steps with the same random generator
        output_photon_rates = []
        output_current = []
        for i, time_step in enumerate(time_steps):
            # set random number generator
            rng = np.random.default_rng(seed)
            # calculate the photon rates
            photon_rate_i = np.ones((1, int(number_of_pulls[i].value))) * photon_rate
            photon_rate_i, time_scale = self.shot_noise_add(
                photon_rate_i, time_step, rng, time_increment=True
            )
            # sum the current rates into 1 second groupings
            photon_rate_grouped = np.sum(
                photon_rate_i.reshape(-1, int(1 / time_steps[i].value)), axis=1
            )
            # calculate the resulting amperage
            current_rate = self.photo_current_photons_per_sec(photon_rate_i, 0.15)
            # sum the current rates into 1 second groupings
            current_rate = np.sum(
                current_rate.reshape(-1, int(1 / time_steps[i].value)), axis=1
            )
            # append photon rates to the output
            output_photon_rates.append(photon_rate_i)
            output_current.append(current_rate)
            # plot shot noise results in a histogram
            self.plot_histogram(
                photon_rate_i,
                "Produced Photon per " + str(time_step),
                "Photons per " + str(time_step),
                "Counts",
            )
            # plot macro photon rate results in a histogram
            self.plot_histogram(
                photon_rate_grouped,
                "Photon Rate as Sum of Photons per " + str(time_step),
                "Photons per Second",
                "Counts",
            )
            # plot macro current rate in a histogram
            self.plot_histogram(
                current_rate,
                "Produced Current [A] at " + str(time_step) + "Step Size",
                "Amperes",
                "Counts",
            )
        return output_photon_rates, output_current

    def multi_poisson_draw_test(
        self, I_dark_rate, num_steps_per_second, number_of_pulls, cutoff_freq, seed=None
    ):

        # Set the random number generator if a seed is not provided
        if not isinstance(seed, int):
            seed_seq = np.random.SeedSequence()
            seed = seed_seq.generate_state(1)
            seed = int(seed)
        rng = np.random.default_rng(seed)

        # Draw from the poisson distribution with the rate subdivided, the rate left
        # at the nominal level, and a gaussian with std sqrt(2*electron_rate)

        # Change the current into number of electrons/time
        electron_rate = I_dark_rate.to(cds.e / u.s)

        # Make the vector length the desired number of pulls from the distribution
        electron_rate_full = np.ones((1, number_of_pulls)) * electron_rate
        # Subdivide the rate to energy per subdivison of 1 second
        electron_rate_sub = (
            np.ones((1, number_of_pulls)) * electron_rate / num_steps_per_second
        )
        # Generate a vector of the std for each pull (these are equivalent at sqrt(2*electron_rate))
        electron_rate_std = np.ones((1, number_of_pulls)) * np.sqrt(
            2 * electron_rate.value
        )
        # Generate a vector of the length of a lower sample rate at 2 times the cutoff frequency
        lower_sample_rate = ((1 / cutoff_freq) * num_steps_per_second * (1 / 2)).value

        # Draw from the poisson distribution
        electron_rate_full_new = (
            rng.poisson(electron_rate_full.value) * electron_rate_full.unit
        )
        # Reset the random number generator and pull for the subdivided rates
        rng = np.random.default_rng(seed)
        electron_rate_sub_new = (
            rng.poisson(electron_rate_sub.value) * electron_rate_sub.unit
        )
        # Reset the random number generator and pull for the subdivided rates using the previously pulled units
        rng = np.random.default_rng(seed)
        electron_rate_sub_new_2x_poisson_pulls = (
            rng.poisson(electron_rate_sub_new.value) * electron_rate_sub.unit
        )
        # Reset the random number generator and pull for the subdivided rates assuming 2x on the variance
        # rng = np.random.default_rng(seed)
        # electron_rate_sub_new_2x_electron_rate = rng.poisson(electron_rate_sub.value*2)*electron_rate_sub.unit
        # Reset the random number generator and pull for the gaussian rates
        rng = np.random.default_rng(seed)
        electron_rate_gaussian = (
            rng.normal(electron_rate_full.value, electron_rate_std)
            * electron_rate_full.unit
        )
        # Reset the random number generator and pull for the gaussian rates at 2 times the cutoff freq
        rng = np.random.default_rng(seed)
        electron_rate_gaussian_low_rate = np.zeros((1, number_of_pulls))
        sample_counter = 0
        i = 0
        while sample_counter < number_of_pulls:
            electron_rate_gaussian_low_rate[
                0, int(lower_sample_rate) * i : int(lower_sample_rate) * (i + 1)
            ] = np.ones((int(lower_sample_rate))) * rng.normal(
                electron_rate.value, np.sqrt(2 * electron_rate.value)
            )
            sample_counter += int(lower_sample_rate)
            i += 1
        electron_rate_gaussian_low_rate = (
            electron_rate_gaussian_low_rate * electron_rate.unit
        )

        # Change the number of electrons/time back into C/time
        I_dark_full = electron_rate_full_new.to(u.A)
        I_dark_sub = electron_rate_sub_new.to(u.A)
        I_dark_sub_2x_poisson_pulls = electron_rate_sub_new_2x_poisson_pulls.to(u.A)
        I_dark_gaussian = electron_rate_gaussian.to(u.A)
        I_dark_gaussian_low_rate = electron_rate_gaussian_low_rate.to(u.A)

        # For the full and gaussian rates
        # Average the number of electrons and current over windows that encompass 1 second

        # For the subdivided rates
        # Add the number of electrons and current over windows that encompass 1 second
        electron_rate_avg_full = []
        I_dark_avg_full = []
        electron_rate_sum = []
        I_dark_sum = []
        electron_rate_sum_2x_poisson_pulls = []
        I_dark_sum_2x_poisson_pulls = []
        electron_rate_plus_gaussian = []
        I_dark_sum_plus_gaussian = []
        electron_rate_avg_gaussian = []
        I_dark_avg_gaussian = []
        electron_rate_avg_gaussian_low_rate = []
        I_dark_avg_gaussian_low_rate = []
        for i in np.arange(number_of_pulls - num_steps_per_second):
            # Full rate
            electron_rate_avg_full.append(
                np.mean(electron_rate_full_new[0, i : i + num_steps_per_second]).value
            )
            I_dark_avg_full.append(
                np.mean(I_dark_full[0, i : i + num_steps_per_second]).value
            )
            # Subdivided rate
            electron_rate_sum.append(
                np.sum(electron_rate_sub_new[0, i : i + num_steps_per_second]).value
            )
            I_dark_sum.append(np.sum(I_dark_sub[0, i : i + num_steps_per_second]).value)
            # Subdivided rate, 2x poisson pull
            electron_rate_sum_2x_poisson_pulls.append(
                np.sum(
                    electron_rate_sub_new_2x_poisson_pulls[
                        0, i : i + num_steps_per_second
                    ]
                ).value
            )
            I_dark_sum_2x_poisson_pulls.append(
                np.sum(
                    I_dark_sub_2x_poisson_pulls[0, i : i + num_steps_per_second]
                ).value
            )
            # Subdivided rate, 2x electrons in one pull
            # electron_rate_plus_gaussian.append(np.sum(electron_rate_sub_new_plus_gaussian[0,i:i+num_steps_per_second]).value)
            # I_dark_sum_plus_gaussian.append(np.sum(I_dark_sub_plus_gaussian[0,i:i+num_steps_per_second]).value)
            # Gaussian rate
            electron_rate_avg_gaussian.append(
                np.mean(electron_rate_gaussian[0, i : i + num_steps_per_second]).value
            )
            I_dark_avg_gaussian.append(
                np.mean(I_dark_gaussian[0, i : i + num_steps_per_second]).value
            )
            # Gaussian rate low rate
            electron_rate_avg_gaussian_low_rate.append(
                np.mean(
                    electron_rate_gaussian_low_rate[0, i : i + num_steps_per_second]
                ).value
            )
            I_dark_avg_gaussian_low_rate.append(
                np.mean(I_dark_gaussian_low_rate[0, i : i + num_steps_per_second]).value
            )

        electron_rate_avg_full = (
            np.array(electron_rate_avg_full) * electron_rate_full_new.unit
        )
        I_dark_avg_full = np.array(I_dark_avg_full) * I_dark_full.unit
        electron_rate_sum = np.array(electron_rate_sum) * electron_rate_sub_new.unit
        I_dark_sum = np.array(I_dark_sum) * I_dark_sub.unit
        electron_rate_sum_2x_poisson_pulls = (
            np.array(electron_rate_sum_2x_poisson_pulls) * electron_rate_sub_new.unit
        )
        I_dark_sum_2x_poisson_pulls = (
            np.array(I_dark_sum_2x_poisson_pulls) * I_dark_sub.unit
        )
        # Add gaussian onto the rolling poisson dark current
        rng = np.random.default_rng(seed)
        electron_rate_rolling_sum = I_dark_sum.to(cds.e / u.s)
        electron_rate_rolling_sum_std = np.ones(
            (1, len(electron_rate_rolling_sum))
        ) * np.sqrt(2 * electron_rate_rolling_sum.value)
        electron_rate_plus_gaussian = (
            rng.normal(electron_rate_rolling_sum.value, electron_rate_rolling_sum_std)
            * electron_rate_rolling_sum.unit
        )
        I_dark_sum_plus_gaussian = electron_rate_plus_gaussian.to(u.A)
        electron_rate_avg_gaussian = (
            np.array(electron_rate_avg_gaussian) * electron_rate_gaussian.unit
        )
        I_dark_avg_gaussian = np.array(I_dark_avg_gaussian) * I_dark_gaussian.unit
        electron_rate_avg_gaussian_low_rate = (
            np.array(electron_rate_avg_gaussian_low_rate) * electron_rate_gaussian.unit
        )
        I_dark_avg_gaussian_low_rate = (
            np.array(I_dark_avg_gaussian_low_rate) * I_dark_gaussian.unit
        )

        x_value = np.arange(number_of_pulls) / num_steps_per_second * u.s

        # Plot the electron rate (nominal, poisson drawn, and subdivided poisson drawn)
        electron_rate_raw_dict = {
            "rolling poisson + gaussian draw": electron_rate_plus_gaussian,
            "nominal gaussian draw": electron_rate_gaussian,
            "nominal poisson draw": electron_rate_full_new,
            "low rate gaussian draw": electron_rate_gaussian_low_rate,
            "rolling poisson draw": electron_rate_sum,
            "2 pulls rolling poisson draw": electron_rate_sum_2x_poisson_pulls,
            "nominal value": np.ones((1, number_of_pulls - num_steps_per_second))
            * electron_rate,
        }
        # self.plot_poisson_comparison(x_value,electron_rate_raw_dict,'Raw Electron Rate Comparison','Electron Rate [e/s]')
        electron_rate_avg_dict = {
            "rolling poisson + gaussian draw": electron_rate_plus_gaussian,
            "nominal gaussian draw": electron_rate_avg_gaussian,
            "nominal poisson draw": electron_rate_avg_full,
            "low rate gaussian draw": electron_rate_avg_gaussian_low_rate,
            "rolling poisson draw": electron_rate_sum,
            "2 pulls rolling poisson draw": electron_rate_sum_2x_poisson_pulls,
            "nominal value": np.ones((1, number_of_pulls - num_steps_per_second))
            * electron_rate,
        }
        # self.plot_poisson_comparison(x_value,electron_rate_avg_dict,'Avg Electron Rate Comparison','Electron Rate [e/s]')

        # Plot the current rate (nominal, poisson drawn, and subdivided poisson drawn)
        I_dark_raw_dict = {
            "rolling poisson + gaussian draw": I_dark_sum_plus_gaussian,
            "nominal gaussian draw": I_dark_gaussian,
            "nominal poisson draw": I_dark_full,
            "rolling poisson + gaussian draw": I_dark_sum_plus_gaussian,
            "low rate gaussian draw": I_dark_gaussian_low_rate,
            "rolling poisson draw": I_dark_sum,
            "2 pulls rolling poisson draw": I_dark_sum_2x_poisson_pulls,
            "nominal value": np.ones((1, number_of_pulls - num_steps_per_second))
            * I_dark_rate,
        }
        # self.plot_poisson_comparison(x_value,I_dark_raw_dict,'Raw Current Comparison','Current [A]')
        I_dark_avg_dict = {
            "rolling poisson + gaussian draw": I_dark_sum_plus_gaussian,
            "nominal gaussian draw": I_dark_avg_gaussian,
            "nominal poisson draw": I_dark_avg_full,
            "low rate gaussian draw": I_dark_avg_gaussian_low_rate,
            "rolling poisson draw": I_dark_sum,
            "2 pulls rolling poisson draw": I_dark_sum_2x_poisson_pulls,
            "nominal value": np.ones((1, number_of_pulls - num_steps_per_second))
            * I_dark_rate,
        }
        # self.plot_poisson_comparison(x_value,I_dark_avg_dict,'Avg Current Comparison','Current [A]')
        # [-len(electron_rate_avg_full):]
        # Plot the logarithmic current rate (nominal, poisson drawn, and subdivided poisson drawn)
        I_dark_nominal_log = np.log(I_dark_rate.value) * I_dark_rate.unit
        I_dark_rate_log = np.ones((1, number_of_pulls)) * I_dark_nominal_log
        I_dark_full_log = np.log(I_dark_full.value) * I_dark_full.unit
        I_dark_sum_log = np.log(I_dark_sum.value) * I_dark_sum.unit
        I_dark_sum_log_2x_poisson_pulls = (
            np.log(I_dark_sum_2x_poisson_pulls.value) * I_dark_sum_2x_poisson_pulls.unit
        )
        I_dark_sum_log_plus_gaussian = (
            np.log(I_dark_sum_plus_gaussian.value) * I_dark_sum_plus_gaussian.unit
        )
        I_dark_gaussian_log = np.log(I_dark_gaussian.value) * I_dark_gaussian.unit
        I_dark_gaussian_low_log = (
            np.log(I_dark_gaussian_low_rate.value) * I_dark_gaussian.unit
        )
        I_dark_log_dict = {
            "rolling poisson + gaussian draw": I_dark_sum_log_plus_gaussian,
            "nominal gaussian draw": I_dark_gaussian_log,
            "nominal poisson draw": I_dark_full_log,
            "low rate gaussian draw": I_dark_gaussian_low_log,
            "rolling poisson draw": I_dark_sum_log,
            "2 pulls rolling poisson draw": I_dark_sum_log_2x_poisson_pulls,
            "nominal value": I_dark_rate_log,
        }
        # self.plot_poisson_comparison(x_value,I_dark_log_dict,'Log Current Poisson Comparison','Log Current [logA]')

        # Run a low pass filter on the different signals
        I_dark_full_low = self.apply_batch_low_pass_filter(
            I_dark_full_log,
            I_dark_nominal_log,
            cutoff_freq,
            1 / num_steps_per_second * u.s,
        )
        I_dark_sum_low = self.apply_batch_low_pass_filter(
            np.reshape(I_dark_sum_log, (1, len(I_dark_sum))),
            I_dark_nominal_log,
            cutoff_freq,
            1 / num_steps_per_second * u.s,
        )
        I_dark_sum_low_2x_poisson_pulls = self.apply_batch_low_pass_filter(
            np.reshape(
                I_dark_sum_log_2x_poisson_pulls,
                (1, len(I_dark_sum_log_2x_poisson_pulls)),
            ),
            I_dark_nominal_log,
            cutoff_freq,
            1 / num_steps_per_second * u.s,
        )
        I_dark_sum_low_plus_gaussian = self.apply_batch_low_pass_filter(
            I_dark_sum_log_plus_gaussian,
            I_dark_nominal_log,
            cutoff_freq,
            1 / num_steps_per_second * u.s,
        )
        I_dark_gaussian_low = self.apply_batch_low_pass_filter(
            I_dark_gaussian_log,
            I_dark_nominal_log,
            cutoff_freq,
            1 / num_steps_per_second * u.s,
        )
        I_dark_gaussian_low_low_rate = self.apply_batch_low_pass_filter(
            I_dark_gaussian_low_log,
            I_dark_nominal_log,
            cutoff_freq,
            1 / num_steps_per_second * u.s,
        )

        # Plot low pass results
        # pdb.set_trace()
        I_dark_low_dict = {
            "nominal gaussian draw": I_dark_gaussian_low,
            "nominal poisson draw": I_dark_full_low,
            "low rate gaussian draw": I_dark_gaussian_low_low_rate,
            "rolling poisson draw": I_dark_sum_low,
            "2 pulls rolling poisson draw": I_dark_sum_low_2x_poisson_pulls,
            "rolling poisson + gaussian draw": I_dark_sum_low_plus_gaussian,
            "nominal value": I_dark_rate_log,
        }
        # self.plot_poisson_comparison(x_value,I_dark_low_dict,'Low Passed Poisson Comparison','Log Current [logA]')

        # Package data for output by type of pull
        # pdb.set_trace()
        df_rolling_sum = pd.DataFrame(
            {
                "Electron Rate": electron_rate_sum,
                "Current": I_dark_sum,
                "Log Current": I_dark_sum_log,
                "Low Passed Current": I_dark_sum_low,
            }
        )
        df_rolling_sum_plus_gaussian = pd.DataFrame(
            {
                "Electron Rate": electron_rate_plus_gaussian[0],
                "Current": I_dark_sum_plus_gaussian[0],
                "Log Current": I_dark_sum_log_plus_gaussian[0],
                "Low Passed Current": I_dark_sum_low_plus_gaussian,
            }
        )
        df_nominal_poisson = pd.DataFrame(
            {
                "Electron Rate": electron_rate_full_new[0],
                "Current": I_dark_full[0],
                "Log Current": I_dark_full_log[0],
                "Low Passed Current": I_dark_full_low,
            }
        )
        df_low_rate_gaussian = pd.DataFrame(
            {
                "Electron Rate": electron_rate_gaussian_low_rate[0],
                "Current": I_dark_gaussian_low_rate[0],
                "Log Current": I_dark_gaussian_low_log[0],
                "Low Passed Current": I_dark_gaussian_low_low_rate,
            }
        )
        df_rolling_sum_2_draws = pd.DataFrame(
            {
                "Electron Rate": electron_rate_sum_2x_poisson_pulls,
                "Current": I_dark_sum_2x_poisson_pulls,
                "Log Current": I_dark_sum_log_2x_poisson_pulls,
                "Low Passed Current": I_dark_sum_low_2x_poisson_pulls,
            }
        )
        df_nominal_gaussian = pd.DataFrame(
            {
                "Electron Rate": electron_rate_gaussian[0],
                "Current": I_dark_gaussian[0],
                "Log Current": I_dark_gaussian_log[0],
                "Low Passed Current": I_dark_gaussian_low,
            }
        )

        return (
            seed,
            df_rolling_sum,
            df_rolling_sum_plus_gaussian,
            df_nominal_gaussian,
            df_nominal_poisson,
            df_low_rate_gaussian,
            df_rolling_sum_2_draws,
        )

    def multi_poisson_draw_step_test(
        self,
        I_dark_rate,
        I_signal_rate,
        num_steps_per_second,
        number_of_pulls,
        cutoff_freq,
        seed=None,
    ):

        # Set the random number generator if a seed is not provided
        if not isinstance(seed, int):
            seed_seq = np.random.SeedSequence()
            seed = seed_seq.generate_state(1)
            seed = int(seed)
        rng = np.random.default_rng(seed)

        # Draw from the poisson distribution with the rate subdivided, the rate left
        # at the nominal level, and a gaussian with std sqrt(2*electron_rate)

        # Change the current into number of electrons/time
        electron_dark_rate = I_dark_rate.to(cds.e / u.s)
        electron_signal_rate = I_signal_rate.to(cds.e / u.s)

        # Apply the signal rate at 1/3 of the number of pulls
        # and return to the nominal at 2/3 of the number of pulls
        div = int(number_of_pulls / 3)
        if number_of_pulls % 3 == 1:
            electron_rate = np.concatenate(
                (
                    np.ones((1, div)) * electron_dark_rate,
                    np.ones((1, div)) * electron_signal_rate,
                    np.ones((1, div + 1)) * electron_dark_rate,
                ),
                axis=1,
            )
        elif number_of_pulls % 3 == 2:
            electron_rate = np.concatenate(
                (
                    np.ones((1, div)) * electron_dark_rate,
                    np.ones((1, div + 1)) * electron_signal_rate,
                    np.ones((1, div + 1)) * electron_dark_rate,
                ),
                axis=1,
            )
        else:
            electron_rate = np.concatenate(
                (
                    np.ones((1, div)) * electron_dark_rate,
                    np.ones((1, div)) * electron_signal_rate,
                    np.ones((1, div)) * electron_dark_rate,
                ),
                axis=1,
            )

        # Change the number of electrons/time back into C/time
        I = electron_rate.to(u.A)

        # Apply noise
        electron_rate_new = (
            rng.poisson(electron_rate.value / num_steps_per_second) * electron_rate.unit
        )
        I_new = electron_rate_new.to(u.A)

        # Sum over 1 second intervals
        electron_rate_sum = []
        I_sum = []
        for i in np.arange(number_of_pulls - num_steps_per_second):
            electron_rate_sum.append(
                np.sum(electron_rate_new[0, i : i + num_steps_per_second]).value
            )
            I_sum.append(np.sum(I_new[0, i : i + num_steps_per_second]).value)

        electron_rate_sum = np.array(electron_rate_sum) * electron_rate.unit
        I_sum = np.array(I_sum) * I.unit
        x_value = (np.arange(number_of_pulls) / num_steps_per_second * u.s)[
            num_steps_per_second:
        ]

        # Convert to logarithmic values
        I_sum_log = np.log(I_sum.value) * I_sum.unit

        # Pass values through lowpass filter
        I_dark_nominal_log = np.log(I.value) * I.unit
        # pdb.set_trace()
        I_sum_low = self.apply_batch_low_pass_filter(
            np.reshape(I_sum_log, (1, len(I_sum_log))),
            I_dark_nominal_log[0, 0],
            cutoff_freq,
            1 / num_steps_per_second * u.s,
        )

        # Plots
        self.plot_noise(
            x_value,
            electron_rate[0, num_steps_per_second:],
            electron_rate_sum,
            "Electron Rate for Step Response",
            "Electron Rate [e/s]",
        )
        self.plot_noise(
            x_value,
            I[0, num_steps_per_second:],
            I_sum,
            "Current for Step Response",
            "Current [A]",
        )
        self.plot_noise(
            x_value,
            I_dark_nominal_log[0, num_steps_per_second:],
            I_sum_log,
            "Logarithmic Current for Step Response",
            "Log Current [log A]",
        )
        self.plot_noise(
            x_value,
            I_dark_nominal_log[0, num_steps_per_second:],
            I_sum_low,
            "Low Passed Step Response",
            "Log Current [log A]",
        )
        return seed

    def find_dark_rate(
        self,
        I_dark_approx,
        event_rate_desired,
        num_pix_x,
        num_pix_y,
        num_steps_per_second,
        number_of_pulls,
        cutoff_freq,
        pos_threshold,
        neg_threshold,
        seed=None,
    ):
        # Set the random number generator if a seed is not provided
        if not isinstance(seed, int):
            seed_seq = np.random.SeedSequence()
            seed = seed_seq.generate_state(1)
            seed = int(seed)

        # Run a least squares minimization on the definition of the dark current
        bnds = opt.Bounds((1e-17,), (7e-17,))
        I_optimized = opt.minimize(
            self.dark_rate_tuning,
            I_dark_approx,
            args=(
                event_rate_desired,
                num_pix_x,
                num_pix_y,
                num_steps_per_second,
                number_of_pulls,
                cutoff_freq,
                pos_threshold,
                neg_threshold,
                seed,
            ),
            bounds=bnds,
            tol=1e-12,
        )
        return I_optimized, seed

    def rolling_poisson_shot_noise_sim(
        self,
        I_dark,
        Leak_rate,
        I_in,
        num_pix_x,
        num_pix_y,
        num_steps_per_second,
        number_of_pulls,
        cutoff_freq,
        pos_threshold,
        neg_threshold,
        extra_noise,
        varying_cuttoff,
        junction_leak,
        parasitic_leak,
        seed,
    ):
        # initialize random number generator
        rng = np.random.default_rng(seed)

        # Establish the initial array of poisson pulls
        electron_dark_rate_sub = I_dark.to(cds.e / u.s) / num_steps_per_second
        dark_rate_pulls = (
            rng.poisson(
                np.ones((num_steps_per_second, num_pix_y, num_pix_x))
                * electron_dark_rate_sub.value
            )
            * electron_dark_rate_sub.unit
        )
        dark_rate_sum = np.sum(dark_rate_pulls, axis=0)
        electron_dark_rate_array = (
            np.ones((1, num_pix_y, num_pix_x)) * electron_dark_rate_sub
        )
        I_dark_log_value = np.log(I_dark.value)

        # Initialize cutoff freqency array
        cutoff_freq_og = cutoff_freq

        # Set memorized current to the nominal current rate in log scale
        I_mem_last = self.initialize_mem_current(
            I_dark + I_in, np.ones((num_pix_y, num_pix_x)), rng
        )
        if not junction_leak or parasitic_leak:
            I_mem_current = I_mem_last
        else:
            # If the junction leak or the parasitic leak is being used, all events will occur
            # at approximately the same time if the current memorized current is set at the same as the last memorized current
            # beacuse it is a linear change, uniformly distribute the pixels between 0 and -.1 in the log scale from the last memorized current
            I_mem_current = rng.uniform(I_mem_last - 0.1, I_mem_last)

        # Set the last current value to the memorized current
        I_ref = I_mem_last

        # Initialize incident energy on the array, make it a rolling average too
        I_in_ref = I_in
        I_in = np.ones((num_pix_y, num_pix_x)) * I_in
        # I_in_ref = I_in
        # electron_I_in_rate_sub = I_in.to(cds.e/u.s)/num_steps_per_second
        # I_in_pulls = rng.poisson(np.ones((num_steps_per_second,num_pix_y,num_pix_x))*electron_I_in_rate_sub.value)*electron_I_in_rate_sub.unit
        # I_in_sum = np.sum(I_in_pulls,axis=0)
        # electron_I_in_rate_array = np.ones((1,num_pix_y,num_pix_x))*electron_I_in_rate_sub

        # Initialize arrays that track events
        N_event_total = np.zeros((num_pix_y, num_pix_x))
        N_event_time = np.zeros((num_pix_y, num_pix_x))
        N_event_polarity = np.zeros((num_pix_y, num_pix_x))

        # Initialize values for memorized current comparison
        time_last = 0
        I_filter_last = 0
        thres = pos_threshold
        pos_threshold = pos_threshold * np.ones((num_pix_y, num_pix_x))
        neg_threshold = neg_threshold * np.ones((num_pix_y, num_pix_x))

        # Initialize arbiter
        last_row = 0
        last_column = 0
        # Create structure to hold events
        # events = pd.DataFrame(list(),columns = ['t','x','y','p','attr'])
        events = []
        events_prev = []  # Events from previous step to send into the next step (simplier than taking a window of events, but that is also possible)
        events_attr_list = []
        events_attr_list_prev = []

        # Calculate the simulation total time
        sim_time = 0
        sim_step = 1 / num_steps_per_second
        sim_total_time = number_of_pulls / num_steps_per_second
        itr = 0
        i = 0

        # Loop through the simulation steps
        while sim_time <= sim_total_time:
            if itr % num_steps_per_second == 0.0:
                print(
                    "     Rolling poisson sim with {} dark current, {} threshold, and {} induced current is {}% done, {} events produced.".format(
                        I_dark,
                        thres,
                        I_in_ref,
                        sim_time / sim_total_time * 100,
                        len(events),
                    )
                )
                print(
                    "          frequency cutoff variable: {}, junction leak: {}, parasitic leak: {}, extra noise: {}".format(
                        varying_cuttoff, junction_leak, parasitic_leak, extra_noise
                    )
                )
            if i == num_steps_per_second:
                i = 0
            # Poisson draw on dark current
            dark_rate_pull = (
                rng.poisson(electron_dark_rate_array.value)
                * electron_dark_rate_array.unit
            )
            # Calculate the dark rate sum
            dark_rate_sum = (
                dark_rate_sum - dark_rate_pulls[i, :, :] + dark_rate_pull[0, :, :]
            )
            # Update the pull array
            dark_rate_pulls[i, :, :] = dark_rate_pull
            # Convert dark electrons/second rate to Amperes
            I_diode = dark_rate_sum.to(u.A)
            # # Poisson draw on induced current
            # I_in_i = self.white_noise_add(I_in, rng, double_noise=True)
            I_in_i = I_in
            # I_in_rate_pull = rng.poisson(electron_I_in_rate_array.value)*electron_I_in_rate_array.unit
            # # Calculate the dark rate sum
            # I_in_sum = I_in_sum - I_in_pulls[i,:,:] + I_in_rate_pull[0,:,:]
            # # Update the pull array
            # I_in_pulls[i,:,:] = I_in_rate_pull
            # # Convert dark electrons/second rate to Amperes
            # I_in_i = I_in_sum.to(u.A)
            # Calculate the cuttoff frequency if variable
            if varying_cuttoff:
                cutoff_freq = self.update_cutoff_freq(
                    I_in_i + I_diode, I_diode, cutoff_freq_og
                )
            else:
                cutoff_freq = cutoff_freq_og
            # Add in the induced photocurrent
            I_diode = I_diode + I_in_i
            # If additional noise in feedback amplifier, add gaussian on top based on the current
            if extra_noise:
                # I found the peak to be at a std of 4.041912066331482e-15*u.A
                # The std taken from the rise of the curve results in a linear fit in the log space to find the standard deviation
                # slope : 1.0412595726748335 intercept: -2.1577612630018854
                # convert back to linear to take gaussians
                I_in_i_log = self.log_Intensity(I_in_i)
                # std_extra_noise = np.exp(1.0412595726748335*I_in_i_log - 2.1577612630018854)
                # std_extra_noise = np.exp(-2.830387044159785 + 0.9919021331472112*I_in_i_log) #Based on data not read off by eye. There is a peak I cannot match at the highest levels. This matches pretty closely. Lets look at that output.
                # std_extra_noise = np.exp(-3.63313 + 0.984117*I_in_i_log)
                # std_extra_noise = np.exp(-3.63313 + 0.984117*I_in_i_log)
                # std_extra_noise = np.exp(-3.8 + .96*I_in_i_log)
                # std_extra_noise = np.exp(-0.0113066*I_in_i_log**2 + 0.273553*I_in_i_log -14.8335) #Threshold 0.1
                std_extra_noise = np.exp(
                    -0.0113066 * I_in_i_log**2 + 0.273553 * I_in_i_log - 14.1891
                )  # Threshold 0.2
                I_diode = rng.normal(loc=I_diode.value, scale=std_extra_noise) * u.A
                # I_diode = self.white_noise_add(I_diode, rng)
            # Convert the current into the log scale
            I_log_current = self.log_Intensity(I_diode)
            # Low pass the converted current
            I_filter_current, I_ref = self.low_pass_filter(
                I_log_current, I_ref, cutoff_freq, sim_step * u.s
            )
            # Only compare to the previous time, if past the initial time step
            if sim_time != 0 * u.s:
                if junction_leak:
                    I_mem_current = self.junction_leak(
                        sim_step,
                        Leak_rate,
                        pos_threshold,
                        I_mem_current,
                        I_dark_log_value,
                    )
                if parasitic_leak:
                    # Calculate the leak rate
                    R_lambda = 0.11977332746963579 * u.A / u.W
                    fperW = 81960240 * u.Hz / u.W
                    para_leak_rate = (I_in_i / R_lambda) * fperW
                    I_mem_current = self.junction_leak(
                        sim_step,
                        para_leak_rate,
                        pos_threshold,
                        I_mem_current,
                        I_dark_log_value,
                    )
                # pdb.set_trace()
                (
                    del_I_filter,
                    del_I_mem,
                    N_event_total,
                    N_event_time,
                    N_event_polarity,
                ) = self.threshold_check(
                    time_last,
                    sim_time,
                    I_mem_last,
                    I_mem_current,
                    I_filter_last,
                    I_filter_current,
                    pos_threshold,
                    neg_threshold,
                    N_event_total,
                    N_event_time,
                    N_event_polarity,
                )
                attribution_dict = {}
                attribution_dict[sim_time] = {}
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
                    time_last,
                    sim_time,
                    2,
                    I_filter_last,
                    I_filter_current,
                    del_I_filter,
                    I_mem_current,
                    del_I_mem,
                    pos_threshold,
                    neg_threshold,
                    last_row,
                    last_column,
                    num_pix_x - 1,
                    0.003,
                    3e6,
                    attribution_dict,
                    rng,
                    maximum_events=1000000,
                )
                # reduce updated lists to the new information
                events_prev = events_prev[previous_list_len:]
                events_attr_list_prev = events_attr_list_prev[previous_list_len:]
                events.extend(events_prev)
                events_attr_list.extend(events_attr_list_prev)
                # events, events_attr_list, N_event_time, N_event_polarity, I_mem_current, last_row, last_column, curr_time = self.arbiter(events, events_attr_list, N_event_time, N_event_polarity, time_last, sim_time, 2, I_filter_last, I_filter_current, del_I_filter, I_mem_current, del_I_mem, pos_threshold, neg_threshold, last_row, last_column, num_pix_x-1, 0.003, 3e6, attribution_dict, rng)
            # If the number of events exceeds a threshold break simulation and calculate metrics with less events to save time
            if len(events) > 1000000:
                sim_time = curr_time
                break
            # Increment time
            time_last = sim_time
            sim_time += sim_step
            i += 1
            itr += 1
            I_filter_last = I_filter_current
            I_mem_last = I_mem_current

        # make the event output a dataframe
        if not events:
            events = pd.DataFrame(list(), columns=["t", "x", "y", "p", "attr"])
        else:
            events = pd.DataFrame(np.array(events), columns=["t", "x", "y", "p"])
            events["attr"] = events_attr_list
        return events, sim_time

    def discrete_shot_noise_sim(
        self,
        I_dark,
        Leak_rate,
        I_in,
        num_pix_x,
        num_pix_y,
        num_steps_per_second,
        number_of_pulls,
        cutoff_freq,
        pos_threshold,
        neg_threshold,
        noise_type,
        extra_noise,
        varying_cuttoff,
        junction_leak,
        parasitic_leak,
        seed,
    ):
        # initialize random number generator
        rng = np.random.default_rng(seed)

        # Establish the initial array of electron rate from dark current
        electron_dark_rate = I_dark.to(cds.e / u.s)
        electron_dark_rate_array = np.ones((num_pix_y, num_pix_x)) * electron_dark_rate
        I_dark_log_value = np.log(I_dark.value)

        # Initialize cutoff freqency array
        cutoff_freq_og = cutoff_freq

        # Set memorized current to the nominal current rate in log scale
        I_mem_last = self.initialize_mem_current(
            I_dark + I_in, (num_pix_y, num_pix_x), rng
        )
        I_mem_current = I_mem_last
        # Set the last current value to the memorized current
        I_ref = I_mem_last

        # Initialize incident energy on the array
        I_in_ref = I_in
        I_in = np.ones((num_pix_y, num_pix_x)) * I_in

        # Initialize arrays that track events
        N_event_total = np.zeros((num_pix_y, num_pix_x))
        N_event_time = np.zeros((num_pix_y, num_pix_x))
        N_event_polarity = np.zeros((num_pix_y, num_pix_x))

        # Initialize values for memorized current comparison
        time_last = 0
        I_filter_last = 0
        thres = pos_threshold
        pos_threshold = pos_threshold * np.ones((num_pix_y, num_pix_x))
        neg_threshold = neg_threshold * np.ones((num_pix_y, num_pix_x))

        # Initialize arbiter
        last_row = 0
        last_column = 0
        # Create structure to hold events
        events = []
        events_prev = (
            []
        )  # Events from previous step to send into the next step (simplier than taking a window of events, but that is also possible)
        events_attr_list = []
        events_attr_list_prev = []

        # Calculate the simulation total time
        sim_time = 0
        sim_step = 1 / num_steps_per_second
        sim_total_time = number_of_pulls / num_steps_per_second
        itr = 0
        i = 0

        # Determine how often the gaussian should be pulled (2 times the cutoff frequency)
        refresh_noise_rate = (1 / cutoff_freq.value) * num_steps_per_second * (1 / 2)

        # Loop through the simulation steps
        while sim_time <= sim_total_time:
            if itr % num_steps_per_second == 0.0:
                if not extra_noise:
                    print(
                        "     Discrete {} sim with {} dark current, {} threshold, and {} induced current is {}% done.".format(
                            noise_type,
                            I_dark,
                            thres,
                            I_in_ref,
                            sim_time / sim_total_time * 100,
                        )
                    )
                else:
                    print(
                        "     Discrete {} with additional gaussian sim with {} dark current, {} threshold, and {} induced current is {}% done.".format(
                            noise_type,
                            I_dark,
                            thres,
                            I_in_ref,
                            sim_time / sim_total_time * 100,
                        )
                    )
            if i == num_steps_per_second:
                i = 0
            # Type of draw is taken based on requested call
            # Only update the Guassian at half the cutoff_frequency
            # # Poisson draw on induced current
            # I_in_i = self.white_noise_add(I_in, rng, double_noise=True)
            I_in_i = I_in
            if itr % refresh_noise_rate == 0.0:
                if noise_type == "poisson":
                    dark_rate_discrete = (
                        rng.poisson(electron_dark_rate_array.value)
                        * electron_dark_rate_array.unit
                    )
                    # Convert dark electrons/second rate to Amperes
                    I_diode = dark_rate_discrete.to(u.A)
                    # Update the cuttoff frequency if variable
                    if varying_cuttoff:
                        cutoff_freq = self.update_cutoff_freq(
                            I_in + I_diode, I_diode, cutoff_freq_og
                        )
                    else:
                        cutoff_freq = cutoff_freq_og
                    I_diode = I_diode + I_in
                    if extra_noise:
                        I_diode = self.white_noise_add(I_diode, rng)
                elif noise_type == "gaussian":
                    I_diode = self.white_noise_add(electron_dark_rate_array, rng)
                    # Update the cuttoff frequency if variable
                    if varying_cuttoff:
                        cutoff_freq = self.update_cutoff_freq(
                            I_in + I_diode, I_diode, cutoff_freq_og
                        )
                    else:
                        cutoff_freq = cutoff_freq_og
                    I_diode = I_diode + I_in
                    if extra_noise:
                        I_diode = self.white_noise_add(I_diode, rng)
                else:
                    print("Noise type is not an accepted value")
                    break
                # Convert the current into the log scale
                I_log_current = self.log_Intensity(I_diode)
            # Low pass the converted current
            I_filter_current, I_ref = self.low_pass_filter(
                I_log_current, I_ref, cutoff_freq, sim_step * u.s
            )
            # Only compare to the previous time, if past the initial time step
            if sim_time != 0 * u.s:
                if junction_leak:
                    I_mem_current = self.junction_leak(
                        sim_step,
                        Leak_rate,
                        pos_threshold,
                        I_mem_current,
                        I_dark_log_value,
                    )
                if parasitic_leak:
                    # Calculate the leak rate
                    R_lambda = 0.11977332746963579 * u.A / u.W
                    fperW = 81960240 * u.Hz / u.W
                    para_leak_rate = (I_in_i / R_lambda) * fperW
                    I_mem_current = self.junction_leak(
                        sim_step,
                        para_leak_rate,
                        pos_threshold,
                        I_mem_current,
                        I_dark_log_value,
                    )
                # pdb.set_trace()
                (
                    del_I_filter,
                    del_I_mem,
                    N_event_total,
                    N_event_time,
                    N_event_polarity,
                ) = self.threshold_check(
                    time_last,
                    sim_time,
                    I_mem_last,
                    I_mem_current,
                    I_filter_last,
                    I_filter_current,
                    pos_threshold,
                    neg_threshold,
                    N_event_total,
                    N_event_time,
                    N_event_polarity,
                )
                attribution_dict = {}
                attribution_dict[sim_time] = {}
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
                    time_last,
                    sim_time,
                    2,
                    I_filter_last,
                    I_filter_current,
                    del_I_filter,
                    I_mem_current,
                    del_I_mem,
                    pos_threshold,
                    neg_threshold,
                    last_row,
                    last_column,
                    num_pix_x - 1,
                    0.003,
                    3e6,
                    attribution_dict,
                    rng,
                )
                # reduce updated lists to the new information
                events_prev = events_prev[previous_list_len:]
                events_attr_list_prev = events_attr_list_prev[previous_list_len:]
                events.extend(events_prev)
                events_attr_list.extend(events_attr_list_prev)
                # events, events_attr_list, N_event_time, N_event_polarity, I_mem_current, last_row, last_column, curr_time = self.arbiter(events, events_attr_list, N_event_time, N_event_polarity, time_last, sim_time, 2, I_filter_last, I_filter_current, del_I_filter, I_mem_current, del_I_mem, pos_threshold, neg_threshold, last_row, last_column, num_pix_x-1, 0.003, 3e6, attribution_dict, rng)
            # If the number of events exceeds a threshold break simulation and calculate metrics with less events to save time
            # if len(events) > 50000:
            #     sim_time = curr_time
            #     break
            # Increment time
            time_last = sim_time
            sim_time += sim_step
            i += 1
            itr += 1
            I_filter_last = I_filter_current
            I_mem_last = I_mem_current

        # make the event output a dataframe
        if not events:
            events = pd.DataFrame(list(), columns=["t", "x", "y", "p", "attr"])
        else:
            events = pd.DataFrame(np.array(events), columns=["t", "x", "y", "p"])
            events["attr"] = events_attr_list
        return events, sim_time

    def dark_rate_tuning(
        self,
        I_dark,
        event_rate_desired,
        num_pix_x,
        num_pix_y,
        num_steps_per_second,
        number_of_pulls,
        cutoff_freq,
        pos_threshold,
        neg_threshold,
        seed,
    ):

        if type(I_dark) != u.quantity.Quantity:
            I_dark = I_dark * u.A

        # Run the shot noise simulation
        print("Dark current for this iteration is {}.".format(I_dark))
        event_array = self.run_shot_noise_sim(
            I_dark,
            num_pix_x,
            num_pix_y,
            num_steps_per_second,
            number_of_pulls,
            cutoff_freq,
            pos_threshold,
            neg_threshold,
            seed,
        )

        # Calculate the events per pixel per second rate
        sim_total_time = number_of_pulls * 1 / num_steps_per_second
        sim_total_pix = num_pix_x * num_pix_y
        event_rate_calc = self.calculate_event_rate(
            event_array, sim_total_time, sim_total_pix
        )
        event_rate_difference = self.event_rate_diff(
            event_rate_desired, event_rate_calc
        )
        print(
            "A dark current of {} produced an event rate of {} making a difference of {} from the desired event rate".format(
                I_dark, event_rate_calc, event_rate_difference
            )
        )
        # Poorly scaled instead bring the desired event rate to a desired total event number
        # desired_event_number = self.calculate_desired_total_events(event_rate_desired, sim_total_time, sim_total_pix)
        # event_rate_difference = self.event_rate_diff(desired_event_number, event_array.shape[0])
        # print('A dark current of {} produced {} events making a difference of {} in the total number of events'.format(I_dark,event_array.shape[0],event_rate_difference))

        return event_rate_difference

    def dark_rate_analysis(
        self,
        I_dark,
        Leak_rate,
        I_in,
        lock,
        num_pix_x,
        num_pix_y,
        num_steps_per_second,
        number_of_pulls,
        cutoff_freq,
        pos_threshold,
        neg_threshold,
        seed,
        file_name,
        full_event_out=False,
    ):
        # pdb.set_trace()
        # Check if a second gaussian should be applied for the feedback in the photodiode
        if "plus" in file_name:
            additional_gaussian = True
        else:
            additional_gaussian = False
        # Check if variable cuttoff frequency should be applied
        if "var_cuttoff" in file_name:
            changing_cuttoff = True
        else:
            changing_cuttoff = False
        # Check if junction leak should be applied
        if "junction_leak" in file_name:
            junction_leak = True
        else:
            junction_leak = False
        # Check if parasitic leask should be applied
        if "parasitic_leak" in file_name:
            parasitic_leak = True
        else:
            parasitic_leak = False
        # Run the shot noise simulation
        tic = time.time()
        if "rolling_poisson" in file_name:
            event_array, sim_total_time = self.rolling_poisson_shot_noise_sim(
                I_dark,
                Leak_rate,
                I_in,
                num_pix_x,
                num_pix_y,
                num_steps_per_second,
                number_of_pulls,
                cutoff_freq,
                pos_threshold,
                neg_threshold,
                additional_gaussian,
                changing_cuttoff,
                junction_leak,
                parasitic_leak,
                seed,
            )
        elif "discrete_poisson" in file_name:
            event_array, sim_total_time = self.discrete_shot_noise_sim(
                I_dark,
                Leak_rate,
                I_in,
                num_pix_x,
                num_pix_y,
                num_steps_per_second,
                number_of_pulls,
                cutoff_freq,
                pos_threshold,
                neg_threshold,
                "poisson",
                additional_gaussian,
                changing_cuttoff,
                junction_leak,
                parasitic_leak,
                seed,
            )
        elif "discrete_gaussian" in file_name:
            event_array, sim_total_time = self.discrete_shot_noise_sim(
                I_dark,
                Leak_rate,
                I_in,
                num_pix_x,
                num_pix_y,
                num_steps_per_second,
                number_of_pulls,
                cutoff_freq,
                pos_threshold,
                neg_threshold,
                "gaussian",
                additional_gaussian,
                changing_cuttoff,
                junction_leak,
                parasitic_leak,
                seed,
            )
        else:
            print("Type of noise not a given choice")
        toc = time.time()

        # Calculate the mean and standard deviation of the event rate on all the pixels
        # in the dataset
        # sim_total_time = number_of_pulls*1/num_steps_per_second
        mean, std = self.calculate_event_rate_per_pixel(
            event_array, num_pix_x, num_pix_y, sim_total_time
        )
        # Calculate the event rate by events per pixel per second
        event_rate = self.calculate_event_rate(
            event_array, sim_total_time, num_pix_x * num_pix_y
        )

        # Save the outputs
        # Save the raw event array if requested
        lock.acquire()
        if full_event_out:
            event_array.to_csv(
                file_name.split(".csv")[0]
                + "_events_"
                + str(np.around(I_in.value, decimals=20)).replace("-", "eneg")
                + ".csv",
                index=False,
            )
        # Acquire lock to ensure this is the only process writing to a file
        try:
            # Add the dark current, mean, and standard deviation to the pandas datafile
            # Open the pandas file
            df = pd.read_csv(file_name)
            # Add the new row of data
            new_row = pd.DataFrame(
                {
                    "Dark Current": np.around(I_dark, decimals=20),
                    "Threshold": pos_threshold,
                    "Induced Current": np.around(I_in, decimals=20),
                    "Mean Event Rate": mean,
                    "Standard Deviation": std,
                    "Events Per Pixel Per Second": event_rate,
                    "Simulation Duration": (toc - tic),
                    "Total Simulation Events": len(event_array),
                    "Final Simulation Time": sim_total_time,
                },
                index=[0],
            )
            df = pd.concat([df, new_row])
            # Save the pandas file
            df.to_csv(file_name, index=False)
            print(" ")
            print(
                "Saving dark current: {}, threshold: {}, induced current: {}, sucessful.".format(
                    I_dark, pos_threshold, I_in
                )
            )
            print(" ")
        except Exception as e:
            print(
                "Saving dark current: {}, threshold: {}, induced current: {}, failed because {}.".format(
                    I_dark, pos_threshold, I_in, e
                )
            )
        finally:
            # Release the lock to ensure another process can save
            lock.release()

        return

    def calculate_event_rate(self, event_array, sim_total_time, sim_total_pix):
        # This matches what is given in Tyler Brewer's paper
        return len(event_array) / sim_total_pix / sim_total_time

    def calculate_desired_total_events(
        self, event_rate_desired, sim_total_time, sim_total_pix
    ):
        # Opposite of calculating the event rate in Brewer's paper
        return event_rate_desired * sim_total_pix * sim_total_time

    def calculate_event_rate_per_pixel(
        self, event_array, num_pix_x, num_pix_y, sim_total_time
    ):
        # For each unique pixel calculate the number of events/sec
        # Find the unique pixels and their counts
        if len(event_array) != 0:
            # pixels, counts = np.unique(event_array[:,1:3],axis=0,return_counts=True)
            # event_rates = counts/sim_total_time

            pixels = event_array.groupby(["x", "y"]).size().reset_index(name="counts")
            event_rates = (pixels["counts"] / sim_total_time).to_numpy()

            # Make sure to account for pixels without events in the average
            pix_no_events = num_pix_x * num_pix_y - np.shape(event_rates)[0]
            event_rates = np.pad(event_rates, (0, pix_no_events))

            # Output the mean and standard deviation of the events/sec
            mean = np.mean(event_rates)
            std = np.std(event_rates)
        else:
            mean = 0
            std = 0
        return mean, std

    def event_rate_diff(self, event_rate_desired, event_rate_calc):
        return np.abs(event_rate_desired - event_rate_calc)

    def apply_batch_low_pass_filter(self, signal, signal_ref, cutoff_freq, time_step):
        """
        Apply low pass filter to batch produced signals.

        Parameters
        ----------
        signal : np.array
            Array that captures the signal.
        signal_ref : float
            Previous value of the current to start the low pass filter.
        cutoff_freq : float
            Cutoff frequency of the filter.
        time_step : float
            Duration of the simulation timestep.

        Returns
        -------
        signal_low_pass : np.array
            Low passed array.

        """
        signal_low_pass = []
        for i in np.arange(len(np.transpose(signal))):
            signal_i, signal_ref = self.low_pass_filter(
                signal[0, i], signal_ref, cutoff_freq, time_step
            )
            signal_low_pass.append(signal_i.value)
        signal_low_pass = np.array(signal_low_pass) * signal.unit
        return signal_low_pass

    def calculate_event_pairs(self, events):
        """
        Record neighboring event pairs' timestamps

        Parameters
        ----------
        events : list
            List with time, event location (x,y), polarity, and attribution.

        Returns
        -------
        events : Pandas DataFrame
            DataFrame with time, event location (x,y), polarity, and attribution.
        onon : list
            Time differences between two on events.
        offoff : list
            Time differences between two off events.
        onoff : list
            Time differences between on then off events.
        offon : list
            Time differences between off then on events.

        """
        # Initialize lists to record pairs of events
        onon = []
        offoff = []
        onoff = []
        offon = []

        x_values = pd.unique(events["x"])
        for i, x in enumerate(x_values):
            y_values = pd.unique(events[events["x"] == x]["y"])
            for j, y in enumerate(y_values):
                # Select all events on this pixel and sort by time
                events_on_pixel = events[
                    (events["x"] == x) * (events["y"] == y)
                ].sort_values("t")
                # Find the time difference between sequential events
                t_diff = events_on_pixel["t"].diff()
                # Find the difference in polarity between sequential events
                p_diff = events_on_pixel["p"].diff()
                # Record the different event pairs
                if t_diff[(p_diff == 0) * (events_on_pixel["p"] == 1)].to_list() != []:
                    onon = (
                        onon
                        + t_diff[(p_diff == 0) * (events_on_pixel["p"] == 1)].to_list()
                    )
                if t_diff[(p_diff == 0) * (events_on_pixel["p"] == 0)].to_list() != []:
                    offoff = (
                        offoff
                        + t_diff[(p_diff == 0) * (events_on_pixel["p"] == 0)].to_list()
                    )
                if t_diff[p_diff == -2].to_list() != []:
                    onoff = onoff + t_diff[p_diff == -2].to_list()
                if t_diff[p_diff == 2].to_list():
                    offon = offon + t_diff[p_diff == 2].to_list()

        return onon, offoff, onoff, offon

    def plot_histogram(self, data, title, x_label, y_label):
        # First bin the data
        counts, bins = np.histogram(data, bins=20)
        # Plot the data using the plt.bar
        fig, ax = plt.subplots(dpi=200)
        ax.stairs(counts, bins, fill=True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        # Save the resulting plot
        plt.savefig(self.file_directory + "test_plots/Shot_Noise/" + title + ".pdf")
        return

    def plot_low_pass_filter(
        self,
        data_low_pass,
        data_log,
        sim_length,
        signal_type,
        num_steps_max,
        impulse_time,
    ):
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.serif": ["Arial"],
            }
        )

        # Set default font size
        plt.rcParams["font.size"] = 12

        # Set Commonalities for Every Plot
        x_label = "Simulation Time [sec]"
        y_label = "Log Current [log A]"

        # Plot all the cutoff frequencies for one simulation step size
        title = signal_type + " Low Pass Influence at Step Size "
        data_label = "cutoff freq = "
        for i, num_steps in enumerate(data_low_pass):
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=200)
            x_vec = np.linspace(0, sim_length.value, num_steps) * sim_length.unit
            ax.step(
                x_vec.value,
                data_log[num_steps],
                where="post",
                label="Log Scaled Signal",
            )
            for j, freq in enumerate(data_low_pass[num_steps]):
                ax.plot(
                    x_vec, data_low_pass[num_steps][freq], label=data_label + str(freq)
                )
            ax.legend()
            plt.title(title + str(sim_length / num_steps))
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            # Save the resulting plot
            plt.savefig(
                self.file_directory
                + "/test_plots/Low_Pass_Filter/"
                + title
                + str(sim_length / num_steps)
                + ".pdf"
            )

        # Plot all the simulation step sizes for one low pass filter frequency
        title = signal_type + " Step Size Influence at cutoff Frequency "
        data_label = "step size = "
        cutoff_freqs = data_low_pass[num_steps].keys()
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=200)
        for i, freq in enumerate(cutoff_freqs):
            fig, ax = plt.subplots(dpi=200)
            x_vec = np.linspace(0, sim_length.value, num_steps_max) * sim_length.unit
            if signal_type == "Impulse":
                plt.axvline(
                    impulse_time,
                    c="black",
                    linestyle="--",
                    label="Impulse of "
                    + str(np.around(np.max(data_log[num_steps_max]), 2))
                    + "[log A]",
                )
            else:
                ax.step(
                    x_vec.value,
                    data_log[num_steps_max],
                    where="post",
                    label="Log Scaled Signal Max Steps",
                )
            for j, num_steps in enumerate(data_low_pass):
                if freq not in data_low_pass[num_steps]:
                    continue
                x_vec = np.linspace(0, sim_length.value, num_steps) * sim_length.unit
                ax.plot(
                    x_vec,
                    data_low_pass[num_steps][freq],
                    label=data_label + str(sim_length / num_steps),
                )
            ax.legend()
            plt.title(title + str(freq))
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            # Save the resulting plot
            plt.tight_layout()
            plt.savefig(
                self.file_directory
                + "/test_plots/Low_Pass_Filter/"
                + title
                + str(freq)
                + ".pdf"
            )
        return

    def plot_poisson_comparison(self, x_value, data, title, ylabel):
        fig, ax = plt.subplots(dpi=200)
        legend_list = []
        for i, key_id in enumerate(data):
            if np.shape(x_value) != np.shape(data[key_id]):
                if len(data[key_id]) == 1:
                    ax.plot(
                        x_value[-len(data[key_id][0]) :],
                        np.reshape(
                            data[key_id], np.shape(x_value[-len(data[key_id][0]) :])
                        ),
                    )
                else:
                    ax.plot(
                        x_value[-len(data[key_id]) :],
                        np.reshape(
                            data[key_id], np.shape(x_value[-len(data[key_id]) :])
                        ),
                    )
            else:
                ax.plot(x_value, data[key_id])
            legend_list.append(key_id)
        ax.legend(legend_list)
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plot_name = (
            self.file_directory
            + "test_plots\Distribution Comparison\_"
            + "True_signal_levels_"
            + title
            + ".pdf"
        )
        plot_name.encode("unicode_escape")
        plot_name = plot_name.replace("\\", "/")
        plt.savefig(plot_name)
        return

    def plot_noise(self, x_value, nominal_sig, noisy_sig, title, ylabel):
        fig, ax = plt.subplots(dpi=200)
        ax.plot(x_value, np.transpose(noisy_sig), x_value, np.transpose(nominal_sig))
        ax.legend(["noisy signal", "nominal signal"])
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.savefig(
            self.file_directory + "/test_plots/Poisson_Comparison/" + title + ".pdf"
        )
        return
