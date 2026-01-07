from astropy import units as u
import circuitry as ctry
import numpy as np 
from astropy.units import cds
import pandas as pd
import h5py

class OccultationSimulation(ctry.ebCircuitSim):
    def occultation_sim(
        self,
        I_in,
        I_dark,
        Leak_rate,
        num_pix_x,
        num_pix_y,
        num_steps_per_second,
        cutoff_freq,
        pos_threshold,
        neg_threshold,
        seed,
        extra_noise = True,
        varying_cuttoff = True,
        junction_leak = True,
        parasitic_leak = False,
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
            I_dark + I_in[0], np.ones((num_pix_y, num_pix_x)), rng
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
        I_in_ref = I_in[0]

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
        itr = 0
        i = 0

        # Loop through the simulation steps
        for j, I_in_i in enumerate(I_in):
            # if itr % num_steps_per_second == 0.0:
            #     print(
            #         "     Rolling poisson sim with {} dark current, {} threshold, and {} induced current is {}% done, {} events produced.".format(
            #             I_dark,
            #             thres,
            #             I_in_ref,
            #             sim_time / sim_total_time * 100,
            #             len(events),
            #         )
            #     )
            #     print(
            #         "          frequency cutoff variable: {}, junction leak: {}, parasitic leak: {}, extra noise: {}".format(
            #             varying_cuttoff, junction_leak, parasitic_leak, extra_noise
            #         )
            #     )
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
                # convert back to linear to take gaussians
                I_in_i_log = self.log_Intensity(I_in_i)
                std_extra_noise = np.exp(
                    -0.0113066 * I_in_i_log**2 + 0.273553 * I_in_i_log - 14.1891
                )  # Threshold 0.2
                I_diode = rng.normal(loc=I_diode.value, scale=std_extra_noise) * u.A
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