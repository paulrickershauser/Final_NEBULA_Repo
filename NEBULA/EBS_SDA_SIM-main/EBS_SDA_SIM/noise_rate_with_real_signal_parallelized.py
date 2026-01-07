# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:55:38 2024

@author: User
"""

import run_test_circuitry as rtc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import os
import multiprocessing as mp
import time
import pdb
import pickle


def init_pool(
    dark,
    leak,
    pix_x,
    pix_y,
    steps_per_second,
    pulls,
    freq,
    pos_thres,
    neg_thres,
    rng_seed,
    events_out,
    data_name,
):
    global I_dark
    global Leak_rate
    global num_pix_x
    global num_pix_y
    global num_steps_per_second
    global number_of_pulls
    global cuttoff_freq
    global pos_threshold
    global neg_threshold
    global seed
    global raw_events_out
    global file_names

    I_dark = dark
    Leak_rate = leak
    num_pix_x = pix_x
    num_pix_y = pix_y
    num_steps_per_second = steps_per_second
    number_of_pulls = pulls
    cuttoff_freq = freq
    pos_threshold = pos_thres
    neg_threshold = neg_thres
    seed = rng_seed
    raw_events_out = events_out
    file_names = data_name


def runDarkCurrentAnalysis(lock, I_in):
    # Load the proper class to run the noise generation code
    circuit_test = rtc.circuitTest()
    # Run Each Simulation
    for i, file_i in enumerate(file_names):
        tic = time.time()
        circuit_test.dark_rate_analysis(
            I_dark,
            Leak_rate,
            I_in,
            lock,
            num_pix_x,
            num_pix_y,
            num_steps_per_second,
            number_of_pulls,
            cuttoff_freq,
            pos_threshold,
            neg_threshold,
            seed,
            file_i,
            raw_events_out,
        )
        toc = time.time()
        if "rolling" in file_i:
            test_style = "rolling poisson style of noise"
        elif "discrete_poisson" in file_i:
            test_style = "discrete poisson style of noise"
        else:
            test_style = "discrete gaussian style of noise"
        if "plus_gaussian" in file_i:
            test_style += (
                " and additional gaussian for feedback amplifier of photodiode"
            )
        if "junction_leak" in file_i:
            test_style += " and junction leak"
        if "parasitic_leak" in file_i:
            test_style += " and parasitic leak"
        print(
            "Induced Current {} with a {} is done processing. Process took {} seconds.".format(
                I_in, test_style, (toc - tic)
            )
        )


def plot_data(file_names, file_location, test_name):
    # plot data from each of the files
    fig, ax = plt.subplots(dpi=200)
    for i, file_i in enumerate(file_names):
        dark_stats = pd.read_csv(file_i)
        dark_stats = dark_stats.sort_values("Induced Current")
        file_label = file_i.replace(file_location + "/" + str(test_name) + "_", "")
        file_label = file_label.replace(".csv", "")
        file_label = file_label.replace("_", " ")
        ax.plot(
            dark_stats["Induced Current"],
            dark_stats["Mean Event Rate"],
            "-",
            label=file_label.title(),
        )
    ax.set_title(
        "Shot Noise Mean Event Rate with Varying Induced Current with Different Sampling Methods"
    )
    ax.set_xlabel("Induced Current [A]")
    ax.set_ylabel("Mean Event Rate [events/sec]")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(
        file_location
        + "/test_plots/"
        + test_name
        + "_sampling_comparison_mean_event_rate.pdf"
    )
    plt.xticks(rotation=45, ha="right")

    fig, ax = plt.subplots(dpi=200)
    for i, file_i in enumerate(file_names):
        dark_stats = pd.read_csv(file_i)
        dark_stats = dark_stats.sort_values("Induced Current")
        file_label = file_i.replace(file_location + "/" + str(test_name) + "_", "")
        file_label = file_label.replace(".csv", "")
        file_label = file_label.replace("_", " ")
        ax.plot(
            dark_stats["Induced Current"],
            dark_stats["Events Per Pixel Per Second"],
            "-",
            label=file_label.title(),
        )
    ax.set_title(
        "Events per Pixel per Second with Varying Induced Current with Different Sampling Methods"
    )
    ax.set_xlabel("Induced Current [A]")
    ax.set_ylabel("Event Rate [events/per pixel/sec]")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(
        file_location
        + "/test_plots/"
        + test_name
        + "_sampling_comparison_events_per_pixel_per_sec.pdf"
    )
    plt.xticks(rotation=45, ha="right")

    for i, file_i in enumerate(file_names):
        fig, ax = plt.subplots(dpi=200)
        dark_stats = pd.read_csv(file_i)
        dark_stats = dark_stats.sort_values("Induced Current")
        file_label = file_i.replace(file_location + "/" + str(test_name) + "_", "")
        file_label = file_label.replace(".csv", "")
        file_label = file_label.replace("_", " ")
        ax.plot(dark_stats["Induced Current"], dark_stats["Mean Event Rate"], "-")
        ax.fill_between(
            dark_stats["Dark Current"],
            dark_stats["Mean Event Rate"] - dark_stats["Standard Deviation"],
            dark_stats["Mean Event Rate"] + dark_stats["Standard Deviation"],
            alpha=0.2,
        )
        ax.set_title("Shot Noise Mean Event Rate with Varying Induced Current")
        ax.set_xlabel("Induced Current [A]")
        ax.set_ylabel("Mean Event Rate [events/sec]")
        ax.set_yscale("log")
        plt.savefig(
            file_location
            + "/test_plots/"
            + test_name
            + "_"
            + file_label
            + "_mean_event_rate_with_std.pdf"
        )
        plt.xticks(rotation=45, ha="right")


if __name__ == "__main__":
    # Choose if output should include event array
    raw_events_out = True

    # Ensure all local variables exist to run the simulation
    with open(
        "collection_parameters_2021_04_11_real_signal.pickle", "rb"
    ) as picklefile:
        collect_params = pickle.load(picklefile)
    num_pix_x = collect_params["x pix"]
    num_pix_y = collect_params["y pix"]
    num_steps_per_second = collect_params["Sim Steps Per Second"]
    # num_steps_per_second = num_steps_per_second * 4 # Increase number of steps per second
    number_of_pulls = collect_params["Number of Sim Steps"]
    # number_of_pulls = number_of_pulls * 4 # If increased number of steps per second also increase the number of overall steps to get the same length simulation
    cuttoff_freq = collect_params["Cutoff Frequency"]
    cuttoff_freq = 100 * u.Hz
    pos_threshold = collect_params["Positive Event Threshold"]
    neg_threshold = collect_params["Negative Event Threshold"]
    try:
        seed = collect_params["Seed"]
        print("Seed: {}".format(seed))
    except NameError:
        # Establish seed
        seed_seq = np.random.SeedSequence()
        seed = seed_seq.generate_state(1)
        seed = int(seed)
    # Establish File Names for Saving Data
    directory = os.getcwd()
    file_location = 0
    while os.path.exists(directory + "/" + str(file_location)) == False:
        file_location = input(
            "Please input the simulation directory name within the current working directory: "
        )
    file_location = directory + "/" + str(file_location)
    file_name = 0
    while isinstance(file_name, str) != True:
        file_name = input("Please input the file name without file extension (.csv): ")
    file_names = [  # file_location + '/' + str(file_name) + '_rolling_poisson_var_cuttoff.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_var_cuttoff_junction_leak.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_var_cuttoff_parasitic_leak.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_var_cuttoff_junction_leak_parasitic_leak.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_var_cuttoff_junction_leak_parasitic_leak_plus_gaussian.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_junction_leak.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_parasitic_leak.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_junction_leak_parasitic_leak.csv',
        # file_location + '/' + str(file_name) + '_rolling_poisson_plus_gaussian.csv',
        file_location
        + "/"
        + str(file_name)
        + "_rolling_poisson_plus_gaussian_var_cuttoff.csv",
        # file_location + '/' + str(file_name) + '_rolling_poisson_plus_gaussian_var_cuttoff_parasitic_leak.csv',
        # file_location + '/' + str(file_name) + '_discrete_poisson.csv',
        # file_location + '/' + str(file_name) + '_discrete_poisson_var_cuttoff.csv',
        # file_location + '/' + str(file_name) + '_discrete_poisson_plus_guassian.csv',
        # file_location + '/' + str(file_name) + '_discrete_poisson_plus_guassian_var_cuttoff.csv',
        # file_location + '/' + str(file_name) + '_discrete_gaussian.csv',
        # file_location + '/' + str(file_name) + '_discrete_gaussian_var_cuttoff.csv',
        # file_location + '/' + str(file_name) + '_discrete_gaussian_plus_guassian.csv',
        # file_location + '/' + str(file_name) + '_discrete_gaussian_plus_guassian_var_cuttoff.csv']
    ]
    # Ensure there is a csv file at that location
    df = pd.DataFrame(
        list(),
        columns=[
            "Dark Current",
            "Threshold",
            "Induced Current",
            "Mean Event Rate",
            "Standard Deviation",
            "Events Per Pixel Per Second",
            "Simulation Duration",
            "Total Simulation Events",
            "Final Simulation Time",
        ],
    )
    for i, file_i in enumerate(file_names):
        try:
            df = pd.read_csv(file_i)
        except:
            df.to_csv(file_i, index=False)

    # Create a lock object to enable saving
    m = mp.Manager()
    lock = m.Lock()

    # Generate a tuple with each input variable for all the runs
    argsList = []
    # for specific list
    I_dark = 1e-15 * u.A
    leak_rate = 0.1 * u.Hz
    # pdb.set_trace()
    # signal_background = 1.8873918221350995e-12
    # argsList.append((lock, signal_background * u.A))
    signal_backgrounds = np.logspace(-17, -10, 35)
    signal_backgrounds = signal_backgrounds[signal_backgrounds > 3e-15]
    for i, signal_background in enumerate(signal_backgrounds):
        argsList.append((lock, signal_background * u.A))
    with mp.pool.Pool(
        processes=22,
        initializer=init_pool,
        initargs=(
            I_dark,
            leak_rate,
            num_pix_x,
            num_pix_y,
            num_steps_per_second,
            number_of_pulls,
            cuttoff_freq,
            pos_threshold,
            neg_threshold,
            seed,
            raw_events_out,
            file_names,
        ),
    ) as pool:
        pool.starmap(runDarkCurrentAnalysis, argsList)
        pool.close()
        pool.join()

    # Plot resulting distributions
    plot_data(file_names, file_location, file_name)

    # To determine the max incident energy to simulate, use Peter's graph going to 3000lux
    # 3000lux (lumens/m**2) -> W/m**2 assuming an ideal monochromatic source 683.002 lumens/W
    # Multiplying by the pixel pitch (1.5e-5m) to get W incident on each pixel
    # Convert to amps for the Prophosee Gen 3 (QE = .27)
    # Solve for the corresponding R_lambda = QE*wvl*e/(h*c) (e = elementary charge, h = plank's constant, c = speed of light)
    # Approximate maximum induced current from AFRL test = 1.18e-10 A, ranging from a 10th of the
    # dark current through currents with e-11 values. Final plot will note the maximum current
    # values seen in simulated datasets with a discussion on the resulting noise behavior

    # To test with the added leak rate
    # The leak rate across all event-based sensors is claimed to be .1Hz according to
    # Delbruck's V2E paper. I don't think that is right. I'll run the simulation with
    # that value and no additional parasitic photocurrent.

    # To test with Parasitic Photocurrent
    # According to the thermal leak paper an early generation of had a leak rate of
    # fperlum = 2.7e-3 #Hz/lux. Using the same conversions as above we can translate this
    # into a Hz/W = fperW = 81960240. Now that we are working inside the circuit with amps
    # We need to go backwards to calculate the incident watts.
    # For the gen3 Prophesee, R_lambda = 0.11977332746963579 A/W, dividing by this value
    # will return to the watts prior to hitting the photodiode. This value can be used to
    # calculate parasitic photocurrent.
