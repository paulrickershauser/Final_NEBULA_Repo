# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:08:40 2021
Load modules and workspace variables from a previous save to run an EBS simulation.
@author: rache
"""

import radiometry as rmtry
import opticalSystem as optsys
import circuitry as ctry
import circuit_params as cparams
import satelliteModelDynamics as dynamics
import satelliteSelection as satselect
import os
import pickle
import pandas as pd
import astropy.units as u

# sat_select = satselect.SatelliteSelection()
# sat_dynamics = dynamics.SatelliteDynamics()
# obs_define = optsys.opticalSystem()
radiometry = rmtry.RadiometryModel()
circuitry = ctry.ebCircuitSim()
circuitry_params = cparams.circuitParameters()

# Load the simulation data from a particular file folder
directory = os.getcwd()
file_location = directory + "/2021_01_12T02_40_07"
os.chdir(file_location)
with open("sim_name", "rb") as pickle_file:
    sim_name = pickle.load(pickle_file)

with open("t_array", "rb") as pickle_file:
    t_array = pickle.load(pickle_file)

with open("sat_parameters", "rb") as pickle_file:
    sat_parameters = pickle.load(pickle_file)

with open("sat_obs", "rb") as pickle_file:
    sat_obs = pickle.load(pickle_file)

with open("sat_array", "rb") as pickle_file:
    sat_array = pickle.load(pickle_file)

with open("prop_parameters_test_2", "rb") as pickle_file:
    prop_parameters = pickle.load(pickle_file)

with open("observer.pickle", "rb") as pickle_file:
    observer = pickle.load(pickle_file)

with open("Dz", "rb") as pickle_file:
    Dz = pickle.load(pickle_file)

with open("observer_loc", "rb") as pickle_file:
    observer_loc = pickle.load(pickle_file)

with open("ebs_circuit_parameters.pickle", "rb") as pickle_file:
    ebs_cir_params = pickle.load(pickle_file)


# Return to parent directory and run the satellite loading modules
# os.chdir(directory)
# tle_name = directory + "/TLE Files/tle_2021_012.txt"
# satellites, sat_parameters = sat_select.choose_sat(filename = tle_name)

# Load the proper phase screen file path for this scenario if they have already been generated
# pscrn_file_name = file_location + "/PhaseScreens/pscrns_2021_01_12_test_2.hdf5"

# Load the proper final frames if they have already been generated
frame_file_name = file_location + "/FinalImageFields/FinalFrames3.hdf5"
attribution_dict = file_location + "/FinalImageFields/Attributions3"

# Run the simulation to generate the events
events = circuitry.generate_events(observer,t_array,
        frame_file_name,
        attribution_dict,
        sim_name,
        ebs_cir_params,
        time_window=2 * u.s,
        seed=None,
        shot_noise=True,
        high_freq_noise=True,
        junction_leak=False,
        parasitic_leak=False,
        plot=False,
        plot_freq=100,
    )

pd.save_csv(file_location + "/events.csv", events)