import circuitry as ctry
import circuit_params as cparams
import numpy as np
import astropy.units as u
import os
import pickle
import h5py
import pandas as pd

if __name__ == '__main__':
    # Load the correct simulation hdf5 files
    circuitry = ctry.ebCircuitSim()
    circuitry_params = cparams.circuitParameters()
    # File paths
    simulation_folder = "2021_01_12T02_40_07"
    h5_path = simulation_folder + "/FinalImageFields/FinalFrames3.hdf5"
    attr_path = simulation_folder + "/FinalImageFields/Attributions3.pickle"
    events_path = simulation_folder + "/Events/"
    # Load the simulation parameters
    directory = os.getcwd()
    os.chdir(simulation_folder)
    with open("observer.pickle", "rb") as pickle_file:
        observer = pickle.load(pickle_file)
    with open("t_array", "rb") as pickle_file:
        t_array = pickle.load(pickle_file)
    with open("sim_name", "rb") as pickle_file:
        sim_name = pickle.load(pickle_file)
    with open("ebs_circuit_parameters.pickle", "rb") as pickle_file:
        circuit_para = pickle.load(pickle_file)
    # Change threshold values for testing
    circuit_para.theta_on = 0.15  # Threshold to turn on event
    circuit_para.theta_off = 0.15  # Threshold to turn off event
    
    seed = 42
    os.chdir(directory)
    # Run the circuitry simulation
    events = circuitry.generate_events(observer, t_array, h5_path, attr_path, sim_name, circuit_para, time_window=2 * u.s, seed=None, shot_noise=True, high_freq_noise=True, junction_leak=True, parasitic_leak=False, plot=False, plot_freq=100)
    # Save events to HDF5 with attributes of the circuit simulation parameters
    # Format circuit parameters to 2 significant digits for filename
    param_str = "pthreshold_{:f}_nthreshold_{:f}".format(circuit_para.theta_on,circuit_para.theta_off)
    with h5py.File(h5_path, "a") as f:
        events_grp_string = 'events'
        events_grp = f.require_group(events_grp_string)
        if param_str in events_grp:
            del events_grp[param_str]
        events.to_hdf(h5_path, key=events_grp_string+'/'+param_str, mode='a')
        # Save attributes
        # for key, value in circuit_para.__dict__.items():
        #     events_grp["events/"+param_str].attrs[key] = value
    # Save events to CSV for easy viewing
    csv_path = os.path.join(events_path, param_str + ".csv")
    # Ensure events_path exists
    os.makedirs(events_path, exist_ok=True)
    # Save events to CSV
    events.to_csv(csv_path, index=False)