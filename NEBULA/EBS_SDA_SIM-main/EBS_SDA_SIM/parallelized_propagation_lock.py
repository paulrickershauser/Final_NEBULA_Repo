# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:55:49 2023

@author: User
"""

import radiometry as rmtry
import opticalSystem as optsys
import satelliteModelDynamics as dynamics
import os
import pickle
import multiprocessing as mp
import time
import pdb
import h5py


def init_pool(file_location,full_name_pscrn,full_name_hdf5,full_name_dict):
    # Set the variables that will be globally used for this simulation
    global sim_name
    global t_array
    global sat_parameters
    global sat_obs
    global sat_array
    global prop_parameters
    global observer_optics
    global Dz
    global observer_loc
    global pscrn_name
    global hdf5_name
    global dict_name
    
    # Assign the input variables
    pscrn_name = full_name_pscrn
    hdf5_name = full_name_hdf5
    dict_name = full_name_dict
    
    # Initialize the packages to load in the necessary parameters
    sat_dynamics = dynamics.SatelliteDynamics()
    # obs_define = optsys.opticalSystem()
    
    # Load the simulation data from a particular file folder
    directory = os.getcwd()
    os.chdir(file_location)
    with open('sim_name','rb') as pickle_file:
        sim_name = pickle.load(pickle_file)

    with open('t_array','rb') as pickle_file:
        t_array = pickle.load(pickle_file)

    with open('sat_parameters','rb') as pickle_file:
        sat_parameters = pickle.load(pickle_file)

    with open('sat_obs','rb') as pickle_file:
        sat_obs = pickle.load(pickle_file)

    with open('sat_array','rb') as pickle_file:
        sat_array = pickle.load(pickle_file)

    with open('prop_parameters_test_2','rb') as pickle_file:
        prop_parameters = pickle.load(pickle_file)

    with open("observer.pickle", "rb") as pickle_file:
        observer_optics = pickle.load(pickle_file)

    with open('Dz','rb') as pickle_file:
        Dz = pickle.load(pickle_file)
    # Change back to the original directory
    os.chdir(directory)
    
def runPropagation(lock,t_step):
    # Load the proper class to run the propagation code
    radiometry = rmtry.RadiometryModel()
    
    # Run with only one timestep
    if t_step % 100 == 0:
        plot_psf = True
    else:
        plot_psf = False
        
    radiometry.frame_generation_one_step_lock(prop_parameters, observer_optics, sat_parameters, sat_obs, sat_array, t_array, t_step, lock, hdf5_name, dict_name, sim_name = sim_name, pscrn_file_name = pscrn_name, plot_psf=False)

if __name__ == '__main__':
    # Load the correct simulation
    directory = os.getcwd()
    file_location = 0
    while os.path.exists(directory + '/' + str(file_location)) == False:
        file_location = input('Please input the simulation directory name within the current working directory: ')
    file_location = directory + '/' + str(file_location)
    
    # Load the proper phase screen file path for this scenario
    pscrn_file_name = 0
    while os.path.exists(file_location + '/PhaseScreens/' + str(pscrn_file_name) + '.hdf5') == False:
        pscrn_file_name = input('Please input the phase screen file name without file extension (.hdf5): ')
    pscrn_file_name = file_location + '/PhaseScreens/' + str(pscrn_file_name) + '.hdf5'
    
    # Choose the number of processors that will be involved in the computation
    num_processors = 'num_processors'
    while isinstance(num_processors,int) == False:
        num_processors = input('Please input the number of processors that will run the simulation. Note this number will be rounded to the nearest integer: ')
        try:
            num_processors = int(num_processors)
            break
        except:
            print('Please input an integer for the number of processors.')
    
    # Load time array to determine number of simulations being run
    os.chdir(file_location)
    with open('t_array','rb') as pickle_file:
        t_array = pickle.load(pickle_file)
    t_steps = len(t_array)
    
    # Create an HDF5 File to store the PSFs
    os.chdir(directory)
    watt_file_name = 0
    while isinstance(watt_file_name, str) == False:
        watt_file_name = input('Please input a file name for the file that will hold the final images: ')
    newdirectory = file_location + '/FinalImageFields/'
    if not os.path.exists(newdirectory):
        os.makedirs(newdirectory)
    full_name_hdf5 = newdirectory + watt_file_name + '.hdf5'
    # Create an empty HDF5 file
    with h5py.File(full_name_hdf5, 'w') as f:
        f.close()
    
    # Create a dictionary to store the different timesteps, attribuation dictionaries
    attr_dict_file_name = 0
    while isinstance(attr_dict_file_name, str) == False:
        attr_dict_file_name = input('Please input a file name for the dictionary that will hold the attributions by pixel at each timestep: ')
    full_name_dict = newdirectory + attr_dict_file_name + '.pickle'
    Attribution_dict = {}
    with open(full_name_dict, "wb") as picklefile:
        pickle.dump(Attribution_dict, picklefile)
    
    # Create a lock object to enable saving 
    m = mp.Manager()
    lock = m.Lock()
    
    # Add the lock to each time step
    argsList = []
    os.chdir(file_location)
    # Open the time array to determine the number of steps
    with open('t_array','rb') as pickle_file:
        t_array = pickle.load(pickle_file)
    os.chdir(directory)
    # for i in range(t_steps):
    for i in range(len(t_array)):
        argsList.append((lock,i))
    
    # Start the chosen number of processes, keep a job list to be able to close the processes
    tic = time.time()
    with mp.pool.Pool(processes=num_processors,initializer=init_pool,initargs=(file_location,pscrn_file_name,full_name_hdf5,full_name_dict)) as pool:      
        pool.starmap(runPropagation,argsList)
        pool.close()
        pool.join()
    toc = time.time()
    total_time = toc-tic
    print('Total Run Time = {}'.format(total_time))