# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 01:00:50 2023

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

def init_pool(pix_x,pix_y,steps_per_second,pulls,freq,pos_thres,neg_thres,rng_seed,data_name):
    global num_pix_x
    global num_pix_y
    global num_steps_per_second
    global number_of_pulls
    global cutoff_freq
    global pos_threshold
    global neg_threshold
    global seed
    global file_name
    
    num_pix_x = pix_x
    num_pix_y = pix_y
    num_steps_per_second = steps_per_second
    number_of_pulls = pulls
    cutoff_freq = freq
    pos_threshold = pos_thres
    neg_threshold = neg_thres
    seed = rng_seed
    file_name = data_name

def runDarkCurrentAnalysis(lock,I_dark):
    # Load the proper class to run the noise generation code
    circuit_test = rtc.circuitTest()
    # Run the simulation
    tic = time.time()
    circuit_test.dark_rate_analysis(I_dark, lock, num_pix_x, num_pix_y, num_steps_per_second, number_of_pulls, cutoff_freq, pos_threshold, neg_threshold, seed, file_name)
    toc = time.time()
    print('Dark Current {} is done processing. Process took {} seconds.'.format(I_dark, (toc-tic)))
    
def plot_data(dark_stats,directory):
    fig, ax = plt.subplots(dpi=200)
    ax.plot(dark_stats['Dark Current'], dark_stats['Mean Event Rate'], '-')
    ax.fill_between(dark_stats['Dark Current'], dark_stats['Mean Event Rate'] - dark_stats['Standard Deviation'], dark_stats['Mean Event Rate'] + dark_stats['Standard Deviation'], alpha=0.2)
    ax.set_title('Shot Noise Event Rate from Dark Current')
    ax.set_xlabel('Dark Current [A]')
    ax.set_ylabel('Event Rate [events/s]')
    plt.savefig(directory +'/test_plots/dark_current_analysis.pdf')
    plt.xticks(rotation=45, ha='right')
    
def prompt_user_input(variable_name,variable_type,variable_units=0,upper_bound=0,lower_bound=0,bounds=False):
    '''
    Prompt the user to input variables for the circuit object definition.

    Parameters
    ----------
    variable_name : sting
        Name of the variable the user will be prompted to enter.
    variable_type : string
        The expected type of variable the user will enter.
    variable_units : astropy units, optional
        The units the input is expected to be in if applicable. The default is 0.
    upper_bound : float, optional
        Maximum value of the user input.
    lower_bound : float, optional
        Minimum value of the user input.
    bounds : bool, optional
        The units the input is expected to be in if applicable. The default is 0.

    Returns
    -------
    user_input : TYPE
        DESCRIPTION.

    '''
    user_input = variable_name
    if variable_units != 0:
        user_input = variable_name
        input_check = variable_name
        while not isinstance(input_check, variable_type) :
            user_input = input('Please input a {} for the sensor {} in units of {}: '.format(str(variable_type),variable_name, str(variable_units)))
            try:
                if variable_type == int:
                    input_check = int(user_input)
                    user_input = int(user_input)*variable_units
                elif variable_type == float:
                    input_check = float(user_input)
                    user_input = float(user_input)*variable_units
                if bounds :
                    if user_input.value <= upper_bound and user_input.value >= lower_bound:
                        break
                    else:
                        print('User input is not between {} and {}. Please enter a new value.'.format(lower_bound, upper_bound))
                        input_check = variable_name
            except:
                print('Input was not a {}. Please try input again.'.format(str(variable_type)))
    else:
        user_input = variable_name
        while not isinstance(user_input, variable_type) :
            user_input = input('Please input a {} for the sensor {}: '.format(str(variable_type),variable_name))
            try:
                if variable_type == int:
                    user_input = int(user_input)
                elif variable_type == float:
                    user_input = float(user_input)
                elif variable_type == bool:
                    user_input = bool(user_input)
                if bounds :
                    if user_input <= upper_bound and user_input >= lower_bound:
                        break
                    else:
                        print('User input is not between {} and {}. Please enter a new value.'.format(lower_bound, upper_bound))
                        user_input = variable_name
            except:
                print('Input was not a {}. Please try input again.'.format(str(variable_type)))
    return user_input 
 
if __name__ == '__main__':
    # Ensure all local variables exist to run the simulation
    try:
        print('Number of x pixels: {}'.format(num_pix_x))
    except NameError:
        num_pix_x = prompt_user_input('number of pixels in the x direction', int)
    try:
        print('Number of y pixels: {}'.format(num_pix_y))
    except NameError:
        num_pix_y = prompt_user_input('number of pixels in the y direction', int)
    try:
        print('Number of simulation steps per second of simulated time: {}'.format(num_steps_per_second))
    except NameError:
        num_steps_per_second = prompt_user_input('number of simulation steps per second of simulated time', int)
    try:
        print('Number of simulated steps: {}'.format(number_of_pulls))
    except NameError:
        number_of_pulls = prompt_user_input('number of simulated step:', int)
    try:
        print('Cutoff frequency: {}'.format(cutoff_freq))
    except NameError:
        cutoff_freq = prompt_user_input('cutoff frequency', float, variable_units=u.Hz)
    try:
        print('Positive event threshold: {}'.format(pos_threshold))
    except NameError:
        pos_threshold = prompt_user_input('positive threshold', float)
    try:
        print('Negative event threshold: {}'.format(neg_threshold))
    except NameError:
        neg_threshold = prompt_user_input('negative threshold', float)
    try:
        print('Seed: {}'.format(seed))
    except NameError:
        # Establish seed
        seed_seq = np.random.SeedSequence()
        seed = seed_seq.generate_state(1)
        seed = int(seed)
    try:
        print('File name: {}'.format(file_name))
    except NameError:
        directory = os.getcwd()
        file_location = 0
        while os.path.exists(directory + '/' + str(file_location)) == False:
            file_location = input('Please input the simulation directory name within the current working directory: ')
        file_location = directory + '/' + str(file_location)
        file_name = 0
        while isinstance(file_name,str) != True:
            file_name = input('Please input the file name without file extension (.csv): ')
        file_name = file_location + '/' + str(file_name) + '.csv'
    # Ensure there is a csv file at that location
    try:
        df = pd.read_csv(file_name)
    except:
        df = pd.DataFrame(list(),columns=['Dark Current','Mean Event Rate','Standard Deviation','Simulation Duration'])
        df.to_csv(file_name,index=False)
    
    # Create a lock object to enable saving 
    m = mp.Manager()
    lock = m.Lock()
    
    # Generate a tuple with each input variable for all the runs
    argsList = []
    # for i in range(t_steps):
    #for i in range(10):
    #    if i != 0:
    #        argsList.append((lock,i*1e-17*u.A))
    # for specific list
    noise_rates = [2e-16,1.9e-16,1.8e-16,1.7e-16,1.6e-16,1.5e-16,1.4e-16,1.3e-16,1.2e-16,1.1e-16,1e-16,9.9e-17,9.8e-17,9.7e-17,9.6e-17,9.5e-17,9.4e-17,9.3e-17,9.2e-17,9.1e-17,9e-17,8.9e-17,8.8e-17,8.7e-17,8.6e-17,8.5e-17,8.4e-17,8.3e-17,8.2e-17,8.1e-17,8e-17,7.9e-17,7.8e-17,7.7e-17,7.6e-17,7.5e-17,7.4e-17,7.3e-17,7.2e-17,7.1e-17,7e-17,6.9e-17,6.8e-17,6.7e-17,6.6e-17,6.5e-17,6.4e-17,6.3e-17,6.2e-17,6.1e-17,6e-17]
    for i, noise in enumerate(noise_rates):
    	argsList.append((lock,noise*u.A))
    
    with mp.pool.Pool(processes=20,initializer=init_pool,initargs=(num_pix_x, num_pix_y, num_steps_per_second, number_of_pulls, cutoff_freq, pos_threshold, neg_threshold, seed, file_name)) as pool:      
        pool.starmap(runDarkCurrentAnalysis,argsList)
        pool.close()
        pool.join()
        
    df = pd.read_csv(file_name)
    plot_data(df,file_location)
    