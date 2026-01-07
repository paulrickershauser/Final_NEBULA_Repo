# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:55:09 2021

@author: rache
"""

import radiometry as rmtry
import opticalSystem as optsys
import circuitry as ctry
import satelliteModelDynamics as dynamics
import satelliteSelection as satselect

class simRun():
    '''
    This class contains differnent methods to easily set up and run iterations 
    of the EBS SDA simulation
    '''
    def __init__(self,**kwargs):
        return
    
    def __str__(self):
        print('This object contains standard methods for running EBS SDA.')
        return
    
    def setup_EBS_SDA_SIM(self,t_array='undefined', tle_file_name = 'filename', pscrn_file_name = 'filename', watt_file_name = 'filename', sim_name = 'simname'):
        
        # Define all the required objects 
        sat_select = satselect.SatelliteSelection()
        sat_dynamics = dynamics.SatelliteDynamics()
        obs_define = optsys.opticalSystem()
        radiometry = rmtry.RadiometryModel()
        circuitry = ctry.ebCircuitSim()
        
        # Define Necessary Simulation Values
        # Define a date time object for this simulation
        if t_array == 'undefined':
            t_array = sat_dynamics.time_array_def()
        
        # Choose a satellite to simulate
        if tle_file_name == 'filename':
            # A specific tle file was not provided, download the date and time requested
            satellites, sat_parameters = sat_select.choose_sat() #TLE filename 'tle_2021_8_31.txt'
        else:
            satellites, sat_parameters = sat_select.choose_sat(filename=tle_file_name)
        
        # Define the observing conditions
        observer_loc = obs_define.define_observer()
        observer_optics = obs_define.define_optics(t_array)
        
        # Run the satellite propagation (Get the average zenith for the sat to set the prop length)
        sat_obs, sat_array, Dz = sat_dynamics.sat_observation(observer_loc,observer_optics,satellites,t_array,sim_name)
        
        # Conduct the geometric analysis to choose spacing for phase screens (Run the schmidt analysis, use geometry to determine Diameter at different length scales to determine phase screen sizes)
        prop_parameters = radiometry.grid_spacing_analysis(observer_optics,sim_name=sim_name)
        
        # Create extended phase screens
        # Output the name of the HDF5 file that contains the screens
        if pscrn_file_name == 'filename':
            pscrn_file_name, prop_parameters = radiometry.extended_phase_screen(observer_optics,prop_parameters,t_array,sim_name)

            
        # Propagate the point sources to create a set of PSFs
        if watt_file_name == 'filename':
            watt_file_name = radiometry.propogate_turbluence(Dz, prop_parameters, observer_optics, satellites, sat_parameters, sat_obs, sat_array, t_array, sim_name = sim_name, pscrn_file_name = pscrn_file_name, plot_point_source = True, plot_psf=True)

        
        return sat_select, sat_dynamics, obs_define, radiometry, circuitry, satellites, observer_loc, observer_optics, pscrn_file_name, watt_file_name
    
    def run_EBS_SDA_SIM():
        
        # Input of PSFs run the circuitry model 
        
        return
    
    def sensitivity_Analysis():
        
        return