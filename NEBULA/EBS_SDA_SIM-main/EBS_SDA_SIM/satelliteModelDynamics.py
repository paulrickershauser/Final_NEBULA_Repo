# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:40:09 2021

@author: rache
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import os
import pdb
from astropy.coordinates import SkyCoord
from skyfield.api import load, wgs84

class SatelliteDynamics(SkyCoord):
    """
    This class contains all variables and functions needed to compute a 
    satellite's location at a specific observation time. Additionally this 
    object will ensure the attributes required to define the satellite in Az
    and El coordinates are satisfied prior to computing intersection with the
    optical system frame.
    
    Args:
        kwargs:
            user specified values, specifically satellite information 
            
        
    Attributes:
        
    
    """
    
    def __init__(self,**kwargs):
        # self.satellite = satellite
        # # Satellite being propagated, created as a satelliteSelection object
        # self.observer = observer
        # # Oberving site of the satellite, created as a satelliteSelection object
        # self.timearray = timearray
        # Time of the Propagation
        
        return
    
    def __str__(self):
        """
        String Representation of the Satellte Dynamics Outputs
        
        When the command "print" is used this method will return the values
        contained in the object

        """
        print('This object will propagate a satellite object.')
        print('Ensure inputs include a satellite selection and an array of time.')
        print('Outputs include a time series set of positions in latitude, longitude, and .')
        return
    
    def satellite_prop(self):
        '''
        Using the skyfield objects defined with an updated TLE, the satelite GCRS position is reported at each time step.

        Returns
        -------
        sat_pos_geocentric : WGS object
            WGS Object with methods that output latitude and longitude of the satellite's subpoint on the geocentric sphere.

        '''
        # Satellite Propagation in SkyField outputs into 
        # GCRS (Geocentric Celestial Reference System) = (x,y,z)
        sat_pos_geocentric = self.satellite.at(self.timearray)
        subpoint = wgs84.subpoint(sat_pos_geocentric)
        lat = subpoint.latitude
        lon = subpoint.longitude
        plt.plot(lat.degrees,lon.degrees)
        
        return sat_pos_geocentric
    
    def sat_observation(self,observer,optics,satellites,t_array,sim_name):
        '''
        This method uses the skyfield satellite objects and WCS objects to propogate each satellite position and register the projected pixel locations on the focal array.
        The projection, when compared to a real dataset, is a good way to check if the pointing parameters are close enough.

        Parameters
        ----------
        observer : skyfield.toposlib.GeographicPosition
            Skyfield geographic position for the observer.
        optics : dict
            Dictionary of optical properties of the observer as defined in self.define_optics.
        satellites : dict
            Dictionary of satellite skyfield objects.
        t_array : skyfield.timelib.Time
            Array with all the time steps defined.
        sim_name : string
            Simulation name to create a unique folder for this simulation data to be saved.

        Returns
        -------
        sat_obs : dict
            Dictionary with each satellite's observable condition at each time step. If the satellite is in the field of FOV and illuminated by the sun at a specific time step it is assigned a True boolean.
            The dictionary has each satellite as a key and the boolean assignments at each timestep as a list.
        sat_array : dict
            Dictionary with each satellite's right ascension, declination, and zenith distance in topocentric coordinates at each timestep. 
            The dictionary has each satellite as a key and the topocentric coordinates as a list of tuples.
        Dz : float
            Maximum distance through the atmosphere between the observer and satellite at each time step.

        '''
        # Unpack the pointing dictionary from the optics object
        pointing = optics['optics_pointing_dict']
        # Unpack the field of view that restricting the right ascension and 
        # declination at each time step
        AFOV = optics['AFOV']
        aspect_ratio = optics['aspect ratio']
        DAFOV = AFOV*aspect_ratio
        # Use WCS object to convert pixels to RA,DEC
        proj = optics['optics_wcs_dict'][0]
        x_total = optics['num of pixels x dir']
        y_total = optics['num of pixels y dir']
        sensor_outline = []
        for i in np.arange(1,x_total,1):
            sensor_outline.append((i,1))
        for i in np.arange(1,y_total,1):
            sensor_outline.append((x_total,i))
        for i in np.arange(x_total,0,-1):
            sensor_outline.append((i,y_total))
        for i in np.arange(y_total,0,-1):
            sensor_outline.append((1,i))
        sensor_ra_dec = proj.wcs_pix2world(sensor_outline,0)
        sensor_ra = sensor_ra_dec[:,0]
        sensor_dec = sensor_ra_dec[:,1]
        
        # For each satellite
        sat_obs = {}
        sat_array = {}
        Dz = []
        for i, satkey in enumerate(list(satellites.keys())):
            # Unpack specific satellite for iteration
            satellite = satellites[satkey]
            # Calculate the vector between the observer
            difference = satellite - observer
            topocentric = difference.at(t_array)
            # Produce the ICRF coordinates, ICRF is a substantiation of ICRS
            ra , dec, zenith = topocentric.radec() 
            Dz.append(np.mean(np.array(zenith.m)))
            
            # Check if the satellite is sunlit
            eph = load('de421.bsp')
            sunlit = satellite.at(t_array).is_sunlit(eph)
        
            # If the satellite is sunlit, Check if the latitude/longitude 
            # intersects the optical FOV during the sunlit times
            # If so then an observation was possible
        
            sat_obs[satkey] = {}
            sat_array[satkey] = {}
            first_true = True
            if sunlit.any() == True:
                for t in np.arange(0,len(t_array.utc[0,:])):
                    Ra = pointing[t][0]
                    Dec = pointing[t][1]
                    ra_max = Ra + AFOV/2
                    ra_min = Ra - AFOV/2
                    dec_max = Dec + DAFOV/2
                    dec_min = Dec - DAFOV/2
                    if ra._degrees[t] < ra_max.value and ra._degrees[t] > ra_min.value and dec._degrees[t] < dec_max.value and dec._degrees[t] > dec_min.value and sunlit[t] == True:
                        if first_true == True:
                            print('The first timestep with an observation is {}'.format(t))
                            first_true = False
                        sat_obs[satkey][t] = True
                    else:
                        sat_obs[satkey][t] = False
                    sat_array[satkey][t] = (ra._degrees[t]*u.deg,dec._degrees[t]*u.deg,zenith.m[t]*u.m)
                print('The satellite is observable during this period.')
            else:
                for t in np.arange(0,len(t_array[0,:])):
                    sat_obs[satkey][t] = False
                    sat_array[satkey][t] = (ra._degrees[t]*u.deg,dec._degrees[t]*u.deg,zenith.m[t]*u.m)
                print('The satellite is not observable.')
        
        # Keep maximum distance for propagation parameter    
        Dz = max(Dz)*u.m
        
        # Plot the satellite track and the observer FOV to help user understand adjustments needed
        # Save in an output folder
        # For each Satellite Observed
        fig, ax = plt.subplots(dpi=200)
        any_sat_observed = False
        for i, satkey in enumerate(list(satellites.keys())):
            sat_observed = False
            if any(list(sat_obs[satkey].values())) == True:
                sat_observed = True
            if i == 0:
                plt.plot(ra._degrees,dec._degrees,'b',zorder = 0)
            else:
                plt.plot(ra._degrees,dec._degrees,'b',zorder = 3)
            if sat_observed == True:
                if any_sat_observed == False:
                    any_sat_observed = True
        # For each Star Observed in first time step, plot its (ra, dec)
        # Load earth from Skyfield to retrieve Ra and Dec
        planets = load('de421.bsp')
        earth = planets['earth']
        star_objs = optics['optics_star_dict'][0]
        star_ras, star_decs, star_dists = earth.at(t_array[0]).observe(star_objs).radec()
        plt.scatter(star_ras._degrees,star_decs._degrees,s=2,c='gold',marker='*',zorder=1)
        if any_sat_observed == True:
            plt.plot(sensor_ra,sensor_dec,'r',zorder = 2)
            plt.legend(['Satellite Track','Stars','Sensor FOV Extrema at t=0'],loc='upper right')
        else:
            plt.legend(['Sensor FOV Extrema at t=0','Stars'],loc='upper right')
        plt.gca().invert_xaxis() # Invert because image is inverted after going through optic
        plt.xlabel('ICRF right ascension [deg]')
        plt.ylabel('ICRF declination [deg]')
        plt.title('Satellite Track in ICRS')
        ax.set_aspect('equal')
        directory = os.getcwd()
        newdirectory = directory + '/' + sim_name + '/ProcessingImages/Satellite_Prop'
        if not os.path.exists(newdirectory):
            os.makedirs(newdirectory)
        plt.savefig(newdirectory + '/satellitePropagation.pdf')
        plt.close()
        
        return sat_obs, sat_array, Dz
    
    def time_array_def(self):
        '''
        Build a time class object for the full simulation. 
        It is defined in UTC (Greenwich) date and time.

        Parameters
        ----------
        year_start : TYPE
            DESCRIPTION.
        month_start : TYPE
            DESCRIPTION.
        day_start : TYPE
            DESCRIPTION.
        hour_start : TYPE
            DESCRIPTION.
        minute_start : TYPE
            DESCRIPTION.
        second_start : TYPE
            DESCRIPTION.
        number_of_steps : TYPE
            DESCRIPTION.
        year_end : TYPE, optional
            DESCRIPTION. The default is 0.
        month_end : TYPE, optional
            DESCRIPTION. The default is 0.
        day_end : TYPE, optional
            DESCRIPTION. The default is 0.
        hour_end : TYPE, optional
            DESCRIPTION. The default is 0.
        minute_end : TYPE, optional
            DESCRIPTION. The default is 0.
        second_end : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        t : TYPE
            DESCRIPTION.

        '''
        ts = load.timescale()
        time_parameters_complete = False
        year_end = False
        month_end = False
        day_end = False
        hour_end = False
        minute_end = False
        second_end = False
        
        while time_parameters_complete == False:
            # Ask for time scale of the simulation
            print('The simulation duration will be set by the unit of time that will be varied (year, month, day, hour, minute, or second).')
            time_scale = 'time_scale'
            time_scale_list = ['year','month','day','hour','minute','second','Year','Month','Day','Hour','Minute','Second']
            while time_scale not in time_scale_list:
                time_scale = input('PLease input the unit of time that will define the end of the simulation: ')
                if time_scale not in time_scale_list:
                    print('The input was not one of the time scale options. Please try again.')
            if time_scale == 'year' or time_scale == 'Year':
                year_end = True
            elif time_scale == 'month' or time_scale == 'Month':
                month_end = True
            elif time_scale == 'day' or time_scale == 'Day':
                day_end = True
            elif time_scale == 'hour' or time_scale == 'Hour':
                hour_end = True
            elif time_scale == 'minute' or time_scale == 'Minute':
                minute_end = True
            else:
                second_end = True
            # Ask for the number of time steps set by this time scale
            number_of_steps = 'number_of_steps'
            while isinstance(number_of_steps, int) == False:
                number_of_steps = input('Please input an integer for the number of timesteps that should be simulated. Assume 3000 timesteps per second to simulate the effective frame rate.: ')
                try:
                    number_of_steps = int(number_of_steps)
                    time_parameters_complete = True
                except:
                    print('Please input an integer for the number of timesteps.')

        year_start = 'year_start'
        while isinstance(year_start, int) == False:
            year_start = input('Please input an integer for the starting year of the simulation (UTC): ')
            try:
                year_start = int(year_start)
            except:
                print('Please input an integer for the starting simulation year.')
        if year_end == True:
            year_end = 'year_end'
            while isinstance(year_end, int) == False:
                year_end = input('Please input an integer for the ending year of the simulation (UTC): ')
                try:
                    year_end = int(year_end)
                except:
                    print('Please input an integer for the ending simulation year.')
            year_list = np.ndarray.tolist(np.linspace(year_start,year_end,number_of_steps))
            t = ts.utc(year_list)
        else:
            month_start = 'month_start'
            while isinstance(month_start, int) == False:
                month_start = input('Please input an integer for the starting month of the simulation (UTC): ')
                try:
                    month_start = int(month_start)
                except:
                    print('Please input an integer for the starting simulation month.')
            if month_end == True:
                month_end = 'month_end'
                while isinstance(month_end, int) == False:
                    month_end = input('Please input an integer for the ending month of the simulation (UTC): ')
                    try:
                        month_end = int(month_end)
                    except:
                        print('Please input an integer for the ending simulation month.')
                month_list = np.ndarray.tolist(np.linspace(month_start,month_end,number_of_steps))
                t = ts.utc(year_start, month_list)
            else:
                day_start = 'day_start'
                while isinstance(day_start, int) == False:
                    day_start = input('Please input an integer for the starting day of the simulation (UTC): ')
                    try:
                        day_start = int(day_start)
                    except:
                        print('Please input an integer for the starting simulation day.')
                if day_end == True:
                    day_end = 'day_end'
                    while isinstance(day_end, int) == False:
                        day_end = input('Please input an integer for the ending day of the simulation (UTC): ')
                        try:
                            day_end = int(day_end)
                        except:
                            print('Please input an integer for the ending simulation day.')
                    day_list = np.ndarray.tolist(np.linspace(day_start,day_end,number_of_steps))
                    t = ts.utc(year_start, month_start, day_list)
                else:
                    hour_start = 'hour_start'
                    while isinstance(hour_start, int) == False:
                        hour_start = input('Please input an integer for the starting hour of the simulation (UTC): ')
                        try:
                            hour_start = int(hour_start)
                        except:
                            print('Please input an integer for the starting simulation hour.')
                    if hour_end == True:
                        hour_end = 'hour_end'
                        while isinstance(hour_end, int) == False:
                            hour_end = input('Please input an integer for the ending hour of the simulation (UTC): ')
                            try:
                                hour_end = int(hour_end)
                            except:
                                print('Please input an integer for the ending simulation hour.')
                        hour_list = np.ndarray.tolist(np.linspace(hour_start,hour_end,number_of_steps))
                        t = ts.utc(year_start, month_start, hour_list)
                    else:
                        minute_start = 'minute_start'
                        while isinstance(minute_start, int) == False:
                            minute_start = input('Please input an integer for the starting minute of the simulation (UTC): ')
                            try:
                                minute_start = int(minute_start)
                            except:
                                print('Please input an integer for the starting simulation minute.')
                        if minute_end == True:
                            minute_end = 'minute_end'
                            while isinstance(minute_end, int) == False:
                                minute_end = input('Please input an integer for the ending minute of the simulation (UTC): ')
                                try:
                                    minute_end = int(minute_end)
                                except:
                                    print('Please input an integer for the ending simulation minute.')
                            minute_list = np.ndarray.tolist(np.linspace(minute_start,minute_end,number_of_steps))
                            t = ts.utc(year_start,month_start,day_start,hour_start,minute_list)
                        else:
                            second_start = 'second_start'
                            while isinstance(second_start, float) == False:
                                second_start = input('Please input a float for the starting second of the simulation (UTC): ')
                                try:
                                    second_start = float(second_start)
                                except:
                                    print('Please input a float for the starting simulation second.')
                            if second_end == True:
                                second_end = 'second_end'
                                while isinstance(second_end, float) == False:
                                    second_end = input('Please input a float for the ending second of the simulation (UTC): ')
                                    try:
                                        second_end = float(second_end)
                                    except:
                                        print('Please input a float for the ending simulation second.')
                                seconds_list = np.ndarray.tolist(np.linspace(second_start,second_end,number_of_steps))
                                t = ts.utc(year_start,month_start,day_start,hour_start,minute_start,seconds_list)

        return t
            
        
        
        
        
        