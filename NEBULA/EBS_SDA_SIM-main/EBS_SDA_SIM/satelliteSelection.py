# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:40:09 2021

@author: rache
"""

import numpy as np
import pdb
import astropy.units as u
from astropy.coordinates import SkyCoord
import skyfield as sf
import datetime as dt
from skyfield.api import load
from sgp4.api import Satrec, WGS72

class SatelliteSelection(SkyCoord):
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
    
        
        return
    
    def __str__(self):
        """
        String Representation of the Satellte Dynamics Outputs
        
        When the command "print" is used this method will return the values
        contained in the object

        """
        print('This is a satellite object.')
        print('Given a TLE and time since the TLE, the orbital position is returned.')
        return
    
    def load_tle(self,filename = 0):
        '''
        This program will query the celestrak catalog to load a set of current
        TLE files. The code structure is from a guide found in the documentation
        for SkyField at 
        https://rhodesmill.org/skyfield/earth-satellites.html#loading-a-tle-file

        Returns
        -------
        sats : Dictionary of satellites and associated TLE elements

        '''
        website_url = 'http://celestrak.com/NORAD/elements/active.txt'
        # Load a given file name or prompt user to download a specific TLE
        if filename == 0:
            today_bool = 'today_bool'
            while isinstance(today_bool, bool) == False:
                today_bool = input("Use today's date? [True or False]: ")
                if today_bool == 'True' or today_bool == 'true' or today_bool == 'Yes' or today_bool == 'yes':
                    today_bool = True
                elif today_bool == 'False' or today_bool == 'false' or today_bool == 'No' or today_bool == 'no':
                    today_bool = False
                else:
                    print('Please enter a boolean')
            if today_bool == True:
                # Use today's date as the TLE selection
                todaydate = dt.datetime.today()
                filename = 'tle_{}_{}_{}.txt'.format(todaydate.year,todaydate.month,todaydate.day)
                print('The TLE catalog will be saved using the filename {}'.format(filename))
                sats = load.tle_file(website_url,filename = filename)
            else:
                # Prompt user for year, month, and date
                # year = 'year'
                # while isinstance(year, int) == False:
                #     year = input("Please input the observation year: ")
                #     try:
                #         year = int(year)
                #     except:
                #         print('Please enter an integer for the year.')
                # month = 'month'
                # while isinstance(month, int) == False:
                #     month = input("Please input the observation month: ")
                #     try:
                #         month = int(month)
                #     except:
                #         print('Please enter an integer for the month.')
                # date = 'date'
                # while isinstance(date, int) == False:
                #     date = input("Please input the observation date: ")
                #     try:
                #         date = int(date)
                #     except:
                #         print('Please enter an integer for the date.')
                # filename = 'tle_{}_{}_{}.txt'.format(year,month,date)
                print('User must request download of historical TLE files manually or use a previously downloaded file')
                filename = 0
                while isinstance(filename, str) == False:
                    filename = input("Please input the file location of the relevant TLE file: ")
                    try:
                        sats = load.tle_file(filename)
                    except:
                        print('Load failed check path')
                        filename = 0  
        else:
            sats = load.tle_file(filename)
        
        print('Loaded', len(sats),'satellites')
        return sats
    
    def choose_sat(self,sats = 0,sat_name = 0,sat_num = 0, filename = 0):
        '''
        This method will select a satellite and it's TLE information for future
        methods to employ in orbital propagation. The code structure is from a 
        guide found in the documentation for SkyField at 
        https://rhodesmill.org/skyfield/earth-satellites.html#loading-a-tle-file

        Parameters
        ----------
        sats : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_name : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_num : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        satellite : TYPE
            DESCRIPTION.

        '''
        if sats == 0:
            if isinstance(filename,str):
                sats = self.load_tle(filename)
            else:
                sats = self.load_tle()
            # print('Run load_tle first to populate satellite choices.')
        else:
            print('Satellites already loaded.')
        sat_choice = {}
        sat_parameters = {}
        sat_choice_done = False
        num_satellites = 0
        while sat_choice_done == False:
            if sat_name == 0 and sat_num == 0:
                sat_choice_itr = input("Please enter a catalog number (5 digits) or satellite name: ")
                try:
                    sat_choice[num_satellites] = int(sat_choice_itr)
                    print('You have chosen satellite {}.'.format(sat_choice_itr))
                    num_satellites += 1
                except:
                    print('Invalid number. Please enter an integer.')
                # Prompt user to exit
                sat_choice_done = 'sat_choice_done'
                while isinstance(sat_choice_done, bool) == False:
                    sat_choice_done = input('Are you done selecting satellites (True or False): ')
                    if sat_choice_done == 'True' or sat_choice_done == 'true':
                        sat_choice_done = True
                    elif sat_choice_done == 'False' or sat_choice_done == 'false':
                        sat_choice_done = False
                    else:
                        print('Please enter a boolean.')
            elif sat_name != 0 and sat_num == 0:
                # If given list loop through the satellite numbers
                for i in np.arange(0,len(sat_name),1):
                    try:
                        sat_choice[num_satellites] = str(sat_name[i])
                        print('You have chosen {}.'.format(sat_choice))
                        num_satellites += 1
                    except:
                        print('Invalid entry. Not a string. Satellite {} not added to satellite list'.format(sat_name[i]))
                # Prompt user to exit
                sat_choice_done = 'sat_choice_done'
                while isinstance(sat_choice_done, bool) == False:
                    sat_choice_done = input('Are you done selecting satellites (True or False): ')
                    if sat_choice_done == 'True' or sat_choice_done == 'true':
                        sat_choice_done = True
                    elif sat_choice_done == 'False' or sat_choice_done == 'false':
                        sat_choice_done = False
                        # set sat_name = 0 to prompt different selection method
                        sat_name = 0
                    else:
                        print('Please enter a boolean.')
            elif sat_num != 0 and sat_name == 0:
                for i in np.arange(0,len(sat_num),1):
                    if sat_num[i]<=0 or sat_num[i]>=100000:
                        print('Invalid number. Outside acceptable NORAD catalog list. Moving to next number.')
                    else:
                        try:
                            sat_choice[num_satellites] = int(sat_num[i])
                            print('You have chosen {}.'.format(sat_num[i]))
                            num_satellites += 1
                        except:
                            print('Invalid entry. Not an integer. Satellite {} not added to satellite list'.format(sat_num[i]))
                # Prompt user to exit
                sat_choice_done = 'sat_choice_done'
                while isinstance(sat_choice_done, bool) == False:
                    sat_choice_done = input('Are you done selecting satellites (True or False): ')
                    if sat_choice_done == 'True' or sat_choice_done == 'true':
                        sat_choice_done = True
                    elif sat_choice_done == 'False' or sat_choice_done == 'false':
                        sat_choice_done = False
                        # set sat_num = 0 to prompt different selection method
                        sat_num = 0
                    else:
                        print('Please enter a boolean.')
            else:
                print('Please input only the satellite number or the satellite name not both. ')
        
        by_num = {sat.model.satnum:sat for sat in sats}
        by_name = {sat.name:sat for sat in sats}
        satellites = {}
        for i in np.arange(0,num_satellites,1):
            if isinstance(sat_choice[i],int):
                try:
                    name_sat = by_num[sat_choice[i]].name
                    satellites[name_sat] = by_num[sat_choice[i]]
                    proj_area, rho_para = self.defineSatParameters(sat_choice[i])
                    sat_parameters[name_sat] = {}
                    sat_parameters[name_sat]['satellite approx projected area'] = proj_area
                    sat_parameters[name_sat]['surface rho properties'] = rho_para
                except:
                    print('Satellite number chosen was not in tle file. Moving to next satellite.')
            elif isinstance(sat_choice[i],str):
                try:
                    satellites[sat_choice[i]] = by_name[sat_choice[i]]
                    proj_area, rho_para = self.defineSatParameters(sat_choice[i])
                    sat_parameters[sat_choice[i]] = {}
                    sat_parameters[sat_choice[i]]['satellite approx projected area'] = proj_area
                    sat_parameters[sat_choice[i]]['surface rho properties'] = rho_para
                except:
                    print('Satellite name chosen was not in tle file. Moving to next satellite.')
        if list(satellites.keys()) == []:
            print('No satellites are were sucessfully selected. Please try the selection process again. ')
            
        return satellites, sat_parameters
        
    def define_sat(self,sat_num = 0, sat_epoch = 0, sat_drag = 0, sat_ball = 0, sat_meanmotddot = 0, sat_ecc = 0, sat_argofp = 0, sat_inc = 0, sat_meanan = 0, sat_meanmot = 0, sat_rightasc = 0 ):
        '''
        This method will generate a satellite and y defining orbital parameters
        to employ in orbital propagation. The code structure is from a 
        guide found in the documentation for SkyField at 
        https://rhodesmill.org/skyfield/earth-satellites.html#loading-a-tle-file
        

        Parameters
        ----------
        sat_num : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_epoch : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_drag : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_ball : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_meanmotddot : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_ecc : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_argofp : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_inc : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_meanan : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_meanmot : TYPE, optional
            DESCRIPTION. The default is 0.
        sat_rightasc : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        satellite : TYPE
            DESCRIPTION.

        '''
        
        satrec = Satrec()
        ts = load.timescale()
        
        if sat_num == 0:
            sat_num = input('Please input a catalog number.')
            if isinstance(sat_num,int):
                print('')
            else:
                print('Please input a number.')
            print('You have chosen a satellite number of {}.'.format(sat_num))
        else:
            print('There is a provided satellite number.')
        print('The satellite number is {}.'.format(sat_num))
        
        if sat_epoch == 0:
            sat_epoch = input('Please input a epoch (time since 31 DEC 1949 at 00:00 UT).')
            if isinstance(sat_epoch,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen an epoch of {}.'.format(sat_epoch))
        else:
            print('There is a provided satellite epoch.')
        print('The epoch is {}.'.format(sat_epoch))
        
        if sat_drag == 0:
            sat_drag = input('Please input a drag coefficient (/earth radii).')
            if isinstance(sat_drag,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen a drag coefficient of {}.'.format(sat_drag))
        else:
            print('There is a provided satellite drag coefficient.')
        print('The drag coefficient is {}.'.format(sat_drag))
        
        if sat_ball == 0:
            sat_ball = input('Please input a ballistic coefficient.')
            if isinstance(sat_ball,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen a ballistic coefficient of {}.'.format(sat_ball))
        else:
            print('There is a provided satellite ballistic coefficient.')
        print('The ballistic coefficient is {}.'.format(sat_ball))
        
        if sat_meanmotddot == 0:
            sat_meanmotddot = input('Please input a 2nd derivative of mean motion.')
            if isinstance(sat_meanmotddot,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen an 2nd derivative of {}.'.format(sat_meanmotddot))
        else:
            print('There is a provided 2nd derivative of mean motion.')
        print('The 2nd derivative of mean motion is {}.'.format(sat_meanmotddot))
        
        if sat_ecc == 0:
            sat_ecc = input('Please input an eccentricity of the orbit.')
            if isinstance(sat_ecc,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen an eccentricity of {}.'.format(sat_ecc))
        else:
            print('There is a provided orbit eccentricity.')
        print('The eccentricity is {}.'.format(sat_ecc))
        
        if sat_argofp == 0:
            sat_argofp = input('Please input an argument of perigee.')
            if isinstance(sat_argofp,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen an argument of perigee, {}.'.format(sat_argofp))
        else:
            print('There is a provided argument of perigee.')
        print('The argument of perigee is {}.'.format(sat_argofp))
        
        if sat_inc == 0:
            sat_inc = input('Please input an inclination of the orbit.')
            if isinstance(sat_inc,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen an inclination of {}.'.format(sat_inc))
        else:
            print('There is a provided inclination.')
        print('The inclination is {}.'.format(sat_inc))
        
        if sat_meanan == 0:
            sat_meanan = input('Please input a mean anomaly of the orbit.')
            if isinstance(sat_meanan,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen a mean anomaly of {}.'.format(sat_meanan))
        else:
            print('There is a provided mean anomaly.')
        print('The mean anomaly is {}.'.format(sat_meanan))
        
        if sat_meanmot == 0:
            sat_meanmot = input('Please input a mean motion of the orbit (radians/minute).')
            if isinstance(sat_meanmot,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen a mean motion of {}.'.format(sat_meanmot))
        else:
            print('There is a provided mean motion.')
        print('The mean motion is {}.'.format(sat_meanmot))
        
        if sat_rightasc == 0:
            sat_rightasc = input('Please input a right ascension of the ascending node of the orbit.')
            if isinstance(sat_rightasc,float):
                print('')
            else:
                print('Please input a float.')
            print('You have chosen a right ascension of the ascending node, {}.'.format(sat_rightasc))
        else:
            print('There is a provided right ascension.')
        print('The right ascension is {}.'.format(sat_rightasc))
        
        satrec.sgp4init(WGS72,'i',sat_num,sat_epoch,sat_drag,sat_ball,sat_meanmotddot,sat_ecc,sat_argofp,sat_inc,sat_meanan,sat_meanmot,sat_rightasc)
        
        satellite = sf.api.EarthSatellite.from_satrec(satrec,ts)
        return satellite
    
    def defineSatParameters(self,sat_id):
        '''
        Define the area and surface properties of a specific satellite

        Parameters
        ----------
        sat_id : int
            Norad satellite id to identify the parameters with the satellite.

        Returns
        -------
        proj_areas : list
            All corresponding areas associated with a satellite reflecting energy.
        surf_rho_prop : list
            All corresponding reflectivity properties associated with a satellite reflecting energy.

        '''
        # prompt user to input areas and surface properties
        sat_parameters_done = False
        proj_areas = []
        surf_rho_prop = []
        print('Please input satellite {} reflecting components (area and associated reflectivity) one at a time.'.format(sat_id))
        while sat_parameters_done == False:
            # Prompt user for Projected Area
            area = 'area'
            while isinstance(area, float) == False:
                area = input('Please input a float for the projected area component (m^2): ')
                try:
                    area = float(area)
                except:
                    print('Input was not a float. Please input a float for the projected area.')
            proj_areas.append(area)
            # Prompt user for Projected Area
            rho = 'rho'
            while isinstance(rho, float) == False:
                rho = input('Please input a float for the reflectivity of this component: ')
                try:
                    rho = float(rho)
                except:
                    print('Input was not a float. Please input a float for the reflectivity.')
            surf_rho_prop.append(rho)
            # Prompt user to exit
            sat_parameters_done = 'sat_parameters_done'
            while isinstance(sat_parameters_done, bool) == False:
                sat_parameters_done = input('Are you done defining satellite properties? (True or False): ')
                if sat_parameters_done == 'True' or sat_parameters_done == 'true':
                    sat_parameters_done = True
                elif sat_parameters_done == 'False' or sat_parameters_done == 'false':
                    sat_parameters_done = False
                else:
                    print('Please enter a boolean.')
        proj_areas = proj_areas*u.m*u.m
        
        return proj_areas, surf_rho_prop