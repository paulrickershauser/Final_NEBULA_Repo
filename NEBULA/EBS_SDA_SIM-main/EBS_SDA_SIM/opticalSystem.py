# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:39:07 2021

@author: rache
"""

import pdb
import numpy as np
import astropy.wcs as wcs
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, ICRS, SkyCoord
from skyfield.api import wgs84, N, W, Star, load
from skyfield.data import hipparcos
from astroquery.simbad import Simbad

class opticalSystem():
    
    
    def __init__(self,latitude = 0, longitude = 0, elevation = 0):
        self.define_observer(latitude = latitude, longitude = longitude, elevation = elevation)
        return
    
    def __str__(self):
        print('This is the optical system object')
        print('Outputs can include an observer object that can be paired with a satellite object to propagate the satellite across a focal plane.')
        return
    
    def define_observer(self,latitude = 0, longitude = 0, elevation = 0):
        '''
        Define the observer's location on Earth for Skyfield and Astropy transformations.

        Parameters
        ----------
        latitude : float, optional
            Latitude of the observing location in degrees. The default is 0.
        longitude : float, optional
            Longitude of the observing location in degrees. The default is 0.
        elevation : float, optional
            Elevation of the observing location in meters. The default is 0.

        Returns
        -------
        observer : skyfield object
            This object will have the latitude, longitude, and elevation which can then 
            be converted to other coordinate systems using Skyfield.

        '''
        
        if latitude == 0:
            latitude = self.__prompt_user_input('latitude of the observing location in degrees', float)
            print('{} is your selected latitude'.format(latitude))
        if longitude == 0:
            longitude = self.__prompt_user_input('longitude of the observing location in degrees', float)            
            print('{} is your selected longitude'.format(longitude))
        if elevation == 0:
            elevation = self.__prompt_user_input('elevation of the observing location in meters', float)            
            print('{} is your selected elevation'.format(elevation))
        
        observer = wgs84.latlon(latitude * N, longitude * W,elevation)
        earth_loc = EarthLocation(lat = latitude, lon = longitude, height = elevation, ellipsoid = 'WGS84')
        self.observer = observer
        self.earth_loc = earth_loc
        self.latitude = latitude*u.degree
        self.longitude = longitude*u.degree
        self.elevation = elevation*u.m
        return observer
    
    def define_optics(self,time_array,AFOV=0,tele_aperature = 0,tele_transmission=0,lim_mag=0,observation_plane_sampling=0,tfocus=0,eyefocus=0,azimuth=[],altitude=[],rightascension=[],declination=[],theta=[],dAz=[],dAlt=[],dRa=[],dDec=[],dTh=[],pixelheight=0,pixelwidth=0, pixel_num_y=0, pixel_num_x=0,fillfactor=0,wavelength=0,bandpass=0):
        '''
        Define the optical parameters of the observer to contain them the optical system
        object.
        A transformation dictionary of the optical pointing is generated to determine
        the objects in the field of view during the orbital simulation. User will be
        prompted for input on any parameters left with the default condition.

        Parameters
        ----------
        time_array : skyfield time object
            Object containing all time steps to be simulated .
        AFOV : float, optional
            Projected of view based on the optical train in degrees. The default is 0.
        tele_aperature : float, optional
            Telescope aperature in meters. The default is 0.
        tele_transmission : float, optional
            Percentage of energy transmission through the optical train between 0 and 1.
            The default is 0.
        lim_mag : float, optional
            Limiting magnitude of the sensor. The default is 0.
        observation_plane_sampling : int, optional
            Number of cells in the array sampling the observation plane (At the front 
            end of the optic). The default is 0.
        tfocus : float, optional
            Focal length of the telescope in meters. The default is 0.
        eyefocus : float, optional
            Focal length of the eyepiece in meters if used. The default is 0.
        azimuth : float, optional
            If Az/El telescope, degrees of the azimuthal setting at the start of the
            simulaiton. The default is [].
        altitude : float, optional
            If Az/El telescope, degrees of the altitude (elevation) setting at the start
            of the simulaiton. The default is [].
        rightascension : float, optional
            If RA/DEC telescope, degrees of the right ascension setting at the start of 
            the simulaiton. The default is [].
        declination : float, optional
            If RA/DEC telescope, degrees of the declination setting at the start of the
            simulaiton. The default is [].
        theta : float, optional
            The rotation of the sensor along the boresight in degrees. The default is [].
        dAz : float, optional
            If Az/El telescope, degrees per second of motion around the azimuthal axis
            during the simulation. The default is [].
        dAlt : float, optional
            If Az/El telescope, degrees per second of motion around the altitude 
            (elevation) axis during the simulation. The default is [].
        dRa : float, optional
            If RA/DEC telescope, degrees per second of motion around the right ascension
            axis during the simulation. The default is [].
        dDec : float, optional
            If RA/DEC telescope, degrees per second of motion around the declination
            axis during the simulation. The default is [].
        dTh : float, optional
            The rotation of the sensor along the boresight in degrees per second. The 
            default is [].
        pixelheight : float, optional
            The pixel height in meters. The default is 0.
        pixelwidth : float, optional
            The pixel width in meters. The default is 0.
        pixel_num_y : int, optional
            The number of pixels along the y axis. The default is 0.
        pixel_num_x : int, optional
            The number of pixels along the x axis. The default is 0.
        fillfactor : float, optional
            The percentage of photodiode to pixel area between 0 and 1. The default is 0.
        wavelength : float, optional
            The peak wavelength in the observed band in meters. The default is 0.

        Returns
        -------

        '''
        
        # Start by Defining the Dimensions of the Focal Plane                
        if pixel_num_y == 0:
            pixel_num_y = self.__prompt_user_input('number of pixels in the y direction of the focal array', int)
            print('{} is the number of pixels in the y direction of the focal plane array.'.format(pixel_num_y))
        self.num_y_pix = pixel_num_y
        
        if pixelheight == 0:
            pixelheight = self.__prompt_user_input('pixel height of the focal array', float, variable_units=u.m)
            print('{} is the pixel height of the focal plane array.'.format(pixelheight))
        pixelheight = self.__unit_check(pixelheight, u.m)
        self.pixel_height = pixelheight
            
        if pixel_num_x == 0:
            pixel_num_x = self.__prompt_user_input('number of pixels in the x direction of the focal array', int)
            print('{} is the number of pixels in the x direction of the focal plane array.'.format(pixel_num_x))   
        self.num_x_pix = pixel_num_x
            
        if pixelwidth == 0:
            pixelwidth = self.__prompt_user_input('pixel width of the focal array', float, variable_units=u.m)
            print('{} is the pixel width of the focal plane array.'.format(pixelwidth))          
        pixelwidth = self.__unit_check(pixelwidth, u.m)
        self.pixel_width = pixelwidth
        
        # if fillfactor == 0:
        #     fillfactor = self.__prompt_user_input('percentage of photodiode area on the pixel, fill factor, as a float between 0 and 1', float, upper_bound=1, lower_bound=0, bounds=True)
        # optics['fill factor'] = fillfactor
        
        self.pixel_total = pixel_num_x*pixel_num_y
        self.aspect_ratio = (pixel_num_y*pixelheight)/(pixel_num_x*pixelwidth)
        x_vector = np.arange(-pixel_num_x/2*pixelwidth.value,pixel_num_x/2*pixelwidth.value-pixelwidth.value,pixelwidth.value)*u.m
        y_vector = np.arange(-pixel_num_y/2*pixelheight.value,pixel_num_y/2*pixelheight.value-pixelheight.value,pixelheight.value)*u.m
        x_mat, y_mat = np.meshgrid(x_vector,y_vector)
        self.x_observation_grid_spacing = x_mat
        self.y_observation_grid_spacing = y_mat
        
        # Collect information about the camera for propagation
        # Assign the telescope aperature diameter as a limiting factor for the propagation dimensions
        if tele_aperature == 0:
            tele_aperature = self.__prompt_user_input('telescope or lens aperature diameter', float, variable_units=u.m)
            print('{} is the telescope aperature diameter.'.format(tele_aperature))  
        tele_aperature = self.__unit_check(tele_aperature, u.m)
        self.aperature_diameter = tele_aperature
        
        # Collect information about the sampling of the observation plane
        if observation_plane_sampling == 0:
            samp_choice = 'no choice'
            while not isinstance(samp_choice,bool) == True:
                samp_choice = input('Do you want the subsampling of the observation plane to be done at the pixel pitch? (True or False): ')
                if samp_choice == 'True' or samp_choice == 'true':
                    samp_choice = True
                elif samp_choice == 'False' or samp_choice == 'false':
                    samp_choice = False
                else:
                    print('Please enter a boolean.')
            if samp_choice == True:
                # Use the smallest pixel dimension to determine the sampling
                if pixelwidth >= pixelheight:
                    obs_sample = pixelheight
                else:
                    obs_sample = pixelwidth
                observation_plane_sampling = int(np.ceil(tele_aperature/obs_sample))
                if observation_plane_sampling % 2 != 0:
                    observation_plane_sampling += 1
            else:
                observation_plane_sampling = 'observation_plane_sampling'
                while not isinstance(observation_plane_sampling, int) == True:
                    observation_plane_sampling = input('Please input an even integer for the observation plane sampling: ')
                    try:
                        observation_plane_sampling = int(observation_plane_sampling)
                        if observation_plane_sampling % 2 != 0:
                            observation_plane_sampling += 1
                        break
                    except:
                        print('PLease enter an even integer for the observation plane sampling.')
        self.observation_plane_sampling = observation_plane_sampling      
        
        if tfocus == 0:
            tfocus = self.__prompt_user_input('focus length of the telescope or lens', float, variable_units=u.m)
            print('{} is your selected telescope focus length'.format(tfocus))
        tfocus = self.__unit_check(tfocus, u.m)
        self.focal_length = tfocus
        self.f_number = ((tfocus)/(tele_aperature)).decompose()
        
        if wavelength == 0:
            wavelength = self.__prompt_user_input('wavelength of peak intensity being observed. Note Gia G Passband has an effective filter wavelength of 0.673um (6.73e-7m). The value should be input', float, variable_units=u.m)
            print('{} is the wavelength of light being observed.'.format(wavelength))          
        wavelength = self.__unit_check(wavelength, u.m)
        self.wavelength = wavelength
        self.wavenumber = (2*np.pi*u.rad)/(wavelength)
        
        if bandpass == 0:
            bandpass = self.__prompt_user_input('width of the bandpass observed by the focal plane', float, variable_units=u.Angstrom)
            print('{} is the width of the bandpass of light being observed.'.format(bandpass))
        bandpass = self.__unit_check(bandpass, u.Angstrom)
        self.bandpass = bandpass
        
        # if tele_transmission == 0:
        #     tele_transmission  = 'telescopetransmission'
        #     while not isinstance(tele_transmission,float) == True:
        #         tele_transmission  = input('Please enter the transmission (percentage of energy transmitted) of the telescope or lens between 0 and 1 at the modeled wavelength: ')
        #         try:
        #             tele_transmission  = float(tele_transmission)
        #             if tele_transmission >= 0 and tele_transmission <= 1:
        #                 break
        #             else:
        #                 print('Transmission value is not between 0 and 1. Please input a different value.')
        #                 tele_transmission = 'telescopetransmission'
        #         except:
        #             print('Please enter a float or integer for the focal length.')
        #     print('{} is your selected telescope focus length'.format(tele_transmission))
        # optics['optical transmission'] = tele_transmission
        
        # Use the wavelength definition to define the angular resolution and the spatial
        # equivalent of the airy disk created by this optic
        self.angular_resolution = (1.22*wavelength/(tele_aperature)).decompose()
        # The spatial resolution is the full diameter of the airy disk
        self.spatial_resolution = (2.44*tfocus*wavelength/(tele_aperature)).decompose()
        
        # Check the relative size of the airy disk diameter to the size of the pixels
        # Check largest pixel dimension, it must have at least 4 pixels spanning the 
        # airy disk diameter defined by the spatial resolution
        if pixelwidth >= pixelheight:
            max_pixel_dim = pixelwidth
        else:
            max_pixel_dim = pixelheight 
        if self.spatial_resolution/max_pixel_dim < 4:
            self.sufficient_airy_disk_discretization = False
            # Then subsample the pixel plane until a chosen gamma is reached (at least 4)
            num_pix_per_airy_disk = self.__prompt_user_input('number of subpixels that should span the airy disk. This number should be even. Floats will be rounded. Odd numbers will have one added to it', int)
            if num_pix_per_airy_disk % 2 != 0:
                num_pix_per_airy_disk += 1
            self.gamma = num_pix_per_airy_disk
            # Take the number of subpixels spanning the airy disk to be gamma
            # Calculate the number of airy disks that span a pixel and multiply by gamma
            # To determine the subdividion of pixels in the x direction
            subpixel_per_pixel_x = int(np.ceil((pixelwidth*num_pix_per_airy_disk)/self.spatial_resolution))
            if subpixel_per_pixel_x % 2 != 0:
                subpixel_per_pixel_x += 1
            self.subpixel_per_pixel_x_direction = subpixel_per_pixel_x
            image_array_total_subpixels_x_direction = int(subpixel_per_pixel_x*pixel_num_x)
            subpixel_per_pixel_y = int(np.ceil((pixelheight*num_pix_per_airy_disk)/self.spatial_resolution))
            if subpixel_per_pixel_y % 2 != 0:
                subpixel_per_pixel_y += 1
            self.subpixel_per_pixel_y_direction = subpixel_per_pixel_y
            image_array_total_subpixels_y_direction = int(subpixel_per_pixel_y*pixel_num_y)
            # Ask the user how many of the regular pixels should be simulated in the 
            # image plane to conserve computational resources
            sim_num_pix_x = self.__prompt_user_input('even number of the image array pixels in the x direction that should be simulated. Simulating the whole array with subpixel discretization creates an [{'+str(image_array_total_subpixels_y_direction)+'},{'+str(image_array_total_subpixels_x_direction)+'}] array', int)
            if sim_num_pix_x % 2 != 0:
                sim_num_pix_x += 1
            sim_num_pix_y = self.__prompt_user_input('even number of the image array pixels in the y direction that should be simulated. Simulating the whole array with subpixel discretization creates an [{'+str(image_array_total_subpixels_y_direction)+'},{'+str(image_array_total_subpixels_x_direction)+'}] array', int)
            if sim_num_pix_y % 2 != 0:
                sim_num_pix_y += 1
            self.image_plane_pixels_simulated_x_direction = sim_num_pix_x
            self.image_plane_pixels_simulated_y_direction = sim_num_pix_y
            self.mx = sim_num_pix_x*subpixel_per_pixel_x/num_pix_per_airy_disk
            self.my = sim_num_pix_y*subpixel_per_pixel_y/num_pix_per_airy_disk
        else:
            self.sufficient_airy_disk_discretization = True
            pixels_in_airy = int(np.ceil(self.spatial_resolution/pixelwidth))
            if pixels_in_airy % 2 != 0:
                pixels_in_airy += 1
            self.gamma = pixels_in_airy
            if pixel_num_x >= pixel_num_y:
                sim_num_pix = pixel_num_x
            else:
                sim_num_pix = pixel_num_y
            self.image_plane_pixels_simulated_x_direction  = sim_num_pix
            self.image_plane_pixels_simulated_y_direction  = sim_num_pix
            self.mx = sim_num_pix/pixels_in_airy
            self.my = sim_num_pix/pixels_in_airy
        
        if lim_mag == 0:
            lim_mag = self.__prompt_user_input('limiting magnitude of the stars in Gaia G Broadband that can be sensed by the optics (9.8 for DVS)', float, variable_units=u.mag)
            print('{} is the limiting magnitude of the stars that the camera can sense.'.format(lim_mag))
        lim_mag = self.__unit_check(lim_mag,u.mag)
        self.limiting_magnitude  = lim_mag
        
        # Either Calculate or Define the Field of View
        # Ask user which they would prefer.
        # 1) If working from a dataset, they may know the subtended angle from the data
        # 2) If details of the optical train are known then the FOV can be derived from that
        FOV_choice = 'no choice'
        FOV_choice_list = ['Angle','angle','Optic','optic']
        while FOV_choice not in FOV_choice_list:
            FOV_choice = input('Do you have an angle to define the field of view or optical specifications? (angle or optic): ')
            if FOV_choice not in FOV_choice_list:
                print('Please choose either angle or optic.')
                
        if FOV_choice == 'angle' or FOV_choice == 'Angle':
            Rect = self.__prompt_user_input('boolean to determine if the subtended angle should be assumed square (True or False)', bool)
            self.Rectangle_Subtended_Angle = Rect
            if Rect == True:
                h_deg = self.__prompt_user_input('solid angle associated with the FOV height', float, variable_units=u.degree)
                print('{} is your selected solid angle associated with the FOV height.'.format(h_deg))
                self.height_fov = h_deg
                w_deg = self.__prompt_user_input('solid angle associated with the FOV width', float, variable_units=u.degree)
                print('{} is your selected solid angle associated with the FOV width.'.format(w_deg))
                self.width_fov = w_deg*u.deg
                solid_angle = np.deg2rad(h_deg.value)*np.deg2rad(w_deg.value)*u.steradian
                print('The solid angle subtended by this sensor is {} [sr]'.format(solid_angle))
                self.AFOV = solid_angle
            else:
                sub_deg = self.__prompt_user_input('solid angle associated with the FOV', float, variable_units=u.degree)
                print('{} is your selected solid angle associated with the FOV height.'.format(sub_deg))
                self.round_fov = sub_deg
                solid_angle = 2*np.pi*(1-np.cos(np.deg2rad(sub_deg.value)/2))*u.steradian
                print('The solid angle subtended by this sensor is {} [sr]'.format(solid_angle))
                self.AFOV = solid_angle
        if FOV_choice == 'optic' or FOV_choice == 'Optic':
            # Use the larger dimension (height or width) of the focal array to define the AFOV
            if pixel_num_y*pixelheight > pixel_num_x*pixelwidth:
                H = pixel_num_y*pixelheight
            else:
                H = pixel_num_x*pixelwidth
            AFOV = 2*np.arctan(H/(2*tfocus))
            self.AFOV  = np.rad2deg(AFOV)*u.deg
        
        
        # Collect pointing of the camera information, to connect ICRS coordinates 
        # to the focal plane at each timestep.
        if azimuth != [] and altitude != [] and theta != [] and dAz != [] and dAlt != [] and dTh != []:
            # Mount is Az/El or the coordinates have been converted. 
            # This is ideal for assigning an Az/El for each pixel
            self.mount_type  = 'AltAz'
            azimuth = self.__unit_check(azimuth, u.deg)
            self.azimuth  = azimuth
            altitude = self.__unit_check(altitude, u.deg)
            self.altitude  = altitude
            theta = self.__unit_check(theta, u.deg)
            self.rotation  = theta
            dAz = self.__unit_check(dAz, u.deg/u.s)
            self.Az_slew_rate  = dAz
            dAlt = self.__unit_check(dAlt, u.deg/u.s)
            self.Alt_slew_rate  = dAlt
            dTh = self.__unit_check(dTh, u.deg/u.s)
            self.rotation_rate  = dTh
        elif rightascension != [] and declination != [] and theta != [] and dRa != [] and dDec != [] and dTh != []:
            # The RA/DEC mount needs a bit more care to understand the 
            # orientation of the optic with respect to the FOV.
            self.mount_type  = 'RaDec'
            rightascension = self.__unit_check(rightascension, u.deg)
            self.right_ascension  = rightascension
            declination = self.__unit_check(declination, u.deg)
            self.declination  = declination
            theta = self.__unit_check(theta, u.deg)
            self.rotation  = theta
            dRa = self.__unit_check(dRa, u.deg/u.s)
            self.Ra_slew_rate  = dRa
            dDec = self.__unit_check(dDec, u.deg/u.s)
            self.Dec_slew_rate  = dDec
            dTh = self.__unit_check(dTh, u.deg/u.s)
            self.rotation_rate  = dTh
        else:
            mountselect = False
            while mountselect == False:
                mounttype = input('Please enter the mount type, AltAz or RaDec: ')
                if mounttype == 'AltAz':
                    mountselect = True
                    self.mount_type  = 'AltAz'
                    self.azimuth = self.__prompt_user_input('initial azimuth of the telescope. Note 0 degrees is towards the clestial North Pole. The value should be reported', float, variable_units=u.deg, upper_bound=360, lower_bound=0, bounds=True)
                    self.altitude = self.__prompt_user_input('initial altitude of the telescope', float, variable_units=u.deg, upper_bound=90, lower_bound=0, bounds=True)
                    self.rotation = self.__prompt_user_input('initial rotation of the camera with respect to the clestial sphere lines of declination', float, variable_units=u.deg, upper_bound=360, lower_bound=0, bounds=True)
                    self.Az_slew_rate = self.__prompt_user_input('slew rate for the azimuth axis of the telescope', float, variable_units=u.deg/u.s)
                    self.Alt_slew_rate = self.__prompt_user_input('slew rate for the altitude axis of the telescope', float, variable_units=u.deg/u.s)
                    self.rotation_rate = self.__prompt_user_input('slew rate for the camera along its nadir axis', float, variable_units=u.deg/u.s)
                elif mounttype == 'RaDec':
                    mountselect = True
                    self.mount_type  = 'RaDec'
                    self.right_ascension = self.__prompt_user_input('initial right ascension of the telescope', float, variable_units=u.deg, upper_bound=360, lower_bound=0, bounds=True)
                    self.declination = self.__prompt_user_input('initial declination of the telescope', float, variable_units=u.deg, upper_bound=90, lower_bound=-90, bounds=True)
                    self.rotation = self.__prompt_user_input('initial rotation of the camera with respect to the clestial sphere lines of declination. Note 0 degrees is towards the clestial North Pole. The value should be reported', float, variable_units=u.deg, upper_bound=360, lower_bound=0, bounds=True)
                    self.Ra_slew_rate = self.__prompt_user_input('slew rate for the right ascension axis of the telescope', float, variable_units=u.deg/u.s)
                    self.Dec_slew_rate = self.__prompt_user_input('slew rate for the declination axis of the telescope,', float, variable_units=u.deg/u.s)
                    self.rotation_rate = self.__prompt_user_input('slew rate for the camera along its nadir axis', float, variable_units=u.deg/u.s)
        
        # Create a dictionary of transformation objects instead of a mapping array
        # at each timestep.
        # Create a dictionary of stars in the frame at each timestep
        optics_pointing_dict, optics_wcs_dict, optics_star_dict, optics_star_mag_dict, optics_star_id_dict = self.optics_propagation(time_array)
        # Store these dictionaries in the optics object
        self.optics_pointing_dict = optics_pointing_dict
        self.optics_wcs_dict = optics_wcs_dict
        self.optics_star_dict = optics_star_dict
        self.optics_star_mag_dict = optics_star_mag_dict
        self.optics_star_id_dict = optics_star_id_dict
                        
        return
    
    def define_proj_object(self,ra0,dec0,rotation):
        '''
        Defines a World Coordinate System that describes a gnomonic projection of ICRS to the local focal plane frame for each point in time of the simulation.

        Parameters
        ----------
        ra0 : float
            Right ascension of the telescope pointing (center of FOV) in degrees.
        dec0 : float
            Declination of the telescope pointing (center of FOV) in degrees.
        rotation : float
            Rotation of the telescope pointing in degrees.
        optics : dict
            Dictionary of optical properties of the observer as defined in self.define_optics.

        Returns
        -------
        Frame_2_ICRS : wcs object
            World Coordinate System that enables translation of RA/DEC and pixel coordinates at this timestep.

        '''
        # Unpack relevant optics conditions including pointing
        radius = self.AFOV  # Radius of the projection in degrees
        width = self.num_x_pix  # Number of pixels in width. NAXIS1
        height = self.num_y_pix  # Number of pixels in hieght. NAXIS2     
        
        # Create a new world coordinate system.
        Frame_2_ICRS = wcs.WCS()
        
        # Image dimensions.
        Frame_2_ICRS.naxis = 2
        Frame_2_ICRS.naxis1 = width
        Frame_2_ICRS.naxis2 = height
        
        # Gnomonic projection.
        Frame_2_ICRS.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        
        # RA, DEc of reference point (must be in degrees).
        Frame_2_ICRS.wcs.cunit = ['deg', 'deg']
        Frame_2_ICRS.wcs.crval = [ra0.value, dec0.value] # Tangential, reference, point of the projection
        
        # Pixel coordinates of reference point. Note that in WCS integer pixel indices
        # refer to the center of the pixel, and that WCS assumes 1-based indexing.
        crpix1 = 0.5 * (width + 1) # Centered on the focal plane
        crpix2 = 0.5 * (height + 1) # Centered on the focal plane
        Frame_2_ICRS.wcs.crpix = [crpix1, crpix2]
        Frame_2_ICRS.tunit = ['prob deg-2']
        
        # Apply the pixel coordinate rotation matrix to align the stars with constant lines of latitude
        rotation_rad = np.deg2rad(rotation)
        cd1 = np.cos(rotation_rad)
        cd2 = np.sin(rotation_rad)
        Frame_2_ICRS.pc = [[cd1, -cd2],[cd2, cd1]]
        
        # Coordinate increment at reference pixel, so that the circle with the given
        # radius passes through the centers of the pixels on the edge of the image.
        #
        # Valid for a gnomonic ('TAN') projection. See Equation (54) of
        # Calabretta & Greisen (2002, http://dx.doi.org/10.1051/0004-6361:20021327).
        cdelt = np.rad2deg(np.tan(np.deg2rad(radius.value/2))) / (0.5 * (width - 1))
        Frame_2_ICRS.wcs.cdelt = [-cdelt, cdelt]
        
        # Use ICRS equatorial coordinates.
        Frame_2_ICRS.wcs.radesys = 'ICRS'
        
        return Frame_2_ICRS
    
    def convert_alt_az_2_ra_dec(self,Alt,Az,time_inc):
        '''
        Conversion from Alt/Az to RA/DEC is conducted via astropy's SkyCoord.

        Parameters
        ----------
        Alt : float
            Local altitude of the telescope pointing (center of FOV) in degrees.
        Az : float
            Local azimuth of the telescope pointing (center of FOV) in degrees.
        time_inc : float
            Time increment from the previous time step simulated.

        Returns
        -------
        Ra : float
            Right ascension of the telescope pointing (center of FOV) in degrees.
        Dec : float
            Declination of the telescope pointing (center of FOV) in degrees.

        '''
        # Create an astropy Alt/Az object
        # Need the earth location since AltAz is defined locally
        earth_loc = self.earth_loc
        # Need to define an astropy time object since the location of earth changes with respect to ICRS
        time_str = str(time_inc[0]) + '-' + str(time_inc[1]) + '-' + str(time_inc[2]) + 'T' + str(time_inc[3]) + ':' + str(time_inc[4]) + ':' + str(time_inc[5])
        time_now = Time(time_str, format='isot', scale='utc')
        # Define the current AltAz frame
        AltAz_coord = SkyCoord(AltAz(obstime = time_now,az=Az,alt=Alt,location=earth_loc))
        RaDec_coord = AltAz_coord.transform_to(ICRS)
        Ra = RaDec_coord.ra.degree*u.deg
        Dec = RaDec_coord.dec.degree*u.deg
        return Ra, Dec
    
    def convert_ra_dec_2_alt_az(self,Ra,Dec,time_inc):
        '''
        Conversion from RA/DEC to Alt/Az is conducted via astropy's SkyCoord.

        Parameters
        ----------
        Ra : float
            Right ascension of the telescope pointing (center of FOV) in degrees.
        Dec : float
            Declination of the telescope pointing (center of FOV) in degrees.
        time_inc : float
            Time increment from the previous time step simulated.

        Returns
        -------
        Alt : float
            Local altitude of the telescope pointing (center of FOV) in degrees.
        Az : float
            Local azimuth of the telescope pointing (center of FOV) in degrees.

        '''
        # Create an astropy Alt/Az object
        # Need the earth location since AltAz is defined locally
        earth_loc = self.earth_loc
        # Need to define an astropy time object since the location of earth changes with respect to ICRS
        time_str = str(time_inc[0]) + '-' + str(time_inc[1]) + '-' + str(time_inc[2]) + 'T' + str(time_inc[3]) + ':' + str(time_inc[4]) + ':' + str(time_inc[5])
        time_now = Time(time_str, format='isot', scale='utc')
        # Define the current AltAz frame
        aa = AltAz(obstime = time_now,location=earth_loc)
        RaDec_coord = SkyCoord(ra = Ra, dec = Dec, frame = ICRS)
        AltAz_coord = RaDec_coord.transform_to(aa)
        Alt = AltAz_coord.alt.degree*u.deg
        Az = AltAz_coord.az.degree*u.deg
        return Alt, Az
        
    def create_star_objects(self, Ra, Dec, df_stars):
        '''
        From the current pointing of the observer, the FOV, and the limiting magnitude
        reduce the stars in the Hipparcos catalog to those possibly seen by the sensor.

        Parameters
        ----------
        Ra : float
            Right ascension of the telescope pointing (center of FOV) in degrees.
        Dec : float
            Declination of the telescope pointing (center of FOV) in degrees.
        optics : dict
            Dictionary of optical properties of the observer as defined in self.define_optics.
        df_stars : dataframe
            A custom list of stars within the FOV during the simulation and reduced by the limiting magnitude.

        Returns
        -------
        bright_stars_in_FOV : starlib object
            Contains time and propagation information of the relevant stars.
        star_magnitudes : list
            Ordered list of the star magnitudes in the FOV.
        star_ids : list
            Ordered list of the star identification number in the hipparcos catalog.

        '''
            
        # From the reduced list of stars
        # Reduce them further for each timestep, by filtering by the step's
        # right ascension and declination of pointing in ICRS + AFOV
        AFOV = self.AFOV 
        
        ra_max = Ra.value + AFOV.value/2
        ra_min = Ra.value - AFOV.value/2
        dec_max = Dec.value + AFOV.value/2
        dec_min = Dec.value - AFOV.value/2
        
        stars_bounded = df_stars[df_stars['ra_degrees'] < ra_max] 
        stars_bounded = stars_bounded[stars_bounded['ra_degrees'] > ra_min]
        stars_bounded = stars_bounded[stars_bounded['dec_degrees'] < dec_max]
        stars_bounded = stars_bounded[stars_bounded['dec_degrees'] > dec_min]
        
        bright_stars_in_FOV = Star.from_dataframe(stars_bounded)  
        star_magnitudes = list(stars_bounded['Gia G Mag'])
        star_ids = list(stars_bounded['Star ID'])
        
        return bright_stars_in_FOV, star_magnitudes, star_ids
    
    def optics_propagation(self, time_array):
        '''
        Using the timestep, definition of the pointing and slew rates, step the pointing state forward to define the new time step's WCS projection.

        Parameters
        ----------
        time_array : skyfield.timelib.CalendarArray
            Array with all the time steps defined.

        Returns
        -------
        optics_pointing_dict : dict
            Dictionary storing the pointing information for each timestep.
        optics_wcs_dict : dict
            Dictionary storing the WCS object for each timestep.
        optics_star_dict : dict
            Dictionary storing the stars in the FOV for each timestep.
        optics_star_mag_dict : dict
            Dictionary storing the star magnitudes for each timestep.
        optics_star_id_dict : dict
            Dictionary storing the star idenitification number in the Hipparcos catalog for each timestep.

        '''
        # If the optics are moving, use this function to propagate the slew
        # and establish a new pointing vector for the optic
        # Unpack Optics Definition
        # Type of mount determines the Euler Equation being used
        # Create the pointing dictionary in ICRS coordinates
        # Create a dictionary of wcs objects to map current coordinates to pixel coordinates
        optics_pointing_dict = {}
        optics_wcs_dict = {}
        optics_star_dict = {}
        optics_star_id_dict = {}
        optics_star_mag_dict = {}
        if self.mount_type == 'AltAz':
            # Propagate the azimuth and elevation to create pointing lists
            Az,Alt,Th = self.euler_propagate_pointing(self.azimuth, self.altitude, self.rotation, self.Az_slew_rate, self.Alt_slew_rate, self.rotation_rate, time_array)
            # Convert all azimuth and elevation to right ascension and declination lists
            Ra = []
            Dec = []
            for i in np.arange(0,len(Az)):
                Ra_i, Dec_i = self.convert_alt_az_2_ra_dec(Alt[i],Az[i],time_array[i].utc)
                Ra.append(Ra_i)
                Dec.append(Dec_i)
        elif self.mount_type == 'RaDec':
            # Propagate the right ascension and delincation to create pointing dictionaries
            Ra,Dec,Th = self.euler_propagate_pointing(self.right_ascension, self.declination, self.rotation, self.Ra_slew_rate, self.Dec_slew_rate, self.rotation_rate, time_array)
        # Create a catalog of stars within the FOV of the whole simulation
        df_stars = self.create_star_catalog(Ra[0],Ra[-1],Dec[0],Dec[-1])
        # Iterate through the pointing lists
        for i in np.arange(0,len(Ra)):
            # Store Ra and Dec
            optics_pointing_dict[i] = (Ra[i],Dec[i])
            # Create and store the wcs object to convert between pixels and RA/DEC
            ICRS_2_Frame = self.define_proj_object(Ra[i],Dec[i],Th[i])
            optics_wcs_dict[i] = ICRS_2_Frame
            # Create and store the star objects relevant to this timestep
            bright_stars_in_FOV, star_magnitudes, star_ids = self.create_star_objects(Ra[i],Dec[i],df_stars)
            if i == 0:
                optics_star_dict[i] = bright_stars_in_FOV
            else:
                # Only store all the star objects if it is a different set of stars. Otherwise, point to the dictionary item that has the appropriate stars
                new_stars = True
                for j in np.arange(0,len(optics_star_id_dict.keys())):
                    if optics_star_id_dict[j] == star_ids:
                        optics_star_dict[i] = j
                        new_stars = False
                        break
                if new_stars == True:
                    optics_star_dict[i] = bright_stars_in_FOV
            optics_star_mag_dict[i] = star_magnitudes
            optics_star_id_dict[i] = star_ids
            
        return optics_pointing_dict, optics_wcs_dict, optics_star_dict, optics_star_mag_dict, optics_star_id_dict
    
    def euler_propagate_pointing(self,st_1,st_2,st_3,dst_1,dst_2,dst_3,t_array):
        '''
        Propagate the pointing of the optical system through the simulation duration.

        Parameters
        ----------
        st_1 : astropy.units.quantity.Quantity
            Initial state of angle 1.
        st_2 : astropy.units.quantity.Quantity
            Initial state of angle 2.
        st_3 : astropy.units.quantity.Quantity
            Initial state of angle 3.
        dst_1 : astropy.units.quantity.Quantity
            Initial rate of angle 1.
        dst_2 : astropy.units.quantity.Quantity
            Initial rate of angle 2.
        dst_3 : astropy.units.quantity.Quantity
            Initial rate of angle 3.
        t_array : skyfield time object
            Object containing all time steps to be simulated .

        Returns
        -------
        state_1 : list
            list of all values of state 1 through the simulation.
        state_2 : list
            list of all values of state 2 through the simulation.
        state_3 : list
            list of all values of state 3 through the simulation.

        '''
        # Create lists for each state parameter
        state_1 = []
        state_2 = []
        state_3 = []
        for i in np.arange(0,len(t_array)):
            if i != 0:
                # Calculate time difference since previous step
                t_step = t_array.utc[:,i] - t_array.utc[:,i-1]
                # Convert difference into seconds
                if t_step[1] != 0:
                    # If there are months in the time object covert it to days
                    t_step_days = self.month2day(t_step)
                    t_step = (t_step[0]*u.year+t_step_days+t_step[3]*u.hour+t_step[4]*u.minute+t_step[5]*u.second).decompose()
                else:
                    t_step = (t_step[0]*u.year+t_step[2]*u.day+t_step[3]*u.hour+t_step[4]*u.minute+t_step[5]*u.second).decompose()
                # Multiply the time difference by the rate of change to update the 
                st_1 = st_1 + dst_1*t_step
                st_2 = st_2 + dst_2*t_step
                st_3 = st_3 + dst_3*t_step
                # Ensure the angles are wrapped
                st_1 = self.wrap_angle(st_1)
                st_2 = self.wrap_angle(st_2)
                st_3 = self.wrap_angle(st_3)
            # Append the updated state to the list
            state_1.append(st_1)
            state_2.append(st_2)
            state_3.append(st_3)
        return state_1, state_2, state_3
            
    def wrap_angle(self,angle):
        '''
        Wrap angle to ensure its value is between 0 and 360 degrees.

        Parameters
        ----------
        angle : astropy.units.quantity.Quantity
            Angle that may need to be wrapped.

        Returns
        -------
        angle : astropy.units.quantity.Quantity
            Corrected angle if needed.

        '''
        if angle.value > 360:
            angle = (angle.value-360)*angle.unit
        elif angle.value < 0:
            angle = (angle.value+360)*angle.unit
        return angle
    
    def create_star_catalog(self,RA_o,RA_f,DEC_o,DEC_f):
        '''
        Create the star catalog to use for the simulation that is limited by the extents
        of the right ascension and declination covered and the limiting magnitude of the
        optical system.

        Parameters
        ----------
        RA_o : astropy.units.quantity.Quantity
            Right ascension at first time step.
        RA_f : astropy.units.quantity.Quantity
            Right ascension at last time step.
        DEC_o : astropy.units.quantity.Quantity
            Declination at first time step.
        DEC_f : astropy.units.quantity.Quantity
            Declination at last time step.

        Returns
        -------
        star_catalog : pandas dataframe
            List of stars within the field of view and under the limiting magnitude.

        '''
            
        # Determine the maximum and minimum RA/DEC values from the two extreme 
        # timeframes
        AFOV = self.AFOV 
        # Determine the limiting RA/DEC values
        ra_max = np.max([RA_o.value + AFOV.value/2,RA_f.value + AFOV.value/2])
        ra_min = np.min([RA_o.value - AFOV.value/2,RA_f.value - AFOV.value/2])
        dec_max = np.max([DEC_o.value + AFOV.value/2,DEC_f.value + AFOV.value/2])
        dec_min = np.min([DEC_o.value - AFOV.value/2,DEC_f.value - AFOV.value/2])
        
        # Download the hipparcos catalog
        with load.open(hipparcos.URL) as f:
            df_hipparcos = hipparcos.load_dataframe(f)
        
        # Remove stars that do not have RA,DEC in ICRS
        df_hipparcos = df_hipparcos[df_hipparcos['ra_degrees'].notnull()]
        
        # Reduce the stars being considered for a Simbad Query
        stars_bounded = df_hipparcos[df_hipparcos['ra_degrees'] < ra_max] 
        stars_bounded = stars_bounded[stars_bounded['ra_degrees'] > ra_min]
        stars_bounded = stars_bounded[stars_bounded['dec_degrees'] < dec_max]
        stars_bounded = stars_bounded[stars_bounded['dec_degrees'] > dec_min]
        
        # Query SIMBAD database to pull more recent Johnson V and Gaia G Broadband
        # for each HIP object
        hip_num = hip_num = list(stars_bounded.index)
        for i, num in enumerate(hip_num):
            hip_num[i] = 'HIP '+str(num)
        Simbad.add_votable_fields('flux(V)','flux(G)')
        result_table = Simbad.query_objects(hip_num)
        
        # Add Gia G, Updated Johnson V, and the Star Name to the data frame
        stars_bounded.insert(1,'Gia G Mag',list(result_table['FLUX_G']))
        stars_bounded.insert(1,'Johnson V Mag',list(result_table['FLUX_V']))
        stars_bounded.insert(1,'Star ID',list(result_table['MAIN_ID']))
        del result_table
        
        # Reduce the dataframe to the stars under the limiting magntiude of the sensor
        star_catalog = stars_bounded[stars_bounded['Gia G Mag']<=self.limiting_magnitude]
        
        return star_catalog
        
    def __prompt_user_input(self,variable_name,variable_type,variable_units=0,upper_bound=0,lower_bound=0,bounds=False):
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
                user_input = input('Please input a {} for the {} in units of {}: '.format(str(variable_type),variable_name, str(variable_units)))
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
                user_input = input('Please input a {} for the {}: '.format(str(variable_type),variable_name))
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
    
    def __unit_check(self,variable,variable_unit):
        '''
        Check whether the variable has units assigned to it. If not, apply the 
        appropriate units.

        Parameters
        ----------
        variable : various
            Variable being checked for astropy units.
        variable_unit : u.quantity.Quantity
            Astropy unit to be applied to the variable.

        Returns
        -------
        variable : TYPE
            DESCRIPTION.

        '''
        if type(variable) != u.quantity.Quantity:
            variable = variable*variable_unit
        elif variable.unit != variable_unit:
            try:
                variable = variable.to(variable_unit)
            except:
                print('Variable with units {} cannot be converted into {}'.format(variable.unit, variable_unit))
        return variable
        