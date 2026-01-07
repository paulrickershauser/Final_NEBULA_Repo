# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 08:48:17 2022

@author: rache
"""

from radiometry import RadiometryModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy import units as u
import pdb
import pickle
import os

class propTest(RadiometryModel):
    '''
    This class contains differnent methods to easily set up and run tests of the propagation functions
    '''
    def __init__(self,**kwargs):
        return
    
    def __str__(self):
        print('This object contains standard methods for running EBS SDA.')
        return
    
    def ptSourceVacProp(self):
        '''
        Test case for a vacuum propagation with a central point source.

        Returns
        -------
        None.

        '''
        # Initial inputs taken from Schmidt 2010 (pg.175- ):
        D2 = 0.5*u.m # observation aperature [m]
        wvl = 1e-6*u.m # optical wavelength [m]
        k = 2*np.pi*u.rad / wvl # optical wavenumber [rad/m]
        Dz = 50e3*u.m # propagation distance [m]
        d1 = .01*u.m
        d2 = .01*u.m
        N = 512 # grid points
        DROI = 4*D2 # diameter of observation-plane region of interest [m]
        arg = DROI/(wvl*Dz) # ensure U units are sqrt(W)/m [1/m]
        D1 = wvl*Dz/DROI # width of central lobe [m]
        R = Dz # wavefront radius of curvature [m]
        nscrn = 11 # number of screens
        
        # create meshgrid of all coordinates for the source plane
        delta1 = d1 # source-plane grid spacing
        delta2 = d2 # observation-plane grid spacing
        n = nscrn # number of planes
        
        grid = np.arange(-N/2+0.5,N/2+.05)*delta1 # grid units [m]
        x1, y1 = np.meshgrid(grid,grid,copy=False) # mesh units [m]
        # convert to polar coordinates
        r1, theta1 = self.cart2pol(x1,y1)
        
        # Define the point source
        power = 1*u.W
        A = wvl*Dz*np.sqrt(power)/u.m # Scaled to the squareroot of power/meter^2, overall unit [sqrt(W)*m]
        sinc_x = np.sinc((x1/D1)*u.rad) # Unitless
        #sinc_x_2 = np.sinc((x1*arg).value) #Unitless
        sinc_y = np.sinc((y1/D1)*u.rad) # Unitless
        #sinc_y_2 = np.sinc((y1*arg).value) #Unitless
        phase_dim = np.exp(-(r1/(4*D1))**2) # Unitless
        #phase_dim_2 = np.exp(-((r1*arg)/4)**2) # Unitless
        phase_spat = np.exp(((-1j*k)/(2*R)*r1**2).value)/D1**2 # [1/m^2]
        #phase_spat_2 = np.exp(((-1j*k)/(2*R)*r1**2).value)*arg**2 # [1/m^2] Note arg^2 = DROI^2/(wvl*Dz)^2 
        pt = A*np.multiply(phase_spat,np.multiply(sinc_x,np.multiply(sinc_y,phase_dim)))
        pt2 = self.central_point_source_sinc_func(x1,y1,r1,Dz,D2,wvl,k,power,time=0,source_number=0,sim_name = 'sim_name',plot_source=False)
        
        if (pt == pt2).all():
            print('Central point source sinc function is working properly')
        else:
            print('Central point source sinc function is not working properly.')
            if pt.unit == pt2.unit:
                print('Units of the function are correct.')
            else:
                print('Output units are not correct. Check how power units are treated.')
        
        # Plot the point source (Real and Complex)
        self.plotField(pt, x1, y1, 'Source Plane')
        
        # Define the propagation steps
        z = np.linspace(0,Dz,nscrn+2)
        
        # Run the angular spectrum multi propagation
        xn,yn,Uout = self.angular_spec_multi_prop_vac(pt,k,delta1,delta1,delta2,delta2,z)
        
        # Plot the resulting field (Real and Complex)
        self.plotField(Uout, xn, yn, 'Uncollimated Observation Plane')
        
        # Collimate the beam
        quadratic_phase_factor = np.exp((-1j*np.pi)/(wvl*Dz)*(xn**2+yn**2))
        Ucoll = np.multiply(Uout,quadratic_phase_factor)
        
        # Plot the collimated field (Real and Complex) and PSF
        self.plotField(Ucoll, xn, yn, 'Collimated Observation Plane')
        
        # Convolve the field with the pupil function to create final image
        # This section assumes some optical parameters that will be specific 
        # for each simulation
        # Define the image plane coordinates
        pixel_size = 1e-6*u.m
        grid = np.arange(-N/2,N/2)*pixel_size
        xi, yi = np.meshgrid(grid,grid,copy=False)
        # Calculate the pupil numerical aperature
        # Numerical aperature for a telescope is approximately f = 1/(2*NA)
        # For this example assume an f number of 2
        f = 10
        NA = 1/(2*f)
        focal_length = f*D2
        # Translate the object to image plane coordinates through geometric optic
        # magnification equation Goodman, Fourier Optics, Equation 6-39
        # Due to far distance of the point source assume 
        M = 1
        Uimage, xout, yout = self.magnifyField(Uout, xn, yn, focal_length, focal_length)
        # Define the pupil plane coordinates
        delta_p, N = self.pupilPlaneCorrd(pixel_size, pixel_size, N, N, wvl, focal_length, D2)
        grid = np.arange(-N/2,N/2)*delta_p
        xp, yp = np.meshgrid(grid,grid,copy=False)
        # The pupil function should be defined in pupil plane coordinates
        pupil_mask = self.circ(xp,yp,D2)
        pdb.set_trace()
        impulse_response = self.impulseResponse(pupil_mask,wvl,focal_length,delta_p,delta_p,N,N)
        # Resample the magnified field to the image coordinate size (pixel pitch)
        Uimage = self.resampleField(Uimage, xn, yn, xp, yp)
        # The fourier transform of the pupil function is taken with respect to tranformed dependent variables
        image = self.conv2(Uimage, impulse_response, pixel_size, pixel_size, pixel_size,N,N)
        
        # Scale the image to have the same intensity as the energy intercepted by the aperature
        masked_aperature = self.fieldMask(Uout,x1,y1,D2)
        masked_aperature_2 = self.fieldMask(Uimage, xp, yp, D2)
        power_aperature = self.intensityCalc(masked_aperature,delta2,delta2)
        power_image = self.intensityCalc(image,pixel_size,pixel_size)
        scaled_image, power_image_scaled = self.scaleField(image,pixel_size,pixel_size,power_aperature,power_image)
        
        # Plot the final image
        self.plotField(scaled_image, xi, yi, 'Image')
        
        return
    
    def ptSourceOffAxisVacProp(self):
        '''
        Test case for a vacuum propagation with a point source off the central axis.
        Due to the small FOV, the point source is defined as on axis and shifted in
        the image plane.

        Returns
        -------
        None.

        '''
        # Using the same inputs for the on axis point source, 
        # but source is offset to 1/2 the maximum x and 1/2 of the maximum y
        # Initial inputs taken from Schmidt 2010 (pg.175- ):
        D2 = 0.5*u.m # observation aperature [m]
        wvl = 1e-6*u.m # optical wavelength [m]
        k = 2*np.pi*u.rad / wvl # optical wavenumber [rad/m]
        Dz = 50e3*u.m # propagation distance [m]
        d1 = .01*u.m
        d2 = .01*u.m
        N = 512 # grid points
        DROI = 4*D2 # diameter of observation-plane region of interest [m]
        arg = DROI/(wvl*Dz) # ensure U units are sqrt(W)/m [1/m]
        D1 = wvl*Dz/DROI # width of central lobe [m]
        R = Dz # wavefront radius of curvature [m]
        nscrn = 11 # number of screens
        
        # create meshgrid of all coordinates for the source plane
        delta1 = d1 # source-plane grid spacing
        delta2 = d2 # observation-plane grid spacing
        n = nscrn # number of planes
        
        grid = np.arange(-N/2,N/2)*delta1 # grid units [m]
        x1, y1 = np.meshgrid(grid,grid,copy=False) # mesh units [m]
        # convert to polar coordinates
        r1, theta1 = self.cart2pol(x1,y1)
        # Define the center of the point source
        xc = np.max(np.max(x1))/2
        yc = np.max(np.max(y1))/2
        
        # # Define the point source
        # xc_mat = x1 - xc
        # yc_mat = y1 - yc
        # rc_mat, thetac_mat = self.cart2pol(xc_mat,yc_mat)
        # # Scaling factor to get the right units
        # power = 1*u.W
        # A = wvl*Dz*np.sqrt(power)/u.m # Scaled to the squareroot of power/meter^2, overall unit [sqrt(W)*m]
        # # Limits field amplitude in offset x direction
        # sinc_x = np.sinc((x1-xc)/D1*u.rad) # Unitless
        # sinc_x_2 = np.sinc(arg*(x1-xc)*u.rad) # Unitless
        # sinc_x_3 = np.sinc(x1/D1*u.rad)
        # # Limits field amplitude in offset y direction
        # sinc_y = np.sinc((y1-yc)/D1*u.rad) # Unitless
        # sinc_y_2 = np.sinc(arg*(y1-yc)*u.rad) # Unitless
        # sinc_y_3 = np.sinc(y1/D1*u.rad) 
        # # Scale amplitude based on radial distance from center of point source
        # phase_dim = np.exp(-(r1/(4*D1))**2) # Unitless
        # phase_dim_2 = np.exp(-((r1*arg)/4)**2) # Unitless
        # phase_dim_3 = np.exp(-((rc_mat*arg)/4)**2) # Unitless
        # # Prescribe phase based on spatial coordiantes
        # phase_spat = np.exp(((-1j*k)/(2*Dz)*r1**2).value)*np.exp(((1j*k)/(2*Dz)*rc_mat**2).value)*np.exp(((-1j*k)/(Dz)*np.multiply(rc_mat,r1)).value)/D1**2
        # phase_spat_2 = np.exp(((-1j*k)/(2*Dz)*r1**2).value)*np.exp(((1j*k)/(2*Dz)*rc_mat**2).value)*np.exp(((-1j*k)/(Dz)*np.multiply(rc_mat,r1)).value)*arg**2
        # pt = A*np.multiply(phase_spat,np.multiply(sinc_x,np.multiply(sinc_y,phase_dim_3)))
        
        # Define the point source
        power = 1*u.W
        A = wvl*Dz*np.sqrt(power)/u.m # Scaled to the squareroot of power/meter^2, overall unit [sqrt(W)*m]
        sinc_x = np.sinc((x1/D1)*u.rad) # Unitless
        #sinc_x_2 = np.sinc((x1*arg).value) #Unitless
        sinc_y = np.sinc((y1/D1)*u.rad) # Unitless
        #sinc_y_2 = np.sinc((y1*arg).value) #Unitless
        phase_dim = np.exp(-(r1/(4*D1))**2) # Unitless
        #phase_dim_2 = np.exp(-((r1*arg)/4)**2) # Unitless
        phase_spat = np.exp(((-1j*k)/(2*R)*r1**2).value)/D1**2 # [1/m^2]
        #phase_spat_2 = np.exp(((-1j*k)/(2*R)*r1**2).value)*arg**2 # [1/m^2] Note arg^2 = DROI^2/(wvl*Dz)^2 
        pt = A*np.multiply(phase_spat,np.multiply(sinc_x,np.multiply(sinc_y,phase_dim)))
        pt2 = self.central_point_source_sinc_func(x1,y1,r1,Dz,D2,wvl,k,power,time=0,source_number=0,sim_name = 'sim_name',plot_source=False)
        
        if (pt == pt2).all():
            print('Central point source sinc function is working properly')
        else:
            print('Central point source sinc function is not working properly.')
            if pt.unit == pt2.unit:
                print('Units of the function are correct.')
            else:
                print('Output units are not correct. Check how power units are treated.')
        
        # Plot the point source (Real and Complex)
        self.plotField(pt, x1, y1, 'Source Plane')
        
        # Define the propagation steps
        z = np.linspace(0,Dz,nscrn+2)
        
        # Run the angular spectrum multi propagation
        xn,yn,Uout = self.angular_spec_multi_prop_vac(pt,k,delta1,delta1,delta2,delta2,z)
        
        # Plot the resulting field (Real and Complex)
        self.plotField(Uout, xn, yn, 'Uncollimated Observation Plane')
        
        # Collimate the beam
        quadratic_phase_factor = np.exp((-1j*np.pi)/(wvl*Dz)*(xn**2+yn**2))
        Ucoll = np.multiply(Uout,quadratic_phase_factor)
        
        # Plot the collimated field (Real and Complex) and PSF
        self.plotField(Ucoll, xn, yn, 'Collimated Observation Plane')
        
        # Convolve the field with the pupil function to create final image
        # This section assumes some optical parameters that will be specific 
        # for each simulation
        # Define the image plane coordinates
        pixel_size = 1e-6*u.m
        grid = np.arange(-N/2,N/2)*pixel_size
        xi, yi = np.meshgrid(grid,grid,copy=False)
        # Calculate the pupil numerical aperature
        # Numerical aperature for a telescope is approximately f = 1/(2*NA)
        # For this example assume an f number of 2
        f = 10
        NA = 1/(2*f)
        focal_length = f*D2
        # Translate the object to image plane coordinates through geometric optic
        # magnification equation Goodman, Fourier Optics, Equation 6-39
        # Due to far distance of the point source assume 
        M = 1
        Uimage, xout, yout = self.magnifyField(Uout, xn, yn, focal_length, focal_length)
        # Define the pupil plane coordinates
        delta_p = (wvl*focal_length)/(N*pixel_size)
        # Determine how many grid points are required
        N_min = np.ceil(D2*1.5/delta_p)
        if N_min > N:
            N = N_min
        grid = np.arange(-N/2,N/2)*delta_p
        xp, yp = np.meshgrid(grid,grid,copy=False)
        # The pupil function should be defined in pupil plane coordinates
        pupil_mask = self.circ(xp,yp,D2)
        impulse_response = self.impulseResponse(pupil_mask,wvl,focal_length,delta_p,delta_p)
        # Resample the magnified field to the image coordinate size (pixel pitch)
        Uimage, xout, yout = self.resampleField(Uimage, xout, yout, xp, yp)
        # The fourier transform of the pupil function is taken with respect to tranformed dependent variables
        image = self.conv2(Uimage, impulse_response, pixel_size, pixel_size, pixel_size)
        # Scale the image to have the same intensity as the energy intercepted by the aperature
        masked_aperature = self.fieldMask(Uout,x1,y1,D2)
        power_aperature = self.intensityCalc(masked_aperature,delta2,delta2)
        power_image = self.intensityCalc(image,pixel_size,pixel_size)
        scaled_image, power_image_scaled = self.scaleField(image,pixel_size,pixel_size,power_aperature,power_image)
        
        # Shift the geometric center of the image to the projected (x,y) 
        # location of the point source.
        # Crop the image to the detector dimensions (Nx,Ny)
        Ix = N
        Iy = N
        xc = N*(1/4)
        yc = N*(3/4)
        shifted_image = self.imageShift(scaled_image,xc,yc)
        cropped_image = self.imageCrop(shifted_image, Ix, Iy)
        
        # Plot the final image
        self.plotField(cropped_image, xi, yi, 'Image')
        
        return
    
    def ptSourceTurbulence(self):
        '''
        Test case for a vacuum propagation with a zernike phase screen applied
        at each propagation step.
        
        Note this program should be run inside the folder with the 
        "test_propagations" folder below it. This ensures the phase screens and
        additional parameters are correctly loaded.

        Returns
        -------
        None.

        '''
        # Initial inputs taken from Schmidt 2010 (pg.175- ):
        D2 = 0.5*u.m # observation aperature [m]
        wvl = 1e-6*u.m # optical wavelength [m]
        k = 2*np.pi*u.rad / wvl # optical wavenumber [rad/m]
        Dz = 50e3*u.m # propagation distance [m]
        d1 = .01*u.m
        d2 = .01*u.m
        N = 512 # grid points
        DROI = 4*D2 # diameter of observation-plane region of interest [m]
        arg = DROI/(wvl*Dz) # ensure U units are sqrt(W)/m [1/m]
        D1 = wvl*Dz/DROI # width of central lobe [m]
        R = Dz # wavefront radius of curvature [m]
        nscrn = 11 # number of screens
        
        # create meshgrid of all coordinates for the source plane
        delta1 = d1 # source-plane grid spacing
        delta2 = d2 # observation-plane grid spacing
        n = nscrn # number of planes
        
        grid = np.arange(-N/2,N/2)*delta1 # grid units [m]
        x1, y1 = np.meshgrid(grid,grid,copy=False) # mesh units [m]
        # convert to polar coordinates
        r1, theta1 = self.cart2pol(x1,y1)
        
        # Define the point source
        power = 1*u.W
        A = wvl*Dz*np.sqrt(power)/u.m # Scaled to the squareroot of power/meter^2, overall unit [sqrt(W)*m]
        sinc_x = np.sinc((x1/D1)*u.rad) # Unitless
        #sinc_x_2 = np.sinc((x1*arg).value) #Unitless
        sinc_y = np.sinc((y1/D1)*u.rad) # Unitless
        #sinc_y_2 = np.sinc((y1*arg).value) #Unitless
        phase_dim = np.exp(-(r1/(4*D1))**2) # Unitless
        #phase_dim_2 = np.exp(-((r1*arg)/4)**2) # Unitless
        phase_spat = np.exp(((-1j*k)/(2*R)*r1**2).value)/D1**2 # [1/m^2]
        #phase_spat_2 = np.exp(((-1j*k)/(2*R)*r1**2).value)*arg**2 # [1/m^2] Note arg^2 = DROI^2/(wvl*Dz)^2 
        pt = A*np.multiply(phase_spat,np.multiply(sinc_x,np.multiply(sinc_y,phase_dim)))
        pt2 = self.central_point_source_sinc_func(x1,y1,r1,Dz,D2,wvl,k,power,time=0,source_number=0,sim_name = 'sim_name',plot_source=False)
        
        if (pt == pt2).all():
            print('Central point source sinc function is working properly')
        else:
            print('Central point source sinc function is not working properly.')
            if pt.unit == pt2.unit:
                print('Units of the function are correct.')
            else:
                print('Output units are not correct. Check how power units are treated.')
        
        # Plot the point source (Real and Complex)
        self.plotField(pt, x1, y1, 'Source Plane')
        
        # Define the propagation steps
        z1 = np.linspace(0,Dz,nscrn+2)
        # Load additional parameters from saved propagation parameters
        # First define the file location
        directory = os.getcwd()
        test_prop_params_loc = directory + '/test_propagations/test_prop_parameters'
        with open(test_prop_params_loc,'rb') as picklefile:
            prop_parameters = pickle.load(picklefile)
        # Open files from dictionary
        delta_n = prop_parameters['delta_n']
        delta_n_pscrn = prop_parameters['deltan_pscrn']
        scale = prop_parameters['scale_pscrn']
        z = prop_parameters['z']
        delta_z = prop_parameters['delta_z']
        # Define the rest of the inputs
        sg = self.super_gaussian(x1,y1,N,N,delta1,delta1)
        pscrn_file_name = directory + '/test_propagations/PhaseScreens/test_pscrns.hdf5'
        psn_itr = 0
        x_y_prop_vec = [(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2),(N/2,N/2)] # Modeled as centralized point source (this parameter may be removed)
        x_y_pscrn_vec = []
        for i,n_pscrn in enumerate(prop_parameters['N_total_pscrn'][0]):
            x_y_pscrn_vec.append((n_pscrn/2,n_pscrn/2))
        x_prop = prop_parameters['N_total_n']/2
        y_prop = prop_parameters['N_total_n']/2
        
        # Run the angular spectrum multi propagation
        pdb.set_trace()
        xn,yn,Uout,point_in_frame = self.angular_spec_multi_prop(pt, k, delta_n, delta_n_pscrn, scale, z, delta_z, sg, pscrn_file_name, psn_itr, x_y_prop_vec, x_y_pscrn_vec, x_prop, y_prop)
        # Plot the resulting field (Real and Complex)
        self.plotField(Uout, xn, yn, 'Uncollimated Observation Plane')
        
        # Collimate the beam
        quadratic_phase_factor = np.exp((-1j*np.pi)/(wvl*Dz)*(xn**2+yn**2))
        Ucoll = np.multiply(Uout,quadratic_phase_factor)
        
        # Plot the collimated field (Real and Complex) and PSF
        self.plotField(Ucoll, xn, yn, 'Collimated Observation Plane')
        
        # Convolve the field with the pupil function to create final image
        # This section assumes some optical parameters that will be specific 
        # for each simulation
        # Define the image plane coordinates
        pixel_size = 1e-6*u.m
        grid = np.arange(-N/2,N/2)*pixel_size
        xi, yi = np.meshgrid(grid,grid,copy=False)
        # Calculate the pupil numerical aperature
        # Numerical aperature for a telescope is approximately f = 1/(2*NA)
        # For this example assume an f number of 2
        f = 10
        NA = 1/(2*f)
        focal_length = f*D2
        # Translate the object to image plane coordinates through geometric optic
        # magnification equation Goodman, Fourier Optics, Equation 6-39
        # Due to far distance of the point source assume 
        M = 1
        
        Uimage, xout, yout = self.magnifyField(Uout, xi, yi, focal_length, focal_length)
        # Define the pupil plane coordinates
        delta_p, N = self.pupilPlaneCorrd(pixel_size, pixel_size, N, N, wvl, focal_length, D2)
        grid = np.arange(-N/2,N/2)*delta_p
        xp, yp = np.meshgrid(grid,grid,copy=False)
        # The pupil function should be defined in pupil plane coordinates
        pupil_mask = self.circ(xp,yp,D2)
        impulse_response = self.impulseResponse(pupil_mask,wvl,focal_length,delta_p,delta_p)
        # Resample the magnified field to the image coordinate size (pixel pitch)
        Uimage = self.resampleField(Uimage, xn, yn, xp, yp)
        # The fourier transform of the pupil function is taken with respect to tranformed dependent variables
        image = self.conv2(Uimage, impulse_response, pixel_size, pixel_size, pixel_size)
        
        # Scale the image to have the same intensity as the energy intercepted by the aperature
        masked_aperature = self.fieldMask(Uout,x1,y1,D2)
        power_aperature = self.intensityCalc(masked_aperature,delta2,delta2)
        power_image = self.intensityCalc(image,pixel_size,pixel_size)
        scaled_image, power_image_scaled = self.scaleField(image,pixel_size,pixel_size,power_aperature,power_image)
        
        # Plot the final image
        self.plotField(scaled_image, xi, yi, 'Image')
        return
        
    
    def plotField(self,Ui,xi,yi,plane_title,log_plot = False):
        '''
        Plot the corresponding field and label it appropriately

        Parameters
        ----------
        Ui : np.array
            Array containing amplitude and phase information encoded as a complex number.
        xi : np.array
            Array with x distance values from the center of the array for each grid space.
        yi : np.array
            Array with y distance values from the center of the array for each grid space.
        plane_title : string
            Description of the plane which is being depicted

        Returns
        -------
        None.

        '''
        
        # Plot the intensity on this plane
        # Intensity or irradiance is U^2 [W/m^2]
        I = np.real(np.multiply(Ui,np.conjugate(Ui)))
        if np.min(I.value) == 0.0:
            I = (I.value + 1e-50)*I.unit
        fig, ax = plt.subplots(dpi=200)
        if log_plot == True:
            pcm = ax.contourf(xi,yi,I,locator=ticker.LogLocator(),cmap=plt.get_cmap('gray'))
        else:
            pcm = ax.contourf(xi,yi,I,cmap=plt.get_cmap('gray'))
        cbar = fig.colorbar(pcm)
        color_bar_label = 'Irradiance ['+str(I.unit)+']'
        cbar.set_label(color_bar_label, rotation=270)
        ax.axis('equal')
        ax.set_title('Irradiance in ' + plane_title)
        ax.set_xlabel(plane_title + '[' + str(xi.unit) + ']')
        ax.set_ylabel(plane_title + '[' + str(yi.unit) + ']')
        plt.show()
        
        # Plot the phase component of the field
        U_phase = np.arctan2(-np.imag(Ui),np.real(Ui))
        fig, ax = plt.subplots(dpi=200)
        pcm = ax.contourf(xi,yi,U_phase,cmap=plt.get_cmap('gray'))
        cbar = fig.colorbar(pcm)
        color_bar_label = 'Phase [rad]'
        cbar.set_label(color_bar_label, rotation=270)
        ax.axis('equal')
        ax.set_title('Phase in ' + plane_title)
        ax.set_xlabel(plane_title + '[' + str(xi.unit) + ']')
        ax.set_ylabel(plane_title + '[' + str(yi.unit) + ']')
        plt.show()
        
        return