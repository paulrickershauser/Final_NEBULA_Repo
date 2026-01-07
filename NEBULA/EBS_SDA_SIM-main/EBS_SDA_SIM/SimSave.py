# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 18:50:45 2023

@author: Rachel Oliver

After all the parameters of a simulation are created. Save them to a folder for future 
use.
"""

import os
import pickle

# Save the simulation data into a particular file folder
directory = os.getcwd()
file_location = directory + '/' + sim_name
os.chdir(file_location)
with open('sim_name.pickle','rb') as pickle_file:
    pickle.dump(sim_name,pickle_file)

with open('t_array.pickle','rb') as pickle_file:
    pickle.dump(t_array,pickle_file)

with open('sat_parameters.pickle','rb') as pickle_file:
    pickle.dump(sat_parameters,pickle_file)

with open('sat_obs.pickle','rb') as pickle_file:
    pickle.dump(sat_obs,pickle_file)

with open('sat_array.pickle','rb') as pickle_file:
    pickle.dump(sat_array,pickle_file)

with open('prop_parameters.pickle','rb') as pickle_file:
    pickle.dump(prop_parameters,pickle_file)

with open('observer.pickle','rb') as pickle_file:
    pickle.dump(observer,pickle_file)

with open('Dz.pickle','rb') as pickle_file:
    pickle.dump(Dz,pickle_file)

with open('ebs_circuit_parameters.pickle','rb') as pickle_file:
    pickle.dump(ebs_circuit_parameters,pickle_file)
    
with open('pscrn_file_name.pickle','rb') as pickle_file:
    pickle.dump(pscrn_file_name,pickle_file)
    
with open('frame_file_name.pickle','rb') as pickle_file:
    pickle.dump(frame_file_name,pickle_file)
    
with open('attribution_dict_file_name.pickle','rb') as pickle_file:
    pickle.dump(attribution_dict_file_name,pickle_file)
    
    