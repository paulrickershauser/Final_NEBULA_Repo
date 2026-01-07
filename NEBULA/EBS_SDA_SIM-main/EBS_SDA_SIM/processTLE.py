# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 20:54:05 2022

@author: rache
"""

import os
import io
import pdb
import numpy as np

for (root,dirs,files) in os.walk('TLE Files'):
    for i, file in enumerate(files):
        open_file = open('TLE Files/'+ file,'r')
        lines = open_file.readlines()
        # Handle lines in groups of three
        num = len(lines)
        m = num/3
        for j in np.arange(0,m,1):
            three_line_set = lines[int(j*3):int(j*3+3)]
            # Find time of tle
            year = '20' + three_line_set[1][18:20]
            if year != '2021':
                pdb.set_trace()
            day = three_line_set[1][20:23]
            tle_file = 'tle_' + year + '_' + day + '.txt'
            path_tle_file = 'TLE Files/' + tle_file
            if os.path.isfile(path_tle_file) == False:
                f = open(path_tle_file, mode = 'w')
                new_lines = three_line_set
            else:
                f = open(path_tle_file, mode = 'r')
                current_lines = f.readlines()
                curr_num = len(current_lines)
                curr_m = curr_num/3
                # Replace old tle with newer one
                if three_line_set[0] in current_lines:
                    match = False
                    for k in np.arange(0,curr_m,1):
                        current_three_lines = current_lines[int(k*3):int(k*3+3)]
                        if current_three_lines[0] == three_line_set[0] and current_three_lines[1][2:7] == three_line_set[1][2:7]:
                            current_lines[int(k*3+1)] = three_line_set[1]
                            current_lines[int(k*3+2)] = three_line_set[2]
                            new_lines = current_lines
                            match = True
                            break
                    if match == False:
                        new_lines = current_lines + three_line_set
                else:
                    new_lines = current_lines + three_line_set
                f = open(path_tle_file, mode = 'w')
            f.writelines(new_lines)
            f.close()