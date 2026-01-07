# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 23:29:42 2023

@author: User
"""
import cProfile
import pstats
import run_test_circuitry_v2 as rtc
from astropy import units as u

def run_test():
    I_dark = 3.5e-17*u.A
    num_pix_x = 640
    num_pix_y = 480
    num_steps_per_second = 500
    number_of_pulls = 10000
    cutoff_freq = 50*u.Hz
    pos_threshold = .2
    neg_threshold = .2
    seed = 2695430341
    
    run_cir_test = rtc.circuitTest()
    
    events = run_cir_test.run_shot_noise_sim(I_dark, num_pix_x, num_pix_y, num_steps_per_second, number_of_pulls, cutoff_freq, pos_threshold, neg_threshold, seed)

if __name__ == '__main__':
    cProfile.run('run_test()',"{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.sort_stats("time").print_stats()