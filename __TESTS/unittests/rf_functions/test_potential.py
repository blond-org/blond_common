# coding: utf8
# Copyright 2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for the blond_common.rf_functions.potential module

:Authors: **Alexandre Lasheen**

"""

# General imports
# ---------------
import sys
import unittest
import numpy as np
import os
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

from blond_common.rf_functions.potential import find_potential_wells_cubic

# Input data folder
input_folder = this_directory+'/../../input/rf_functions/'


class TestPotential(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):
        '''
        We generate three different distributions, Gaussian, Parabolic
        Amplitude, and Binomial that will be used to test the fitting functions
        '''
        self._generation_tests_findpotwell()
    
    # Tests for find_potential_wells_cubic ------------------------------------
        
    def _generation_tests_findpotwell(self):
        '''
        Generate the tests for all the potential walls cases
        with results found in the input/rf_functions folder
        '''
 
        self.n_tests_findpotwell=0
        self.test_list=[]
         
        list_files = os.listdir(input_folder)
         
        for filename in list_files:
            if ('results' in filename) or (filename[-4:] != '.npy'):
                continue
             
            if 'results_'+filename[:-4]+'.npz' in list_files: 
                self.n_tests_findpotwell+=1
                self.test_list.append(filename)
    
    def test_findpotwell(self):
        
        for index_test in range(self.n_tests_findpotwell):
            filename = self.test_list[index_test]
            loaded_data = np.load(input_folder+filename, allow_pickle=True)
           
            (potential_well_locs, potential_well_vals,
             potential_well_inner_max, potential_well_min,
             potential_well_min_val) = find_potential_wells_cubic(
                loaded_data[0, :], loaded_data[1, :], mest=200, verbose=False)
               
            n_potentials = len(potential_well_locs)
            
            results = np.load(input_folder+'/results_'+filename[:-4]+'.npz')
            
            with self.subTest(filename + ' - n_potentials'):    
                np.testing.assert_equal(
                    n_potentials, results['n_potentials'])
            
            with self.subTest(filename + ' - potential_well_locs'):    
                np.testing.assert_equal(
                    potential_well_locs, results['potential_well_locs'])
            
            with self.subTest(filename + ' - potential_well_vals'):    
                np.testing.assert_equal(
                    potential_well_vals, results['potential_well_vals'])
            
            with self.subTest(filename + ' - potential_well_inner_max'):    
                np.testing.assert_equal(
                    potential_well_inner_max, results['potential_well_inner_max'])
            
            with self.subTest(filename + ' - potential_well_min'):    
                np.testing.assert_equal(
                    potential_well_min, results['potential_well_min'])
            
            with self.subTest(filename + ' - potential_well_min_val'):    
                np.testing.assert_equal(
                    potential_well_min_val, results['potential_well_min_val'])


if __name__ == '__main__':
    
    # Run tests
    unittest.main()
    
#     # Commented part used to generate reference results
#     # or for testing/debugging
#     
#     for filename in os.listdir(input_folder):
#   
#         if ('results' in filename) or (filename[-4:] != '.npy'):
#             continue
#          
#         loaded_data = np.load(input_folder+filename, allow_pickle=True)
#       
#         (potential_well_locs, potential_well_vals,
#          potential_well_inner_max, potential_well_min,
#          potential_well_min_val) = find_potential_wells_cubic(
#             loaded_data[0, :], loaded_data[1, :], mest=200, verbose=False)
#           
#         n_potentials = len(potential_well_locs)
#          
#         np.savez(input_folder+'/results_'+filename[:-4]+'.npz',
#                  n_potentials=n_potentials,
#                  potential_well_locs=potential_well_locs,
#                  potential_well_vals=potential_well_vals,
#                  potential_well_inner_max=potential_well_inner_max,
#                  potential_well_min=potential_well_min,
#                  potential_well_min_val=potential_well_min_val)
#            
#         plt.figure('Case')
#         plt.clf()
#         plt.plot(loaded_data[0, :], loaded_data[1, :], 'k')
#          
#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']
#      
#         for index_pot in range(n_potentials):
#             plt.plot(potential_well_locs[index_pot],
#                      potential_well_vals[index_pot],
#                     'o', color=colors[index_pot])
#             plt.axhline(potential_well_inner_max[index_pot],
#                         color=colors[index_pot])
#             plt.plot(potential_well_min[index_pot],
#                      potential_well_min_val[index_pot],
#                      'o', color=colors[index_pot])
#         plt.savefig(input_folder+filename[:-4]+'.png')
# #          
#         plt.show()


