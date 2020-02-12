# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test preprocess.py

'''

# General imports
# ---------------
import sys
import unittest
import unittest.mock as mk
import numpy as np
import scipy.interpolate as interp
import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

import blond_common.beam_dynamics.bucket as bucket
import blond_common.devtools.exceptions as excpt


class test_bucket(unittest.TestCase):
    
    def setUp(self):
        
        inTime = np.linspace(0, 2*np.pi, 100)
        inWell = np.cos(inTime)
        inWell += np.cos(inTime*2)
        inWell -= np.min(inWell)
        
        self.buck = bucket.Bucket(inTime, inWell, 3, 4, 5)
    
    
    def test_attributes(self):
        
        self.assertEqual(len(self.buck.separatrix), 2, 
                         'separatrix length incorrect')
        
        self.assertEqual(len(self.buck.separatrix[0]), 200, 
                         'separatrix[0] length incorrect')
        
        self.buck.smooth_well(200)
        self.buck.calc_separatrix()
        
        self.assertEqual(len(self.buck.separatrix[0]), 400, 
                         'separatrix[0] length incorrect after reinterpolation')
        
        
    
    def test_exceptions(self):
        
        with self.assertRaises(excpt.InputError, 
                           msg='single valued time should raise InputError'):
            bucket.Bucket(0, [1, 2, 3], 1, 1, 1)

        with self.assertRaises(excpt.InputError, 
                           msg='single valued well should raise InputError'):
            bucket.Bucket([1, 2, 3], 1, 1, 1, 1)
            
        with self.assertRaises(excpt.InputError, 
                       msg='different length inputs should raise InputError'):
            bucket.Bucket([1, 2, 3], [1, 2], 1, 1, 1)

        with self.assertRaises(excpt.BunchSizeError, 
                           msg='Too long bunch should raise BunchSizeError'):
            self.buck.outline_from_length(100)
        
        with self.assertRaises(excpt.BunchSizeError, 
                           msg='Too tall bunch should raise BunchSizeError'):
            self.buck.outline_from_dE(100)
        
        with self.assertRaises(excpt.BunchSizeError, 
                           msg='Too large bunch should raise BunchSizeError'):
            self.buck.outline_from_emittance(100)
    
        with self.assertRaises(excpt.InputError, 
                           msg='Too high potential should raise InputError'):
            self.buck._interp_time_from_potential(100)
            
        with self.assertRaises(excpt.InputError, 
                           msg='Negative potential should raise InputError'):
            self.buck._interp_time_from_potential(-1)

    
    def test_smooth_well(self):
        
        if hasattr(self.buck, '_well_smooth_func'):
            self.fail("_well_smooth_func should not exist yet")
        self.buck.smooth_well()
        if not hasattr(self.buck, '_well_smooth_func'):
            self.fail("_well_smooth_func should exist now")
        
        self.buck._well_smooth_func = mk.MagicMock(name='_well_smooth_mock')
        self.buck.smooth_well(1000)
        
        exptTime = np.linspace(self.buck.time[0], self.buck.time[-1], 1000)
        
        self.buck._well_smooth_func.assert_called_once()
        
        calledWith = self.buck._well_smooth_func.call_args[0][0]
        np.testing.assert_array_equal(calledWith, exptTime, 
                      err_msg='_well_smooth_func called with unexpected array')
        
        self.buck.smooth_well(reinterp=True)
        
        if not isinstance(self.buck._well_smooth_func, interp._cubic.CubicSpline):
            self.fail("_well_smooth_func has not been recreated")
        
    
    
    def test__interp_time_from_potential(self):
        
        l, r = self.buck._interp_time_from_potential(self.buck.well[0])
        self.assertEqual(l, self.buck.time[0], 'time limit should be returned')
        self.assertEqual(r, self.buck.time[-1], 'time limit should be returned')
        l, r = self.buck._interp_time_from_potential(self.buck.well[-1])
        self.assertEqual(l, self.buck.time[0], 'time limit should be returned')
        self.assertEqual(r, self.buck.time[-1], 'time limit should be returned')
    
        time = self.buck._interp_time_from_potential(3, 100)
        self.assertEqual(len(time), 100, 'returned array should have length 100')
        
        self.buck.smooth_well()
        well = self.buck._well_smooth_func(time)
        self.assertAlmostEqual(well[0], 3, places=2, 
                               msg= 'insufficient precision in '\
                               + '_interp_time_from_potential')
        self.assertAlmostEqual(well[-1], 3, places=2, 
                               msg= 'insufficient precision in '\
                               + '_interp_time_from_potential')

    
    def test_outline_from_length(self):
        
        contour = self.buck.outline_from_length(3)
        retLen = np.max(contour[0]) - np.min(contour[0])
        self.assertAlmostEqual(retLen, 3, places=2, 
                               msg='Length of returned contour too imprecise')
        self.assertEqual(np.max(contour[1]), -np.min(contour[1]), 
                         msg='Contour top and bottom should be mirror images')
    
    
    def test_outline_from_dE(self):
        
        contour = self.buck.outline_from_dE(3)
        self.assertAlmostEqual(np.max(contour[1]), 3, places=2, 
                               msg='Length of returned contour too imprecise')
        self.assertEqual(np.max(contour[1]), -np.min(contour[1]), 
                         msg='Contour top and bottom should be mirror images')
    

    def test_outline_from_emittance(self):
        
        contour = self.buck.outline_from_emittance(30)
        retEmit = np.trapz(contour[1], contour[0])
        self.assertAlmostEqual(retEmit, 30, places=2, 
                               msg='Emittance of returned contour too imprecise')
        self.assertEqual(np.max(contour[1]), -np.min(contour[1]), 
                         msg='Contour top and bottom should be mirror images')

    
            
if __name__ == '__main__':
    
    unittest.main()