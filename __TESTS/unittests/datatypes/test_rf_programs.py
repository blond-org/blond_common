# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test dataypes._core.py

'''

# General imports
# ---------------
import sys
import unittest
import numpy as np
import numpy.testing as npTest
import os
import scipy.constants as cont

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

import blond_common.datatypes.rf_programs as rfProg
from blond_common.devtools import exceptions

class test_rf_programs(unittest.TestCase):

    def test_basic_rf_function(self):

        func = rfProg._RF_function(1, harmonics = 1)
        expectDict = {'timebase': 'single', 'harmonics': (1,),
                      'interpolation': 'linear'}
        self.assertEqual(func, 1, msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        func = rfProg._RF_function(1, 1, harmonics = [1, 2])
        expectDict = {'timebase': 'single', 'harmonics': [1, 2],
                      'interpolation': 'linear'}
        np.testing.assert_array_equal(func, [1, 1],
                                      err_msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        func = rfProg._RF_function(1, 1, harmonics = [1, 2], time=[1, 2])
        expectDict = {'timebase': 'by_time', 'harmonics': [1, 2],
                      'interpolation': 'linear'}
        np.testing.assert_array_equal(func, [[[1, 2], [1, 1]],
                                             [[1, 2], [1, 1]]],
                                      err_msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        func = rfProg._RF_function(1, 1, harmonics = [1, 2], n_turns=2)
        expectDict = {'timebase': 'by_turn', 'harmonics': [1, 2],
                      'interpolation': 'linear'}
        np.testing.assert_array_equal(func, [[1, 1], [1, 1]],
                                      err_msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')


    def test_reshape(self):

        func = rfProg._RF_function([2, 4], harmonics = 1, time=[1, 2])

        func2 = func.reshape(use_time = [1.5])
        expectDict = {'timebase': 'interpolated', 'harmonics': (1,),
                      'interpolation': 'linear'}
        np.testing.assert_array_equal(func2, [[3]],
                                      err_msg='Function value incorrect')
        self.assertEqual(func2.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        func2 = func.reshape(harmonics = [1, 2], use_time = [1.5])
        expectDict = {'timebase': 'interpolated', 'harmonics': [1, 2],
                      'interpolation': 'linear'}
        np.testing.assert_array_equal(func2, [[3], [0]],
                                      err_msg='Function value incorrect')
        self.assertEqual(func2.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        with self.assertRaises(exceptions.InputError,
                               msg='Reshape with use_turns only should raise'
                               + 'InputError for a "by_time" function'):
            func.reshape(use_turns = [1, 2, 3])

        func = rfProg._RF_function([2, 4], harmonics = 1)

        func2 = func.reshape(use_turns=[1])
        expectDict = {'timebase': 'interpolated', 'harmonics': (1,),
                      'interpolation': 'linear'}
        np.testing.assert_array_equal(func2, [[4]],
                                      err_msg='Function value incorrect')
        self.assertEqual(func2.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        with self.assertRaises(exceptions.InputError,
                               msg='Reshape with use_time only should raise'
                               + 'InputError for a "by_turn" function'):
            func.reshape(use_time = [1, 2, 3])


if __name__ == '__main__':

    unittest.main()