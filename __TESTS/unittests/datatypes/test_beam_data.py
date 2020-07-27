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
import matplotlib.pyplot as plt

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

import blond_common.datatypes.beam_data as bDat
from blond_common.devtools import exceptions

class test_beam_data(unittest.TestCase):

    def test_basic_beam_data(self):

        func = bDat._beam_data(1, units='test')
        expectDict = {'timebase': 'single', 'units': 'test',
                      'interpolation': 'linear', 'bunching': 'single_bunch'}
        self.assertEqual(func, 1, msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        func = bDat._beam_data(1, 1, units='test')
        expectDict = {'timebase': 'single', 'units': 'test',
                      'interpolation': 'linear', 'bunching': 'multi_bunch'}
        np.testing.assert_array_equal(func, [1, 1],
                                      err_msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        func = bDat._beam_data(1, 1, units = 'test', time=[1, 2])
        expectDict = {'timebase': 'by_time', 'units': 'test',
                      'interpolation': 'linear', 'bunching': 'multi_bunch'}
        np.testing.assert_array_equal(func, [[[1, 2], [1, 1]],
                                              [[1, 2], [1, 1]]],
                                      err_msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')

        func = bDat._beam_data(1, 1, units = 'test', n_turns=2)
        expectDict = {'timebase': 'by_turn', 'units': 'test',
                      'interpolation': 'linear', 'bunching': 'multi_bunch'}
        np.testing.assert_array_equal(func, [[1, 1], [1, 1]],
                                      err_msg='Function value incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                    +' incorrect')


    def test_acceptance(self):

        func = bDat.acceptance(1)
        expectDict = {'timebase': 'single', 'units': 'eVs',
                      'interpolation': 'linear', 'bunching': 'single_bunch'}
        self.assertEqual(func.data_type, expectDict, msg = 'data_type dict '
                                                         +'contents incorrect')


    def test_emittance(self):

        func = bDat.emittance(1)
        expectDict = {'timebase': 'single', 'units': 'eVs',
                      'interpolation': 'linear', 'bunching': 'single_bunch',
                      'emittance_type': 'matched_area'}
        self.assertEqual(func.data_type, expectDict, msg = 'data_type dict '
                                                         +'contents incorrect')


    def test_length(self):

        func = bDat.length(1)
        expectDict = {'timebase': 'single', 'units': 's',
                      'interpolation': 'linear', 'bunching': 'single_bunch',
                      'length_type': 'full_length'}
        self.assertEqual(func.data_type, expectDict, msg = 'data_type dict '
                                                         +'contents incorrect')


    def test_height(self):

        func = bDat.height(1)
        expectDict = {'timebase': 'single', 'units': 'eV',
                      'interpolation': 'linear', 'bunching': 'single_bunch',
                      'height_type': 'half_height'}
        self.assertEqual(func.data_type, expectDict, msg = 'data_type dict '
                                                          +'contents incorrect')


    def test_synchronous_phase(self):

        func = bDat.synchronous_phase(1)
        expectDict = {'timebase': 'single', 'units': 's',
                      'interpolation': 'linear', 'bunching': 'single_bunch'}
        self.assertEqual(func.data_type, expectDict, msg = 'data_type dict '
                                                         +'contents incorrect')


if __name__ == '__main__':

    unittest.main()