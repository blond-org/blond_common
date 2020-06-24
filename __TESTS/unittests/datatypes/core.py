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
import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

import blond_common.datatypes._core as core
from blond_common.devtools import exceptions

class test_core(unittest.TestCase):

    def test_creation(self):

        core._function(np.array([1, 2, 3]), data_type = {},
                                       interpolation = 'linear')
        core._function(np.array([[1, 2, 3], [2, 3, 4]]), data_type = {})
        core._function(np.zeros([3, 2, 10]), data_type = {})
        core._function(np.array([[1, 2, 3], [2, 3]]), data_type = {})


    def test_zeros(self):

        test = core._function.zeros([3, 3])
        self.assertEqual(len(test.shape), 2,
                                 'length of test.shape should be 2')
        self.assertEqual(test.shape[0], 3,
                                 'first element of shape should be 3')
        self.assertEqual(test.shape[1], 3,
                                 'second element of shape should be 3')

        self.assertIsNone(test.timebase, msg='timebase should be None')

        test = core._function.zeros([3, 3], data_type = {'timebase': 'test'})
        self.assertEqual(test.timebase, 'test',
                                 msg='timebase has not been correctly set')

        with self.assertRaises(exceptions.InputDataError, \
                               msg='Passing an invalid parameter should ' \
                               + 'raise an InputDataError'):

            test = core._function.zeros([3, 3], data_type = {'fake_parameter':
                                                                 'test'})

    def test_slicing(self):

        test = core._function.zeros(10, data_type={'timebase': 'fake'})

        slice1 = test[:5]
        slice2 = test[5:]

        self.assertEqual(slice1.data_type['timebase'], 'fake1',
                             msg='slice1 data_type not copied correctly')
        self.assertEqual(slice2.data_type['timebase'], 'fake',
                             msg='slice1 data_type not copied correctly')

if __name__ == '__main__':

    unittest.main()