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


import blond_common.datatypes._core as core
import blond_common.datatypes.ring_programs as rProg
import blond_common.datatypes.functions as funcs
from blond_common.devtools import exceptions


class test_functions(unittest.TestCase):

    def test_vstack(self):

        alpha0_0 = rProg.momentum_compaction(0, time = [0, 1])
        alpha0_1 = rProg.momentum_compaction(1, time = [2, 3])

        alpha0 = funcs.vstack(alpha0_0, alpha0_1)
        expectDict = {'timebase': 'by_time', 'sectioning': 'single_section',
                      'order': 0, 'interpolation': 'linear'}
        expectArray = [[[0, 1, 2, 3], [0, 0, 1, 1]]]

        self.assertEqual(alpha0.data_type, expectDict, msg='Final data_type '
                         +'dict does not match expected')
        np.testing.assert_array_equal(alpha0, expectArray, err_msg = 'Stacked'
                                      +' array does not match expected.')

        alpha0_0 = rProg.momentum_compaction(0, 1, time = [0, 1, 2, 3])
        alpha0_1 = rProg.momentum_compaction(1, 0, time = [5, 6, 7])

        alpha0 = funcs.vstack((1, alpha0_0), (5.5, alpha0_1, 8))
        expectDict = {'timebase': 'by_time', 'sectioning': 'multi_section',
                      'order': 0, 'interpolation': 'linear'}
        expectArray = [[[1, 2, 3, 5.5, 6, 7, 8], [0, 0, 0, 1, 1, 1, 1]],
                       [[1, 2, 3, 5.5, 6, 7, 8], [1, 1, 1, 0, 0, 0, 0]]]

        self.assertEqual(alpha0.data_type, expectDict, msg='Final data_type '
                         +'dict does not match expected')
        np.testing.assert_array_equal(alpha0, expectArray, err_msg = 'Stacked'
                                      +' array does not match expected.')

        with self.assertRaises(exceptions.InputError, msg='Non stackable '
                               +'input should raise an InputError'):
            funcs.vstack(1, 2)

        with self.assertRaises(exceptions.InputError, msg='Input without a '
                               +'datatype array should raise in InputError'):
            funcs.vstack(alpha0_0, (2, 3))

        with self.assertRaises(exceptions.InputError, msg='Wrong length input '
                               +'should raise in InputError'):
            funcs.vstack(alpha0_0, (2, alpha0_1, 3, 4))

        with self.assertRaises(RuntimeError, msg='Non-monotonically '
                           +'increasing times should raise a RuntimeError'):
            funcs.vstack(alpha0_0, (2, alpha0_1, 3))

        alpha1_0 = alpha0_0.copy()
        alpha1_0.order = 1
        with self.assertRaises(exceptions.InputError, msg='Mismatched '
                               +'data_type should raise in InputError'):
            funcs.vstack(alpha0_0, alpha1_0)


if __name__ == '__main__':

    unittest.main()