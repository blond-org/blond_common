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

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

import blond_common.datatypes.ring_programs as ringProg
from blond_common.devtools import exceptions

class test_ring_programs(unittest.TestCase):

    def test_basic_ring_function_creation(self):

        func = ringProg._ring_function(1E9)
        expectDict = {'timebase': 'single', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1,), 'Single value shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                     +' incorrect')

        func = ringProg._ring_function([1E9, 1E9], [1E9, 1E9])
        expectDict = {'timebase': 'by_turn', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2), 'Multi-section by turn shape '
                                           + 'incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                     +' incorrect')

        func = ringProg._ring_function([1E9, 1E9], time = [0, 1])
        expectDict = {'timebase': 'by_time', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1, 2, 2), 'Single-section by time shape '
                                              + 'incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                     +' incorrect')

        func = ringProg._ring_function([[0, 1], [1E9, 1E9]],
                                        [[0, 1], [1E9, 1E9]])
        expectDict = {'timebase': 'by_time', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2, 2), 'Multi-section by time shape '
                                              + 'incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        func = ringProg._ring_function(1, n_turns = 100)
        expectDict = {'timebase': 'by_turn', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1, 100), 'single section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        func = ringProg._ring_function(1, time = [0, 1])
        expectDict = {'timebase': 'by_time', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1, 2, 2), 'single section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        func = ringProg._ring_function(1, 2, time = [0, 1])
        expectDict = {'timebase': 'by_time', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2, 2), 'multi section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')


    def test_mixed_input(self):

        with self.assertRaises(exceptions.DataDefinitionError, \
                               msg='Declaring a mix of turn based and single'
                                 + ' valued input should raise a '
                                 + 'DataDefinitionError with default of '
                                 + 'allow_single = False'):
            func = ringProg._ring_function(1, [1, 2])

        func = ringProg._ring_function(1, [1, 2], allow_single = True)
        expectDict = {'timebase': 'by_turn', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2), 'multi section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        with self.assertRaises(exceptions.DataDefinitionError, \
                               msg='Declaring a mix of time based and single'
                                 + ' valued input should raise a '
                                 + 'DataDefinitionError with default of '
                                 + 'allow_single = False'):
            func = ringProg._ring_function(1, [[0, 1], [1, 2]])

        func = ringProg._ring_function(1, [[0, 1], [1, 2]], allow_single=True)
        expectDict = {'timebase': 'by_time', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2, 2), 'multi section by time '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')


    def test_synchronous_data(self):

        mom = ringProg.momentum_program(1E9)
        totEn = ringProg.total_energy_program([2E9]*10)
        kinEn = ringProg.kinetic_energy_program([[0, 1], [160E6, 1E9]])
        bend = ringProg.bending_field_program(0.5, n_turns = 1000)

        self.assertEqual(mom.source, 'momentum', msg='momentum_program source'
                                                 + ' str incorrect')
        self.assertEqual(totEn.source, 'energy', msg='momentum_program source'
                                                 + ' str incorrect')
        self.assertEqual(kinEn.source, 'kin_energy', msg='momentum_program'
                                                 + ' source str incorrect')
        self.assertEqual(bend.source, 'B_field', msg='momentum_program source'
                                                 + ' str incorrect')


    def test_conversions(self):

        conversions = {'momentum': ringProg}



if __name__ == '__main__':

    unittest.main()