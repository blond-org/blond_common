# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for impedance_sources.py
:Authors: **Simon Albright**
"""

# General imports
# ---------------
import sys
import os
import unittest
import numpy as np

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../../'))

import blond_common.interfaces.impedances.impedance_sources as impSource
import blond_common.devtools.exceptions as exceptions


   

class TestImpedanceSources(unittest.TestCase):

    def _input_except_test(self, function, exception, msg, *args, **kwargs):
        with self.assertRaises(exception, msg=msg):
            function(*args, **kwargs)
    
    def test_impedance_object(self):
        
        imp = impSource._ImpedanceObject()
        exp = exceptions.WrongCalcError
        msg = 'Expected WrongCalcError exception'
        self._input_except_test(imp.wake_calc, exp, msg)
        self._input_except_test(imp.wake_calc, exp, msg, 1, 'a', range, time=0)
        self._input_except_test(imp.wake_calc, exp, msg, \
                                new_time_array = [1, 2, 3])

        self._input_except_test(imp.imped_calc, exp, msg)
        self._input_except_test(imp.imped_calc, exp, msg, 1, 'a', range, time=0)
        self._input_except_test(imp.imped_calc, exp, msg, \
                                new_frequency_array = [1, 2, 3])
        
        self.assertEqual(imp.time_array, 0, "Expected time array to be 0")
        self.assertEqual(imp.wake, 0, "Expected wake to be 0")
        self.assertEqual(imp.frequency_array, 0, "Expected freuency array to be 0")
        self.assertEqual(imp.impedance, 0, "Expected impedance to be 0")
        self.assertEqual(imp.Re_Z_array, 0, "Expected Re_Z to be 0")
        self.assertEqual(imp.Im_Z_array, 0, "Expected Im_Z to be 0")


    def test_input_table_wake(self):

        self._input_except_test(impSource._InputTable, exceptions.InputError,
                                "Expected InputError exception", [1], [1, 2])
        
        imp = impSource._InputTable([1, 2, 3], [4, 5, 6])
        
        self.assertEqual(imp.time_array_loaded.tolist(), [1, 2, 3], 
                         "Expected time array to be [1, 2, 3]")
        self.assertEqual(imp.wake_array_loaded.tolist(), [4, 5, 6],
                         "Expected time array to be [4, 5, 6]")

        self.assertEqual(imp.frequency_array, 0, "Expected time array to be 0")
        self.assertEqual(imp.impedance, 0, "Expected impedance to be 0")

        exp = exceptions.WrongCalcError
        msg = 'Expected WrongCalcError exception'

        self._input_except_test(imp.imped_calc, exp, msg)
        self._input_except_test(imp.imped_calc, exp, msg, 1, 'a', range, time=0)
        self._input_except_test(imp.imped_calc, exp, msg, \
                                new_frequency_array = [1, 2, 3])

        imp.wake_calc([1, 2, 3])
        
        self.assertEqual(imp.frequency_array, 0, "Expected freuency array to be 0")
        self.assertEqual(imp.impedance, 0, "Expected impedance to be 0")

        
    def test_input_table_imped(self):
        
        self._input_except_test(impSource._InputTable, exceptions.InputError,
                        "Expected InputError exception", [1], [1, 2], [3, 4])

        imp = impSource._InputTable([1, 2, 3], [4, 5, 6], [7, 8, 9])
        
        self.assertEqual(imp.frequency_array_loaded.tolist(), [0, 1, 2, 3], 
                         "Expected frequency array to to be [0, 1, 2, 3]")
        self.assertEqual(imp.Re_Z_array_loaded.tolist(), [0, 4, 5, 6], 
                         "Expected real impedance array to be [0, 4, 5, 6]")
        self.assertEqual(imp.Im_Z_array_loaded.tolist(), [0, 7, 8, 9], 
                         "Expected imag impedance array to be [0, 7, 8, 9]")

        self.assertEqual(imp.time_array, 0, "Expected time array to be 0")
        self.assertEqual(imp.wake, 0, "Expected impedance to be 0")

        exp = exceptions.WrongCalcError
        msg = 'Expected WrongCalcError exception'

        self._input_except_test(imp.wake_calc, exp, msg)
        self._input_except_test(imp.wake_calc, exp, msg, 1, 'a', range, time=0)
        self._input_except_test(imp.wake_calc, exp, msg, \
                                new_frequency_array = [1, 2, 3])
        
        imp.imped_calc([1, 2, 3])
        
        self.assertEqual(imp.time_array, 0, "Expected freuency array to be 0")
        self.assertEqual(imp.wake, 0, "Expected impedance to be 0")

    
    def test_impedance_table(self):

        imp = impSource.ImpedanceTable(range(10), range(10), 
                                       np.arange(0, -10, -1))

        exp = exceptions.WrongCalcError
        msg = 'Expected WrongCalcError exception'
        self._input_except_test(imp.wake_calc, exp, msg)
        self._input_except_test(imp.wake_calc, exp, msg, 1, 'a', range, time=0)
        self._input_except_test(imp.wake_calc, exp, msg, \
                                new_time_array = [1, 2, 3])

        imp.imped_calc([1, 2, 3])
        
        self.assertEqual(imp.frequency_array.tolist(), [1, 2, 3])
        self.assertEqual(imp.Re_Z_array.tolist(), [1, 2, 3])
        self.assertEqual(imp.Im_Z_array.tolist(), [-1, -2, -3])\
        
        self.assertEqual(imp.impedance.real.tolist(), [1, 2, 3])
        self.assertEqual(imp.impedance.imag.tolist(), [-1, -2, -3])
        
        
    def test_wakefield_table(self):
        
        imp = impSource.WakefieldTable(range(10), range(10))
        
        exp = exceptions.WrongCalcError
        msg = 'Expected WrongCalcError exception'
        self._input_except_test(imp.imped_calc, exp, msg)
        self._input_except_test(imp.imped_calc, exp, msg, 1, 'a', range, time=0)
        self._input_except_test(imp.imped_calc, exp, msg, \
                                new_frequency_array = [1, 2, 3])
        
        imp.wake_calc([1, 2, 3])
        self.assertEqual(imp.time_array.tolist(), [1, 2, 3])
        self.assertEqual(imp.wake_array.tolist(), [1, 2, 3])

    
if __name__ == '__main__':

    unittest.main()