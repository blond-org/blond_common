# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for blond_common.interfaces.machine_parameters.ring_section.py
:Authors: **Alexandre Lasheen**
"""

# General imports
# ---------------
import sys
import unittest
import numpy as np
import os
import warnings

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../../'))

from blond_common.interfaces.machine_parameters.rf_system import RFSystem
# from blond_common.devtools import exceptions as excpt
# from blond_common import datatypes as dTypes


class TestRFSystem(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):

        pass

    def assertIsNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """

        standardMsg = "%s is not NaN" % str(value)

        if not np.isnan(value):
            self.fail(self._formatMessage(msg, standardMsg))

    # Input test --------------------------------------------------------------

    def test_simple_input_harmonic(self):
        # Test the simplest input with an input harmonic

        voltage = 1e3  # V
        phase = np.pi  # rad
        harmonic = 2

        rf_system = RFSystem(voltage, phase, harmonic)

        with self.subTest('Simple input - voltage'):
            np.testing.assert_equal(
                voltage, rf_system.voltage)

        with self.subTest('Simple input - phase'):
            np.testing.assert_equal(
                phase, rf_system.phase)

        with self.subTest('Simple input - harmonic'):
            np.testing.assert_equal(
                harmonic, rf_system.harmonic)

        with self.subTest('Simple input - frequency'):
            np.testing.assert_equal(
                None, rf_system.frequency)

    def test_simple_input_fixed_freq(self):
        # Test the simplest input with a fixed rf frequency

        voltage = 1e3  # V
        phase = np.pi  # rad
        frequency = 1e3  # Hz

        rf_system = RFSystem(voltage, phase, frequency=frequency)

        with self.subTest('Simple input - voltage'):
            np.testing.assert_equal(
                voltage, rf_system.voltage)

        with self.subTest('Simple input - phase'):
            np.testing.assert_equal(
                phase, rf_system.phase)

        with self.subTest('Simple input - harmonic'):
            np.testing.assert_equal(
                None, rf_system.harmonic)

        with self.subTest('Simple input - frequency'):
            np.testing.assert_equal(
                frequency, rf_system.frequency)

    def test_turn_by_turn_prog(self):
        # Test turn by turn program
        # Note that shape is defined by the datatype shaping
        # inside the RingSection object
        # -> (n_rf, n_turns)

        voltage = [1e3, 1.1e3, 1.2e3]  # V
        phase = [np.pi, np.pi - 0.1, np.pi - 0.2]  # rad
        harmonic = [2, 3, 4]
        frequency = [1e3, 1.1e3, 1.2e3]  # Hz

        with self.subTest('Turn by turn program - Only voltage'):
            rf_system = RFSystem(voltage, phase[0], harmonic[0])
            np.testing.assert_equal(
                voltage, rf_system.voltage[0, :])
            np.testing.assert_equal(
                phase[0], rf_system.phase)
            np.testing.assert_equal(
                harmonic[0], rf_system.harmonic)
            np.testing.assert_equal(
                None, rf_system.frequency)

        with self.subTest('Turn by turn program - Only phase'):
            rf_system = RFSystem(voltage[0], phase, harmonic[0])
            np.testing.assert_equal(
                voltage[0], rf_system.voltage)
            np.testing.assert_equal(
                phase, rf_system.phase[0, :])
            np.testing.assert_equal(
                harmonic[0], rf_system.harmonic)
            np.testing.assert_equal(
                None, rf_system.frequency)

        with self.subTest('Turn by turn program - Only harmonic'):
            rf_system = RFSystem(voltage[0], phase[0], harmonic)
            np.testing.assert_equal(
                voltage[0], rf_system.voltage)
            np.testing.assert_equal(
                phase[0], rf_system.phase)
            np.testing.assert_equal(
                harmonic, rf_system.harmonic[0, :])
            np.testing.assert_equal(
                None, rf_system.frequency)

        with self.subTest('Turn by turn program - Only frequency'):
            rf_system = RFSystem(voltage[0], phase[0], frequency=frequency)
            np.testing.assert_equal(
                voltage[0], rf_system.voltage)
            np.testing.assert_equal(
                phase[0], rf_system.phase)
            np.testing.assert_equal(
                None, rf_system.harmonic)
            np.testing.assert_equal(
                frequency, rf_system.frequency[0, :])

        with self.subTest('Turn by turn program - All programs except freq'):
            rf_system = RFSystem(voltage, phase, harmonic)
            np.testing.assert_equal(
                voltage, rf_system.voltage[0, :])
            np.testing.assert_equal(
                phase, rf_system.phase[0, :])
            np.testing.assert_equal(
                harmonic, rf_system.harmonic[0, :])
            np.testing.assert_equal(
                None, rf_system.frequency)

    # Exception raising test --------------------------------------------------


if __name__ == '__main__':

    unittest.main()
