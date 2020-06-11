# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for blond_common.interfaces.input_parameters.ring_section.py
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

from blond_common.interfaces.input_parameters.ring_section import RingSection
from blond_common.devtools import exceptions as excpt
from blond_common import datatypes as dTypes


class TestRingSection(unittest.TestCase):

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

    def test_simple_input(self):
        # Test the simplest input

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV

        section = RingSection(length, alpha_0, momentum)

        with self.subTest('Simple input - length'):
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Simple input - momentum'):
            np.testing.assert_equal(
                momentum, section.synchronous_data)

        with self.subTest('Simple input - orbit_length'):
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Simple input - alpha_0'):
            np.testing.assert_equal(
                alpha_0, section.alpha_0)

    def test_other_sychronous_data(self):
        # Test other synchronous data
        # kinetic energy, total energy, bending field

        length = 6900  # m
        alpha_0 = 1e-3
        energy = 26e9  # eV
        kin_energy = 25e9  # eV
        bending_field = 1.  # T
        bending_radius = 749  # m

        with self.subTest('Other sychronous data - Total energy'):
            section = RingSection(length, alpha_0, energy=energy)
            np.testing.assert_equal(
                energy, section.synchronous_data)

        with self.subTest('Other sychronous data - Kinetic energy'):
            section = RingSection(length, alpha_0, kin_energy=kin_energy)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data)

        with self.subTest('Other sychronous data - Bending field'):
            section = RingSection(length, alpha_0,
                                  bending_field=bending_field,
                                  bending_radius=bending_radius)
            np.testing.assert_equal(
                bending_field, section.synchronous_data)
            np.testing.assert_equal(
                bending_radius, section.bending_radius)

    def test_orbit_length(self):

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV
        orbit_length = 300.001  # m

        section = RingSection(length, alpha_0, momentum,
                              orbit_length=orbit_length)

        with self.subTest('Orbit length input - section.length'):
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Orbit length input - section.orbit_length'):
            np.testing.assert_equal(
                orbit_length, section.orbit_length)

    def test_turn_by_turn_prog(self):
        # Test turn by turn program
        # Note that shape is defined by the datatype shaping
        # inside the RingSection object
        # -> (n_sections, n_turns)

        length = 6900  # m
        alpha_0 = [1e-3, 1e-3, 1e-3]
        momentum = [26e9, 27e9, 28e9]
        energy = [26e9, 27e9, 28e9]  # eV
        kin_energy = [25e9, 26e9, 27e9]  # eV
        bending_field = [1.0, 1.1, 1.2]  # T
        bending_radius = 749  # m
        orbit_length = [6900.001, 6900.002, 6900.003]  # T

        with self.subTest('Turn by turn program - Only momentum'):
            section = RingSection(length, alpha_0[0], momentum)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Turn by turn program - Only total energy'):
            section = RingSection(length, alpha_0[0], energy=energy)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                energy, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Turn by turn program - Only kinetic energy'):
            section = RingSection(length, alpha_0[0],
                                  kin_energy=kin_energy)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Turn by turn program - Only bending field'):
            section = RingSection(length, alpha_0[0],
                                  bending_field=bending_field,
                                  bending_radius=bending_radius)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                bending_field, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Turn by turn program - Only orbit length'):
            section = RingSection(length, alpha_0[0], momentum[0],
                                  orbit_length=orbit_length)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum[0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                orbit_length, section.orbit_length[0, :])

        with self.subTest('Turn by turn program - Only momentum compaction'):
            section = RingSection(length, alpha_0, momentum[0])
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum[0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :])
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest(
                'Turn by turn program - All programs'):
            section = RingSection(length, alpha_0, momentum,
                                  orbit_length=orbit_length)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :])
            np.testing.assert_equal(
                orbit_length, section.orbit_length[0, :])

    def test_time_based_prog(self):
        # Test time based program
        # Note that shape is defined by the datatype shaping
        # inside the RingSection object
        # -> (n_sections, 2, n_turns)

        time_base = [0, 1, 2]  # s

        length = 6900  # m
        alpha_0 = [time_base, [1e-3, 1e-3, 1e-3]]
        momentum = [time_base, [26e9, 27e9, 28e9]]
        energy = [time_base, [26e9, 27e9, 28e9]]  # eV
        kin_energy = [time_base, [25e9, 26e9, 27e9]]  # eV
        bending_field = [time_base, [1.0, 1.1, 1.2]]  # T
        bending_radius = 749  # m
        orbit_length = [time_base, [6900.001, 6900.002, 6900.003]]  # T

        with self.subTest('Time based program - Only momentum'):
            section = RingSection(length, alpha_0[1][0], momentum)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Time based program - Only total energy'):
            section = RingSection(length, alpha_0[1][0],
                                  energy=energy)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                energy, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Time based program - Only kinetic energy'):
            section = RingSection(length, alpha_0[1][0],
                                  kin_energy=kin_energy)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Time based program - Only bending field'):
            section = RingSection(length, alpha_0[1][0],
                                  bending_field=bending_field,
                                  bending_radius=bending_radius)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                bending_field, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest('Time based program - Only orbit length'):
            section = RingSection(length, alpha_0[1][0], momentum[1][0],
                                  orbit_length=orbit_length)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum[1][0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                orbit_length, section.orbit_length[0, :, :])

        with self.subTest('Time based program - Only momentum compaction'):
            section = RingSection(length, alpha_0, momentum[1][0])
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum[1][0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :, :])
            np.testing.assert_equal(
                length, section.orbit_length)

        with self.subTest(
                'Time based program - All programs'):
            section = RingSection(length, alpha_0, momentum,
                                  orbit_length=orbit_length)
            np.testing.assert_equal(
                length, section.length)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :, :])
            np.testing.assert_equal(
                orbit_length, section.orbit_length[0, :, :])

    def test_non_linear_alpha(self):
        # Passing non-linear momentum compaction factors

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV
        alpha_1 = 1e-6
        alpha_2 = 1e-9
        alpha_5 = 1e-12

        with self.subTest('Non-linear momentum compaction - alpha_1'):
            section = RingSection(length, alpha_0, momentum,
                                  alpha_1=alpha_1)
            np.testing.assert_equal(
                alpha_0, section.alpha_0)
            np.testing.assert_equal(
                alpha_1, section.alpha_1)
            np.testing.assert_equal(
                1, section.alpha_order)

        with self.subTest('Non-linear momentum compaction - alpha_2'):
            section = RingSection(length, alpha_0, momentum,
                                  alpha_2=alpha_2)
            np.testing.assert_equal(
                alpha_0, section.alpha_0)
            np.testing.assert_equal(
                alpha_2, section.alpha_2)
            np.testing.assert_equal(
                2, section.alpha_order)

        with self.subTest('Non-linear momentum compaction - alpha_n'):
            section = RingSection(length, alpha_0, momentum,
                                  alpha_5=alpha_5)
            np.testing.assert_equal(
                alpha_0, section.alpha_0)
            np.testing.assert_equal(
                alpha_5, section.alpha_5)
            np.testing.assert_equal(
                5, section.alpha_order)

        with self.subTest('Non-linear momentum compaction - multiple alpha_n'):
            section = RingSection(length, alpha_0, momentum,
                                  alpha_1=alpha_1,
                                  alpha_2=alpha_2,
                                  alpha_5=alpha_5)
            np.testing.assert_equal(
                alpha_0, section.alpha_0)
            np.testing.assert_equal(
                alpha_1, section.alpha_1)
            np.testing.assert_equal(
                alpha_2, section.alpha_2)
            np.testing.assert_equal(
                0, section.alpha_3)
            np.testing.assert_equal(
                0, section.alpha_4)
            np.testing.assert_equal(
                alpha_5, section.alpha_5)
            np.testing.assert_equal(
                5, section.alpha_order)

    # Exception raising test --------------------------------------------------

    def test_assert_synchronous_data_input(self):
        # Test the exception that at least one synchronous data is passed

        length = 300  # m
        alpha_0 = 1e-3

        error_message = "Exactly one of \('momentum', 'kin_energy', " +\
            "'energy', 'B_field'\) must be declared"

        with self.assertRaisesRegex(excpt.InputError, error_message):
            RingSection(length, alpha_0)

    def test_assert_missing_bending_radius(self):
        # Test the exception that the bending radius is not passed

        length = 300  # m
        alpha_0 = 1e-3
        bending_field = 1.  # T

        error_message = "If bending_field is used, bending_radius must " +\
            "be defined."

        with self.assertRaisesRegex(excpt.InputError, error_message):
            RingSection(length, alpha_0, bending_field=bending_field)

    def test_assert_wrong_turn_by_turn_alpha_length(self):
        # Test the exception that an alpha_n and synchronous data are turn
        # based and have different lengths

        length = 300  # m
        alpha_0 = [1e-3, 1e-3]
        momentum = [26e9, 27e9, 28e9]  # eV
        alpha_1 = [1e-6, 1e-6]

        with self.subTest('Wrong turn-by-turn momentum compaction - alpha_0'):
            order = 0

            error_message = (
                'The momentum compaction alpha_' + str(order) +
                ' was passed as a turn based program but with ' +
                'different length than the synchronous data. ' +
                'Turn based programs should have the same length.')

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_0, momentum)

        with self.subTest('Wrong turn-by-turn momentum compaction - alpha_1'):
            order = 1

            error_message = (
                'The momentum compaction alpha_' + str(order) +
                ' was passed as a turn based program but with ' +
                'different length than the synchronous data. ' +
                'Turn based programs should have the same length.')

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_0[0], momentum, alpha_1=alpha_1)

    def test_warning_turn_time_mix(self):
        # Test the warning that time based programs and turn based
        # were mixed

        length = 300  # m
        momentum = [[0, 1, 2], [26e9, 27e9, 28e9]]  # eV
        alpha_0 = [1e-3, 1e-3]
        alpha_1 = [1e-6, 1e-6]

        warn_message = (
            'The synchronous data was defined time based while the ' +
            'momentum compaction was defined turn base, this may' +
            'lead to errors in the Ring object after interpolation')

        with self.subTest('Turn/time program mix - alpha_0'):
            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0, momentum)

        with self.subTest('Turn/time program mix - alpha_1'):
            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0[0], momentum, alpha_1=alpha_1)

    def test_assert_wrong_alpha_n(self):
        # Test the exception that an alpha_n is incorrectly passed

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV
        alpha5 = 1e-12

        error_message = ('The keyword argument alpha5 was interpreted ' +
                         'as non-linear momentum compaction factor. ' +
                         'The correct syntax is alpha_n.')

        with self.assertRaisesRegex(excpt.InputError, error_message):
            RingSection(length, alpha_0, momentum, alpha5=alpha5)

    def test_assert_wrong_alpha_order_datatype(self):
        # Test the exception that the user defined momentum compaction
        # with a different order than the kwarg passed

        length = 300  # m
        alpha_0 = dTypes.ring_programs.momentum_compaction(1e-3, order=0)
        momentum = 26e9  # eV
        alpha_1 = dTypes.ring_programs.momentum_compaction(1e-6, order=1)

        with self.subTest('Wrong datatype order - alpha_0'):
            order = 0
            error_message = ("The order of the datatype passed as keyword " +
                             "argument alpha_%s do not match" % (order))

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_1, momentum)

        with self.subTest('Wrong datatype order - alpha_1'):
            order = 1
            error_message = ("The order of the datatype passed as keyword " +
                             "argument alpha_%s do not match" % (order))

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_0, momentum, alpha_1=alpha_0)

        with self.subTest('Wrong datatype order - alpha_n'):
            order = 3
            error_message = ("The order of the datatype passed as keyword " +
                             "argument alpha_%s do not match" % (order))

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_0, momentum, alpha_1=alpha_1,
                            alpha_3=alpha_1)


if __name__ == '__main__':

    unittest.main()
