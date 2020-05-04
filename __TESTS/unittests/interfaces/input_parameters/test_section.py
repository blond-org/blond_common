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

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../../'))

from blond_common.interfaces.input_parameters.ring_section import Section
from blond_common.devtools import exceptions as excpt


class TestSection(unittest.TestCase):

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

        section_length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV

        section = Section(section_length, alpha_0, momentum)

        with self.subTest('Simple input - section_length'):
            np.testing.assert_equal(
                section_length, section.section_length)

        with self.subTest('Simple input - momentum'):
            np.testing.assert_equal(
                momentum, section.synchronous_data)

        with self.subTest('Simple input - alpha_0'):
            np.testing.assert_equal(
                alpha_0, section.alpha_0)

    def test_other_sychronous_data(self):
        # Test other synchronous data
        # kinetic energy, total energy, bending field

        section_length = 6900  # m
        alpha_0 = 1e-3
        energy = 26e9  # eV
        kin_energy = 25e9  # eV
        bending_field = 1.  # T
        bending_radius = 749  # m

        with self.subTest('Other sychronous data - Total energy'):
            section = Section(section_length, alpha_0, energy=energy)
            np.testing.assert_equal(
                energy, section.synchronous_data)

        with self.subTest('Other sychronous data - Kinetic energy'):
            section = Section(section_length, alpha_0, kin_energy=kin_energy)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data)

        with self.subTest('Other sychronous data - Bending field'):
            section = Section(section_length, alpha_0,
                              bending_field=bending_field,
                              bending_radius=bending_radius)
            np.testing.assert_equal(
                bending_field, section.synchronous_data)
            np.testing.assert_equal(
                bending_radius, section.bending_radius)

    def test_turn_by_turn_prog(self):
        # Test turn by turn program
        # Note that shape is defined by the datatype shaping
        # inside the Section object
        # -> (n_sections, n_turns)

        section_length = [6900.0, 6900.01, 6900.02]  # m
        alpha_0 = [1e-3, 1e-3, 1e-3]
        momentum = [26e9, 27e9, 28e9]
        energy = [26e9, 27e9, 28e9]  # eV
        kin_energy = [25e9, 26e9, 27e9]  # eV
        bending_field = [1.0, 1.1, 1.2]  # T
        bending_radius = 749  # m

        with self.subTest('Turn by turn program - Only momentum'):
            section = Section(section_length[0], alpha_0[0], momentum)
            np.testing.assert_equal(
                section_length[0], section.section_length)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)

        with self.subTest('Turn by turn program - Only total energy'):
            section = Section(section_length[0], alpha_0[0], energy=energy)
            np.testing.assert_equal(
                section_length[0], section.section_length)
            np.testing.assert_equal(
                energy, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)

        with self.subTest('Turn by turn program - Only kinetic energy'):
            section = Section(section_length[0], alpha_0[0],
                              kin_energy=kin_energy)
            np.testing.assert_equal(
                section_length[0], section.section_length)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)

        with self.subTest('Turn by turn program - Only bending field'):
            section = Section(section_length[0], alpha_0[0],
                              bending_field=bending_field,
                              bending_radius=bending_radius)
            np.testing.assert_equal(
                section_length[0], section.section_length)
            np.testing.assert_equal(
                bending_field, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)

        with self.subTest('Turn by turn program - Only section length'):
            section = Section(section_length, alpha_0[0], momentum[0])
            np.testing.assert_equal(
                section_length, section.section_length[0, :])
            np.testing.assert_equal(
                momentum[0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)

        with self.subTest('Turn by turn program - Only momentum compaction'):
            section = Section(section_length[0], alpha_0, momentum[0])
            np.testing.assert_equal(
                section_length[0], section.section_length)
            np.testing.assert_equal(
                momentum[0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :])

        with self.subTest(
                'Turn by turn program - All programs'):
            section = Section(section_length, alpha_0, momentum)
            np.testing.assert_equal(
                section_length, section.section_length[0, :])
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :])

    def test_time_based_prog(self):
        # Test time based program
        # Note that shape is defined by the datatype shaping
        # inside the Section object
        # -> (n_sections, 2, n_turns)

        time_base = [0, 1, 2]  # s

        section_length = [time_base, [6900.0, 6900.01, 6900.02]]  # m
        alpha_0 = [time_base, [1e-3, 1e-3, 1e-3]]
        momentum = [time_base, [26e9, 27e9, 28e9]]
        energy = [time_base, [26e9, 27e9, 28e9]]  # eV
        kin_energy = [time_base, [25e9, 26e9, 27e9]]  # eV
        bending_field = [time_base, [1.0, 1.1, 1.2]]  # T
        bending_radius = 749  # m

        with self.subTest('Time based program - Only momentum'):
            section = Section(section_length[1][0], alpha_0[1][0], momentum)
            np.testing.assert_equal(
                section_length[1][0], section.section_length)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)

        with self.subTest('Time based program - Only total energy'):
            section = Section(section_length[1][0], alpha_0[1][0],
                              energy=energy)
            np.testing.assert_equal(
                section_length[1][0], section.section_length)
            np.testing.assert_equal(
                energy, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)

        with self.subTest('Time based program - Only kinetic energy'):
            section = Section(section_length[1][0], alpha_0[1][0],
                              kin_energy=kin_energy)
            np.testing.assert_equal(
                section_length[1][0], section.section_length)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)

        with self.subTest('Time based program - Only bending field'):
            section = Section(section_length[1][0], alpha_0[1][0],
                              bending_field=bending_field,
                              bending_radius=bending_radius)
            np.testing.assert_equal(
                section_length[1][0], section.section_length)
            np.testing.assert_equal(
                bending_field, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)

        with self.subTest('Time based program - Only section length'):
            section = Section(section_length, alpha_0[1][0], momentum[1][0])
            np.testing.assert_equal(
                section_length, section.section_length[0, :, :])
            np.testing.assert_equal(
                momentum[1][0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)

        with self.subTest('Time based program - Only momentum compaction'):
            section = Section(section_length[1][0], alpha_0, momentum[1][0])
            np.testing.assert_equal(
                section_length[1][0], section.section_length)
            np.testing.assert_equal(
                momentum[1][0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :, :])

        with self.subTest(
                'Time based program - All programs'):
            section = Section(section_length, alpha_0, momentum)
            np.testing.assert_equal(
                section_length, section.section_length[0, :, :])
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :, :])

    # Exception raising test --------------------------------------------------

    def test_assert_synchronous_data_input(self):
        # Test the assertion that at least one synchronous data is passed

        section_length = 300  # m
        alpha_0 = 1e-3

        with self.assertRaises(excpt.InputError):
            Section(section_length, alpha_0)


if __name__ == '__main__':

    unittest.main()
