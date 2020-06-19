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

from blond_common.interfaces.machine_parameters.ring_section import RingSection, \
    machine_program
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
                length, section.length_design)

        with self.subTest('Simple input - momentum'):
            np.testing.assert_equal(
                momentum, section.synchronous_data)

        with self.subTest('Simple input - orbit_bump'):
            np.testing.assert_equal(
                length, section.length)

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
        orbit_bump = 0.001  # m

        section = RingSection(length, alpha_0, momentum,
                              orbit_bump=orbit_bump)

        with self.subTest('Orbit length input - section.length_design'):
            np.testing.assert_equal(
                length, section.length_design)

        with self.subTest('Orbit length input - section.length'):
            np.testing.assert_equal(
                length + orbit_bump, section.length)
            np.testing.assert_equal(
                orbit_bump, section.orbit_bump)

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
        orbit_bump = [0.001, 0.002, 0.003]  # m

        with self.subTest('Turn by turn program - Only momentum'):
            section = RingSection(length, alpha_0[0], momentum)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Turn by turn program - Only total energy'):
            section = RingSection(length, alpha_0[0], energy=energy)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                energy, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Turn by turn program - Only kinetic energy'):
            section = RingSection(length, alpha_0[0],
                                  kin_energy=kin_energy)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Turn by turn program - Only bending field'):
            section = RingSection(length, alpha_0[0],
                                  bending_field=bending_field,
                                  bending_radius=bending_radius)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                bending_field, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Turn by turn program - Only orbit bump'):
            section = RingSection(length, alpha_0[0], momentum[0],
                                  orbit_bump=orbit_bump)
            # The warning raised here is expected and treated by a specific
            # unittest

            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum[0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0[0], section.alpha_0)
            np.testing.assert_equal(
                orbit_bump, section.orbit_bump[0, :])
            np.testing.assert_equal(
                length + np.array(orbit_bump), section.length[0, :])

        with self.subTest('Turn by turn program - Only momentum compaction'):
            section = RingSection(length, alpha_0, momentum[0])
            # The warning raised here is expected and treated by a specific
            # unittest

            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum[0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :])
            np.testing.assert_equal(
                length, section.length)

        with self.subTest(
                'Turn by turn program - All programs'):
            section = RingSection(length, alpha_0, momentum,
                                  orbit_bump=orbit_bump)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :])
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :])
            np.testing.assert_equal(
                orbit_bump, section.orbit_bump[0, :])
            np.testing.assert_equal(
                length + np.array(orbit_bump), section.length[0, :])

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
        orbit_bump = [time_base, [0.001, 0.002, 0.003]]  # m

        with self.subTest('Time based program - Only momentum'):
            section = RingSection(length, alpha_0[1][0], momentum)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Time based program - Only total energy'):
            section = RingSection(length, alpha_0[1][0],
                                  energy=energy)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                energy, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Time based program - Only kinetic energy'):
            section = RingSection(length, alpha_0[1][0],
                                  kin_energy=kin_energy)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                kin_energy, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Time based program - Only bending field'):
            section = RingSection(length, alpha_0[1][0],
                                  bending_field=bending_field,
                                  bending_radius=bending_radius)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                bending_field, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                length, section.length)

        with self.subTest('Time based program - Only orbit length'):
            section = RingSection(length, alpha_0[1][0], momentum[1][0],
                                  orbit_bump=orbit_bump)
            # The warning raised here is expected and treated by a specific
            # unittest

            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum[1][0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0[1][0], section.alpha_0)
            np.testing.assert_equal(
                orbit_bump, section.orbit_bump[0, :, :])
            np.testing.assert_equal(
                [time_base, length + np.array(orbit_bump)[1, :]],
                section.length[0, :, :])

        with self.subTest('Time based program - Only momentum compaction'):
            section = RingSection(length, alpha_0, momentum[1][0])
            # The warning raised here is expected and treated by a specific
            # unittest

            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum[1][0], section.synchronous_data)
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :, :])
            np.testing.assert_equal(
                length, section.length)

        with self.subTest(
                'Time based program - All programs'):
            section = RingSection(length, alpha_0, momentum,
                                  orbit_bump=orbit_bump)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                momentum, section.synchronous_data[0, :, :])
            np.testing.assert_equal(
                alpha_0, section.alpha_0[0, :, :])
            np.testing.assert_equal(
                orbit_bump, section.orbit_bump[0, :, :])
            np.testing.assert_equal(
                [time_base, length + np.array(orbit_bump)[1, :]],
                section.length[0, :, :])

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
                [0, 1], section.alpha_orders)

        with self.subTest('Non-linear momentum compaction - alpha_2'):
            section = RingSection(length, alpha_0, momentum,
                                  alpha_2=alpha_2)
            np.testing.assert_equal(
                alpha_0, section.alpha_0)
            np.testing.assert_equal(
                alpha_2, section.alpha_2)
            np.testing.assert_equal(
                [0, 2], section.alpha_orders)

        with self.subTest('Non-linear momentum compaction - alpha_n'):
            section = RingSection(length, alpha_0, momentum,
                                  alpha_5=alpha_5)
            np.testing.assert_equal(
                alpha_0, section.alpha_0)
            np.testing.assert_equal(
                alpha_5, section.alpha_5)
            np.testing.assert_equal(
                [0, 5], section.alpha_orders)

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
                [0, 1, 2, 5], section.alpha_orders)

    def test_machine_program_input(self):
        # Passing a machine_program as input

        length = 300  # m

        alpha_0_single = machine_program(1e-3)
        alpha_0_turn = machine_program(1e-3, n_turns=5)
        alpha_0_time = machine_program([[0, 1, 2],
                                        [1e-3, 1.1e-3, 1.2e-3]])

        momentum_single = machine_program(26e9)  # eV
        momentum_turn = machine_program(26e9, n_turns=5)  # eV
        momentum_time = machine_program([[0, 1, 2],
                                         [26e9, 27e9, 28e9]])  # [s, eV]

        orbit_single = machine_program(0.001)  # m
        orbit_turn = machine_program(0.001, n_turns=5)  # m
        orbit_time = machine_program([[0, 1, 2],
                                      [0., 0.001, 0.002]])  # [s, m]

        with self.subTest('Machine program input - Single value'):
            section = RingSection(length, alpha_0_single, momentum_single,
                                  orbit_bump=orbit_single)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                alpha_0_single, section.alpha_0)
            np.testing.assert_equal(
                momentum_single, section.synchronous_data)
            np.testing.assert_equal(
                length + orbit_single, section.length)

        with self.subTest('Machine program input - Turn based'):
            section = RingSection(length, alpha_0_turn, momentum_turn,
                                  orbit_bump=orbit_turn)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                alpha_0_turn, section.alpha_0)
            np.testing.assert_equal(
                momentum_turn, section.synchronous_data)
            np.testing.assert_equal(
                length + np.array(orbit_turn), section.length)

        with self.subTest('Machine program input - Time based'):
            section = RingSection(length, alpha_0_time, momentum_time,
                                  orbit_bump=orbit_time)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                alpha_0_time, section.alpha_0)
            np.testing.assert_equal(
                momentum_time, section.synchronous_data)
            np.testing.assert_equal(
                length + np.array(orbit_time)[:, 1, :],
                section.length[:, 1, :])

    def test_datatype_input(self):
        # Passing a momentum_program as input

        length = 300  # m

        alpha_0_single = dTypes.ring_programs.momentum_compaction(1e-3)
        alpha_0_turn = dTypes.ring_programs.momentum_compaction(
            1e-3, n_turns=5)
        alpha_0_time = dTypes.ring_programs.momentum_compaction(
            [[0, 1, 2],
             [1e-3, 1.1e-3, 1.2e-3]])

        momentum_single = dTypes.ring_programs.momentum_program(26e9)  # eV
        momentum_turn = dTypes.ring_programs.momentum_program(
            26e9, n_turns=5)  # eV
        momentum_time = dTypes.ring_programs.momentum_program(
            [[0, 1, 2],
             [26e9, 27e9, 28e9]])  # [s, eV]

        orbit_single = dTypes.ring_programs.orbit_length_program(0.001)  # m
        orbit_turn = dTypes.ring_programs.orbit_length_program(
            0.001, n_turns=5)  # m
        orbit_time = dTypes.ring_programs.orbit_length_program(
            [[0, 1, 2],
             [0., 0.001, 0.002]])  # [s, m]

        with self.subTest('Datatype input - Single value'):
            section = RingSection(length, alpha_0_single, momentum_single,
                                  orbit_bump=orbit_single)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                alpha_0_single, section.alpha_0)
            np.testing.assert_equal(
                momentum_single, section.synchronous_data)
            np.testing.assert_equal(
                orbit_single, section.orbit_bump)
            np.testing.assert_equal(
                length + orbit_single, section.length)

        with self.subTest('Datatype input - Turn based'):
            section = RingSection(length, alpha_0_turn, momentum_turn,
                                  orbit_bump=orbit_turn)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                alpha_0_turn, section.alpha_0)
            np.testing.assert_equal(
                momentum_turn, section.synchronous_data)
            np.testing.assert_equal(
                orbit_turn, section.orbit_bump)
            np.testing.assert_equal(
                length + orbit_turn, section.length)

        with self.subTest('Datatype input - Time based'):
            section = RingSection(length, alpha_0_time, momentum_time,
                                  orbit_bump=orbit_time)
            np.testing.assert_equal(
                length, section.length_design)
            np.testing.assert_equal(
                alpha_0_time, section.alpha_0)
            np.testing.assert_equal(
                momentum_time, section.synchronous_data)
            np.testing.assert_equal(orbit_time, section.orbit_bump)
            np.testing.assert_equal(
                length + np.array(orbit_time)[:, 1, :],
                section.length[:, 1, :])

    # Exception raising test -------------------------------------------

    def test_assert_synchronous_data_input(self):
        # Test the exception that at least one synchronous data is passed

        length = 300  # m
        alpha_0 = 1e-3

        error_message = "Exactly one of \('momentum', 'kinetic_energy', " + \
            "'total_energy', 'bending_field'\) must be declared"

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

    def test_assert_wrong_turn_by_turn_length(self):
        # Test the exception that an alpha_n, orbit length and synchronous data
        # are turn
        # based and have different lengths

        length = 300  # m
        alpha_0 = [1e-3, 1e-3]
        momentum = [26e9, 27e9, 28e9]  # eV
        alpha_1 = [1e-6, 1e-6]
        orbit_bump = [0.01, 0.01]  # m

        with self.subTest('Wrong turn-by-turn momentum compaction - alpha_0'):
            order = 0
            attr_name = 'alpha_' + str(order)

            error_message = (
                'The input ' + attr_name +
                ' was passed as a turn based program but with ' +
                'different length than the synchronous data. ' +
                'Turn based programs should have the same length.')

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_0, momentum)

        with self.subTest('Wrong turn-by-turn momentum compaction - alpha_1'):
            order = 1
            attr_name = 'alpha_' + str(order)

            error_message = (
                'The input ' + attr_name +
                ' was passed as a turn based program but with ' +
                'different length than the synchronous data. ' +
                'Turn based programs should have the same length.')

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_0[0], momentum, alpha_1=alpha_1)

        with self.subTest(
                'Wrong turn-by-turn momentum compaction - orbit_bump'):

            attr_name = 'orbit_bump'

            error_message = (
                'The input ' + attr_name +
                ' was passed as a turn based program but with ' +
                'different length than the synchronous data. ' +
                'Turn based programs should have the same length.')

            with self.assertRaisesRegex(excpt.InputError, error_message):
                RingSection(length, alpha_0[0], momentum,
                            orbit_bump=orbit_bump)

    def test_warning_turn_time_mix(self):
        # Test the warning that time based programs and turn based
        # were mixed

        length = 300  # m
        momentum = [[0, 1, 2], [26e9, 27e9, 28e9]]  # eV
        alpha_0 = [1e-3, 1e-3]
        alpha_1 = [1e-6, 1e-6]
        orbit_bump = [0.01, 0.01]  # m

        with self.subTest('Turn/time program mix - alpha_0'):

            order = 0
            attr_name = 'alpha_' + str(order)
            warn_message = (
                'The synchronous data was defined time based while the ' +
                'input ' + attr_name + ' was defined turn base, this may' +
                'lead to errors in the Ring object after interpolation.')

            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0, momentum)

        with self.subTest('Turn/time program mix - alpha_1'):

            order = 1
            attr_name = 'alpha_' + str(order)
            warn_message = (
                'The synchronous data was defined time based while the ' +
                'input ' + attr_name + ' was defined turn base, this may' +
                'lead to errors in the Ring object after interpolation.')

            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0[0], momentum, alpha_1=alpha_1)

        with self.subTest('Turn/time program mix - orbit_bump'):

            attr_name = 'orbit_bump'
            warn_message = (
                'The synchronous data was defined time based while the ' +
                'input ' + attr_name + ' was defined turn base, this may' +
                'lead to errors in the Ring object after interpolation.')

            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0[0], momentum,
                            orbit_bump=orbit_bump)

    def test_warning_single_prog_mix(self):
        # Test the warning that single value programs and turn/time based
        # were mixed

        length = 300  # m
        momentum = 26e9  # eV
        alpha_0 = [1e-3, 1e-3]
        alpha_1 = [1e-6, 1e-6]
        orbit_bump = [0.01, 0.01]  # m

        with self.subTest('Single and Turn/time program mix - alpha_0'):

            order = 0
            attr_name = 'alpha_' + str(order)
            warn_message = (
                'The synchronous data was defined as single element while the ' +
                'input ' + attr_name + ' was defined turn or time based. ' +
                'Only the first element of the program will be taken in ' +
                'the Ring object after treatment.')

            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0, momentum)

        with self.subTest('Turn/time program mix - alpha_1'):

            order = 1
            attr_name = 'alpha_' + str(order)
            warn_message = (
                'The synchronous data was defined as single element while the ' +
                'input ' + attr_name + ' was defined turn or time based. ' +
                'Only the first element of the program will be taken in ' +
                'the Ring object after treatment.')

            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0[0], momentum, alpha_1=alpha_1)

        with self.subTest('Turn/time program mix - orbit_bump'):

            attr_name = 'orbit_bump'
            warn_message = (
                'The synchronous data was defined as single element while the ' +
                'input ' + attr_name + ' was defined turn or time based. ' +
                'Only the first element of the program will be taken in ' +
                'the Ring object after treatment.')

            with self.assertWarnsRegex(Warning, warn_message):
                RingSection(length, alpha_0[0], momentum,
                            orbit_bump=orbit_bump)

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
