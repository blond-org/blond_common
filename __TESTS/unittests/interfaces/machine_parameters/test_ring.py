# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for blond_common.interfaces.machine_parameters.ring.py
:Authors: **Markus Schwarz**, **Alexandre Lasheen**
"""

# General imports
# ---------------
import sys
import unittest
import numpy as np
import os
from scipy.constants import e, c

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../../'))

from blond_common.interfaces.machine_parameters.ring import Ring, RingSection, \
    machine_program
from blond_common.interfaces.beam.beam import Proton, Electron, Particle
from blond_common.devtools import exceptions as excpt
from blond_common import datatypes as dTypes


class TestRing(unittest.TestCase):

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
        momentum = 26e9  # eV/c
        particle = Proton()

        section = RingSection(length, alpha_0, momentum)
        ring = Ring(particle, [section])

        with self.subTest('Simple input - length'):
            np.testing.assert_equal(
                length, ring.circumference)

        with self.subTest('Simple input - alpha_0'):
            np.testing.assert_equal(
                alpha_0, ring.alpha_0)

        with self.subTest('Simple input - momentum'):
            np.testing.assert_equal(
                momentum, ring.momentum)

    def test_simple_input_othermethods(self):
        # Test other methods to pass the simplest input

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()

        section = RingSection(length, alpha_0, momentum)
        ring_reference = Ring(particle, [section])

        with self.subTest('Simple input - Single section'):
            ring = Ring(particle, section)
            np.testing.assert_equal(
                ring_reference.circumference, ring.circumference)
            np.testing.assert_equal(
                ring_reference.alpha_0, ring.alpha_0)
            np.testing.assert_equal(
                ring_reference.momentum, ring.momentum)

        with self.subTest('Simple input - Single section tuple'):
            ring = Ring(particle, (section, ))
            np.testing.assert_equal(
                ring_reference.circumference, ring.circumference)
            np.testing.assert_equal(
                ring_reference.alpha_0, ring.alpha_0)
            np.testing.assert_equal(
                ring_reference.momentum, ring.momentum)

        with self.subTest('Simple input - Direct classmethod'):
            ring = Ring.direct_input(particle, length, alpha_0, momentum)
            np.testing.assert_equal(
                ring_reference.circumference, ring.circumference)
            np.testing.assert_equal(
                ring_reference.alpha_0, ring.alpha_0)
            np.testing.assert_equal(
                ring_reference.momentum, ring.momentum)

    def test_particle_types(self):
        # Test various particle input

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c

        section = RingSection(length, alpha_0, momentum)

        with self.subTest('Particle type - Proton()'):
            particle = Proton()
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                Proton().mass, ring.Particle.mass)
            np.testing.assert_equal(
                Proton().charge, ring.Particle.charge)

        with self.subTest('Particle type - "proton"'):
            particle = "proton"
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                Proton().mass, ring.Particle.mass)
            np.testing.assert_equal(
                Proton().charge, ring.Particle.charge)

        with self.subTest('Particle type - Electron()'):
            particle = Electron()
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                Electron().mass, ring.Particle.mass)
            np.testing.assert_equal(
                Electron().charge, ring.Particle.charge)

        with self.subTest('Particle type - "electron"'):
            particle = "electron"
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                Electron().mass, ring.Particle.mass)
            np.testing.assert_equal(
                Electron().charge, ring.Particle.charge)

        with self.subTest('Particle type - Particle()'):
            user_mass = 208 * Proton().mass
            user_charge = 82
            particle = Particle(user_mass, user_charge)
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                user_mass, ring.Particle.mass)
            np.testing.assert_equal(
                user_charge, ring.Particle.charge)

    def test_other_synchronous_data(self):
        # Test passing other sync data

        length = 300  # m
        alpha_0 = 1e-3
        particle = Proton()
        tot_energy = 26e9  # eV
        kin_energy = 26e9  # eV
        bending_radius = 70  # m
        momentum = 26e9  # eV/c
        b_field = momentum / c / particle.charge / bending_radius  # T

        with self.subTest('Simple input - tot_energy'):
            section = RingSection(length, alpha_0, energy=tot_energy)
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                tot_energy, ring.energy)

        with self.subTest('Simple input - kin_energy'):
            section = RingSection(length, alpha_0, kin_energy=kin_energy)
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                kin_energy, ring.kin_energy)

        with self.subTest('Simple input - b_field'):
            section = RingSection(length, alpha_0, bending_field=b_field,
                                  bending_radius=bending_radius)
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                momentum, ring.momentum)

    def test_non_linear_momentum_compaction(self):
        # Test passing non linear momentum compaction factor

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()
        alpha_1 = 1e-6
        alpha_2 = 1e-9
        alpha_5 = 1e-12

        section = RingSection(length, alpha_0, momentum,
                              alpha_1=alpha_1,
                              alpha_2=alpha_2,
                              alpha_5=alpha_5)
        ring = Ring(particle, [section])

        with self.subTest('Simple input - alpha_1'):
            np.testing.assert_equal(
                alpha_1, ring.alpha_1)

        with self.subTest('Simple input - alpha_2'):
            np.testing.assert_equal(
                alpha_2, ring.alpha_2)

        with self.subTest('Simple input - alpha_5'):
            np.testing.assert_equal(
                alpha_5, ring.alpha_5)

    def test_non_linear_momentum_compaction_other_input(self):
        # Test passing non linear momentum compaction factor

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()
        alpha_1 = 1e-6
        alpha_2 = 1e-9
        alpha_5 = 1e-12

        ring = Ring.direct_input(particle, length, alpha_0, momentum,
                                 alpha_1=alpha_1,
                                 alpha_2=alpha_2,
                                 alpha_5=alpha_5)

        with self.subTest('Simple input - Direct input'):
            np.testing.assert_equal(
                alpha_1, ring.alpha_1)
            np.testing.assert_equal(
                alpha_2, ring.alpha_2)
            np.testing.assert_equal(
                alpha_5, ring.alpha_5)

    # Exception raising test --------------------------------------------------

    def test_assert_wrong_section_list(self):
        # Test the exception that other than RingSection is passed

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()
        section = RingSection(length, alpha_0, momentum)

        error_message = (
            "The RingSection_list should be exclusively composed " +
            "of RingSection object instances.")

        with self.subTest('Wrong RingSection_list - other type'):
            with self.assertRaisesRegex(excpt.InputError, error_message):
                Ring(particle, ['test'])

        with self.subTest('Wrong RingSection_list - multisection other type'):
            with self.assertRaisesRegex(excpt.InputError, error_message):
                Ring(particle, [section, 'test'])


if __name__ == '__main__':

    unittest.main()
