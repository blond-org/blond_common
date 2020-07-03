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
from scipy.constants import c

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
    '''
    TODO:
    - testing parameters_at_... functions
    - direct_input multisection and datatypes tests
    - options tests (t_start, t_stop, interp_time, store_turns...)
    - calculated parameters tests (delta_E, f_rev)
    '''

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

        with self.subTest('Simple input - circumference'):
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
        momentum = 26e9  # eV/c
        tot_energy = 26e9  # eV
        kin_energy = 26e9  # eV
        bending_radius = 70  # m
        b_field = momentum / c / particle.charge / bending_radius  # T

        with self.subTest('Other sync data - tot_energy'):
            section = RingSection(length, alpha_0, energy=tot_energy)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                tot_energy, ring.energy)

        with self.subTest('Other sync data - kin_energy'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, kin_energy=kin_energy)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                kin_energy, ring.kin_energy)

        with self.subTest('Other sync data - b_field'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, bending_field=b_field,
                                  bending_radius=bending_radius)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                momentum, ring.momentum)

    def test_other_synchronous_data_other_input(self):
        # Test passing other sync data

        length = 300  # m
        alpha_0 = 1e-3
        particle = Proton()
        momentum = 26e9  # eV/c
        tot_energy = 26e9  # eV
        kin_energy = 26e9  # eV
        bending_radius = 70  # m
        b_field = momentum / c / particle.charge / bending_radius  # T

        with self.subTest('Other sync data - Direct - tot_energy'):
            ring = Ring.direct_input(particle, length, alpha_0,
                                     energy=tot_energy)
            np.testing.assert_allclose(
                tot_energy, ring.energy)

        with self.subTest('Other sync data - Direct - kin_energy'):
            # NB: allclose due to double conversion
            ring = Ring.direct_input(
                particle, length, alpha_0, kin_energy=kin_energy)
            np.testing.assert_allclose(
                kin_energy, ring.kin_energy)

        with self.subTest('Other sync data - Direct - b_field'):
            # NB: allclose due to double conversion
            ring = Ring.direct_input(particle, length, alpha_0,
                                     bending_field=b_field,
                                     bending_radius=bending_radius)
            np.testing.assert_allclose(
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

        with self.subTest('Nonlinear alpha - alpha_1'):
            np.testing.assert_equal(
                alpha_1, ring.alpha_1)

        with self.subTest('Nonlinear alpha - alpha_2'):
            np.testing.assert_equal(
                alpha_2, ring.alpha_2)

        with self.subTest('Nonlinear alpha - alpha_5'):
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

    def test_turn_based_sync_program(self):
        # Test passing turn based momentum program

        length = 300  # m
        alpha_0 = 1e-3
        particle = Proton()
        momentum = [26e9, 27e9, 28e9]  # eV/c
        tot_energy = [26e9, 27e9, 28e9]  # eV
        kin_energy = [26e9, 27e9, 28e9]  # eV
        bending_radius = 70  # m
        b_field = np.array(momentum) / c / \
            particle.charge / bending_radius  # T

        with self.subTest('Turn based program - momentum'):
            section = RingSection(length, alpha_0, momentum)
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                momentum, ring.momentum[0, :])

        with self.subTest('Turn based program - tot_energy'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, energy=tot_energy)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                tot_energy, ring.energy[0, :])

        with self.subTest('Turn based program - kin_energy'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, kin_energy=kin_energy)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                kin_energy, ring.kin_energy[0, :])

        with self.subTest('Turn based program - bending_field'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, bending_field=b_field,
                                  bending_radius=bending_radius)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                momentum, ring.momentum[0, :])

    def test_time_based_sync_program(self):
        # Test passing non linear momentum compaction factor

        length = 300  # m
        alpha_0 = 1e-3
        particle = Proton()
        momentum = [[0, 100e-6], [26e9, 26e9]]  # eV/c
        tot_energy = [[0, 100e-6], [26e9, 26e9]]  # eV
        kin_energy = [[0, 100e-6], [26e9, 26e9]]  # eV
        bending_radius = 70  # m
        b_field = np.array(momentum)
        b_field[1, :] *= 1 / c / \
            particle.charge / bending_radius  # T

        with self.subTest('Time based program - momentum'):
            section = RingSection(length, alpha_0, momentum)
            ring = Ring(particle, [section])
            np.testing.assert_equal(
                np.mean(momentum[1]), np.mean(ring.momentum))

        with self.subTest('Time based program - tot_energy'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, energy=tot_energy)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                np.mean(tot_energy[1]), np.mean(ring.energy))

        with self.subTest('Time based program - kin_energy'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, kin_energy=kin_energy)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                np.mean(kin_energy[1]), np.mean(ring.kin_energy))

        with self.subTest('Time based program - b_field'):
            # NB: allclose due to double conversion
            section = RingSection(length, alpha_0, bending_field=b_field,
                                  bending_radius=bending_radius)
            ring = Ring(particle, [section])
            np.testing.assert_allclose(
                np.mean(momentum[1]), np.mean(ring.momentum))

    def test_simple_input_multisection(self):
        # Test the simplest input in multisection configuration

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()

        section = RingSection(length / 2, alpha_0, momentum)
        ring = Ring(particle, [section, section])

        with self.subTest('Simple input - Multisection - circumference'):
            np.testing.assert_equal(
                length, ring.circumference)

        with self.subTest('Simple input - Multisection - alpha_0'):
            np.testing.assert_equal(
                alpha_0, ring.alpha_0)

        with self.subTest('Simple input - Multisection - momentum'):
            np.testing.assert_equal(
                momentum, ring.momentum)

    def test_simple_input_multisection_othermethods(self):
        # Test other methods to pass the simplest input with multisection

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()

        section = RingSection(length / 2, alpha_0, momentum)
        ring_reference = Ring(particle, [section, section])

        with self.subTest('Simple input - Multisection tuple'):
            ring = Ring(particle, (section, section))
            np.testing.assert_equal(
                ring_reference.circumference, ring.circumference)
            np.testing.assert_equal(
                ring_reference.alpha_0, ring.alpha_0)
            np.testing.assert_equal(
                ring_reference.momentum, ring.momentum)

    def test_other_synchronous_data_multisection(self):
        # Test passing other sync data in multisection

        length = 300  # m
        alpha_0 = 1e-3
        particle = Proton()
        momentum = 26e9  # eV/c
        tot_energy = 26e9  # eV
        kin_energy = 26e9  # eV
        bending_radius = 70  # m
        b_field = momentum / c / particle.charge / bending_radius  # T

        with self.subTest('Other sync data - Multisection - tot_energy'):
            section_1 = RingSection(length / 2, alpha_0, momentum)
            section_2 = RingSection(length / 2, alpha_0, energy=tot_energy)
            ring = Ring(particle, [section_1, section_2])
            np.testing.assert_allclose(
                momentum, ring.momentum[0, :])
            np.testing.assert_allclose(
                tot_energy, ring.energy[1, :])

        with self.subTest('Other sync data - Multisection - kin_energy'):
            # NB: allclose due to double conversion
            section_1 = RingSection(length / 2, alpha_0, momentum)
            section_2 = RingSection(length / 2, alpha_0, kin_energy=kin_energy)
            ring = Ring(particle, [section_1, section_2])
            np.testing.assert_allclose(
                momentum, ring.momentum[0, :])
            np.testing.assert_allclose(
                kin_energy, ring.kin_energy[1, :])

        with self.subTest('Other sync data - Multisection - b_field'):
            # NB: allclose due to double conversion
            section_1 = RingSection(length / 2, alpha_0, momentum)
            section_2 = RingSection(length / 2, alpha_0, bending_field=b_field,
                                    bending_radius=bending_radius)
            ring = Ring(particle, [section_1, section_2])
            np.testing.assert_allclose(
                momentum, ring.momentum[0, :])
            np.testing.assert_allclose(
                momentum, ring.momentum[1, :])

    def test_non_linear_momentum_compaction_multisection(self):
        # Test passing non linear momentum compaction factor in multisection

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()
        alpha_1 = 1e-6
        alpha_2 = 1e-9
        alpha_5 = 1e-12

        section_1 = RingSection(length, alpha_0, momentum)
        section_2 = RingSection(length, alpha_0, momentum,
                                alpha_1=alpha_1,
                                alpha_2=alpha_2,
                                alpha_5=alpha_5)
        ring = Ring(particle, [section_1, section_2])

        with self.subTest('Non linear alpha - Multisection - alpha_1'):
            np.testing.assert_equal(
                0, ring.alpha_1[0, :])
            np.testing.assert_equal(
                alpha_1, ring.alpha_1[1, :])

        with self.subTest('Non linear alpha - Multisection - alpha_2'):
            np.testing.assert_equal(
                0, ring.alpha_2[0, :])
            np.testing.assert_equal(
                alpha_2, ring.alpha_2[1, :])

        with self.subTest('Non linear alpha - Multisection - alpha_5'):
            np.testing.assert_equal(
                0, ring.alpha_5[0, :])
            np.testing.assert_equal(
                alpha_5, ring.alpha_5[1, :])

    # Functions test ----------------------------------------------------------

    def test_parameters_at_time(self):
        # Test passing non linear momentum compaction factor

        length = 300  # m
        alpha_0 = [[0, 100e-6], [1e-3, 1.5e-3]]
        particle = Proton()
        momentum = [[0, 100e-6], [26e9, 26e9]]  # eV/c

        section = RingSection(length, alpha_0, momentum)
        ring = Ring(particle, [section])

        params = ring.parameters_at_time(50e-6)

        with self.subTest('Time based program - momentum'):
            np.testing.assert_equal(
                np.mean(momentum[1]), params['momentum'])
            np.testing.assert_equal(
                50e-6, params['cycle_time'])

    def test_parameters_at_sample(self):
        # Test passing non linear momentum compaction factor

        length = 300  # m
        alpha_0 = 1e-3
        particle = Proton()
        momentum = [26e9, 27e9, 28e9]  # eV/c

        section = RingSection(length, alpha_0, momentum)
        ring = Ring(particle, [section])

        params = ring.parameters_at_sample(1)

        with self.subTest('Time based program - momentum'):
            np.testing.assert_equal(
                momentum[1], params['momentum'])

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

    def test_unused_kwarg(self):
        # Test the warning that kwargs were not used
        # (e.g. miss-typed or bad option)

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        particle = Proton()
        kwargs = {'bad_kwarg': 0}
        warn_message = (
            "Unused kwargs have been detected, " +
            f"they are \['{list(kwargs.keys())[0]}'\]")

        with self.assertWarnsRegex(Warning, warn_message):
            section = RingSection(length, alpha_0, momentum)
            Ring(particle, [section], **kwargs)

    def test_exception_mix_time_turn(self):
        # Test the exception when time/turn programs are mixed for various
        # sections

        length = 300  # m
        alpha_0 = 1e-3
        momentum_1 = [26e9, 27e9, 28e9]  # eV/c
        momentum_2 = [[0, 100e-6], [26e9, 26e9]]  # eV/c
        particle = Proton()

        error_message = (
            'The synchronous data for' +
            'the different sections is mixing time and turn ' +
            'based programs which is not supported.')

        with self.assertRaisesRegex(excpt.InputError, error_message):
            section_1 = RingSection(length / 2, alpha_0, momentum_1)
            section_2 = RingSection(length / 2, alpha_0, momentum_2)
            Ring(particle, [section_1, section_2])

    def test_warning_time_prog_multisection(self):
        # Test the warning when identical time programs are given
        # for each section

        length = 300  # m
        alpha_0 = 1e-3
        momentum = [[0, 100e-6], [26e9, 26e9]]  # eV/c
        particle = Proton()

        warn_message = 'The synchronous data for all sections ' + \
            'are defined time based and ' + \
            'are identical. Presently, ' + \
            'the momentum is assumed constant for one turn over ' + \
            'all sections, no increment in delta_E from ' + \
            'one section to the next. Please use custom ' + \
            'turn based program if needed.'

        with self.assertWarnsRegex(Warning, warn_message):
            section_1 = RingSection(length / 2, alpha_0, momentum)
            section_2 = RingSection(length / 2, alpha_0, momentum)
            Ring(particle, [section_1, section_2])

    def test_error_time_prog_multisection(self):
        # Test the error when different time programs are given
        # for each section

        length = 300  # m
        alpha_0 = 1e-3
        momentum_1 = [[0, 100e-6], [26e9, 26e9]]  # eV/c
        momentum_2 = [[0, 100e-6], [27e9, 27e9]]  # eV/c
        particle = Proton()

        error_message = ('The synchronous data for all sections ' +
                         'are defined time based and ' +
                         'are not identical. This case is not yet ' +
                         'implemented.')

        with self.assertRaisesRegex(NotImplementedError, error_message):
            section_1 = RingSection(length / 2, alpha_0, momentum_1)
            section_2 = RingSection(length / 2, alpha_0, momentum_2)
            Ring(particle, [section_1, section_2])

    def test_warning_eta_order(self):
        # Test the warning when identical time programs are given
        # for each section

        length = 300  # m
        alpha_0 = 1e-3
        momentum = 26e9  # eV/c
        eta_orders = 4
        particle = Proton()

        warn_message = 'The eta_orders can only be computed up to eta_2!'

        with self.assertWarnsRegex(Warning, warn_message):
            section = RingSection(length, alpha_0, momentum)
            Ring(particle, [section], eta_orders=eta_orders)


if __name__ == '__main__':

    unittest.main()
