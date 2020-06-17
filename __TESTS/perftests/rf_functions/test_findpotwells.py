# coding: utf8
# Copyright 2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Performance tests for the blond_common.rf_functions.potential module

:Authors: **Alexandre Lasheen**

"""

# General imports
# ---------------
import sys
import numpy as np
import os
import time
from scipy.constants import u, c, e

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

from blond_common.rf_functions.potential import find_potential_wells_cubic
from blond_common.interfaces.beam.beam import Particle
from blond_common.interfaces.machine_parameters.ring import Ring, RingSection

from blond_common.interfaces.machine_parameters.rf_parameters import RFStation
from blond_common.rf_functions.potential import rf_voltage_generation
from blond_common.rf_functions.potential import rf_potential_generation


# Input data folder
input_folder = this_directory + '/../../input/rf_functions/'


class TestFindPotWells(object):

    # Initialization ----------------------------------------------------------

    def __init__(self, iterations=100):

        self.iterations = iterations

        # Defining a ramp with a program time vs. energy (warning: initial
        # energy cannot be 0)

        # Machine circumference [m]
        ring_length = 2 * np.pi * 100
        bending_radius = 70.079                     # Bending radius [m]
    #     bending_field = 1.136487 # Bending field [T]
        bending_field = [[0, 1e-3], [1.136487, 1.136487]]  # Bending field [T]
        gamma_transition = 6.1                      # Transition gamma
        alpha_0 = 1 / gamma_transition**2.

        particle_charge = 39                            # Particle charge [e]
        particle_mass = 128.883 * u * c**2. / \
            e               # Particle mass [eV/c2]
        particle = Particle(particle_mass, particle_charge)

        ring = Ring(particle, RingSection(ring_length, alpha_0,
                                          bending_field=bending_field,
                                          bending_radius=bending_radius))

        harmonic = [21, 28, 169]
        # voltage = [80e3, 0, 0]  # V, h21 Single RF
        voltage = [6e3, 20e3, 0]  # V, h21->h28 batch compression
        voltage = [0, 16.1e3, 12.4e3]  # V, h28->h169 rebucketting
        phi_rf = [np.pi, np.pi, np.pi]  # rad

        rf_station = RFStation(ring, harmonic, voltage, phi_rf)

        n_points = 10000
        t_rev = ring.t_rev[0]
        voltage = rf_station.voltage[:, 0]
        harmonic = rf_station.harmonic[:, 0]
        phi_rf = rf_station.phi_rf[:, 0]
        time_bounds = [-ring.t_rev[0] / harmonic[0]
                       * 2, ring.t_rev[0] / harmonic[0] * 2]

        self.time_array, self.rf_voltage_array = rf_voltage_generation(
            n_points, t_rev, voltage, harmonic, phi_rf,
            time_bounds=time_bounds)

        n_points = 10000
        eta_0 = ring.eta_0[0, 0]
        charge = ring.Particle.charge
        energy_increment_bis = charge * 5e3

        self.time_array, self.rf_potential_array = rf_potential_generation(
            n_points, t_rev, voltage, harmonic, phi_rf, eta_0, charge,
            energy_increment_bis, time_bounds=time_bounds)

    def _find_tests(self):
        '''
        The routine find all the object attributes starting with test_*
        '''

        self.alltests = []
        for att in dir(self):
            if 'test_' in att and hasattr(att, '__call__'):
                self.alltests.append(att)

    def run_tests(self):
        '''
        The routine to run all the tests
        '''

        if not hasattr(self, 'test_list'):
            self._find_tests()

        dict_results = {}

        for test in self.alltests:
            (mean_runtime, std_runtime,
             mean_result, std_result) = self._runtest(
                 getattr(self, test))
            print('%s - Runtime: %.5e +- %.5e - Result: %.5e +- %.5e' %
                  (test, mean_runtime, std_runtime, mean_result, std_result))
            dict_results[test] = {'mean_runtime': mean_runtime,
                                  'std_runtime': std_runtime,
                                  'mean_result': mean_result,
                                  'std_result': std_result}

    # Test template -----------------------------------------------------------
    def _runtest(self, test_function):

        runtime = np.zeros(self.iterations)
        result = np.zeros(self.iterations)

        for iteration in range(self.iterations):

            runtime[iteration], result[iteration] = test_function()

        mean_runtime = np.mean(runtime / self.iterations)
        std_runtime = np.std(runtime / self.iterations)
        mean_result = np.mean(result)
        std_result = np.std(result)

        return mean_runtime, std_runtime, mean_result, std_result

    # Tests for RMS -----------------------------------------------------------
    '''
    Testing the RMS function
    '''

    def test_find_potential_wells_cubic(self):
        '''
        Checking the mean obtained from RMS function for the Gaussian
        profile.
        '''

        t0 = time.perf_counter()
        output = find_potential_wells_cubic(
            self.time_array, self.rf_potential_array,
            mest=200)
        t1 = time.perf_counter()

        n_potentials = len(output[0])

        runtime = t1 - t0
        result = n_potentials

        return runtime, result


if __name__ == '__main__':

    tests = TestFindPotWells()
    dict_results = tests.run_tests()
