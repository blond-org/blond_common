# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Performance tests for the blond_common.fitting.profile module

:Authors: **Alexandre Lasheen**

"""

# General imports
# ---------------
import sys
import numpy as np
import os
import time

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

from blond_common.interfaces.beam.analytic_distribution import (
    gaussian, parabolicAmplitude, binomialAmplitudeN, _binomial_full_to_rms)
from blond_common.fitting.profile import RMS, FWHM


class TestFittingProfile(object):

    # Initialization ----------------------------------------------------------

    def __init__(self, iterations=100):
        '''
        We generate three different distributions, Gaussian, Parabolic
        Amplitude, and Binomial that will be used to test the fitting functions
        '''

        self.iterations = iterations

        self.time_array = np.arange(0, 25e-9, 0.1e-9)

        self.amplitude_gauss = 1.
        self.position_gauss = 13e-9
        self.length_gauss = 2e-9
        self.initial_params_gauss = [self.amplitude_gauss, self.position_gauss,
                                     self.length_gauss]
        self.gaussian_dist = gaussian(self.time_array,
                                      *self.initial_params_gauss)
        self.sigma_gauss = self.length_gauss

        self.amplitude_parabamp = 1.3
        self.position_parabamp = 4e-9
        self.length_parabamp = 5e-9
        self.initial_params_parabamp = [self.amplitude_parabamp,
                                        self.position_parabamp,
                                        self.length_parabamp]
        self.parabamp_dist = parabolicAmplitude(self.time_array,
                                                *self.initial_params_parabamp)
        self.sigma_parabamp = _binomial_full_to_rms(self.length_parabamp, 1.5)

        self.amplitude_binom = 0.77
        self.position_binom = 18.3e-9
        self.length_binom = 3.45e-9
        self.exponent_binom = 3.4
        self.initial_params_binom = [self.amplitude_binom, self.position_binom,
                                     self.length_binom, self.exponent_binom]
        self.binom_dist = binomialAmplitudeN(self.time_array,
                                             *self.initial_params_binom)
        self.sigma_binom = _binomial_full_to_rms(self.length_binom,
                                                 self.exponent_binom)

    def _find_tests(self):
        '''
        The routine find all the object attributes starting with test_*
        '''

        self.alltests = []
        for att in dir(self):
            if 'test_' in att:
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

        mean_runtime = np.mean(runtime/self.iterations)
        std_runtime = np.std(runtime/self.iterations)
        mean_result = np.mean(result)
        std_result = np.std(result)

        return mean_runtime, std_runtime, mean_result, std_result

    # Tests for RMS -----------------------------------------------------------
    '''
    Testing the RMS function
    '''

    def test_RMS_gauss_pos(self):
        '''
        Checking the mean obtained from RMS function for the Gaussian
        profile.
        '''

        t0 = time.perf_counter()
        mean_gauss_gauss = RMS(self.time_array, self.gaussian_dist)[0]
        t1 = time.perf_counter()

        runtime = t1-t0
        result = (mean_gauss_gauss-self.position_gauss)/self.position_gauss

        return runtime, result

    def test_RMS_gauss_length(self):
        '''
        Checking the rms obtained from RMS function for the Gaussian
        profile.
        '''

        t0 = time.perf_counter()
        rms_gauss = RMS(self.time_array, self.gaussian_dist)[1]
        t1 = time.perf_counter()

        runtime = t1-t0
        result = (rms_gauss-self.length_gauss)/self.length_gauss

        return runtime, result


if __name__ == '__main__':

    tests = TestFittingProfile()
    dict_results = tests.run_tests()
