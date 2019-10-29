# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for the blond_common.fitting.profile module

- TODO: include the changes from interfaces.beam.analytic_distribution

:Authors: **Alexandre Lasheen**

"""

# General imports
# ---------------
import sys
import unittest
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

from blond_common.interfaces.beam.analytic_distribution import (
    Gaussian, generalizedGaussian, waterbag, parabolicAmplitude, parabolicLine,
    binomialAmplitude2, binomialAmplitudeN, cosine, cosineSquared,
    _binomial_full_to_rms, _binomial_full_to_fwhm, _binomial_integral)

from blond_common.fitting.profile import (FitOptions, PlotOptions,
    RMS, FWHM, peak_value, integrated_profile,
    binomial_from_width_ratio, binomial_from_width_LUT_generation,
    gaussian_fit, generalized_gaussian_fit, waterbag_fit, parabolic_line_fit,
    parabolic_amplitude_fit, binomial_amplitude2_fit, binomial_amplitudeN_fit,
    cosine_fit, cosine_squared_fit, arbitrary_profile_fit)

from blond_common.devtools.exceptions import InputError


class TestFittingProfile(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):
        '''
        We generate three different distributions, Gaussian, Parabolic
        Amplitude, and Binomial that will be used to test the fitting functions
        '''

        # Base time array
        self.time_array = np.arange(0, 25e-9, 0.1e-9)

        # Base Gaussian profile
        self.amplitude_gauss = 1.
        self.position_gauss = 13e-9
        self.length_gauss = 2e-9
        self.initial_params_gauss = [self.amplitude_gauss, self.position_gauss,
                                     self.length_gauss]
        self.gaussian_dist = Gaussian(*self.initial_params_gauss).profile(
            self.time_array)
        self.sigma_gauss = self.length_gauss
        self.fwhm_gauss = Gaussian(*self.initial_params_gauss).FWHM
        self.integral_gauss = Gaussian(*self.initial_params_gauss).integral

        # Base parabolic line profile
        self.amplitude_parabline = 2.5
        self.position_parabline = 9e-9
        self.length_parabline = 7e-9
        self.exponent_parabline = 1.0
        self.initial_params_parabline = [self.amplitude_parabline,
                                         self.position_parabline,
                                         self.length_parabline]
        self.parabline_dist = parabolicLine(self.time_array,
                                            *self.initial_params_parabline)
        self.sigma_parabline = _binomial_full_to_rms(
            self.length_parabline, self.exponent_parabline)
        self.fwhm_parabline = _binomial_full_to_fwhm(
            self.length_parabline, self.exponent_parabline)
        self.integral_parabline = _binomial_integral(
            self.amplitude_parabline, self.length_parabline,
            self.exponent_parabline)

        # Base parabolic amplitude profile
        self.amplitude_parabamp = 1.3
        self.position_parabamp = 4e-9
        self.length_parabamp = 5e-9
        self.exponent_parabamp = 1.5
        self.initial_params_parabamp = [self.amplitude_parabamp,
                                        self.position_parabamp,
                                        self.length_parabamp]
        self.parabamp_dist = parabolicAmplitude(self.time_array,
                                                *self.initial_params_parabamp)
        self.sigma_parabamp = _binomial_full_to_rms(
            self.length_parabamp, self.exponent_parabamp)
        self.fwhm_parabamp = _binomial_full_to_fwhm(
            self.length_parabamp, self.exponent_parabamp)
        self.integral_parabamp = _binomial_integral(
            self.amplitude_parabamp, self.length_parabamp,
            self.exponent_parabamp)

        # Base binomial profile
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
        self.fwhm_binom = _binomial_full_to_fwhm(self.length_binom,
                                                 self.exponent_binom)
        self.integral_binom = _binomial_integral(
            self.amplitude_binom, self.length_binom, self.exponent_binom)

    # Tests for RMS -----------------------------------------------------------
    '''
    Testing the RMS function on the three profiles, the absolute precision
    required was set manually for the time being.

    Each test consists of 2 assertions comparing the mean and rms obtained
    from the RMS function compared to the analytical expectation for the
    3 profiles.

    TODO: the precision is set manually atm and should be reviewed

    '''

    def test_RMS_gauss(self):
        '''
        Checking the mean,rms obtained from RMS function for the Gaussian
        profile
        '''

        mean_gauss, rms_gauss = RMS(self.time_array, self.gaussian_dist)

        np.testing.assert_almost_equal(
            mean_gauss*1e9, self.position_gauss*1e9, decimal=8)

        np.testing.assert_almost_equal(
            rms_gauss*1e9, self.length_gauss*1e9, decimal=7)

    def test_RMS_parabamp(self):
        '''
        Checking the mean,rms obtained from RMS function for the Parabolic
        Amplitude profile
        '''

        mean_parabamp, rms_parabamp = RMS(self.time_array, self.parabamp_dist)

        np.testing.assert_almost_equal(
            mean_parabamp*1e9, self.position_parabamp*1e9, decimal=8)

        np.testing.assert_almost_equal(
            rms_parabamp*1e9, self.sigma_parabamp*1e9, decimal=4)

    def test_RMS_binom(self):
        '''
        Checking the mean,rms obtained from RMS function for a Binomial profile
        '''

        mean_binom, rms_binom = RMS(self.time_array, self.binom_dist)

        np.testing.assert_almost_equal(
            mean_binom*1e9, self.position_binom*1e9, decimal=8)

        np.testing.assert_almost_equal(
            rms_binom*1e9, self.sigma_binom*1e9, decimal=6)

    def test_RMS_misc(self):
        '''
        Miscellaneous tests for missing coverage on non critical elements
        '''

        fitOpt = FitOptions()
        RMS(self.time_array, self.parabamp_dist, fitOpt=fitOpt)

    # Tests for FWHM ----------------------------------------------------------
    '''
    Testing the FWHM function on the three profiles, the absolute precision
    required was set manually for the time being.

    Each test consists of 2 assertions comparing the center and fwhm obtained
    from the FWHM function compared to the the analytical expectation for the
    3 profiles.

    TODO: the precision is set manually atm and should be reviewed

    '''

    def test_FWHM_gauss(self):
        '''
        Checking the center,fwhm obtained from FWHM function for the Gaussian
        profile.
        '''

        center_gauss, fwhm_gauss = FWHM(self.time_array, self.gaussian_dist)

        np.testing.assert_almost_equal(
            center_gauss*1e9, self.position_gauss*1e9, decimal=8)

        np.testing.assert_almost_equal(
            fwhm_gauss*1e9, self.fwhm_gauss*1e9, decimal=3)

    def test_FWHM_parabamp(self):
        '''
        Checking the center,fwhm obtained from FWHM function for the Parabolic
        Amplitude profile.
        '''

        center_parabamp, fwhm_parabamp = FWHM(self.time_array,
                                              self.parabamp_dist)

        np.testing.assert_almost_equal(
            center_parabamp*1e9, self.position_parabamp*1e9, decimal=8)

        np.testing.assert_almost_equal(
            fwhm_parabamp*1e9, self.fwhm_parabamp*1e9, decimal=3)

    def test_FWHM_binom(self):
        '''
        Checking the center,fwhm obtained from FWHM function for a Binomial
        profile.
        '''

        center_binom, fwhm_binom = FWHM(self.time_array, self.binom_dist)

        np.testing.assert_almost_equal(
            center_binom*1e9, self.position_binom*1e9, decimal=8)

        np.testing.assert_almost_equal(
            fwhm_binom*1e9, self.fwhm_binom*1e9, decimal=3)

    def test_FWHM_gaussian_factor(self):
        '''
        Checking the center,fwhm obtained from FWHM function for the Gaussian
        profile. The fwhm is rescaled to 4sigma and compared with the actual
        4sigma of the Gaussian profile.
        '''

        fitOpt = FitOptions(bunchLengthFactor='gaussian')
        center_gauss, fwhm_gauss = FWHM(self.time_array, self.gaussian_dist,
                                        fitOpt=fitOpt)

        np.testing.assert_almost_equal(
            center_gauss*1e9, self.position_gauss*1e9, decimal=8)

        np.testing.assert_almost_equal(
            fwhm_gauss*1e9, 4*self.sigma_gauss*1e9, decimal=3)

    def test_FWHM_parabline_factor(self):
        '''
        Checking the center,fwhm obtained from FWHM function for the Gaussian
        profile.
        '''

        fitOpt = FitOptions(bunchLengthFactor='parabolic_line')
        center_parabline, fwhm_parabline = FWHM(self.time_array,
                                                self.parabline_dist,
                                                fitOpt=fitOpt)

        np.testing.assert_almost_equal(
            center_parabline*1e9, self.position_parabline*1e9, decimal=8)

        np.testing.assert_almost_equal(
            fwhm_parabline*1e9, 4*self.sigma_parabline*1e9, decimal=3)

    def test_FWHM_parabamp_factor(self):
        '''
        Checking the center,fwhm obtained from FWHM function for the Gaussian
        profile.
        '''

        fitOpt = FitOptions(bunchLengthFactor='parabolic_amplitude')
        center_parabamp, fwhm_parabamp = FWHM(self.time_array,
                                              self.parabamp_dist,
                                              fitOpt=fitOpt)

        np.testing.assert_almost_equal(
            center_parabamp*1e9, self.position_parabamp*1e9, decimal=8)

        np.testing.assert_almost_equal(
            fwhm_parabamp*1e9, 4*self.sigma_parabamp*1e9, decimal=3)

    def test_FWHM_errors(self):
        '''
        Checking that the warnings when bunch is at the edge of the frame
        are being raised.
        '''

        fitOpt = FitOptions(bunchLengthFactor='billy')
        with self.assertRaises(InputError):
            FWHM(self.time_array, self.parabamp_dist, fitOpt=fitOpt)

    def test_FWHM_warnings(self):
        '''
        Checking that the warnings when bunch is at the edge of the frame
        are being raised.
        '''

        # Generate profile on the edge of the frame
        amplitude_parabline = 2.5
        position_parabline = self.time_array[0]
        length_parabline = 7e-9
        initial_params_parabline = [amplitude_parabline,
                                    position_parabline,
                                    length_parabline]
        parabline_dist = parabolicLine(self.time_array,
                                       *initial_params_parabline)

        with self.assertWarns(Warning):
            FWHM(self.time_array, parabline_dist)

        # Check the other side
        position_parabline = self.time_array[-1]
        initial_params_parabline = [amplitude_parabline,
                                    position_parabline,
                                    length_parabline]
        parabline_dist = parabolicLine(self.time_array,
                                       *initial_params_parabline)

        with self.assertWarns(Warning):
            FWHM(self.time_array, parabline_dist)

    def test_FWHM_plot(self):
        '''
        Checking that the plots are not returning any error
        '''

        plotOpt = PlotOptions()
        FWHM(self.time_array, self.parabamp_dist, plotOpt=plotOpt)

        plotOpt = PlotOptions(interactive=False)
        FWHM(self.time_array, self.parabamp_dist, plotOpt=plotOpt)

        plotOpt = PlotOptions(clf=False)
        FWHM(self.time_array, self.parabamp_dist, plotOpt=plotOpt)

    # Tests for peak_value ----------------------------------------------------
    '''
    Testing the peak_value function on the three profiles, the absolute
    precision required was set manually for the time being.

    Each test consists of 2 assertions comparing the position and peak obtained
    from the peak_value function compared to the input for the
    3 profiles.

    TODO: the precision is set manually atm and should be reviewed

    '''

    def test_peak_value_gauss(self):
        '''
        Checking the position,peak obtained from peak_value function for a
        Gaussian profile
        '''

        position_gauss, peak_gauss = peak_value(self.time_array,
                                                self.gaussian_dist)

        np.testing.assert_almost_equal(
            position_gauss*1e9, self.position_gauss*1e9, decimal=8)

        np.testing.assert_almost_equal(
            peak_gauss, self.amplitude_gauss, decimal=8)

    def test_peak_value_parabamp(self):
        '''
        Checking the position,peak obtained from peak_value function for a
        Parabolic Amplitude profile
        '''

        position_parabamp, peak_parabamp = peak_value(self.time_array,
                                                      self.parabamp_dist)

        np.testing.assert_almost_equal(
            position_parabamp*1e9, self.position_parabamp*1e9, decimal=8)

        np.testing.assert_almost_equal(
            peak_parabamp, self.amplitude_parabamp, decimal=8)

    def test_peak_value_binom(self):
        '''
        Checking the position,peak obtained from peak_value function for a
        Binomial profile
        '''

        position_binom, peak_binom = peak_value(self.time_array,
                                                self.binom_dist)

        np.testing.assert_almost_equal(
            position_binom*1e9, self.position_binom*1e9, decimal=8)

        np.testing.assert_almost_equal(
            peak_binom, self.amplitude_binom, decimal=8)

    def test_peak_value_misc(self):
        '''
        Miscellaneous tests for missing coverage on non critical elements
        '''

        fitOpt = FitOptions()
        peak_value(self.time_array, self.parabamp_dist, fitOpt=fitOpt)

    def test_peak_value_plot(self):
        '''
        Checking that the plots are not returning any error
        '''

        plotOpt = PlotOptions()
        peak_value(self.time_array, self.parabamp_dist, plotOpt=plotOpt)

        plotOpt = PlotOptions(interactive=False)
        peak_value(self.time_array, self.parabamp_dist, plotOpt=plotOpt)

        plotOpt = PlotOptions(clf=False)
        peak_value(self.time_array, self.parabamp_dist, plotOpt=plotOpt)

    # Tests for integrated_profile --------------------------------------------
    '''
    Testing the integrated_profile function on the three profiles, the absolute
    precision required was set manually for the time being.

    Each test consists of 1 assertion comparing the integration obtained
    from the integrated_profile function compared to the input for the
    3 profiles.

    TODO: the precision is set manually atm and should be reviewed

    '''

    def test_integrated_profile_gauss(self):
        '''
        Checking the integration obtained from integrated_profile function
        for a Gaussian profile
        '''

        integrated_gauss = integrated_profile(
            self.time_array, self.gaussian_dist)

        np.testing.assert_almost_equal(
            integrated_gauss, self.integral_gauss, decimal=15)

    def test_integrated_profile_parabamp(self):
        '''
        Checking the integration obtained from integrated_profile function for
        a Parabolic Amplitude profile
        '''

        integrated_parabamp = integrated_profile(
            self.time_array, self.parabamp_dist)

        np.testing.assert_almost_equal(
            integrated_parabamp, self.integral_parabamp, decimal=12)

    def test_integrated_profile_binom(self):
        '''
        Checking the integration obtained from integrated_profile function for
        a Binomial profile
        '''

        integrated_binom = integrated_profile(
            self.time_array, self.binom_dist)

        np.testing.assert_almost_equal(
            integrated_binom, self.integral_binom, decimal=15)

    def test_integrated_profile_method(self):
        '''
        Checking the integration obtained from integrated_profile function for
        a Binomial profile with all the methods (sum, trapz, ...)
        '''

        integrated_binom = integrated_profile(
            self.time_array, self.binom_dist)

        np.testing.assert_almost_equal(
            integrated_binom, self.integral_binom, decimal=15)

        integrated_binom = integrated_profile(
            self.time_array, self.binom_dist, method='trapz')

        np.testing.assert_almost_equal(
            integrated_binom, self.integral_binom, decimal=15)

        with self.assertRaises(InputError):
            integrated_binom = integrated_profile(
                self.time_array, self.binom_dist, method='joe')

    def test_integrated_profile_misc(self):
        '''
        Miscellaneous tests for missing coverage on non critical elements
        '''

        fitOpt = FitOptions()
        integrated_profile(self.time_array, self.parabamp_dist, fitOpt=fitOpt)

    def test_integrated_profile_plot(self):
        '''
        Checking that the plots are not returning any error
        '''

        plotOpt = PlotOptions()
        integrated_profile(self.time_array, self.parabamp_dist,
                           plotOpt=plotOpt)

        plotOpt = PlotOptions(interactive=False)
        integrated_profile(self.time_array, self.parabamp_dist,
                           plotOpt=plotOpt)

        plotOpt = PlotOptions(clf=False)
        integrated_profile(self.time_array, self.parabamp_dist,
                           plotOpt=plotOpt)

    # Test for binomial_from_width_ratio --------------------------------------
    '''
    Testing the binomial_from_width_ratio function on the parabolic and
    binomial profiles, the absolute precision required was set manually
    for the time being.

    Each test consists of 4 assertions comparing the full bunch length and
    exponents obtained from the binomial_from_width_ratio function compared to
    the input for the 3 profiles.

    NB: for the Binomial profile with large exponent, the rms length is tested
    instead of the full length.

    TODO: the precision is set manually atm and should be reviewed

    '''

    def test_binomial_from_width_ratio_parabline(self):
        '''
        Checking the full bunch length and exponent obtained from
        binomial_from_width_ratio function for a Parabolic line profile
        '''

        amplitude, position, full_length, exponent = binomial_from_width_ratio(
            self.time_array, self.parabline_dist)

        np.testing.assert_almost_equal(
            amplitude, self.amplitude_parabline, decimal=15)

        np.testing.assert_almost_equal(
            position, self.position_parabline, decimal=15)

        np.testing.assert_almost_equal(
            full_length, self.length_parabline, decimal=10)

        np.testing.assert_almost_equal(
            exponent, self.exponent_parabline, decimal=2)

    def test_binomial_from_width_ratio_parabamp(self):
        '''
        Checking the integration obtained from binomial_from_width_ratio
        function for a Parabolic Amplitude profile
        '''

        amplitude, position, full_length, exponent = binomial_from_width_ratio(
            self.time_array, self.parabamp_dist)

        np.testing.assert_almost_equal(
            amplitude, self.amplitude_parabamp, decimal=15)

        np.testing.assert_almost_equal(
            position, self.position_parabamp, decimal=15)

        np.testing.assert_almost_equal(
            full_length, self.length_parabamp, decimal=10)

        np.testing.assert_almost_equal(
            exponent, self.exponent_parabamp, decimal=2)

    def test_binomial_from_width_ratio_binom(self):
        '''
        Checking the integration obtained from binomial_from_width_ratio
        function for a Binomial profile.

        NB: the full bunch length and exponent are difficult to obtain
        precisely for an abitrary Binomial profile with large exponent!
        However it is till sufficient to get a very good estimate of the
        rms length.
        '''

        amplitude, position, full_length, exponent = binomial_from_width_ratio(
            self.time_array, self.binom_dist)

        np.testing.assert_almost_equal(
            amplitude, self.amplitude_binom, decimal=15)

        np.testing.assert_almost_equal(
            position, self.position_binom, decimal=15)

        rms_length = _binomial_full_to_rms(full_length, exponent)

        np.testing.assert_almost_equal(
            rms_length, self.sigma_binom, decimal=10)

    def test_binomial_from_width_ratio_parabline_customLUT(self):
        '''
        Checking the full bunch length and exponent obtained from
        binomial_from_width_ratio function for a Parabolic line profile
        '''
        exponent_min = 0.5
        exponent_max = 2.
        levels_input = [0.7, 0.3]
        exponent_npoints = 100
        ratio_LUT = binomial_from_width_LUT_generation(
            levels=levels_input,
            exponent_min=exponent_min, exponent_max=exponent_max,
            exponent_distrib='linspace',
            exponent_npoints=exponent_npoints)

        amplitude, position, full_length, exponent = binomial_from_width_ratio(
            self.time_array, self.parabline_dist, ratio_LUT=ratio_LUT)

        np.testing.assert_almost_equal(
            amplitude, self.amplitude_parabline, decimal=15)

        np.testing.assert_almost_equal(
            position, self.position_parabline, decimal=15)

        np.testing.assert_almost_equal(
            full_length, self.length_parabline, decimal=10)

        np.testing.assert_almost_equal(
            exponent, self.exponent_parabline, decimal=3)

    def test_binomial_from_width_ratio_misc(self):
        '''
        Miscellaneous tests for missing coverage on non critical elements
        '''

        fitOpt = FitOptions()
        binomial_from_width_ratio(self.time_array, self.parabamp_dist,
                                  fitOpt=fitOpt)

    def test_binomial_from_width_ratio_plot(self):
        '''
        Checking that the plots are not returning any error
        '''

        plotOpt = PlotOptions()
        binomial_from_width_ratio(self.time_array, self.parabamp_dist,
                                  plotOpt=plotOpt)

        plotOpt = PlotOptions(interactive=False)
        binomial_from_width_ratio(self.time_array, self.parabamp_dist,
                                  plotOpt=plotOpt)

        plotOpt = PlotOptions(clf=False)
        binomial_from_width_ratio(self.time_array, self.parabamp_dist,
                                  plotOpt=plotOpt)

    def test_binomial_from_width_LUT_generation(self):
        '''
        Checking that the lookup table for binomial_from_width_ratio works
        as designed
        '''

        exponent_array, ratio_FW, levels = binomial_from_width_LUT_generation(
            exponent_array=np.array([0.5, 10]))

        np.testing.assert_equal(
            np.min(levels), 0.2)
        np.testing.assert_equal(
            np.max(levels), 0.8)
        np.testing.assert_equal(
            exponent_array[-1], 0.5)
        np.testing.assert_equal(
            exponent_array[0], 10.)
        np.testing.assert_equal(
            ratio_FW[0],
            np.sqrt(
                (1-0.8**(1/10.)) /
                (1-0.2**(1/10.))))
        np.testing.assert_equal(
            ratio_FW[-1],
            np.sqrt(
                (1-0.8**(1/0.5)) /
                (1-0.2**(1/0.5))))

        exponent_min = 0.5
        exponent_max = 10.
        levels_input = [0.7, 0.3]
        exponent_npoints = 2
        exponent_array, ratio_FW, levels = binomial_from_width_LUT_generation(
            levels=levels_input,
            exponent_min=exponent_min, exponent_max=exponent_max,
            exponent_distrib='linspace',
            exponent_npoints=exponent_npoints)

        np.testing.assert_equal(
            len(exponent_array), exponent_npoints)
        np.testing.assert_equal(
            np.min(levels), np.min(levels_input))
        np.testing.assert_equal(
            np.max(levels), np.max(levels_input))
        np.testing.assert_equal(
            exponent_array[-1], exponent_min)
        np.testing.assert_equal(
            exponent_array[0], exponent_max)
        np.testing.assert_equal(
            ratio_FW[0],
            np.sqrt(
                (1-np.max(levels_input)**(1/exponent_max)) /
                (1-np.min(levels_input)**(1/exponent_max))))
        np.testing.assert_equal(
            ratio_FW[-1],
            np.sqrt(
                (1-np.max(levels_input)**(1/exponent_min)) /
                (1-np.min(levels_input)**(1/exponent_min))))

    def test_binomial_from_width_LUT_generation_method(self):
        '''
        Checking that the input options for binomial_from_width_LUT_generation
        works as designed
        '''

        exponent_array = binomial_from_width_LUT_generation(
            exponent_distrib='linspace')[0]

        np.testing.assert_equal(
            exponent_array[-1], 0.5)
        np.testing.assert_equal(
            exponent_array[0], 10.)

        exponent_array = binomial_from_width_LUT_generation(
            exponent_distrib='logspace')[0]

        np.testing.assert_equal(
            exponent_array[-1], 0.5)
        np.testing.assert_equal(
            exponent_array[0], 10.)

        with self.assertRaises(InputError):
            binomial_from_width_LUT_generation(exponent_distrib='jimmy')

    # Test fitting ------------------------------------------------------------
    '''
    Testing all fitting functions, the absolute precision is presenty set
    manually.

    This is a benchmark, the fitted parameters should be as close as possible
    to the input values.

    Tests are also performed changing the initial parameters and applying
    errors on them with respect to their implementation in the function.

    TODO: the precision is set manually atm and should be reviewed
    TODO: change the initial parameters for more test robustness

    '''

    def test_gaussian_fit(self):
        '''
        Checking the fittedparameters obtained from gaussian_fit function
        on a Gaussian profile
        '''

        fitted_params = gaussian_fit(self.time_array, self.gaussian_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_gauss, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_gauss*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_gauss*1e9, decimal=8)

    def test_gaussian_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from gaussian_fit function
        on a Gaussian profile, including an error of +20%, -10%, +10% on the
        initial parameters.
        '''

        fitOpt = FitOptions()
        maxProfile = np.max(self.gaussian_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(self.gaussian_dist)),
             0.9*(np.mean(self.time_array[self.gaussian_dist == maxProfile])),
             1.1*FWHM(self.time_array, self.gaussian_dist, level=0.5)[1]])

        fitted_params = gaussian_fit(self.time_array, self.gaussian_dist,
                                     fitOpt=fitOpt)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_gauss, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_gauss*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_gauss*1e9, decimal=8)

    def test_generalized_gaussian_fit(self):
        '''
        Checking the fittedparameters obtained from generalized_gaussian_fit
        function on a Generalized Gaussian profile
        '''

        amplitude = 0.8
        position = 12.7e-9
        length = 1.7e-9
        exponent = 3.0
        initial_params = [amplitude, position, length, exponent]
        generalized_gaussian_dist = generalizedGaussian(
            self.time_array, *initial_params)

        fitted_params = generalized_gaussian_fit(self.time_array,
                                                 generalized_gaussian_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

        np.testing.assert_almost_equal(
            fitted_params[3], initial_params[3], decimal=10)

    def test_generalized_gaussian_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from generalized_gaussian_fit
        function on a Generalized Gaussian profile, including an error of
        +20%, -10%, +10%, -10% on the initial parameters.
        '''

        amplitude = 0.8
        position = 12.7e-9
        length = 1.7e-9
        exponent = 3.0
        initial_params = [amplitude, position, length, exponent]
        generalized_gaussian_dist = generalizedGaussian(
            self.time_array, *initial_params)

        fitOpt = FitOptions()
        maxProfile = np.max(generalized_gaussian_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(generalized_gaussian_dist)),
             0.9*(np.mean(
                 self.time_array[generalized_gaussian_dist == maxProfile])),
             1.1*FWHM(
                 self.time_array, generalized_gaussian_dist, level=0.5)[1],
             0.9*2.0])

        fitted_params = generalized_gaussian_fit(self.time_array,
                                                 generalized_gaussian_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

        np.testing.assert_almost_equal(
            fitted_params[3], initial_params[3], decimal=10)

    def test_waterbag_fit(self):
        '''
        Checking the fittedparameters obtained from waterbag_fit
        function on a Waterbag profile
        '''

        amplitude = 6.5
        position = 6.3e-9
        length = 5.4e-9
        initial_params = [amplitude, position, length]
        waterbag_dist = waterbag(self.time_array, *initial_params)

        fitted_params = waterbag_fit(
            self.time_array, waterbag_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=15)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_waterbag_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from waterbag_fit
        function on a Parabolic Amplitude profile, including an error of
        +20%, -10%, +10% on the initial parameters.
        '''

        amplitude = 6.5
        position = 6.3e-9
        length = 5.4e-9
        initial_params = [amplitude, position, length]
        waterbag_dist = waterbag(self.time_array, *initial_params)

        fitOpt = FitOptions(bunchLengthFactor='parabolic_line')
        maxProfile = np.max(waterbag_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(waterbag_dist)),
             0.9*(np.mean(self.time_array[waterbag_dist == maxProfile])),
             1.1*FWHM(self.time_array, waterbag_dist, level=0.5)[1] *
             np.sqrt(3+2*1.)/2])

        fitted_params = waterbag_fit(
            self.time_array, waterbag_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=15)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_parabolic_line_fit(self):
        '''
        Checking the fittedparameters obtained from parabolic_line_fit
        function on a Parabolic Line profile
        '''

        fitted_params = parabolic_line_fit(
            self.time_array, self.parabline_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_parabline, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_parabline*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_parabline*1e9, decimal=10)

    def test_parabolic_line_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from parabolic_line_fit
        function on a Parabolic Line profile, including an error of
        +20%, -10%, +10% on the initial parameters.
        '''

        fitOpt = FitOptions(bunchLengthFactor='parabolic_line')
        maxProfile = np.max(self.parabline_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(self.parabline_dist)),
             0.9*(np.mean(self.time_array[self.parabline_dist == maxProfile])),
             1.1*FWHM(self.time_array, self.parabline_dist, level=0.5)[1] *
             np.sqrt(3+2*1.)/2])

        fitted_params = parabolic_line_fit(
            self.time_array, self.parabline_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_parabline, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_parabline*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_parabline*1e9, decimal=10)

    def test_parabolic_amplitude_fit(self):
        '''
        Checking the fittedparameters obtained from parabolic_amplitude_fit
        function on a Parabolic Amplitude profile
        '''

        fitted_params = parabolic_amplitude_fit(
            self.time_array, self.parabamp_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_parabamp, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_parabamp*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_parabamp*1e9, decimal=10)

    def test_parabolic_amplitude_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from parabolic_amplitude_fit
        function on a Parabolic Amplitude profile, including an error of
        +20%, -10%, +10% on the initial parameters.
        '''

        fitOpt = FitOptions()
        maxProfile = np.max(self.parabamp_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(self.parabamp_dist)),
             0.9*(np.mean(self.time_array[self.parabamp_dist == maxProfile])),
             1.1*FWHM(self.time_array, self.parabamp_dist, level=0.5)[1] *
             np.sqrt(3+2*1.5)/2])

        fitted_params = parabolic_amplitude_fit(
            self.time_array, self.parabamp_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_parabamp, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_parabamp*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_parabamp*1e9, decimal=10)

    def test_binomial_amplitude2_fit(self):
        '''
        Checking the fittedparameters obtained from binomial_amplitude2_fit
        function on a Binomial Amplitude with exponent 2 profile
        '''

        amplitude = 1.7
        position = 12.8e-9
        length = 7.4e-9
        initial_params = [amplitude, position, length]
        binomial_amplitude2_dist = binomialAmplitude2(self.time_array,
                                                      *initial_params)

        fitted_params = binomial_amplitude2_fit(
            self.time_array, binomial_amplitude2_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=15)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_binomial_amplitude2_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from binomial_amplitude2_fit
        function on a Binomial Amplitude with exponent 2 profile,
        including an error of +20%, -10%, +10% on the initial parameters.
        '''

        amplitude = 1.7
        position = 12.8e-9
        length = 7.4e-9
        initial_params = [amplitude, position, length]
        binomial_amplitude2_dist = binomialAmplitude2(self.time_array,
                                                      *initial_params)

        fitOpt = FitOptions(bunchLengthFactor='parabolic_amplitude')
        maxProfile = np.max(binomial_amplitude2_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(binomial_amplitude2_dist)),
             0.9*(np.mean(self.time_array[
                 binomial_amplitude2_dist == maxProfile])),
             1.1*FWHM(
                 self.time_array, binomial_amplitude2_dist, level=0.5)[1] *
             np.sqrt(3+2*1.5)/2])

        fitted_params = binomial_amplitude2_fit(
            self.time_array, binomial_amplitude2_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=15)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_binomial_amplitudeN_fit(self):
        '''
        Checking the fittedparameters obtained from binomial_amplitudeN_fit
        function on a Binomial profile
        '''

        fitted_params = binomial_amplitudeN_fit(
            self.time_array, self.binom_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_binom, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_binom*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_binom*1e9, decimal=10)

        np.testing.assert_almost_equal(
            fitted_params[3], self.exponent_binom, decimal=10)

    def test_cosine_fit(self):
        '''
        Checking the fittedparameters obtained from cosine_fit
        function on a Cosine profile
        '''

        amplitude = 1.7
        position = 12.8e-9
        length = 7.4e-9
        initial_params = [amplitude, position, length]
        cosine_dist = cosine(self.time_array, *initial_params)

        fitted_params = cosine_fit(
            self.time_array, cosine_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=15)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_cosine_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from cosine_fit
        function on a Cosine profile,
        including an error of +20%, -10%, +10% on the initial parameters.
        '''

        amplitude = 1.7
        position = 12.8e-9
        length = 7.4e-9
        initial_params = [amplitude, position, length]
        cosine_dist = cosine(self.time_array, *initial_params)

        fitOpt = FitOptions(bunchLengthFactor='parabolic_amplitude')
        maxProfile = np.max(cosine_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(cosine_dist)),
             0.9*(np.mean(self.time_array[
                 cosine_dist == maxProfile])),
             1.1*FWHM(
                 self.time_array, cosine_dist, level=0.5)[1] *
             np.sqrt(3+2*1.5)/2])

        fitted_params = cosine_fit(
            self.time_array, cosine_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=15)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_cosine_squared_fit(self):
        '''
        Checking the fittedparameters obtained from cosine_squared_fit
        function on a Cosine Squared profile
        '''

        amplitude = 0.4
        position = 13.4e-9
        length = 4.2e-9
        initial_params = [amplitude, position, length]
        cosine_squared_dist = cosineSquared(self.time_array, *initial_params)

        fitted_params = cosine_squared_fit(
            self.time_array, cosine_squared_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=14)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_cosine_squared_fit_initial_params(self):
        '''
        Checking the fittedparameters obtained from cosine_squared_fit
        function on a Cosine Squared profile,
        including an error of +20%, -10%, +10% on the initial parameters.
        '''

        amplitude = 8.2
        position = 7.5e-9
        length = 3.1e-9
        initial_params = [amplitude, position, length]
        cosine_squared_dist = cosineSquared(self.time_array, *initial_params)

        fitOpt = FitOptions(bunchLengthFactor='parabolic_amplitude')
        maxProfile = np.max(cosine_squared_dist)
        fitOpt.fitInitialParameters = np.array(
            [1.2*(maxProfile-np.min(cosine_squared_dist)),
             0.9*(np.mean(self.time_array[
                 cosine_squared_dist == maxProfile])),
             1.1*FWHM(
                 self.time_array, cosine_squared_dist, level=0.5)[1] *
             np.sqrt(3+2*1.5)/2])

        fitted_params = cosine_squared_fit(
            self.time_array, cosine_squared_dist)

        np.testing.assert_almost_equal(
            fitted_params[0], initial_params[0], decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, initial_params[1]*1e9, decimal=15)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, initial_params[2]*1e9, decimal=10)

    def test_arbitrary_profile_fit(self):
        '''
        Checking the fittedparameters obtained from arbitrary_profile_fit
        function on a Binomial profile and using binomialAmplitudeN
        as a "user input" fitting function.
        '''

        fitOpt = FitOptions()
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [np.max(self.binom_dist)-np.min(self.binom_dist),
             np.mean(self.time_array[
                 self.binom_dist == np.max(self.binom_dist)]),
             FWHM(self.time_array,
                  self.binom_dist,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2,  # Full bunch length!!
             1.5])

        fitted_params = arbitrary_profile_fit(
            self.time_array, self.binom_dist, binomialAmplitudeN,
            fitOpt=fitOpt)

        np.testing.assert_almost_equal(
            fitted_params[0], self.amplitude_binom, decimal=9)

        np.testing.assert_almost_equal(
            fitted_params[1]*1e9, self.position_binom*1e9, decimal=20)

        np.testing.assert_almost_equal(
            fitted_params[2]*1e9, self.length_binom*1e9, decimal=10)

        np.testing.assert_almost_equal(
            fitted_params[3], self.exponent_binom, decimal=10)


if __name__ == '__main__':

    unittest.main()
