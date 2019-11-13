# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for analytic_distribution.py
:Authors: **Markus Schwarz**
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

from blond_common.interfaces.beam import analytic_distribution


class TestDistributionsBaseClass(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):
        self.test_object = analytic_distribution._DistributionObject(None)

    def assertIsNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """

        standardMsg = "%s is not NaN" % str(value)

        if not np.isnan(value):
            self.fail(self._formatMessage(msg, standardMsg))

    def assertHasAttribute(self, obj, intendedAttr):
        testBool = hasattr(obj, intendedAttr)

        self.assertTrue(
            testBool,
            msg=f'{obj} lacking an attribute: {intendedAttr}')
#            msg=f'{obj} lacking an attribute. {obj}: %s, intendedAttr: %s' %
#            (obj, intendedAttr))

    # test helper function ----------------------------------------------------
    def test_check_greater_zero(self):
        # Test if ValueError is raised for negative arguments
        with self.assertRaises(ValueError):
            analytic_distribution.check_greater_zero(-42, '')

    # test base class ---------------------------------------------------------
    def test_base_class_attribute_amplitude(self):
        # Test if base class has amplitude
        self.assertHasAttribute(self.test_object, 'amplitude')

    def test_base_class_attribute_center(self):
        # Test if base class has amplitude
        self.assertHasAttribute(self.test_object, 'center')

    def test_base_class_attribute_RMS(self):
        # Test if base class RMS returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.RMS

    def test_base_class_attribute_FWHM(self):
        # Test if base class FWHM returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.FWHM

    def test_base_class_attribute_fourSigma_RMS(self):
        # Test if base class fourSigma_RMS returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.fourSigma_RMS

    def test_base_class_attribute_fourSigma_FWHM(self):
        # Test if base class fourSigma_FWHM returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.fourSigma_FWHM

    def test_base_class_attribute_full_bunch_length(self):
        # Test if base class full_bunch_length returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.full_bunch_length

    def test_base_class_profile(self):
        # Test if base class.profile() throws RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.profile(0.0)

    def test_base_class_spectrum(self):
        # Test if base class.spectrum() throws RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.spectrum(0.0)

    def test_base_class_parameters(self):
        # Test if base class.spectrum() throws RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.get_parameters()

    def test_base_class_attribute_integral(self):
        # Test if base class.integral throws RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.integral


class TestDistributionsGaussianClass(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):
        self.test_object = analytic_distribution.Gaussian([1, 0.0, 1])

    def test_negative_bunch_length_exception(self):
        with self.assertRaises(ValueError):
            analytic_distribution.Gaussian([1, 0.0, -42])

    def test_amplitude(self):
        self.assertAlmostEqual(self.test_object.amplitude, 1)

    def test_center(self):
        self.assertAlmostEqual(self.test_object.center, 0.0)

    def test_RMS(self):
        self.assertAlmostEqual(self.test_object.RMS, 1)

    def test_FWHM_1(self):
        self.assertAlmostEqual(self.test_object.FWHM,
                               2*np.sqrt(np.log(4))*1)
    
    def test_FWHM_2(self):
        self.assertAlmostEqual(self.test_object.amplitude/2,
                               self.test_object.profile(self.test_object.center+0.5*self.test_object.FWHM))

    def test_RMS_update(self):
        # test if RMS updates correctly when FWHM is changed
        new_FWHM = 42
        self.test_object.FWHM = new_FWHM
        self.assertAlmostEqual(self.test_object.RMS, 17.8357578060484)
        
    def test_profile_amplitude(self):
        self.assertAlmostEqual(self.test_object.profile(0), 1)
    
    def test_get_parameters(self):
        np.testing.assert_almost_equal(self.test_object.get_parameters(),
                                       np.array([1, 0.0, 1]))
    
    def test_integral(self):
        self.assertAlmostEqual(self.test_object.integral,
                               2.5066282746310002)
    
    def test_call_signature1(self):
        time_array = np.linspace(-4, 4, num=10)
        profile = analytic_distribution.Gaussian([1, 0.0, 1], time_array)

        self.assertTrue(type(profile), np.ndarray)
        
        np.testing.assert_almost_equal(profile,
           np.array([3.35462628e-04, 7.91095973e-03, 8.46579886e-02,
                     4.11112291e-01, 9.05955191e-01, 9.05955191e-01,
                     4.11112291e-01, 8.46579886e-02, 7.91095973e-03,
                     3.35462628e-04]))

    def test_initial_fit(self):
        x_data = np.linspace(-4,4,num=50)
        profile = analytic_distribution.Gaussian([2,0.1,1], time_array=x_data)
        
        np.random.seed(1789*1989)
        y_data = profile + np.random.normal(0,0.05, size=profile.size)
        
        fitted_gauss = analytic_distribution.Gaussian(None, x_data, y_data)
        
        self.assertTrue(type(fitted_gauss.get_parameters()), np.ndarray)
        
        np.testing.assert_almost_equal(fitted_gauss.get_parameters(),
           np.array([1.9742753502240824, 0.0832928729810476,
                     0.9837783914364183]))


class TestDistributionsBinominalAmplitudeNClass(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):
        self.test_object = analytic_distribution.BinomialAmplitudeN(
                [1, -4.2, 1, 1.5])

    def test_negative_bunch_length_exception(self):
        with self.assertRaises(ValueError):
            analytic_distribution.BinomialAmplitudeN([1, 4.2, -42, 1.5])

    def test_amplitude(self):
        self.assertAlmostEqual(self.test_object.amplitude, 1)

    def test_center(self):
        self.assertAlmostEqual(self.test_object.center, -4.2)

    def test_RMS(self):
        self.assertAlmostEqual(self.test_object.RMS, 1)
    
    def test_mu(self):
        self.assertAlmostEqual(self.test_object.mu, 1.5)

    def test_FWHM_1(self):
        self.assertAlmostEqual(self.test_object.FWHM, 2.863740583009688)
        
    def test_FWHM_2(self):
        self.assertAlmostEqual(self.test_object.amplitude/2,
                               self.test_object.profile(self.test_object.center+0.5*self.test_object.FWHM))

    def test_RMS_update(self):
        # test if RMS updates correctly when FWHM is changed
        new_FWHM = 42
        self.test_object.FWHM = new_FWHM
        self.assertAlmostEqual(self.test_object.RMS, 14.666132906444869)
        
    def test_profile_amplitude(self):
        self.assertAlmostEqual(self.test_object.profile(-4.2), 1)
    
    def test_get_parameters(self):
        np.testing.assert_almost_equal(self.test_object.get_parameters(),
                                       np.array([1, -4.2, 1, 1.5]))
    
    def test_integral(self):
        self.assertAlmostEqual(self.test_object.integral,
                               2.82213473180223)
    
    def test_call_signature1(self):
        time_array = np.linspace(-0.51*self.test_object.full_bunch_length,
                 +0.51*self.test_object.full_bunch_length, num=10)\
                 + self.test_object.center
#        profile = self.test_object.profile(time_array)
        profile = analytic_distribution.BinomialAmplitudeN([1, -4.2, 1, 1.5],
                                                           time_array)
        
        self.assertTrue(type(profile), np.ndarray)
        
        np.testing.assert_almost_equal(profile,
           np.array([0.0, 0.13736083, 0.46089012, 0.78216336, 0.97447609,
                     0.97447609, 0.78216336, 0.46089012, 0.13736083, 0.0]))

    def test_initial_fit(self):
        x_data = np.linspace(-0.51*self.test_object.full_bunch_length,
                             +0.51*self.test_object.full_bunch_length, num=50)\
                             + self.test_object.center
        profile = self.test_object.profile(x_data)
        
        np.random.seed(1789*1989)
        y_data = profile + np.random.normal(0,0.05, size=profile.size)
        
        fitted_object = analytic_distribution.BinomialAmplitudeN(
                None, x_data, y_data)
        
        self.assertTrue(type(fitted_object.get_parameters()), np.ndarray)
        
        np.testing.assert_almost_equal(fitted_object.get_parameters(),
           np.array([0.95593063, -4.22207792, 0.96342365, 0.96601695]))
    

if __name__ == '__main__':

    unittest.main()
