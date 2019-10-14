# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:01:25 2019

@author: schwarz
"""

import numpy as np
import matplotlib.pyplot as plt

from blond_common.interfaces.beam.analytic_distribution import Gaussian
import blond_common.fitting.profile_fitting as profile_fitting

import blond_common.devtools.BLonD_Rc as blondrc

time_array = np.arange(0, 25e-9, 0.1e-9)

amplitude = 1.
position = 13e-9
lengthParameter = 2e-9  # RMS by default
FWHM = 2*np.sqrt(np.log(4))*lengthParameter
initial_params_gauss = [amplitude, position, lengthParameter]

# create a Gaussian distribution object with these parmeters
gaussian_dist = Gaussian(*initial_params_gauss)

# compute the Gaussian profile
gaussian_profile = gaussian_dist.profile(time_array)

plt.figure('Gaussian profiles', clear=True)
plt.grid()
plt.plot(time_array, gaussian_profile, label='lengthParameter=RMS')

# compare input RMS and FWHM
print('input length and RMS: ', lengthParameter, gaussian_dist.RMS)
print('resulting FWHM and FWHM: ', FWHM, gaussian_dist.FWHM)

# 
print('profile at position where profile is a half maximum: ',
      amplitude/2, gaussian_dist.profile(position+FWHM/2))

# fit profile to a Gaussian
fitparams_gauss = profile_fitting.gaussianFit(time_array, gaussian_profile)

# notice that the 3rd element is the RMS bunch length
print('fit parameters:', fitparams_gauss)

# the fit parameters can be used as input for a new Gaussian distribution object
gaussian_dist_fit = Gaussian(*fitparams_gauss)
gaussian_profile_fit = gaussian_dist_fit.profile(time_array)
plt.plot(time_array, gaussian_profile_fit, '--', label='fit')

# now change the default bunch length parameter to FWHM
blondrc.rc('distribution', **{'scale_means':'FWHM'})
#blondrc.rc('distribution', **{'scale_means':'fourSigma_RMS'})

# now the 3rd argument is interpreted as FWHM, not RMS
gaussian_dist2 = Gaussian(*initial_params_gauss)
gaussian_profile2 = gaussian_dist2.profile(time_array)

print('length parameter is now interpreted as FWHM:', lengthParameter,
      gaussian_dist2.FWHM)
plt.plot(time_array, gaussian_profile2, '--', label='FWHM')

# a fit to the original profile now returns FWHM in the 3rd element
fitparams_gauss2 = profile_fitting.gaussianFit(time_array, gaussian_profile)
print('the fit now returns FWHM:', FWHM, fitparams_gauss2[2])

# since the 3rd argument is correctly interpreted as FWHM,
# using this for the creation of a Gaussian object yields the original
# the 3rd arguement is correctly interpreted as FWHM
gaussian_dist_fit2 = Gaussian(*fitparams_gauss2)
gaussian_profile_fit2 = gaussian_dist_fit2.profile(time_array[::10])

# ... so the actual profiles are the same
print(gaussian_dist_fit2.RMS, gaussian_dist.RMS)
plt.plot(time_array[::10], gaussian_profile_fit2, '.', label='fit 2')

plt.legend()
plt.tight_layout()