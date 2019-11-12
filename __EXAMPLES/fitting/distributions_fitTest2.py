# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:13:23 2019

@author: schwarz
"""

import numpy as np
from matplotlib import pyplot as plt

from blond_common.interfaces.beam.analytic_distribution import Gaussian

input_parameters = [2,0.1,1]
time = np.linspace(-4,4,num=50)
gauss_profile = Gaussian(input_parameters, time_array=time)

# add some noise to the profile
np.random.seed(1789*1989)
y_data = gauss_profile + np.random.normal(0,0.05, 
                                          size=gauss_profile.size)

# create new Gaussian object and perform a Gaussian fit by passing x and y data
fitGauss = Gaussian(None, time, y_data)

print(f"input paramters: {input_parameters}")
print(f"paramters of fitted profile: {np.round(fitGauss.get_parameters(),3)}")
# should be [1.974 0.083 0.984]

plt.figure('fitted profile', clear=True)
plt.grid()
plt.plot(time, gauss_profile, label='analytic')
plt.plot(time, fitGauss.profile(time), '--', label='gaussian fit')
plt.plot(time, y_data, '.', label='nosy data')
plt.legend()
plt.tight_layout()
