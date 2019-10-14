# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:13:23 2019

@author: schwarz
"""

import numpy as np
from matplotlib import pyplot as plt

from blond_common.interfaces.beam.analytic_distribution import Gaussian

objGauss = Gaussian(1,0,3, store_data=False)

time = np.linspace(-4,4,num=50)

objGauss = Gaussian(1,0,1, time, store_data=True)

y_data = objGauss.computed_profile + np.random.normal(0,0.01,
                                          size=objGauss.computed_profile.size)

fitGauss = Gaussian(time, y_data)
tmp = objGauss.profile(4)

plt.figure('fitted profile', clear=True)
plt.grid()
plt.plot(time, y_data)
plt.plot(time, fitGauss.profile(time), '--')