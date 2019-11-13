# -*- coding: utf-8 -*-
'''
Created on 15 oct. 2019

@author: schwarz
'''

from matplotlib import pyplot as plt
import numpy as np
from blond_common.interfaces.beam.analytic_distribution import BinomialAmplitudeN

#binAmpNobj = BinomialAmplitudeN([1,0,1,1.5], scale_means='full_bunch_length')
binAmpNobj = BinomialAmplitudeN([1,-4.2,1,1.5])
print(binAmpNobj.RMS, binAmpNobj.FWHM, binAmpNobj.full_bunch_length)
print(0.5*binAmpNobj.amplitude, binAmpNobj.profile(binAmpNobj.center+0.5*binAmpNobj.FWHM))

x_data = np.linspace(-0.51*binAmpNobj.full_bunch_length,
                +0.51*binAmpNobj.full_bunch_length, num=50) + binAmpNobj.center
#y = binAmpNobj.profile(x)
y = BinomialAmplitudeN([1,-4.2,1,1.5], time_array=x_data)

np.random.seed(1789*1989)
y_data = y + np.random.normal(0,0.05, size=y.size)

fitted_object = BinomialAmplitudeN(None, x_data, y_data)
print(fitted_object.get_parameters())
plt.figure('binom profile', clear=True)
plt.grid()
plt.plot(x_data, y, label='analytic')
plt.plot(x_data, y_data, '.', label='noisy data')
plt.plot(x_data, fitted_object.profile(x_data), '--', label='fit')
plt.legend()
plt.tight_layout()

dx = x_data[1] - x_data[0]
freqs = np.linspace(-1/binAmpNobj.RMS, 1/binAmpNobj.RMS, num=len(x_data))
Ydft = np.zeros(len(freqs), dtype=complex)
for it, f in enumerate(freqs):
    Ydft[it] = np.trapz(y*np.exp(-2j*np.pi*f*x_data), dx=dx)

Y = binAmpNobj.spectrum(freqs)

plt.figure('binom spectrum', clear=True)
plt.grid()
plt.plot(freqs, Ydft.real)
plt.plot(freqs, Y.real, '--')
plt.plot(freqs, Ydft.imag)
plt.plot(freqs, Y.imag, '--')

