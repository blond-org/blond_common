# -*- coding: utf-8 -*-
'''
Created on 15 oct. 2019

@author: schwarz
'''

from matplotlib import pyplot as plt
import numpy as np


binAmpNobj = BinomialAmplitudeN(1,0,1,1.5)

x = np.linspace(-binAmpNobj.full_bunch_length, binAmpNobj.full_bunch_length)
y = binAmpNobj.profile(x)

plt.figure('binom profile', clear=True)
plt.grid()
plt.plot(x, y)

dx = x[1] - x[0]
freqs = np.linspace(-1/binAmpNobj.RMS, 1/binAmpNobj.RMS, num=len(x))
Ydft = np.zeros(len(freqs), dtype=complex)
for it, f in enumerate(freqs):
    Ydft[it] = np.trapz(y*np.exp(-2j*np.pi*f*x), dx=dx)

Y = binAmpNobj.spectrum(freqs)

plt.figure('binom spectrum', clear=True)
plt.grid()
plt.plot(freqs, Ydft.real)
plt.plot(freqs, Y.real, '--')
plt.plot(freqs, Ydft.imag)
plt.plot(freqs, Y.imag, '--')

tmpObj = _DistributionObject()
# gaussObj = Gaussian(1, 0.4, 1, scale_means='fourSigma_FWHM')
gaussObj = Gaussian(1/np.sqrt(2*np.pi), 0.4, 1, scale_means='fourSigma_FWHM')

x = np.linspace(-5*gaussObj.RMS, 5*gaussObj.RMS, num=200)
y = gaussObj.profile(x)

plt.figure('gauss profile', clear=True)
plt.grid()
plt.plot(x, y)
plt.plot(gaussObj.position - gaussObj.FWHM/2,
         gaussObj.profile(gaussObj.position - gaussObj.FWHM/2), 'ro')
plt.plot(gaussObj.position + gaussObj.FWHM/2,
         gaussObj.profile(gaussObj.position + gaussObj.FWHM/2), 'ro')

print(gaussObj.RMS, gaussObj.FWHM, gaussObj.fourSigma_RMS,
      gaussObj.fourSigma_FWHM)

dx = x[1] - x[0]
freqs = np.linspace(-1/gaussObj.RMS, 1/gaussObj.RMS, num=len(x))
Ydft = np.zeros(len(freqs), dtype=complex)
for it, f in enumerate(freqs):
    Ydft[it] = np.trapz(y*np.exp(-2j*np.pi*f*x), x=x, dx=dx)

Y = gaussObj.spectrum(freqs)

plt.figure('spectrum', clear=True)
plt.grid()
plt.plot(freqs, Ydft.real)
plt.plot(freqs, Y.real, '--')
plt.plot(freqs, Ydft.imag)
plt.plot(freqs, Y.imag, '--')
