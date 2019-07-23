# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module containing all base disitrubution functions used for fitting in
the distribution.py module**

:Authors: **Alexandre Lasheen**
'''

# General imports
from __future__ import division
import numpy as np


def gaussian(time, *fitParameters):
    '''
    Gaussian line density
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    sigma = abs(fitParameters[2])

    lineDensityFunction = amplitude * np.exp(
        -(time-bunchPosition)**2/(2*sigma**2))

    return lineDensityFunction


def generalizedGaussian(time, *fitParameters):
    '''
    Generalized gaussian line density
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    alpha = abs(fitParameters[2])
    exponent = abs(fitParameters[3])

    lineDensityFunction = amplitude * np.exp(
        -(np.abs(time - bunchPosition)/(2*alpha))**exponent)

    return lineDensityFunction


def waterbag(time, *fitParameters):
    '''
    Waterbag distribution line density
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
            (bunchLength/2))**2)**0.5

    return lineDensityFunction


def parabolicLine(time, *fitParameters):
    '''
    Parabolic line density
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
            (bunchLength/2))**2)

    return lineDensityFunction


def parabolicAmplitude(time, *fitParameters):
    '''
    Parabolic in action line density
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
            (bunchLength/2))**2)**1.5

    return lineDensityFunction


def binomialAmplitude2(time, *fitParameters):
    '''
    Binomial exponent 2 in action line density
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
            (bunchLength/2))**2)**2.

    return lineDensityFunction


def binomialAmplitudeN(time, *fitParameters):
    '''
    Binomial exponent n in action line density
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    bunchLength = abs(fitParameters[2])
    exponent = abs(fitParameters[3])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
            (bunchLength/2))**2)**exponent

    return lineDensityFunction


def cosine(time, *fitParameters):
    '''
    * Cosine line density *
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
        amplitude * np.cos(
            np.pi*(time[np.abs(time-bunchPosition) < bunchLength/2] -
                   bunchPosition) / bunchLength)

    return lineDensityFunction


def cosineSquared(time, *fitParameters):
    '''
    * Cosine squared line density *
    '''

    amplitude = fitParameters[0]
    bunchPosition = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
        amplitude * np.cos(
            np.pi*(time[np.abs(time-bunchPosition) < bunchLength/2] -
                   bunchPosition) / bunchLength)**2.

    return lineDensityFunction
