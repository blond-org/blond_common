# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to fit distribution functions, such as bunch line densities**

:Authors: **Alexandre Lasheen**, **Juan F. Esteban Mueller**,
          **Markus Schwarz**
'''

# General imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import warnings

# Cedar imports

from .. interfaces.beam import analytic_distribution


class FitOptions():

    def __init__(self, bunchLengthFactor=1., bunchPositionOffset=0.,
                 nPointsNoise=3, fitInitialParameters=None,
                 bounds=None, fittingRoutine='curve_fit',
                 method='Powell', options=None, residualFunction=None,
                 extraOptions=None):

        self.bunchLengthFactor = bunchLengthFactor
        self.bunchPositionOffset = bunchPositionOffset
        self.nPointsNoise = nPointsNoise
        self.fitInitialParameters = fitInitialParameters
        self.bounds = bounds
        self.fittingRoutine = fittingRoutine
        self.method = method
        self.options = options
        self.residualFunction = residualFunction
        self.extraOptions = extraOptions


class PlotOptions():

    def __init__(self, figname='Bunch profile', clf=True, interactive=True,
                 legend=True):

        self.figname = figname
        self.clf = clf
        self.interactive = interactive
        self.legend = legend


def FWHM(time, bunch, level=0.5, fitOpt=None, plotOpt=None):
    '''
    Compute bunch length and bunch position from FWHM of the profile.
    '''

    if fitOpt is None:
        fitOpt = FitOptions()

    # Removing baseline
    profileToFit = bunch-np.mean(bunch[0:fitOpt.nPointsNoise])

    # Time resolution
    timeInterval = time[1] - time[0]

    # Max and xFator times max values (defaults to half max)
    maximumValue = np.max(profileToFit)
    half_max = level * maximumValue

    # First aproximation for the half maximum values
    taux = np.where(profileToFit >= half_max)
    taux1 = taux[0][0]
    taux2 = taux[0][-1]

    # Interpolation of the time where the line density is half the maximun
    if taux1 == 0:
        t1 = time[taux1]
        warnings.warn('FWHM is at left boundary of profile!')
    else:
        t1 = time[taux1] - (profileToFit[taux1]-half_max) \
            / (profileToFit[taux1] - profileToFit[taux1-1]) * timeInterval

    if taux2 == (len(profileToFit)-1):
        t2 = time[taux2]
        warnings.warn('FWHM is at right boundary of profile!')
    else:
        t2 = time[taux2] + (profileToFit[taux2]-half_max) \
            / (profileToFit[taux2] - profileToFit[taux2+1]) * timeInterval

    # Adjusting the FWHM with some scaling factor
    if isinstance(fitOpt.bunchLengthFactor, str):
        if fitOpt.bunchLengthFactor == 'gaussian':
            bunchLengthFactor = 4. / (2. * np.sqrt(-2 * np.log(level)))
        elif fitOpt.bunchLengthFactor == 'parabolic_line':
            bunchLengthFactor = 4. / (
                2*np.sqrt(3+2*1.) * np.sqrt(1-level**(1/1.)))
        elif fitOpt.bunchLengthFactor == 'parabolic_amplitude':
            bunchLengthFactor = 4. / (
                2*np.sqrt(3+2*1.5) * np.sqrt(1-level**(1/1.5)))
    else:
        bunchLengthFactor = fitOpt.bunchLengthFactor

    bunchLength = bunchLengthFactor * (t2-t1)
    bunchPosition = (t1+t2)/2 + fitOpt.bunchPositionOffset
    extraParameters = None

    if plotOpt is not None:
        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time, profileToFit)
        plt.plot([t1, t2], [half_max, half_max], 'r')
        plt.plot([t1, t2], [half_max, half_max], 'ro', markersize=5)
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return bunchPosition, bunchLength, extraParameters


def peakValue(time, bunch, level=1, fitOpt=None, plotOpt=None):
    '''
    Gives the peak of the profile averaged above the 'level' (default is max).
    '''

    if fitOpt is None:
        fitOpt = FitOptions()

    sampledNoise = np.mean(bunch[0:fitOpt.nPointsNoise])

    bunchPosition = 0
    bunchLength = 0
    extraParameters = np.mean(
        bunch[bunch >= (level*np.max(bunch-sampledNoise))]-sampledNoise)

    if plotOpt is not None:
        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time, bunch-sampledNoise)
        plt.plot(time[bunch >= (level*np.max(bunch-sampledNoise))],
                 bunch[bunch >= (level*np.max(bunch-sampledNoise))] -
                 sampledNoise)
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return bunchPosition, bunchLength, extraParameters


def profileSum(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Compute the sum of the profile.
    '''

    if fitOpt is None:
        fitOpt = FitOptions()

    bunchPosition = 0
    bunchLength = 0
    extraParameters = np.sum(bunch - np.mean(bunch[0:fitOpt.nPointsNoise]))

    if plotOpt is not None:
        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time, bunch-np.mean(bunch[0:fitOpt.nPointsNoise]))
        plt.plot(time[0:fitOpt.nPointsNoise],
                 (bunch-np.mean(bunch[0:fitOpt.nPointsNoise]))[
                     0:fitOpt.nPointsNoise])
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return bunchPosition, bunchLength, extraParameters


def RMS(time, bunch, fitOpt=None):
    '''
    Compute the rms bunch length and position from the profile.
    '''

    if fitOpt is None:
        fitOpt = FitOptions()

    deltaX = time[1]-time[0]

    # Removing baseline
    profileToFit = bunch-np.mean(bunch[0:fitOpt.nPointsNoise])

    normalizedProfileInputY = profileToFit / np.trapz(profileToFit, dx=deltaX)

    bunchPosition = np.trapz(time * normalizedProfileInputY, dx=deltaX)
    bunchLength = fitOpt.bunchLengthFactor * np.sqrt(
        np.trapz(((time - bunchPosition)**2) * normalizedProfileInputY,
                 dx=deltaX))
    bunchPosition += fitOpt.bunchPositionOffset
    extraParameters = None

    return bunchPosition, bunchLength, extraParameters


def binomialParametersFromRatio(time, bunch, levels=[0.8, 0.2],
                                ratioLookUpTable=None,
                                fitOpt=None, plotOpt=None):
    '''
    *Compute RMS bunch length, bunch position, exponent assuming a binomial
    line density from Full-Width at different levels.*
    '''

    if fitOpt is None:
        fitOpt = FitOptions()

    if ratioLookUpTable is None:
        level1 = np.max(levels)
        level2 = np.min(levels)
        ratioLookUpTable = _binomialParametersFromRatioLookupTable(
            level1, level2)
        exponentArray, ratioFWArray, levels = ratioLookUpTable
    else:
        ratioLookUpTable = ratioLookUpTable
        exponentArray, ratioFWArray, levels = ratioLookUpTable
        level1 = np.max(levels)
        level2 = np.min(levels)

    # Finding the width at two different levels
    bunchPosition_1, bunchLength_1 = FWHM(
        time, bunch, level=level1)[0:2]
    bunchPosition_2, bunchLength_2 = FWHM(
        time, bunch, level=level2)[0:2]

    ratioFW = bunchLength_1/bunchLength_2

    exponentFromRatio = np.interp(ratioFW, ratioFWArray, exponentArray)

    fullBunchLengthFromRatio = (
        bunchLength_2 / np.sqrt(1-level2**(1/exponentFromRatio)) +
        bunchLength_1 / np.sqrt(1-level1**(1/exponentFromRatio))) / 2

    bunchLength = fitOpt.bunchLengthFactor*fullBunchLengthFromRatio / \
        (2.*np.sqrt(3.+2.*exponentFromRatio))
    bunchPosition = (bunchPosition_1 + bunchPosition_2)/2 + \
        fitOpt.bunchPositionOffset
    returnFitParameters = np.array(
        [fullBunchLengthFromRatio, exponentFromRatio])

    if plotOpt is not None:
        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time, bunch)
        plt.plot([bunchPosition_2-bunchLength_2/2,
                  bunchPosition_2+bunchLength_2/2],
                 [level2*np.max(bunch),
                  level2*np.max(bunch)], 'r')
        plt.plot([bunchPosition_1-bunchLength_1/2,
                  bunchPosition_1+bunchLength_1/2],
                 [level1*np.max(bunch),
                  level1*np.max(bunch)], 'm')
        plt.plot(time, distribution_functions.binomialAmplitudeN(
            time, *[np.max(bunch),
                    bunchPosition,
                    fullBunchLengthFromRatio,
                    exponentFromRatio]))
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return bunchPosition, bunchLength, returnFitParameters


def _binomialParametersFromRatioLookupTable(level1=0.8, level2=0.2,
                                            exponentMin=0.5, exponentMax=10,
                                            exponentDistrib='logspace',
                                            exponentNPoints=100,
                                            exponentArray=None):
    '''
    *Create the lookup table for the binomialParametersFromRatio function.*
    '''

    if exponentArray is None:
        if exponentDistrib == 'linspace':
            exponentArray = np.linspace(exponentMin, exponentMax,
                                        exponentNPoints)
        elif exponentDistrib == 'logspace':
            exponentArray = np.logspace(np.log10(exponentMin),
                                        np.log10(exponentMax),
                                        exponentNPoints)

    ratioFWArray = np.sqrt(
        (1-level1**(1/exponentArray))/(1-level2**(1/exponentArray)))

    sortAscendingRatioFWArray = np.argsort(ratioFWArray)

    return exponentArray[sortAscendingRatioFWArray], \
        ratioFWArray[sortAscendingRatioFWArray], [level1, level2]


def gaussianFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a gaussian function
    '''

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time, bunch, level=0.5)[1]])

    fitDistribtion = analytic_distribution.Gaussian(
        *fitOpt.fitInitialParameters,
        scale_means='FWHM', store_data=False)

    fitParameters = _lineDensityFit(time, bunch, fitDistribtion.profile,
                                    fitOpt=fitOpt, plotOpt=plotOpt)

    return fitParameters


def generalizedGaussianFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a generalizedGaussian function
    '''

    profileFitFunction = distribution_functions.generalizedGaussian

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='gaussian')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]/4.,  # 1 sigma !!
             2.])

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             (time[-1]-time[0])/2., 5.])

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def waterbagFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a waterbag function
    '''

    profileFitFunction = distribution_functions.waterbag

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_line')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.)/2])  # Full bunch length!!

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def parabolicLineFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a parabolicLine function
    '''

    profileFitFunction = distribution_functions.parabolicLine

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_line')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.)/2])  # Full bunch length!!

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def parabolicAmplitudeFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a parabolicAmplitude function
    '''

    profileFitFunction = distribution_functions.parabolicAmplitude

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def binomialAmplitude2Fit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a binomialAmplitude2 function
    '''

    profileFitFunction = distribution_functions.binomialAmplitude2

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def binomialAmplitudeNFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a binomialAmplitudeN function
    '''

    profileFitFunction = distribution_functions.binomialAmplitudeN

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2,  # Full bunch length!!
             1.5])

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def cosineFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a cosine function
    '''

    profileFitFunction = distribution_functions.cosine

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def cosineSquaredFit(time, bunch, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a cosineSquared function
    '''

    profileFitFunction = distribution_functions.cosineSquared

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(bunch)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(bunch),
             np.mean(time[bunch == maxProfile]),
             FWHM(time,
                  bunch,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    bunchPosition, bunchLength, extraParameters = _lineDensityFit(
        time, bunch, profileFitFunction, fitOpt=fitOpt, plotOpt=plotOpt)

    return bunchPosition, bunchLength, extraParameters


def _lineDensityFit(time, bunch, profileFitFunction, fitOpt=None,
                    plotOpt=None):
    '''
    Fit the profile with the profileFitFunction
    '''
    # TODO: since it only returns fit_parameters, need to update all fit functions
    if fitOpt is None:
        fitOpt = FitOptions()

    profileToFit = bunch-np.mean(bunch[0:fitOpt.nPointsNoise])

    # Rescaling so that the fit parameters are around 1
    rescaleFactorX = 1/(time[-1]-time[0])
    rescaleFactorY = 1/np.max(profileToFit)

    fitInitialParameters = np.array(fitOpt.fitInitialParameters)

    fitInitialParameters[0] *= rescaleFactorY
    fitInitialParameters[1] -= time[0]
    fitInitialParameters[1] *= rescaleFactorX
    fitInitialParameters[2] *= rescaleFactorX

    # Fitting
    if fitOpt.fittingRoutine == 'curve_fit':

        fit_parameters = curve_fit(
            profileFitFunction,
            (time-time[0])*rescaleFactorX,
            profileToFit*rescaleFactorY,
            p0=fitInitialParameters)[0]

    elif fitOpt.fittingRoutine == 'minimize':

        if fitOpt.residualFunction is None:
            fitOpt.residualFunction = _leastSquareResidualFunction

        fit_parameters = minimize(
            fitOpt.residualFunction,
            fitInitialParameters,
            args=(profileFitFunction,
                  (time-time[0])*rescaleFactorX,
                  profileToFit*rescaleFactorY),
            bounds=fitOpt.bounds,
            method=fitOpt.method,
            options=fitOpt.options)['x']

    # Abs on fit parameters
    fit_parameters = np.abs(fit_parameters)

    # Rescaling back to the original dimensions
    fit_parameters[0] /= rescaleFactorY
    fit_parameters[1] /= rescaleFactorX
    fit_parameters[1] += time[0]
    fit_parameters[2] /= rescaleFactorX

    if plotOpt is not None:

        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time,
                 profileToFit, label='Data')
        plt.plot(time,
                 profileFitFunction(time, *fitOpt.fitInitialParameters),
                 label='Initial guess')
        plt.plot(time,
                 profileFitFunction(time, *fit_parameters),
                 label='Fit')
        if plotOpt.legend:
            plt.legend(loc='best')
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return fit_parameters


def _leastSquareResidualFunction(fitParameters, *fittingArgList):
    '''
    * Function to be used for fitting in the minimize function (least square).*
    '''

    profileFitFunction = fittingArgList[0]
    time = fittingArgList[1]
    fittedProfileInputY = fittingArgList[2]

    residue = np.sum((fittedProfileInputY -
                      profileFitFunction(time, *fitParameters))**2)

    return residue
