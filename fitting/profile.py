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

# Analytic distributions import
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


def FWHM(time_array, data_array, level=0.5, fitOpt=None, plotOpt=None):
    r""" Function to compute the Full-Width at Half Maximum for a given
    profile. The Maximum corresponds to the numerical maximum of the
    data_array.

    TODO: add an option to use peakValue to get the Maximum, to have the option
    to get an averaged value of the maximum (e.g. at >95%)

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile
    level : float
        Optional: The ratio from the maximum for which the width of the profile
        is returned
        Default is 0.5, as for Half of the Maximum

    Returns
    -------
    center : float
        The center of the Width at Half Maximum, in the units of time_array
    fwhm : float
        The Full Width at Half Maximum, in the units of time_array
        NB: if the "level" option is set to any other value than 0.5,
        the output corresponds to the full width at the specified "level"
        of the maximum

    Example
    -------
    >>> ''' We generate a Gaussian distribution and get its FWHM '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import gaussian
    >>> from blond_common.fitting.profile import FWHM
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = gaussian(time_array, *[amplitude, position, length])
    >>>
    >>> center, fwhm = FWHM(time_array, data_array)

    """

    if fitOpt is None:
        fitOpt = FitOptions()

    # Removing baseline
    profileToFit = data_array-np.mean(data_array[0:fitOpt.nPointsNoise])

    # Time resolution
    time_interval = time_array[1] - time_array[0]

    # Max and xFator times max values (defaults to half max)
    maximum_value = np.max(profileToFit)
    half_max = level * maximum_value

    # First aproximation for the half maximum values
    taux = np.where(profileToFit >= half_max)
    taux1 = taux[0][0]
    taux2 = taux[0][-1]

    # Interpolation of the time where the line density is half the maximum
    if taux1 == 0:
        t1 = time_array[taux1]
        warnings.warn('FWHM is at left boundary of profile!')
    else:
        t1 = time_array[taux1] - (profileToFit[taux1]-half_max) \
            / (profileToFit[taux1] - profileToFit[taux1-1]) * time_interval

    if taux2 == (len(profileToFit)-1):
        t2 = time_array[taux2]
        warnings.warn('FWHM is at right boundary of profile!')
    else:
        t2 = time_array[taux2] + (profileToFit[taux2]-half_max) \
            / (profileToFit[taux2] - profileToFit[taux2+1]) * time_interval

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

    fwhm = bunchLengthFactor * (t2-t1)
    center = (t1+t2)/2 + fitOpt.bunchPositionOffset

    if plotOpt is not None:
        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time_array, profileToFit)
        plt.plot([t1, t2], [half_max, half_max], 'r')
        plt.plot([t1, t2], [half_max, half_max], 'ro', markersize=5)
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return center, fwhm


def peak_value(time_array, data_array, level=1.0, fitOpt=None, plotOpt=None):
    r""" Function to get the peak amplitude of the profile. If a "level"
    is passed, the function averages the points above the speicifed "level".

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile
    level : float
        Optional: The ratio of the maximum above which the profile is averaged
        Default is 1.0 to return the numerical maximum

    Returns
    -------
    position : float
        The position of the peak value, in the units of time_array
    amplitude : float
        The peak amplitude, in the units of the data_array
        NB: if the "level" option is set to any other value than 1.0,
        the output corresponds to the peak, averaged for the points above
        the specified "level" of the maximum

    Example
    -------
    >>> ''' We generate a Gaussian distribution and get its peak amplitude '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import gaussian
    >>> from blond_common.fitting.profile import peak_value
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = gaussian(time_array, *[amplitude, position, length])
    >>>
    >>> peak_position, amplitude = peak_value(time_array, data_array)

    """

    if fitOpt is None:
        fitOpt = FitOptions()

    sampledNoise = np.mean(data_array[0:fitOpt.nPointsNoise])

    selected_points = np.where(
        data_array >= (level*np.max(data_array-sampledNoise)))[0]

    position = np.mean(time_array[selected_points])

    amplitude = np.mean(
        data_array[selected_points] -
        sampledNoise)

    if plotOpt is not None:
        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time_array, data_array-sampledNoise)
        plt.plot(
            time_array[data_array >= (level*np.max(data_array-sampledNoise))],
            data_array[data_array >= (level*np.max(data_array-sampledNoise))] -
            sampledNoise)
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return position, amplitude


def integrated_profile(time_array, data_array, method='sum',
                       fitOpt=None, plotOpt=None):
    r""" Function to compute the integrated bunch profile.

    TODO: add an error message it the "method" input is not correct

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile
    method : str
        The method used to do the integration, the possible inputs are:
        - "sum": uses np.sum
        - "trapz": uses np.trapz

    Returns
    -------
    integrated_value : float
        The integrated bunch profile, in the units of time_array*data_array

    Example
    -------
    >>> ''' We generate a Gaussian distribution and get its peak amplitude '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import gaussian
    >>> from blond_common.fitting.profile import integrated_profile
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = gaussian(time_array, *[amplitude, position, length])
    >>>
    >>> integrated_value = integrated_profile(time_array, data_array)

    """

    if fitOpt is None:
        fitOpt = FitOptions()

    if method == 'sum':
        integrated_value = np.sum(
            data_array - np.mean(data_array[0:fitOpt.nPointsNoise]))
    elif method == 'trapz':
        integrated_value = np.trapz(
            data_array - np.mean(data_array[0:fitOpt.nPointsNoise]))

    if plotOpt is not None:
        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(
            time_array, data_array-np.mean(data_array[0:fitOpt.nPointsNoise]))
        plt.plot(
            time_array[0:fitOpt.nPointsNoise],
            (data_array-np.mean(data_array[0:fitOpt.nPointsNoise]))[
                0:fitOpt.nPointsNoise])
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return integrated_value


def RMS(time_array, data_array, fitOpt=None):
    r""" Function to compute the mean and root mean square (RMS) of a profile.

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    mean : float
        The mean position of the profile, in time_array units
    rms : float
        The rms length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Gaussian distribution and get its peak amplitude '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import gaussian
    >>> from blond_common.fitting.profile import RMS
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = gaussian(time_array, *[amplitude, position, length])
    >>>
    >>> mean, rms = RMS(time_array, data_array)

    """

    if fitOpt is None:
        fitOpt = FitOptions()

    time_interval = time_array[1] - time_array[0]

    # Removing baseline
    profile = data_array-np.mean(data_array[0:fitOpt.nPointsNoise])

    normalized_profile = profile / np.trapz(
        profile, dx=time_interval)

    mean = np.trapz(time_array * normalized_profile,
                    dx=time_interval)

    rms = fitOpt.bunchLengthFactor * np.sqrt(
        np.trapz(((time_array - mean)**2) * normalized_profile,
                 dx=time_interval))

    mean += fitOpt.bunchPositionOffset

    return mean, rms


def binomialParametersFromRatio(time_array, data_array, levels=[0.8, 0.2],
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
        time_array, data_array, level=level1)[0:2]
    bunchPosition_2, bunchLength_2 = FWHM(
        time_array, data_array, level=level2)[0:2]

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
        plt.plot(time_array, data_array)
        plt.plot([bunchPosition_2-bunchLength_2/2,
                  bunchPosition_2+bunchLength_2/2],
                 [level2*np.max(data_array),
                  level2*np.max(data_array)], 'r')
        plt.plot([bunchPosition_1-bunchLength_1/2,
                  bunchPosition_1+bunchLength_1/2],
                 [level1*np.max(data_array),
                  level1*np.max(data_array)], 'm')
        plt.plot(time_array, analytic_distribution.binomialAmplitudeN(
            time_array, *[np.max(data_array),
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


def gaussianFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a gaussian function
    '''

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array, data_array, level=0.5)[1]])

    fitDistribtion = analytic_distribution.Gaussian(
        *fitOpt.fitInitialParameters,
        scale_means='FWHM', store_data=False)

    fitParameters = _lineDensityFit(time_array, data_array,
                                    fitDistribtion.profile,
                                    fitOpt=fitOpt, plotOpt=plotOpt)

    return fitParameters


def generalizedGaussianFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a generalizedGaussian function
    '''

    profileFitFunction = analytic_distribution.generalizedGaussian

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='gaussian')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]/4.,  # 1 sigma !!
             2.])

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             (time_array[-1]-time_array[0])/2., 5.])

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def waterbagFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a waterbag function
    '''

    profileFitFunction = analytic_distribution.waterbag

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_line')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.)/2])  # Full bunch length!!

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def parabolicLineFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a parabolicLine function
    '''

    profileFitFunction = analytic_distribution.parabolicLine

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_line')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.)/2])  # Full bunch length!!

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def parabolicAmplitudeFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a parabolicAmplitude function
    '''

    profileFitFunction = analytic_distribution.parabolicAmplitude

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def binomialAmplitude2Fit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a binomialAmplitude2 function
    '''

    profileFitFunction = analytic_distribution.binomialAmplitude2

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def binomialAmplitudeNFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a binomialAmplitudeN function
    '''

    profileFitFunction = analytic_distribution.binomialAmplitudeN

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2,  # Full bunch length!!
             1.5])

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def cosineFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a cosine function
    '''

    profileFitFunction = analytic_distribution.cosine

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def cosineSquaredFit(time_array, data_array, fitOpt=None, plotOpt=None):
    '''
    Fit the profile with a cosineSquared function
    '''

    profileFitFunction = analytic_distribution.cosineSquared

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='parabolic_amplitude')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array,
                  data_array,
                  level=0.5,
                  fitOpt=fitOptFWHM,
                  plotOpt=None)[1]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = _lineDensityFit(
        time_array, data_array, profileFitFunction,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def _lineDensityFit(time_array, data_array, profileFitFunction, fitOpt=None,
                    plotOpt=None):
    '''
    Fit the profile with the profileFitFunction
    '''
    # TODO: since it only returns fit_parameters, need to update all fit functions
    if fitOpt is None:
        fitOpt = FitOptions()

    profileToFit = data_array-np.mean(data_array[0:fitOpt.nPointsNoise])

    # Rescaling so that the fit parameters are around 1
    rescaleFactorX = 1/(time_array[-1]-time_array[0])
    rescaleFactorY = 1/np.max(profileToFit)

    fitInitialParameters = np.array(fitOpt.fitInitialParameters)

    fitInitialParameters[0] *= rescaleFactorY
    fitInitialParameters[1] -= time_array[0]
    fitInitialParameters[1] *= rescaleFactorX
    fitInitialParameters[2] *= rescaleFactorX

    # Fitting
    if fitOpt.fittingRoutine == 'curve_fit':

        fit_parameters = curve_fit(
            profileFitFunction,
            (time_array-time_array[0])*rescaleFactorX,
            profileToFit*rescaleFactorY,
            p0=fitInitialParameters)[0]

    elif fitOpt.fittingRoutine == 'minimize':

        if fitOpt.residualFunction is None:
            fitOpt.residualFunction = _leastSquareResidualFunction

        fit_parameters = minimize(
            fitOpt.residualFunction,
            fitInitialParameters,
            args=(profileFitFunction,
                  (time_array-time_array[0])*rescaleFactorX,
                  profileToFit*rescaleFactorY),
            bounds=fitOpt.bounds,
            method=fitOpt.method,
            options=fitOpt.options)['x']

    # Abs on fit parameters
    fit_parameters = np.abs(fit_parameters)

    # Rescaling back to the original dimensions
    fit_parameters[0] /= rescaleFactorY
    fit_parameters[1] /= rescaleFactorX
    fit_parameters[1] += time_array[0]
    fit_parameters[2] /= rescaleFactorX

    if plotOpt is not None:

        plt.figure(plotOpt.figname)
        if plotOpt.clf:
            plt.clf()
        plt.plot(time_array,
                 profileToFit, label='Data')
        plt.plot(time_array,
                 profileFitFunction(time_array, *fitOpt.fitInitialParameters),
                 label='Initial guess')
        plt.plot(time_array,
                 profileFitFunction(time_array, *fit_parameters),
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
    time_array = fittingArgList[1]
    fittedProfileInputY = fittingArgList[2]

    residue = np.sum((fittedProfileInputY -
                      profileFitFunction(time_array, *fitParameters))**2)

    return residue
