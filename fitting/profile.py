# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module with functions to fit profiles, such as bunch line densities**

- TODO: improve treatment of the baseline
- TODO: improve the implementation of options as dict and **kwargs
- TODO: For all fitting function, add formula and default initial guess for the fit
- TODO: detail options for each function, especially for fitting
- TODO: include the changes from interfaces.beam.analytic_distribution

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

# Residue function import
from . residue import vertical_least_square

# Devtools imports
from .. devtools.exceptions import InputError


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
    >>> maximum, center, fwhm = FWHM(time_array, data_array)

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
            raise InputError('The bunch length factor in FWHM is not ' +
                             'recognized !')
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

    return maximum_value, center, fwhm


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

    TODO: use the blond_common.maths package for integration functions

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile
    method : str
        The method used to do the integration, the possible inputs are:

        - "sum": uses np.sum (default)

        - "trapz": uses np.trapz

    Returns
    -------
    integrated_value : float
        The integrated bunch profile, in the units of time_array*data_array

    Example
    -------
    >>> ''' We generate a Gaussian distribution and get the integrated
    >>> profile '''
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

    # Time resolution
    time_interval = time_array[1] - time_array[0]

    if method == 'sum':
        integrated_value = time_interval * np.sum(
            data_array - np.mean(data_array[0:fitOpt.nPointsNoise]))
    elif method == 'trapz':
        integrated_value = time_interval * np.trapz(
            data_array - np.mean(data_array[0:fitOpt.nPointsNoise]))
    else:
        raise InputError('The method passed to the integrated_profile ' +
                         'function is not valid.')

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
    >>> ''' We generate a Gaussian distribution and get mean and rms '''
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
    >>> mean_position, rms_length = RMS(time_array, data_array)

    """

    if fitOpt is None:
        fitOpt = FitOptions()

    # Time resolution
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


def binomial_from_width_ratio(time_array, data_array, levels=[0.8, 0.2],
                              ratio_LUT=None,
                              fitOpt=None, plotOpt=None):
    r""" Function to evaluate the parameters of a binomial function, as defined
    in blond_common.interfaces.beam.analytic_distribution.binomial_amplitude,
    by using the width of the profile at two different levels as input.

    The function returns the parameters of a binomial profile as defined in
    blond_common.interfaces.beam.analytic_distribution.binomial_amplitude

    TODO: check if rms should be in the return list as well

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile
    levels : list or np.array with 2 elements
        Optional: The levels at which the width of the profile is used
        to evaluate the parameters of the fitting binomial profile.
        Default is [0.8, 0.2]
    ratio_LUT : output from _binomial_from_width_LUT_generation
        Optional: The function uses internally a lookup table obtained from the
        _binomial_from_width_LUT_generation function to evaluate the binomial
        parameters. The lookup table can be pre-calculated and passed directly
        for efficiency purposes, if the function is used several times in a
        row.

    Returns
    -------
    amplitude : float
        The amplitude of the binomial profile, in data_array units
    position : float
        The central position of the profile, assumed to be binomial,
        in time_array units
    full_length : float
        The full length of the profile, assumed to be binomial,
        in time_array units
    exponent : float
        The exponent of the profile, assumed to be binomial

    Example
    -------
    >>> ''' We generate a Gaussian distribution and get mean and rms '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import gaussian
    >>> from blond_common.fitting.profile import binomial_from_width_ratio
    >>> from blond_common.fitting.profile import _binomial_from_width_LUT_generation
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = gaussian(time_array, *[amplitude, position, length])
    >>>
    >>> amplitude, position, full_length, exponent = binomial_from_width_ratio(
    >>>    time_array, data_array)
    >>>
    >>> # For a gaussian profile, the exponent of the corresponding binomial
    >>> # profile is infinity, higher values of exponents are required in the
    >>> # lookup table to have a better evaluation.
    >>>
    >>> new_LUT = _binomial_from_width_LUT_generation(
    >>>    exponentMin=100, exponentMax=10000)
    >>>
    >>> amplitude, position, full_length, exponent = binomial_from_width_ratio(
    >>>    time_array, data_array, ratio_LUT=new_LUT)

    """

    if fitOpt is None:
        fitOpt = FitOptions()

    if ratio_LUT is None:
        level1 = np.max(levels)
        level2 = np.min(levels)
        ratio_LUT = binomial_from_width_LUT_generation(
            [level1, level2])
        exponentArray, ratioFWArray, levels = ratio_LUT
    else:
        exponentArray, ratioFWArray, levels = ratio_LUT
        level1 = np.max(levels)
        level2 = np.min(levels)

    # Finding the width at two different levels
    bunchPosition_1, bunchLength_1 = FWHM(
        time_array, data_array, level=level1)[1:3]
    bunchPosition_2, bunchLength_2 = FWHM(
        time_array, data_array, level=level2)[1:3]

    ratioFW = bunchLength_1/bunchLength_2

    exponent = np.interp(ratioFW, ratioFWArray, exponentArray)

    full_length = (
        bunchLength_2 / np.sqrt(1-level2**(1/exponent)) +
        bunchLength_1 / np.sqrt(1-level1**(1/exponent))) / 2

    rms = fitOpt.bunchLengthFactor*full_length / \
        (2.*np.sqrt(3.+2.*exponent))

    position, amplitude = peak_value(time_array, data_array)
    position += fitOpt.bunchPositionOffset

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
            time_array, *[amplitude,
                          position,
                          full_length,
                          exponent]))
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return amplitude, position, full_length, exponent


def binomial_from_width_LUT_generation(levels=[0.8, 0.2],
                                       exponent_min=0.5, exponent_max=10.,
                                       exponent_distrib='logspace',
                                       exponent_npoints=100,
                                       exponent_array=None):
    r""" Function to create the lookup table (LUT) for the
    binomial_from_width_ratio function.

    TODO: return error if exponent_min<0.5

    Parameters
    ----------
    levels : list or np.array with 2 elements
        Optional: The levels at which the width of the profile is used
        to evaluate the parameters of the fitting binomial profile.
        Default is [0.8, 0.2]
    exponent_min : float
        Optional: The smallest exponent to consider for the binomial profile
        Default is 0.5 (NB: cannot be smaller than 0.5)
    exponent_max : float
        Optional: The largest exponent to consider for the binomial profile
        Default is 10. (NB: a higher value is necessary for Gaussian profiles)
    exponent_distrib : str
        Optional: Define how the exponent array of the LUT is distributed
        The possible settings are:

        - logspace: uses np.logspace (default)

        - linspace: uses np.linspace

    exponent_npoints: int
        Optional: The number of points for the lookup table
        Default is 100
    exponent_array: list or np.array
        Optional: This option replaces and discards all the other optional
        inputs

    Returns
    -------
    exponent_array : np.array
        The expected binomial profile exponent corresponding to the ratio
        of the Full-Widths for the specified levels
    ratio_FW : np.array
        The ratio of the Full-Widths corresponding to the exponent_array
        for the specified levels
    levels : list
        The levels at which the width of the profile is used
        to evaluate the parameters of the fitting binomial profile.

    Example
    -------
    >>> ''' We generate a Gaussian distribution and get mean and rms '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import gaussian
    >>> from blond_common.fitting.profile import binomial_from_width_ratio
    >>> from blond_common.fitting.profile import _binomial_from_width_LUT_generation
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = gaussian(time_array, *[amplitude, position, length])
    >>>
    >>> position, full_length, exponent = binomial_from_width_ratio(
    >>>    time_array, data_array)
    >>>
    >>> # For a gaussian profile, the exponent of the corresponding binomial
    >>> # profile is infinity, higher values of exponents are required in the
    >>> # lookup table to have a better evaluation.
    >>>
    >>> new_LUT = _binomial_from_width_LUT_generation(
    >>>    exponentMin=100, exponentMax=10000)
    >>>
    >>> position, full_length, exponent = binomial_from_width_ratio(
    >>>    time_array, data_array, ratio_LUT=new_LUT)

    """

    if exponent_array is None:
        if exponent_distrib == 'linspace':
            exponent_array = np.linspace(exponent_min, exponent_max,
                                         exponent_npoints)
        elif exponent_distrib == 'logspace':
            exponent_array = np.logspace(np.log10(exponent_min),
                                         np.log10(exponent_max),
                                         exponent_npoints)
        else:
            raise InputError('The input exponent_distrib in ' +
                             'binomial_from_width_LUT_generation is not valid')

    level1 = np.max(levels)
    level2 = np.min(levels)

    ratio_FW = np.sqrt(
        (1-level1**(1/exponent_array))/(1-level2**(1/exponent_array)))

    sorting_ratio_FW = np.argsort(ratio_FW)

    return exponent_array[sorting_ratio_FW], \
        ratio_FW[sorting_ratio_FW], [level1, level2]


def gaussian_fit(time_array, data_array,
                 fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Gaussian profile.

    The function returns the parameters of a Gaussian profile as defined in
    blond_common.interfaces.beam.analytic_distribution.Gaussian

    TODO: update with the new analytic_distribution implementation

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    rms_length : float
        The rms length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Gaussian profile and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import gaussian
    >>> from blond_common.fitting.profile import gaussian_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = gaussian(time_array, *[amplitude, position, length])
    >>>
    >>> amplitude, position, rms_length = gaussian_fit(
    >>>    time_array, data_array)

    """

    if fitOpt is None:
        fitOpt = FitOptions()

    if fitOpt.fitInitialParameters is None:
        maxProfile = np.max(data_array)
        fitOptFWHM = FitOptions(bunchLengthFactor='gaussian')
        fitOpt.fitInitialParameters = np.array(
            [maxProfile-np.min(data_array),
             np.mean(time_array[data_array == maxProfile]),
             FWHM(time_array, data_array, level=0.5,
                  fitOpt=fitOptFWHM)[2]/4.])

    fitDistribtion = analytic_distribution.Gaussian(
        *fitOpt.fitInitialParameters,
        scale_means='FWHM', store_data=False)

    fitParameters = arbitrary_profile_fit(time_array, data_array,
                                          fitDistribtion.profile,
                                          fitOpt=fitOpt, plotOpt=plotOpt)

    return fitParameters


def generalized_gaussian_fit(time_array, data_array,
                             fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Generalized Gaussian profile.

    The function returns the parameters of a Generalized Gaussian profile
    as defined in
    blond_common.interfaces.beam.analytic_distribution.generalizedGaussian

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    rms_length : float
        The rms length of the profile, in time_array units
    exponent : float
        The exponent of the Generalized Gaussian profile

    Example
    -------
    >>> ''' We generate a Generalized Gaussian profile and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import generalizedGaussian
    >>> from blond_common.fitting.profile import generalized_gaussian_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>> exponent = 2.5
    >>>
    >>> data_array = generalizedGaussian(
    >>>    time_array, *[amplitude, position, length, exponent])
    >>>
    >>> amplitude, position, rms_length, exponent = generalized_gaussian_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.generalizedGaussian

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
                  plotOpt=None)[2]/4.,  # 1 sigma !!
             2.])

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def waterbag_fit(time_array, data_array,
                 fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Waterbag profile.

    The function returns the parameters of a Waterbag profile
    as defined in
    blond_common.interfaces.beam.analytic_distribution.waterbag

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    full_length : float
        The full length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Waterbag profile and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import waterbag
    >>> from blond_common.fitting.profile import waterbag_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = waterbag(
    >>>    time_array, *[amplitude, position, length, exponent])
    >>>
    >>> amplitude, position, full_length = waterbag_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.waterbag

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
                  plotOpt=None)[2]*np.sqrt(3+2*1.)/2])  # Full bunch length!!

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def parabolic_line_fit(time_array, data_array,
                       fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Parabolic profile
    (parabolic line density).

    The function returns the parameters of a Parabolic profile
    as defined in
    blond_common.interfaces.beam.analytic_distribution.parabolicLine

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    full_length : float
        The full length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Parabolic profile and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import parabolicLine
    >>> from blond_common.fitting.profile import parabolic_line_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = parabolicLine(
    >>>    time_array, *[amplitude, position, length, exponent])
    >>>
    >>> amplitude, position, full_length = parabolic_line_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.parabolicLine

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
                  plotOpt=None)[2]*np.sqrt(3+2*1.)/2])  # Full bunch length!!

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def parabolic_amplitude_fit(time_array, data_array, fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Parabolic profile
    (parabolic amplitude density in phase space).

    The function returns the parameters of a Parabolic amplitude profile
    as defined in
    blond_common.interfaces.beam.analytic_distribution.parabolicAmplitude

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    full_length : float
        The full length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Parabolic amplitude distribution and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import parabolicAmplitude
    >>> from blond_common.fitting.profile import parabolic_amplitude_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = parabolicAmplitude(
    >>>    time_array, *[amplitude, position, length, exponent])
    >>>
    >>> amplitude, position, full_length = parabolic_amplitude_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.parabolicAmplitude

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
                  plotOpt=None)[2]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def binomial_amplitude2_fit(time_array, data_array, fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Binomial profile, with exponent
    2 (binomial density in phase space).

    The function returns the parameters of a Binomial amplitude profile
    with exponent 2 as defined in
    blond_common.interfaces.beam.analytic_distribution.binomialAmplitude2

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    full_length : float
        The full length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Binomial amplitude 2 distribution and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import binomialAmplitude2
    >>> from blond_common.fitting.profile import binomial_amplitude2_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = binomial_amplitude2(
    >>>    time_array, *[amplitude, position, length, exponent])
    >>>
    >>> amplitude, position, full_length = binomial_amplitude2_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.binomialAmplitude2

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
                  plotOpt=None)[2]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def binomial_amplitudeN_fit(time_array, data_array, fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Binomial profile (binomial
    amplitude in phase space).

    The function returns the parameters of a Binomial profile
    as defined in
    blond_common.interfaces.beam.analytic_distribution.binomialAmplitudeN

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    full_length : float
        The full length of the profile, in time_array units
    exponent : float
        The exponent of the Binomial profile

    Example
    -------
    >>> ''' We generate a Binomial distribution and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import binomialAmplitudeN
    >>> from blond_common.fitting.profile import binomial_amplitudeN_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>> exponent = 2.5
    >>>
    >>> data_array = binomialAmplitudeN(
    >>>    time_array, *[amplitude, position, length, exponent])
    >>>
    >>> amplitude, position, full_length, exponent = binomial_amplitudeN_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.binomialAmplitudeN

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
                  plotOpt=None)[2]*np.sqrt(3+2*1.5)/2,  # Full bunch length!!
             1.5])

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def cosine_fit(time_array, data_array, fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Cosine profile.

    The function returns the parameters of a Cosine profile
    as defined in
    blond_common.interfaces.beam.analytic_distribution.cosine

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    full_length : float
        The full length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Cosine profile and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import cosine
    >>> from blond_common.fitting.profile import cosine_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = cosine(
    >>>    time_array, *[amplitude, position, length])
    >>>
    >>> amplitude, position, full_length = cosine_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.cosine

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
                  plotOpt=None)[2]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def cosine_squared_fit(time_array, data_array, fitOpt=None, plotOpt=None):
    r""" Function to fit a given profile with a Cosine Squared profile.

    The function returns the parameters of a Cosine Squared profile
    as defined in
    blond_common.interfaces.beam.analytic_distribution.cosineSquared

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile

    Returns
    -------
    amplitude : float
        The amplitude of the profile, in data_array units
    position : float
        The central position of the profile, in time_array units
    full_length : float
        The full length of the profile, in time_array units

    Example
    -------
    >>> ''' We generate a Cosine profile and fit it '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import cosineSquared
    >>> from blond_common.fitting.profile import cosine_squared_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = cosineSquared(
    >>>    time_array, *[amplitude, position, length])
    >>>
    >>> amplitude, position, full_length = cosine_squared_fit(
    >>>    time_array, data_array)

    """

    profile_fit_function = analytic_distribution.cosineSquared

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
                  plotOpt=None)[2]*np.sqrt(3+2*1.5)/2])  # Full bunch length!!

    fit_parameters = arbitrary_profile_fit(
        time_array, data_array, profile_fit_function,
        fitOpt=fitOpt, plotOpt=plotOpt)

    return fit_parameters


def arbitrary_profile_fit(time_array, data_array, profile_fit_function,
                          fitOpt, plotOpt=None):
    r""" Function to fit a given profile with a user defined arbitrary profile.

    Parameters
    ----------
    time_array : list or np.array
        The input time
    data_array : list or np.array
        The input profile
    profile_fit_function : function
        User defined function to fit the input data_array. The function
        should have the input arguments (time_array, *fit_parameters)
        and returns a profile_array which will be compared to data_array

    Returns
    -------
    fitted_parameters : np.array
        The fitted parameters, in the same order as defined by the user in
        his profile_fit_function

    Example
    -------
    >>> ''' We generate a Gaussian profile and fit it, in this example we use
    >>> the same Gaussian function used to generate the input profile '''
    >>> import numpy as np
    >>> from blond_common.interfaces.beam.analytic_distribution import Gaussian
    >>> from blond_common.fitting.profile import arbitrary_profile_fit
    >>>
    >>> time_array = np.arange(0, 25e-9, 0.1e-9)
    >>>
    >>> amplitude = 1.
    >>> position = 13e-9
    >>> length = 2e-9
    >>>
    >>> data_array = Gaussian(
    >>>    time_array, *[amplitude, position, length])
    >>>
    >>> amplitude, position, full_length = arbitrary_profile_fit(
    >>>    time_array, data_array, Gaussian)

    """

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
            profile_fit_function,
            (time_array-time_array[0])*rescaleFactorX,
            profileToFit*rescaleFactorY,
            p0=fitInitialParameters)[0]

    elif fitOpt.fittingRoutine == 'minimize':

        if fitOpt.residualFunction is None:
            fitOpt.residualFunction = vertical_least_square
        else:
            raise InputError('The residualFunction in the FitOptions is not ' +
                             'valid.')

        fit_parameters = minimize(
            fitOpt.residualFunction,
            fitInitialParameters,
            args=(profile_fit_function,
                  (time_array-time_array[0])*rescaleFactorX,
                  profileToFit*rescaleFactorY),
            bounds=fitOpt.bounds,
            method=fitOpt.method,
            options=fitOpt.options)['x']

    else:

        raise InputError('The fittingRoutine in the FitOptions is not ' +
                         'valid.')

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
                 profile_fit_function(time_array,
                                      *fitOpt.fitInitialParameters),
                 label='Initial guess')
        plt.plot(time_array,
                 profile_fit_function(time_array, *fit_parameters),
                 label='Fit')
        if plotOpt.legend:
            plt.legend(loc='best')
        if plotOpt.interactive:
            plt.pause(0.00001)
        else:
            plt.show()

    return fit_parameters
