# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module containing all base distribution functions used for fitting in
the distribution.py module**

:Authors: **Alexandre Lasheen**, **Markus Schwarz**
'''

# General imports
from __future__ import division
import numpy as np
import inspect
from scipy.special import gamma as gamma_function
from scipy.special import hyp0f1

# Other packages import
from ...devtools.BLonD_Rc import rcBLonDparams
from ...fitting import profile


def check_greater_zero(value, func_name):
    if value <= 0.0:
        raise ValueError("%s needs to be positive" % (func_name))


class _DistributionObject(object):
    r'''Base class for all analytic distributions

    This base class is not to be used directly. Instead, all analytic
    distributions should derive from it.

    Attributes
    ----------
    amplitude : float
        amplitude of the profile
    center : float
        center of the profile (usually the center of the maximum)
    RMS : float
        RMS of the distribution
    FWHM : float
        FWHM of the distribution
    fourSigma_RMS : float
        $4\times$ RMS
    fourSigma_FWHM : float
        Gaussian equivalent $4\sigma$ from FWHM of profile; i.e.
        $4\sigma_{FWHM} = 2/\sqrt(\ln 4) FWHM$
    full_bunch_length : float
        the length from the first zero of the profile function to the other
        zero
    integral : float
        the analytic integral based on the parameters

    Methods
    ----------
    profile(t, *args)
        Computes the profile at 't'. If no extra args are passed, it computes
        the profile using the object attributes (amplitdue, etc.). 
    distribution(J)
        Computes the distribution function for action J or Hamiltonian H
    phase_space()
        Computes the 2d phase_space in (dt, dE) coordinates
    spectrum(f)
        Computes the spectrum at frequency 'f'
    parameters()
        Returns all parameters describing the profile (e.g. amplitude, center)
    '''

    def __new__(cls, parameters, time_array=None, data_array=None, 
                num_args=None, scale_means=None, **kwargs):

        if parameters is not None and len(parameters) != num_args:
            raise ValueError(f'got {len(parameters)} parameters, '
                                + f'but need {num_args}')
        
        if time_array is not None:
            if type(time_array) not in [list, np.ndarray]:
                raise TypeError(f"time_array needs to be 'list' or 'np.array',"
                                +" not {type(time_array)}")
        
        if data_array is not None:
            if type(data_array) not in [list, np.ndarray]:
                raise TypeError(f"data_array needs to be 'list' or 'np.array',"
                                +" not {type(data_array)}")
        
        if time_array is None and data_array is None\
            or time_array is not None and data_array is not None:
            # create object
            return super().__new__(cls)            
        
        elif parameters is not None and data_array is None:
            # return profile at time_array
            time_array = np.array(time_array)
            
            if scale_means is None:
                scale_means = rcBLonDparams['distribution.scale_means']
            else:
                scale_means = kwargs['scale_means']
            
            # convert scale parameter to RMS
            parameters[2] = cls._computeBunchlengths(parameters,
                                                     scale_means)[0]

            return cls._profile(time_array, *parameters)
        else:
            raise RuntimeError("ups, didn't see that coming...")        

    def __init__(self, *args, **kwargs):

        self.amplitude = None
        self.center = None

    @property
    def RMS(self):
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    @property
    def FWHM(self):
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    @property
    def fourSigma_RMS(self):
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    @property
    def fourSigma_FWHM(self):
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    @property
    def full_bunch_length(self):
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    @property
    def integral(self):
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    def profile(self, t):
        r""" Computes the profile at t
        
        Every implementation should also accept *args as a second argument,
        to be compadible with the fitting package.
        """
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    def distribution(self):
        r""" Computes the distribution (e.g. in action)
        """
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    def phase_space(self):
        r""" Computes the longitudinal phase space (in dt,dE)
        """
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))

    def spectrum(self, f):
        r""" Returns the spectrum (Fourier transform of the profile)
        """
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))
        
    def get_parameters(self):
        r""" Returns all parameters describing the profile
        """
        raise RuntimeError(
            '%s not implemented' % (inspect.currentframe().f_code.co_name))
        

class Gaussian(_DistributionObject):
    r""" Gaussian profile function
    .. math::
        profile(t) = A\,\exp(-(t-t_c)^2/2\sigma^2) \,,
    with amplitude $A$, center $t_c$ and RMS bunch length $\sigma$.
    
    Parameters
    ----------
    parameters : list or ndarray
        [amplitude, center, scale]. How `scale` is interpreted depends on 
        `scale_means`.
    time_array : ndarray, optional
        Returns Gaussian profile evaluated at `time_array`. No `Gaussian` 
        object is created.
    data_array : ndarray, optional
        Create a `Gaussian` with parameters obtained by fitting to
        (`time_array`, `data_array`). If `parameters` are passed, they are
        used as initial guesses for the fitting method.
    scale_means: str, optional
        Controlls how `scale` is interpreted. Valid options are 'RMS',
        'FWHM', 'fourSigma_RMS', 'fourSigma_FWHM' ('full_bunch_length'
        is not valid for `Gaussian`)

    Attributes
    ----------
    amplitude : float
        maximum $A$ of the Gaussian profile
    center : float
        position $t_c$ of the the maximum
    RMS : float
        RMS of the distribution
    FWHM : float
        FWHM of the distribution
    fourSigma_RMS : float
        $4\times$ RMS
    fourSigma_FWHM : float
        Gaussian equivalent $4\sigma$ from FWHM of profile; i.e.
        $4\sigma_{FWHM} = 2/\sqrt(\ln 4) FWHM$
    full_bunch_length : np.inf
        infinity for Gaussian bunch
    integral : float
        $\sqrt{2\pi} A \sigma$, analytic integral of profile
    """

    def __new__(cls, parameters, time_array=None, data_array=None,
                scale_means=None, **kwargs):

        return super().__new__(cls, parameters, time_array=time_array,
                    data_array=data_array, scale_means=scale_means, num_args=3,
                    **kwargs)

    def __init__(self, parameters, time_array=None, data_array=None,
                 scale_means=None, **kwargs):

        if time_array is None and data_array is None:
            self.amplitude = parameters[0]
            self.center = parameters[1]
            scale = parameters[2]
        elif parameters is None: # fit Gaussian to time and data arrays
            fitPars = profile.gaussian_fit(time_array, data_array)
            self.amplitude, self.center, scale = fitPars
        else:
            fitPars = profile.gaussian_fit(time_array, data_array,
                  fitOpt=profile.FitOptions(fitInitialParameters=parameters))
            self.amplitude, self.center, scale = fitPars

        if scale_means is None:
            scale_means = rcBLonDparams['distribution.scale_means']
        
        if scale_means == 'RMS':
            self.RMS = scale
        elif scale_means == 'FWHM':
            self.FWHM = scale
        elif scale_means == 'fourSigma_RMS':
            self.fourSigma_RMS = scale
        elif scale_means == 'fourSigma_FWHM':
            self.fourSigma_FWHM = scale
        elif scale_means == 'full_bunch_length':
            raise ValueError("'full_bunch_length' argument no possible for "
                             + "Gaussian profile")

    def get_parameters(self, scale_means=None):
        if scale_means is None:
            scale_means = rcBLonDparams['distribution.scale_means']  
        
        if scale_means == 'RMS':
            scale = self.RMS
        elif scale_means == 'FWHM':
            scale = self.FWHM
        elif scale_means == 'fourSigma_RMS':
            scale = self.fourSigma_RMS
        elif scale_means == 'fourSigma_FWHM':
            scale = self.fourSigma_FWHM
        elif scale_means == 'full_bunch_length':
            raise ValueError("'full_bunch_length' argument no possible for "
                             + "Gaussian profile")

        return np.array([self.amplitude, self.center, scale])

    def _computeBunchlenghtsFromRMS(RMS):
        FWHM = 2*np.sqrt(np.log(4)) * RMS
        # return order is:
        # RMS, FWHM, fourSigma_RMS, fourSigma_FWHM, full_bunch_length
        return RMS, FWHM, 4*RMS, 2/np.sqrt(np.log(4)) * FWHM, np.inf

    def _computeBunchlengths(parameters, scale_means):
        """ Computes the various lengths for `parameters` of the distribution.
        """
        
        value = parameters[2]
        if scale_means == 'RMS':
            bls = Gaussian._computeBunchlenghtsFromRMS(value)
        elif scale_means == 'FWHM':
            bls = Gaussian._computeBunchlenghtsFromRMS(value / (2*np.sqrt(np.log(4))))
        elif scale_means == 'fourSigma_RMS':
            bls = Gaussian._computeBunchlenghtsFromRMS(value/4)
        elif scale_means == 'fourSigma_FWHM':
            bls = Gaussian._computeBunchlenghtsFromRMS(value/4)
        return bls

    @property
    def RMS(self):
        return self._RMS

    @RMS.setter
    def RMS(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self._RMS, self._FWHM, self._fourSigma_RMS, self._fourSigma_FWHM,\
            self._full_bunch_length = Gaussian._computeBunchlenghtsFromRMS(value)

    @property
    def FWHM(self):
        return self._FWHM

    @FWHM.setter
    def FWHM(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self.RMS = value / (2*np.sqrt(np.log(4)))  # updates all other parameters

    @property
    def fourSigma_RMS(self):
        return self._fourSigma_RMS

    @fourSigma_RMS.setter
    def fourSigma_RMS(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self.RMS = value / 4  # updates all other parameters

    @property
    def fourSigma_FWHM(self):
        return self._fourSigma_FWHM

    @fourSigma_FWHM.setter
    def fourSigma_FWHM(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self.RMS = value / 4  # updates all other parameters

    @property
    def full_bunch_length(self):
        return self._full_bunch_length

    @full_bunch_length.setter
    def full_bunch_length(self, value):
        self._full_bunch_length = np.inf

    @property
    def integral(self):
        return self.amplitude*self._RMS*np.sqrt(2*np.pi)

    def profile(self, x, *args, **kwargs):
        """ Computes the Gaussian profile at x
        """
        # profile needs to have this format to work with scipy fitfunctions

        if len(args) == 0:
            return Gaussian._profile(x, self.amplitude, self.center, self.RMS)
        else:
            amplitude = args[0]
            center = args[1]

            if 'scale_means' not in kwargs:
                scale_means = rcBLonDparams['distribution.scale_means']
            else:
                scale_means = kwargs['scale_means']
            
            # first return value is RMS
            RMS = Gaussian._computeBunchlengths(args, scale_means)[0]

            return Gaussian._profile(x, amplitude, center, RMS)

    def _profile(x, amplitude, center, RMS):
        return amplitude * np.exp(-0.5*(x-center)**2/RMS**2)

    def spectrum(self, f, **kwargs):
        """ Computes the Gaussian spectrum at frequency f
        """
        #TODO implement different factors of Fouriertransform

        return_value = np.sqrt(2*np.pi)*self.amplitude * self.RMS\
            * np.exp(-0.5*(2*np.pi*f*self.RMS)**2)

        if self.center != 0.0:  # assures real spectrum for symmetric profile
            return_value = return_value * np.exp(-2j*np.pi*self.center * f)

        return return_value


class BinomialAmplitudeN(_DistributionObject):

    r""" Binominal amplitude function
    .. math::
        profile(t) = A \, \left(1-(\frac{2(t-t_c)}{t_L})^2 \right)^{\mu+1/2}\,,
    with amplitude $A$, center $t_c$, exponent $\mu$ and full bunch length 
    $t_L$. It vanishes for $|t-t_c|>=t_L/2$.

    Parameters
    ----------
    parameters : list or ndarray
        [amplitude, center, scale, mu]. How `scale` is interpreted depends on 
        `scale_means`.
    time_array : ndarray, optional
        Returns Binominal Amplitude N profile evaluated at `time_array`. No 
        object is created.
    data_array : ndarray, optional
        Create a object with parameters obtained by fitting to
        (`time_array`, `data_array`). If `parameters` are passed, they are
        used as initial guesses for the fitting method.
    scale_means: str, optional
        Controlls how `scale` is interpreted. Valid options are 'RMS',
        'FWHM', 'fourSigma_RMS', 'fourSigma_FWHM' and 'full_bunch_length'.

    Attributes
    ----------
    amplitude : float
        maximum $A$ of the Binominal Amplitude N profile
    center : float
        position $t_c$ of the the maximum
    RMS : float
        RMS of the distribution
    FWHM : float
        FWHM of the distribution
    fourSigma_RMS : float
        $4\times$ RMS
    fourSigma_FWHM : float
        Gaussian equivalent $4\sigma$ from FWHM of profile; i.e.
        $4\sigma_{FWHM} = 2/\sqrt(\ln 4) FWHM$
    full_bunch_length : float
        range in which the profile is non-zero
    integral : float
        analytic integral of profile

    """

    def __new__(cls, parameters, time_array=None, data_array=None,
                scale_means=None, **kwargs):

        return super().__new__(cls, parameters, time_array=time_array,
                    data_array=data_array, scale_means=scale_means, num_args=4,
                    **kwargs)

    def __init__(self, parameters, time_array=None, data_array=None,
                 scale_means=None, **kwargs):
        if time_array is None and data_array is None:
            # amplitude, center, scale, mu
            self.amplitude = parameters[0]
            self.center = parameters[1]
            scale = parameters[2]
            self.mu = parameters[3]
        #TODO pass appropriate kwargs to fit function
        elif parameters is None: # fit binominalN to time and data arrays
            fitPars = profile.binomial_amplitudeN_fit(time_array, data_array)
            self.amplitude, self.center, scale, self.mu = fitPars
        else:
            fitPars = profile.binomial_amplitudeN_fit(time_array, data_array,
                  fitOpt=profile.FitOptions(fitInitialParameters=parameters))
            self.amplitude, self.center, scale, self.mu = fitPars
        
        if scale_means is None:
            scale_means = rcBLonDparams['distribution.scale_means']

        if scale_means == 'RMS':
            self.RMS = scale
        elif scale_means == 'FWHM':
            self.FWHM = scale
        elif scale_means == 'fourSigma_RMS':
            self.fourSigma_RMS = scale
        elif scale_means == 'fourSigma_FWHM':
            self.fourSigma_FWHM = scale
        elif scale_means == 'full_bunch_length':
            self.full_bunch_length = scale

    def get_parameters(self, scale_means=None):
        if scale_means is None:
            scale_means = rcBLonDparams['distribution.scale_means']  
        
        if scale_means == 'RMS':
            scale = self.RMS
        elif scale_means == 'FWHM':
            scale = self.FWHM
        elif scale_means == 'fourSigma_RMS':
            scale = self.fourSigma_RMS
        elif scale_means == 'fourSigma_FWHM':
            scale = self.fourSigma_FWHM
        elif scale_means == 'full_bunch_length':
            scale = self.full_bunch_length

        return np.array([self.amplitude, self.center, scale, self.mu])

#    def _computeBunchlenghtsFromRMS(self, RMS):
    def _computeBunchlenghtsFromRMS(RMS, mu):
#        RMS = parameters[2]
#        mu = parameters[3]
        full_bunch_length = np.sqrt(16+8*mu)*RMS
#        FWHM = full_bunch_length / np.sqrt(1-1/2**(1/(self.mu+0.5)))
        FWHM = np.sqrt(1-4**(-1/(1+2*mu))) * full_bunch_length
        # return order is:
        # RMS, FWHM, fourSigma_RMS, fourSigma_FWHM, full_bunch_length
        return RMS, FWHM, 4*RMS, 2/np.sqrt(np.log(4)) * FWHM, full_bunch_length

#    def _computeBunchlengths(self, value, scale_means):
    def _computeBunchlengths(parameters, scale_means):
#        _parameters = np.copy(parameters)
        mu = parameters[3]
        if scale_means == 'RMS':
            RMS = parameters[2]
#            bls = BinomialAmplitudeN._computeBunchlenghtsFromRMS(value)
        elif scale_means == 'FWHM':
            RMS = parameters[2] /  np.sqrt(16+8*mu)\
                / np.sqrt(1-0.5**(1/(mu+0.5)))
        elif scale_means == 'fourSigma_RMS':
            RMS = 0.25 * parameters[2]
#            bls = BinomialAmplitudeN._computeBunchlenghtsFromRMS(value/4)
        elif scale_means == 'fourSigma_FWHM':
            RMS = parameters[2] / np.sqrt(16+8*mu) / np.sqrt(1-0.5**(1/(mu+0.5))) \
                * 0.5*np.sqrt(np.log(4))
#            bls = BinomialAmplitudeN._computeBunchlenghtsFromRMS(value
#               / np.sqrt(16+8*self.mu) / np.sqrt(1-0.5**(1/(self.mu+0.5)))
#               * 0.5*np.sqrt(np.log(4)))
        elif scale_means == 'full_bunch_length':
            RMS = parameters[2] / np.sqrt(16+8*mu)
#            bls = BinomialAmplitudeN._computeBunchlenghtsFromRMS(value/np.sqrt(16+8*self.mu))
        bls = BinomialAmplitudeN._computeBunchlenghtsFromRMS(RMS, mu)
        return bls

    @property
    def RMS(self):
        return self._RMS

    @RMS.setter
    def RMS(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self._RMS, self._FWHM, self._fourSigma_RMS, self._fourSigma_FWHM,\
            self._full_bunch_length \
            = BinomialAmplitudeN._computeBunchlenghtsFromRMS(value, self.mu)

    @property
    def FWHM(self):
        return self._FWHM

    @FWHM.setter
    def FWHM(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self.RMS = value / np.sqrt(16+8*self.mu)\
            / np.sqrt(1-0.5**(1/(self.mu+0.5)))  # updates all other parameters

    @property
    def fourSigma_RMS(self):
        return self._fourSigma_RMS

    @fourSigma_RMS.setter
    def fourSigma_RMS(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self.RMS = value / 4  # updates all other parameters

    @property
    def fourSigma_FWHM(self):
        return self._fourSigma_FWHM

    @fourSigma_FWHM.setter
    def fourSigma_FWHM(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
         # updates all other parameters
        self.RMS =  value / np.sqrt(16+8*self.mu)\
            / np.sqrt(1-1/2**(1/(self.mu+0.5))) * 0.5*np.sqrt(np.log(4))

    @property
    def full_bunch_length(self):
        return self._full_bunch_length

    @full_bunch_length.setter
    def full_bunch_length(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self.RMS = value / np.sqrt(16+8*self.mu)

    @property
    def integral(self):
        return self.amplitude * 0.5*np.sqrt(np.pi) \
            * gamma_function(self.mu+1.5) * self.full_bunch_length \
            / gamma_function(self.mu+2) 

    def profile(self, x, *args, **kwargs):
        """ Computes the Binominal amplitude N profile at x
        """
        
        if len(args) == 0:
            return BinomialAmplitudeN._profile(x, self.amplitude, self.center,
                                               self.RMS, self.mu)
        else:
            amplitude = args[0]
            center = args[1]
            mu = args[3]

            if 'scale_means' not in kwargs:
                scale_means = rcBLonDparams['distribution.scale_means']
            else:
                scale_means = kwargs['scale_means']
            
            # first return value is RMS
            RMS = BinomialAmplitudeN._computeBunchlengths(args, scale_means)[0]

            return BinomialAmplitudeN._profile(x, amplitude, center, RMS, mu)

    def _profile(x, amplitude, center, RMS, mu):
        full_bunch_length = RMS * np.sqrt(16+8*mu)

        try:
            return_value = np.zeros(len(x))
            
            indexes = np.abs(x-center) <= full_bunch_length/2
            return_value[indexes] = amplitude\
                * (1- (2*(x[indexes]-center)/full_bunch_length)**2)**(mu+0.5)            
        except TypeError:
            if np.abs(x-center) <= full_bunch_length/2:
                return_value = amplitude\
                    * (1- (2*(x-center)/full_bunch_length)**2)**(mu+0.5)
            else:
                return_value = 0.0
        return return_value

    def spectrum(self, f):
        """ Returns the Binomial amplitude spectrum at frequency f
        """
        
        return_value = self.amplitude * self.full_bunch_length * np.sqrt(np.pi)\
            * gamma_function(self.mu+1.5) / (2*gamma_function(self.mu+2))\
            * hyp0f1(self.mu+2, -(0.5*np.pi*self.full_bunch_length*f)**2)
        
        if self.center != 0.0:
            return_value = return_value * np.exp(-2j*np.pi*self.center * f)
        
        return return_value


def generalizedGaussian(time, *fitParameters):
    '''
    Generalized gaussian line density
    '''

    amplitude = fitParameters[0]
    bunchcenter = fitParameters[1]
    alpha = abs(fitParameters[2])
    exponent = abs(fitParameters[3])

    lineDensityFunction = amplitude * np.exp(
        -(np.abs(time - bunchcenter)/(2*alpha))**exponent)

    return lineDensityFunction


def waterbag(time, *fitParameters):
    '''
    Waterbag distribution line density
    '''

    amplitude = fitParameters[0]
    bunchcenter = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchcenter) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchcenter) < bunchLength/2]-bunchcenter) /
            (bunchLength/2))**2)**0.5

    return lineDensityFunction


def parabolicLine(time, *fitParameters):
    '''
    Parabolic line density
    '''

    amplitude = fitParameters[0]
    bunchcenter = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchcenter) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchcenter) < bunchLength/2]-bunchcenter) /
            (bunchLength/2))**2)

    return lineDensityFunction


def parabolicAmplitude(time, *fitParameters):
    '''
    Parabolic in action line density
    '''

    amplitude = fitParameters[0]
    bunchcenter = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchcenter) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchcenter) < bunchLength/2]-bunchcenter) /
            (bunchLength/2))**2)**1.5

    return lineDensityFunction


def binomialAmplitude2(time, *fitParameters):
    '''
    Binomial exponent 2 in action line density
    '''

    amplitude = fitParameters[0]
    bunchcenter = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchcenter) < bunchLength/2] = \
        amplitude * (1-(
            (time[np.abs(time-bunchcenter) < bunchLength/2]-bunchcenter) /
            (bunchLength/2))**2)**2.

    return lineDensityFunction


#def binomialAmplitudeN(time, *fitParameters):
#    '''
#    Binomial exponent n in action line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchcenter = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#    exponent = abs(fitParameters[3])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchcenter) < bunchLength/2] = \
#        amplitude * (1-(
#            (time[np.abs(time-bunchcenter) < bunchLength/2]-bunchcenter) /
#            (bunchLength/2))**2)**exponent
#
#    return lineDensityFunction


def _binomial_full_to_rms(full_bunch_length, exponent):
    '''
    Returns the RMS bunch length from the full bunch length and exponent

    - TODO: To be included as @property in the Binomial/Parabolic distributions

    '''

    return full_bunch_length/(2*np.sqrt(3+2*exponent))


def _binomial_full_to_fwhm(full_bunch_length, exponent, level=0.5):
    '''
    Returns the FWHM from the full bunch length and exponent

    - TODO: To be included as @property in the Binomial/Parabolic distributions

    '''

    return full_bunch_length*np.sqrt(1-level**(1/exponent))


def _binomial_integral(amplitude, full_bunch_length, exponent):
    '''
    Returns the integrated profile

    - TODO: To be included as @property in the Binomial/Parabolic distributions

    '''

    return amplitude*full_bunch_length*np.sqrt(np.pi)*gamma(1.+exponent)/(
        2.*gamma(1.5+exponent))


def cosine(time, *fitParameters):
    '''
    * Cosine line density *
    '''

    amplitude = fitParameters[0]
    bunchcenter = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchcenter) < bunchLength/2] = \
        amplitude * np.cos(
            np.pi*(time[np.abs(time-bunchcenter) < bunchLength/2] -
                   bunchcenter) / bunchLength)

    return lineDensityFunction


def cosineSquared(time, *fitParameters):
    '''
    * Cosine squared line density *
    '''

    amplitude = fitParameters[0]
    bunchcenter = fitParameters[1]
    bunchLength = abs(fitParameters[2])

    lineDensityFunction = np.zeros(len(time))
    lineDensityFunction[np.abs(time-bunchcenter) < bunchLength/2] = \
        amplitude * np.cos(
            np.pi*(time[np.abs(time-bunchcenter) < bunchLength/2] -
                   bunchcenter) / bunchLength)**2.

    return lineDensityFunction
