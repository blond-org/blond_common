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
import scipy.special as special_fun

#TODO this needs to be cleaned up! This is to run the unittests
try:
    from ... devtools.BLonD_Rc import rcBLonDparams
    from ... fitting import profile_fitting
except ImportError:
    import sys
    sys.path.append('../../../')
    from devtools.BLonD_Rc import rcBLonDparams
#    from fitting import profile_fitting
    
def check_greater_zero(value, func_name):
    if value <= 0.0:
        raise ValueError(f"{func_name} needs to be positive")
    
class _DistributionObject(object):
    r'''Base class for all analytic distributions
    
    This base class is not to be used directly. Instead, all analytic
    distributions should derive from it.
    
    Attributes
    ----------
    amplitude : float
        amplitude of the profile
    position : float
        position of the profile (usually the position of the maximum)
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
        the length from zero of the profile function to the other
    store_data : bool
        if False, calls to functions like 'profile' return the value, otherwise
        an attribute is created
        
    Methods
    ----------
    profile(t, *args)
        Computes the profile at 't'. If no extra args are passed, it computes 
        the profile using the object attributes (amplitdue, etc.). The
        store_data keyword if the profile is returned or stored as a class
        attribute.
    distribution(J)
        Computes the distribution function for action J or Hamiltonian H
    phase_space()
        Computes the 2d phase_space in (dt, dE) coordinates
    spectrum(f)
        Computes the spectrum at frequency 'f'
    '''
    
    def __init__(self, store_data=None, **kwargs):
        self.amplitude = None
        self.position = None
        
        if store_data is None:
            self.store_data = rcBLonDparams['distribution.store_data']
        else:
            self.store_data = store_data

    @property
    def store_data(self):
        return self._store_data
    @store_data.setter
    def store_data(self, value):
        self._store_data = value
        if value==False:
            self._return_function = self._return_value
        else:
            self._return_function = self._store_value
    
    def _return_value(self, arg):
        return arg
    
    def _store_value(self, value):
        setattr(self, 'computed_'+inspect.currentframe().f_back.f_code.co_name,
                value)
        
    @property
    def RMS(self):
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')

    @property
    def FWHM(self):
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
    
    @property
    def fourSigma_RMS(self):
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
    
    @property
    def fourSigma_FWHM(self):
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
                
    @property
    def full_bunch_length(self):
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
    
    def profile(self):
        r""" Computes the profile (e.g. in time)
        """
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
    
    def distribution(self):
        r""" Computes the distribution (e.g. in action)
        """
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
    
    def phase_space(self):
        r""" Computes the longitudinal phase space (in dt,dE)
        """
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
    
    def spectrum(self):
        r""" Returns the spectrum (Fourier transform of the profile)
        """
        raise RuntimeError(f'{inspect.currentframe().f_code.co_name} not implemented')
     
class Gaussian(_DistributionObject):
    
#    def __init__(self, amplitude, position, scale, 
#                 time_array=None, y_array=None,
    def __init__(self, *args, scale_means=None,
                 store_data=None, **kwargs):
        r""" Gaussian profile function
        .. math::
            profile(t) = A\,\exp(-(t-t_0)^2/2\sigma^2) \,
        
        Parameters
        ----------
        args : 
            If args has the three parameters amplitude, position, scale a
            distribution object is created. If passed as
            amplitude, position, scale, time the profile is computed at time
            with these parameters. If passed as time, y_data a Gaussian fit
            to these data is performed.
        scale_means: str
            controlls how 'scale' is interpreted. Valid options are 'RMS',
            'FWHM', 'fourSigma_RMS', 'fourSigma_FWHM' (and 'full_bunch_length'
            for other distributions)
        
        Attributes
        ----------
        amplitude : float
            maximum $A$ of the Gaussian profile
        position : float
            position $t_0$ of the the maximum
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
        """
        _DistributionObject.__init__(self, store_data=store_data)
        
#        self.amplitude = amplitude  # other call signature
#        self.position = position  # other call signature
        
        if len(args) == 3:  # amplitude, position, scale
            self.amplitude = args[0]
            self.position = args[1]
            scale = args[2]
        elif len(args) == 4:  # amplitude, position, scale; profile computed below
            self.amplitude = args[0]
            self.position = args[1]
            scale = args[2]
        elif len(args) == 2:  # fit Gaussian to time, y_data
            fitPars = profile_fitting.gaussianFit(args[0], args[1])
            self.amplitude, self.position, scale = fitPars
        else:
            raise ValueError("invalid number of arguments")
        
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
            raise ValueError("'full_bunch_length' argument no possible for"
                             +" Gaussian")
            
        if len(args) == 4:
            # store_data needs to be True, since __init__ needs to return None
            #TODO: maybe raise a warning, since this overwrites 'store_data'
            self.profile(args[3], store_data=True)
    
    def _computeBunchlenghtsFromRMS(self,RMS):
        FWHM = 2*np.sqrt(np.log(4)) * RMS
        # return order is RMS, FWHM, fourSigma_RMS, fourSigma_FWHM, full_bunch_length
        return RMS, FWHM, 4*RMS, 2/np.sqrt(np.log(4)) * FWHM, np.inf
    
    
    def _computeBunchlengths(self, value, scale_means):
        if scale_means == 'RMS':
            bls = self._computeBunchlenghtsFromRMS(value)
        elif scale_means == 'FWHM':
            bls = self._computeBunchlenghtsFromRMS(value / (2*np.sqrt(np.log(4))))
        elif scale_means == 'fourSigma_RMS':
            bls = self._computeBunchlenghtsFromRMS(value/4)
        elif scale_means == 'fourSigma_FWHM':
            bls = self._computeBunchlenghtsFromRMS(value/4)
        return bls
            
    @property
    def RMS(self):
        return self._RMS
    @RMS.setter
    def RMS(self, value):
        check_greater_zero(value, inspect.currentframe().f_code.co_name)
        self._RMS, self._FWHM, self._fourSigma_RMS, self._fourSigma_FWHM,\
        self._full_bunch_length = self._computeBunchlenghtsFromRMS(value)
        
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
    
    def profile(self, x, *args, **kwargs):
        """ Computes the Gaussian profile at x
        """
        
        if 'store_data' in kwargs:
            self.store_data = kwargs['store_data']
        
        if len(args) == 0:
            return self._return_function(
                    self._profile(x, self.amplitude, self.position, self.RMS))
        else:
            amplitude = args[0]
            position = args[1]
            
            if 'scale_means' not in kwargs:
                scale_means = rcBLonDparams['distribution.scale_means']
            else:
                scale_means = kwargs['scale_means']

            RMS = self._computeBunchlengths(args[2], scale_means)[0]  # first return value is RMS
            
            return self._return_function(
                    self._profile(x, amplitude, position, RMS))
    
    
    def _profile(self, x, amplitude, position, RMS):
        return amplitude * np.exp(-0.5*(x-position)**2/RMS**2)
    
    
    def spectrum(self, f, **kwargs):
        """ Computes the Gaussian spectrum at frequency f
        """
        #TODO implement different factors of Fouriertransform
        
        if 'store_data' in kwargs:
            self.store_data = kwargs['store_data']
        
        return_value = np.sqrt(2*np.pi)*self.amplitude * self.RMS\
            * np.exp(-0.5*(2*np.pi*f*self.RMS)**2)
        
        if self.position != 0.0:  # assures real spectrum for symmetric profile
            return_value = return_value * np.exp(-2j*np.pi*self.position * f)
        
        return self._return_function(return_value)



#class BinomialAmplitudeN(_DistributionObject):
#    
#    def __init__(self, amplitude, position, scale, mu, scale_means=None, **kwargs):
#        r"""
#        amplitude, center, scale, others
#        scale_means='RMS', 'FWHM', ...
#        """
#        _DistributionObject.__init__(self)
#                
#        self.amplitude = amplitude
#        self.position = position
#        self.mu = mu
#        
#        if scale_means is None:
#            scale_means = rcBLonDparams['distribution.scale_means']
#
#        if scale_means == 'RMS':
#            self.RMS = scale
#        elif scale_means == 'FWHM':
#            self.FWHM = scale
#        elif scale_means == 'fourSigma_RMS':
#            self.fourSigma_RMS = scale
#        elif scale_means == 'fourSigma_FWHM':
#            self.fourSigma_FWHM = scale
#        elif scale_means == 'full_bunch_length':
#            self.full_bunch_length = scale
#
#    @property
#    def full_bunch_length(self):
#        return self._full_bunch_length
#    @full_bunch_length.setter
#    def full_bunch_length(self, value):
#        check_greater_zero(value, inspect.currentframe().f_code.co_name)
#        self._full_bunch_length = value
#        self._RMS = value / np.sqrt(16+8*self.mu)
#        self._FWHM = value * np.sqrt(1-1/2**(1/(self.mu+0.5)))
#        self._fourSigma_RMS = value / np.sqrt(1+0.5*self.mu)
#        self._fourSigma_FWHM = 2/np.sqrt(np.log(4)) * self._FWHM
#        
#    @property
#    def RMS(self):
#        return self._RMS
#    @RMS.setter
#    def RMS(self, value):
#        check_greater_zero(value, inspect.currentframe().f_code.co_name)
#        self.full_bunch_length = np.sqrt(16+8*self.mu) * value   # updates all other parameters
#
#    @property
#    def FWHM(self):
#        return self._FWHM
#    @FWHM.setter
#    def FWHM(self, value):
#        check_greater_zero(value, inspect.currentframe().f_code.co_name)
#        self.full_bunch_length = value / np.sqrt(1-1/2**(1/(self.mu+0.5)))  # updates all other parameters
#
#    @property
#    def fourSigma_RMS(self):
#        return self._fourSigma_RMS
#    @fourSigma_RMS.setter
#    def fourSigma_RMS(self, value):
#        check_greater_zero(value, inspect.currentframe().f_code.co_name)
#        self.full_bunch_length = value * np.sqrt(1+0.5*self.mu)  # updates all other parameters
#
#    @property
#    def fourSigma_FWHM(self):
#        return self._fourSigma_FWHM
#    @fourSigma_FWHM.setter
#    def fourSigma_FWHM(self, value):
#        check_greater_zero(value, inspect.currentframe().f_code.co_name)
#        self.FWHM = value * 0.5*np.sqrt(np.log(4))  # updates all other parameters
#
#    def profile(self, x):
#        """ Returns the Binomial amplitude profile at x
#        """
#        
#        try:
#            return_value = np.zeros(len(x))
#            
#            indexes = np.abs(x-self.position) <= self.full_bunch_length/2
#            return_value[indexes] = self.amplitude\
#                * (1- (2*(x[indexes]-self.position)/self.full_bunch_length)**2)**(self.mu+0.5)            
#        except:
#            if np.abs(x-self.position) <= self.full_bunch_length/2:
#                return_value = self.amplitude\
#                    * (1- (2*(x-self.position)/self.full_bunch_length)**2)**(self.mu+0.5)
#            else:
#                return_value = 0.0
#        return return_value
#
#    def spectrum(self, f):
#        """ Returns the Binomial amplitude spectrum at frequency f
#        """
#        
#        return_value = self.amplitude * self.full_bunch_length * np.sqrt(np.pi)\
#            * special_fun.gamma(self.mu+1.5) / (2*special_fun.gamma(self.mu+2))\
#            * special_fun.hyp0f1(self.mu+2, -(0.5*np.pi*self.full_bunch_length*f)**2)
#        
#        if self.position != 0.0:
#            return_value = return_value * np.exp(-2j*np.pi*self.position * f)
#        
#        return return_value
#

#from matplotlib import pyplot as plt
#
#
#binAmpNobj = BinomialAmplitudeN(1,0,1,1.5)
#
#x = np.linspace(-binAmpNobj.full_bunch_length, binAmpNobj.full_bunch_length)
#y = binAmpNobj.profile(x)
#
#plt.figure('binom profile', clear=True)
#plt.grid()
#plt.plot(x, y)
#
#dx = x[1] - x[0]
#freqs = np.linspace(-1/binAmpNobj.RMS,1/binAmpNobj.RMS, num=len(x))
#Ydft = np.zeros(len(freqs), dtype=complex)
#for it, f in enumerate(freqs):
#    Ydft[it] = np.trapz(y*np.exp(-2j*np.pi*f*x), dx=dx)
#    
#Y = binAmpNobj.spectrum(freqs)
#
#plt.figure('binom spectrum', clear=True)
#plt.grid()
#plt.plot(freqs, Ydft.real)
#plt.plot(freqs, Y.real,'--')
#plt.plot(freqs, Ydft.imag)
#plt.plot(freqs, Y.imag, '--')

#tmpObj = _DistributionObject()
##gaussObj = Gaussian(1, 0.4, 1, scale_means='fourSigma_FWHM')
#gaussObj = Gaussian(1/np.sqrt(2*np.pi), 0.4, 1, scale_means='fourSigma_FWHM')
#
#x = np.linspace(-5*gaussObj.RMS, 5*gaussObj.RMS, num=200)
#y = gaussObj.profile(x)


#plt.figure('gauss profile', clear=True)
#plt.grid()
#plt.plot(x, y)
#plt.plot(gaussObj.position - gaussObj.FWHM/2,
#         gaussObj.profile(gaussObj.position - gaussObj.FWHM/2), 'ro')
#plt.plot(gaussObj.position + gaussObj.FWHM/2,
#         gaussObj.profile(gaussObj.position + gaussObj.FWHM/2), 'ro')
#
#print(gaussObj.RMS, gaussObj.FWHM, gaussObj.fourSigma_RMS, gaussObj.fourSigma_FWHM)
#
#dx = x[1] - x[0]
#freqs = np.linspace(-1/gaussObj.RMS,1/gaussObj.RMS, num=len(x))
#Ydft = np.zeros(len(freqs), dtype=complex)
#for it, f in enumerate(freqs):
#    Ydft[it] = np.trapz(y*np.exp(-2j*np.pi*f*x), x=x, dx=dx)
#    
#Y = gaussObj.spectrum(freqs)
#
#plt.figure('spectrum', clear=True)
#plt.grid()
#plt.plot(freqs, Ydft.real)
#plt.plot(freqs, Y.real,'--')
#plt.plot(freqs, Ydft.imag)
#plt.plot(freqs, Y.imag, '--')


#def gaussian(time, *fitParameters):
#    '''
#    Gaussian line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    sigma = abs(fitParameters[2])
#
#    lineDensityFunction = amplitude * np.exp(
#        -(time-bunchPosition)**2/(2*sigma**2))
#
#    return lineDensityFunction


#def generalizedGaussian(time, *fitParameters):
#    '''
#    Generalized gaussian line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    alpha = abs(fitParameters[2])
#    exponent = abs(fitParameters[3])
#
#    lineDensityFunction = amplitude * np.exp(
#        -(np.abs(time - bunchPosition)/(2*alpha))**exponent)
#
#    return lineDensityFunction


#def waterbag(time, *fitParameters):
#    '''
#    Waterbag distribution line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
#        amplitude * (1-(
#            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
#            (bunchLength/2))**2)**0.5
#
#    return lineDensityFunction


#def parabolicLine(time, *fitParameters):
#    '''
#    Parabolic line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
#        amplitude * (1-(
#            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
#            (bunchLength/2))**2)
#
#    return lineDensityFunction


#def parabolicAmplitude(time, *fitParameters):
#    '''
#    Parabolic in action line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
#        amplitude * (1-(
#            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
#            (bunchLength/2))**2)**1.5
#
#    return lineDensityFunction


#def binomialAmplitude2(time, *fitParameters):
#    '''
#    Binomial exponent 2 in action line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
#        amplitude * (1-(
#            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
#            (bunchLength/2))**2)**2.
#
#    return lineDensityFunction


#def binomialAmplitudeN(time, *fitParameters):
#    '''
#    Binomial exponent n in action line density
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#    exponent = abs(fitParameters[3])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
#        amplitude * (1-(
#            (time[np.abs(time-bunchPosition) < bunchLength/2]-bunchPosition) /
#            (bunchLength/2))**2)**exponent
#
#    return lineDensityFunction

#def cosine(time, *fitParameters):
#    '''
#    * Cosine line density *
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
#        amplitude * np.cos(
#            np.pi*(time[np.abs(time-bunchPosition) < bunchLength/2] -
#                   bunchPosition) / bunchLength)
#
#    return lineDensityFunction


#def cosineSquared(time, *fitParameters):
#    '''
#    * Cosine squared line density *
#    '''
#
#    amplitude = fitParameters[0]
#    bunchPosition = fitParameters[1]
#    bunchLength = abs(fitParameters[2])
#
#    lineDensityFunction = np.zeros(len(time))
#    lineDensityFunction[np.abs(time-bunchPosition) < bunchLength/2] = \
#        amplitude * np.cos(
#            np.pi*(time[np.abs(time-bunchPosition) < bunchLength/2] -
#                   bunchPosition) / bunchLength)**2.
#
#    return lineDensityFunction
