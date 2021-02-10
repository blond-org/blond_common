# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Base class for computing induced voltage
:Authors: **Simon Albright**
"""

#General imports
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.constants as cont


#BLonD_Common imports
if __name__ != '__main__':
    from ..impedances import impedance_sources as impSource
    from ..beam import profile as prof
    from ...devtools import exceptions as exceptions
else:
    import blond_common.interfaces.impedances.impedance_sources as impSource
    import blond_common.interfaces.beam.analytic_distribution as analDist
    import blond_common.interfaces.beam.profile as prof
    import blond_common.devtools.exceptions as exceptions


class InducedVoltage:
    
    def __init__(self, impedance_list = [], wake_list = [], 
                 inductive_list = [], var_impedance_list = [],
                 var_wake_list = [], var_inductive_list = []):
        
        self.impedances_loaded = impedance_list
        self.wakes_loaded = wake_list
        self.inductive_loaded = inductive_list

        self.var_impedances_loaded = var_impedance_list
        self.var_wakes_loaded = var_wake_list
        self.var_inductive_loaded = var_inductive_list
        
        self.interp_frequency_array = None
        self.interp_time_array = None


    def sum_impedance_sources(self, f_rev = None, sample = None):
        
        self._induced_calcs = []
        
        impedances = [i for i in self.impedances_loaded]

        for i in self.var_impedances_loaded:
            i.update(f_rev)

        impedances += [i for i in self.var_impedances_loaded]
        
        if len(impedances) > 0:
            try:
                self.total_impedance = np.zeros(len(self.interp_frequency_array), \
                                                dtype='complex')
            except TypeError as e:
                if str(e) == "object of type 'NoneType' has no len()":
                    raise exceptions.MissingParameterError(
                        "interp_frequency_array has not been correctly defined")
                else:
                    raise

            for imp in impedances:
                imp.imped_calc(self.interp_frequency_array)
                self.total_impedance += imp.impedance
            self._induced_calcs.append(self._calc_induced_freq)


        if len(self.wakes_loaded) > 0:
            try:
                self.total_wake = np.zeros(len(self.interp_time_array))
            except TypeError:
                raise exceptions.MissingParameterError(
                        "interp_time_array has not been correctly defined")

            for wake in self.wakes_loaded:

                wake.wake_calc(self.interp_time_array)
                self.total_wake += wake.wake
            self._induced_calcs.append(self._calc_induced_time)

        inductives = [i for i in self.inductive_loaded]

        for i in self.var_inductive_loaded:
            i.update(f_rev)

        inductives += [i for i in self.var_inductive_loaded]

        if len(inductives) > 0:
            self.total_inductive = 0

            for induct in inductives:

                induct.induct_calc(f_rev)
                self.total_inductive += induct.inductive

            self._induced_calcs.append(self._calc_induced_inductive)


    def calc_induced_by_source(self, spectrum = None, profile = None, 
                               normalisation = 1, f_rev = None, sample = None):

        self.imped_induced = []
        self.var_imped_induced = []
        
        impedances = [i for i in self.impedances_loaded]
        varImpedances = [i for i in self.var_impedances_loaded]
        
        for i in impedances:
            # i.update(f_rev)
            i.imped_calc(self.interp_frequency_array)
            self.imped_induced.append(calc_induced_freq(self.beam_spectrum, 
                                                   i.impedance*normalisation))
        
        for i in varImpedances:
            i.update(f_rev)
            i.imped_calc(self.interp_frequency_array)
            self.var_imped_induced.append(calc_induced_freq(self.beam_spectrum, 
                                                   i.impedance*normalisation))
        
        
        self.wake_induced = []
        
        for w in self.wakes_loaded:
            if isinstance(w, impSource.VariableImpedance):
                w = w.update(sample)
                
            w.wake_calc(self.interp_time_array)
            self.wake_induced.append(calc_induced_time(self.beam_profile, 
                                                       w.wake))
        
        
        self.inductive_induced = []
        self.var_inductive_induced = []
        
        inductives = [i for i in self.inductive_loaded]
        varInductives = [i for i in self.var_inductive_loaded]
        
        for i in inductives:
            i.induct_calc(f_rev)
            self.inductive_induced.append(calc_induced_inductive(
                                                self.beam_derivative, 
                                                 i.inductive*normalisation))
        
        for i in varInductives:
            i.update(f_rev)
            i.induct_calc(f_rev)
            self.var_inductive_induced.append(calc_induced_inductive(self.beam_derivative, 
                                                   i.inductive*normalisation))


    def calc_induced(self, normalisation=1):
        
        self.VInduced = 0
        for calc in self._induced_calcs:
            self.VInduced += calc(normalisation)


    def _calc_induced_time(self, normalisation):
        return calc_induced_time(self.beam_profile, 
                                 self.total_wake*normalisation)
    
    def _calc_induced_freq(self, normalisation):
        return calc_induced_freq(self.beam_spectrum, 
                                 self.total_impedance*normalisation)
    
    def _calc_induced_inductive(self, normalisation):
        return calc_induced_inductive(self.beam_derivative,
                                      self.total_inductive*normalisation)
    
    
    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, value):
        
        if type(value) is prof.Profile:
            self.beam_profile = value.profile_array
            self.beam_spectrum = value.beam_spectrum
            self.beam_derivative = value.beam_derivative
            self.interp_frequency_array = value.beam_spectrum_freq
            self.interp_time_array = value.time_array
            self._profile = value

        elif hasattr(value, '__iter__'):
            if len(value) == 2:
                self.profile = prof.Profile(value[0], value[1])

                
def calc_induced_freq(spectrum, impedance):
    return -fft.irfft(spectrum*impedance)

def calc_induced_time(profile, wake):
    return np.convolve(profile, wake)

def calc_induced_inductive(derivative, inductive):
    return -derivative*inductive
    # return derivative*inductive
        
        


#%%

if __name__ == '__main__':
    
    pos = 0.5E-6
    sigma = 0.025E-6
    ampl = 1
    
    rShunt = 1E3
    fRes = 15E6
    Q = 30
    
    reson1 = impSource.Resonators(rShunt, fRes, Q)
    timeRange = np.linspace(0, 20E-6, 10000)
    profile = analDist.gaussian(timeRange, ampl, pos, sigma)
    
    profile/= np.sum(profile)
    profile *= 1E13
    
    plt.plot(timeRange, profile)
    plt.show()
    
    profile = prof.Profile(timeRange, profile)
    
    profile.beam_spectrum_freq_generation()
    profile.beam_spectrum_generation()
    
    plt.semilogx(profile.beam_spectrum_freq, profile.beam_spectrum)
    plt.show()
    
    reson1.imped_calc(profile.beam_spectrum_freq)
    
    plt.semilogx(reson1.frequency_array, reson1.impedance.real)
    plt.semilogx(reson1.frequency_array, reson1.impedance.imag)
    plt.show()
    
    VInd = calc_induced_freq(cont.e*profile.beam_spectrum, 
                             reson1.impedance/(timeRange[1] - timeRange[0]))
    plt.plot(profile.time_array, VInd.real)
    plt.show()
