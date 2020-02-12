# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Base class for constructing buckets and dealing with single particle dynamics
:Authors: **Simon Albright**
"""

#General imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys

#BLonD_Common imports
if __name__ == "__main__":
    import blond_common.rf_functions.potential as pot
    import blond_common.maths.interpolation as interp
    import blond_common.devtools.exceptions as excpt
    import blond_common.devtools.assertions as assrt
else:
    from ..rf_functions import potential as pot
    from ..maths import interpolation as interp
    from ..devtools import exceptions as excpt
    from ..deftools import assertions as assrt

class Bucket:
    
    def __init__(self, time, well, beta, energy, eta):
        
        try:
            assrt.equal_array_lengths(time, well, 
                              msg = "time and well must have the same length",
                              exception = excpt.InputError)
        except TypeError:
            raise excpt.InputError("time and well must both be iterable")
        
        self.time_loaded = np.array(time, dtype=float)
        self.well_loaded = np.array(well, dtype=float)
        
        self.beta = beta
        self.energy = energy
        self.eta = eta
        
        self.time = self.time_loaded.copy()
        self.well = self.well_loaded.copy()
        
        self.calc_separatrix()
        self.basic_parameters()
    

    def smooth_well(self, nPoints = None, reinterp=False):
    
        if reinterp or not hasattr(self, '_well_smooth_func'):
            self._well_smooth_func = interp.prep_interp_cubic(self.time_loaded, 
                                                             self.well_loaded)

        if nPoints is not None:
            self.time = np.linspace(self.time_loaded[0], self.time_loaded[-1], 
                                    nPoints)
            self.well = self._well_smooth_func(self.time)


    def calc_separatrix(self):
        
        hamil = pot.potential_to_hamiltonian(self.time, self.well,
                                             self.beta, self.energy, 
                                             self.eta)

        self.upper_energy_bound = np.sqrt(hamil)
        
        sepTime = self.time.tolist() + self.time[::-1].tolist()
        sepEnergy = self.upper_energy_bound.tolist() \
                    + (-self.upper_energy_bound[::-1]).tolist()
        
        self.separatrix = np.array([sepTime, sepEnergy])
    
    
    def basic_parameters(self):
        
        self.half_height = np.max(self.separatrix[1])
        self.area = 2*np.trapz(self.upper_energy_bound, self.time)
        self.length = self.time[-1] - self.time[0]
        self.center = np.mean(self.time)
        
    ################################################
    ####Functions for calculating bunch outlines####
    ################################################
    
    
    def _outline_minim_func(self, energy, *args):
        
        if len(args) == 2:
            leftPts, rightPts = args
        elif len(args) == 3:
            leftPts, rightPts, nPts = args
            
            
    def _interp_time_from_potential(self, potential, nPts = 0):
        
        if potential > np.max(self.well):
            raise excpt.InputError("Target potential above maximum potential")
        
        if potential < 0:
            raise excpt.InputError("Target potential must be positive")
        
        pts = np.where(self.well <= potential)[0]
        leftPt = pts[0]
        rightPt = pts[-1]
        
        lTime = np.interp(potential, self.well[leftPt-2:leftPt+2][::-1], 
                          self.time[leftPt-2:leftPt+2][::-1])
        rTime = np.interp(potential, self.well[rightPt-2:rightPt+2],
                          self.time[rightPt-2:rightPt+2])

        if nPts == 0:
            return lTime, rTime
        else:
            return np.linspace(lTime, rTime, nPts)
        
    
    def outline_from_length(self, target_length, nPts=1000):
        
        self.smooth_well()
        
        if target_length > self.length:
            raise excpt.BunchSizeError("target_length longer than bucket")
        
        def len_func(potential):

            try:
                lTime, rTime = self._interp_time_from_potential(potential[0])
            except excpt.InputError:
                return self.time[-1] - self.time[0]

            return np.abs(target_length - (rTime - lTime))

        result = opt.minimize(len_func, np.max(self.well)/2, 
                              method='Nelder-Mead')
        interpTime = self._interp_time_from_potential(result['x'][0], nPts)
        interpWell = self._well_smooth_func(interpTime)
        interpWell[interpWell>interpWell[0]] = interpWell[0]
        
        energyContour = np.sqrt(pot.potential_to_hamiltonian(interpTime, 
                                                             interpWell, 
                                                             self.beta, 
                                                             self.energy,
                                                             self.eta))

        outlineTime = interpTime.tolist() + interpTime[::-1].tolist()
        outlineEnergy = energyContour.tolist() \
                        + (-energyContour[::-1]).tolist()
    
        return np.array([outlineTime, outlineEnergy])


    def outline_from_dE(self, target_height):
        
        self.smooth_well()
        
        if target_height > self.half_height:
            raise excpt.BunchSizeError("target_height higher than bucket")

        potential = target_height**2*self.eta/(2*self.beta**2*self.energy)
        
        interpTime = self._interp_time_from_potential(potential, 1000)
        interpWell = self._well_smooth_func(interpTime)
        interpWell[interpWell>interpWell[0]] = interpWell[0]
        
        energyContour = np.sqrt(pot.potential_to_hamiltonian(interpTime, 
                                                             interpWell, 
                                                             self.beta, 
                                                             self.energy,
                                                             self.eta))
        
        outlineTime = interpTime.tolist() + interpTime[::-1].tolist()
        outlineEnergy = energyContour.tolist() \
                        + (-energyContour[::-1]).tolist()
    
        
        return np.array([outlineTime, outlineEnergy])
    
    
    def outline_from_emittace(self, target_emittance, nPts = 1000):

        self.smooth_well()

        if target_emittance > self.area:
            raise excpt.BunchSizeError("target_emittance exceeds bucket area")
        
        def emit_func(potential, *args):

            nPts = args[0]
            try:
                interpTime = self._interp_time_from_potential(potential[0], nPts)
            except excpt.InputError:
                return self.area
            
            interpWell = self._well_smooth_func(interpTime)
            interpWell[interpWell>interpWell[0]] = interpWell[0]
            
            energyContour = np.sqrt(pot.potential_to_hamiltonian(interpTime, 
                                                             interpWell, 
                                                             self.beta, 
                                                             self.energy,
                                                             self.eta))

            emittance = 2*np.trapz(energyContour, interpTime)
            
            return np.abs(target_emittance - emittance)
    
        result = opt.minimize(emit_func, np.max(self.well)/2, 
                              method='Nelder-Mead', args=(nPts,))
        
        interpTime = self._interp_time_from_potential(result['x'][0], nPts)
        interpWell = self._well_smooth_func(interpTime)
        interpWell[interpWell>interpWell[0]] = interpWell[0]
        
        energyContour = np.sqrt(pot.potential_to_hamiltonian(interpTime, 
                                                             interpWell, 
                                                             self.beta, 
                                                             self.energy,
                                                             self.eta))

        outlineTime = interpTime.tolist() + interpTime[::-1].tolist()
        outlineEnergy = energyContour.tolist() \
                        + (-energyContour[::-1]).tolist()
    
        return np.array([outlineTime, outlineEnergy])    



if __name__ == '__main__':

    inTime = np.linspace(0, 2*np.pi, 100)
    inWell = np.cos(inTime)
    inWell += np.cos(inTime*2)
    inWell -= np.min(inWell)
    
    buck = Bucket(inTime, inWell, 3, 4, 5)
    buck.smooth_well(1000)
    buck.calc_separatrix()
    targetEmit = 30
    bunch = buck.outline_from_emittace(targetEmit)
#    targetLength = 5
#    bunch = buck.outline_from_length(targetLength)
#    targetHeight = 3
#    bunch = buck.outline_from_dE(targetHeight)
    plt.plot(buck.separatrix[0], buck.separatrix[1])
#    plt.axhline(targetHeight)
#    plt.axvline(np.pi - targetLength/2)
#    plt.axvline(np.pi + targetLength/2)
    plt.plot(bunch[0], bunch[1])
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    