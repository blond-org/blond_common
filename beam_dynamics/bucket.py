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
#import time

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
    

    def smooth_well_cubic(self, nPoints = None, reinterp=False):
    
        if reinterp or not hasattr(self, '_well_cubic_func'):
            self._well_cubic_func = interp.prep_interp_cubic(self.time_loaded, 
                                                             self.well_loaded)

        if nPoints is not None:
            self.time = np.linspace(self.time_loaded[0], self.time_loaded[-1], 
                                    nPoints)
            self.well = self._well_cubic_func(self.time)


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
    
    def outline_from_length(self, target_length):
        
        if target_length > self.length:
            raise excpt.BunchSizeError("target_length longer than bucket")
        
        else:
#            raise RuntimeError("Function not yet implemented")
            for w in self.well:
                useTime = self.time[self.well <= w]
                if useTime[-1] - useTime[0] < target_length:
                    break
            pts = np.where(self.well <= w)[0]
            leftPt = pts[0]
            rightPt = pts[-1]
            plt.plot(self.time, self.well, '.')
            plt.plot(useTime, self.well[self.well<=w])
            plt.axvline(np.pi-target_length/2)
            plt.axvline(np.pi+target_length/2)
            plt.plot(self.time[leftPt], self.well[leftPt], '.',
                     color='red')
            plt.plot(self.time[rightPt], self.well[rightPt], '.', 
                     color='red')
            plt.show()                
                

    
    
    def outline_from_dE(self, target_height):
        
        if target_height > self.half_height:
            raise excpt.BunchSizeError("target_height higher than bucket")

        else:
            raise RuntimeError("Function not yet implemented")
    

if __name__ == '__main__':

    inTime = np.linspace(0.5, 2*np.pi, 100)
    inWell = np.cos(inTime)
    inWell -= np.min(inWell)
    
    buck = Bucket(inTime, inWell, 3, 4, 5)
    buck.smooth_well_cubic(30)
    buck.calc_separatrix()
    buck.outline_from_length(4)
    
#    plt.plot(buck.separatrix[0], buck.separatrix[1])
#    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    