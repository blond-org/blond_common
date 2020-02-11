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

#BLonD_Common imports
if __name__ == "__main__":
    import blond_common.rf_functions.potential as pot
    import blond_common.maths.interpolation as interp
    import blond_common.devtools.exceptions as excpt
    import blond_common.devtools.assertions as assrt
else:
    from ..rf_functions import potential as pot
    from ..devtools import exceptions as excpt
    from ..deftools import assertions as assrt

#import scipy.interpolate as interp
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
    

    def smooth_well_cubic(self, nPoints, reinterp=False):
    
        self.time = np.linspace(self.time_loaded[0], self.time_loaded[-1], 
                                nPoints)

        if reinterp or not hasattr(self, 'well_cubic_func'):
            self.well_cubic_func = interp.prep_interp_cubic(self.time_loaded, 
                                                            self.well_loaded)

        self.well = self.well_cubic_func(self.time)

    
    def calc_separatrix(self):
        
        pot.potential_to_hamiltonian(self.time_loaded, self.well_loaded,
                                     self.beta, self.energy, self.eta)
        

if __name__ == '__main__':

    inTime = np.linspace(0, 10, 100)
    inWell = np.cos(inTime)
#    plt.plot(inTime, inWell)
#    plt.show()
    
#    interp.CubicSpline(inTime, inWell)
    
    buck = Bucket(inTime, inWell, 3, 4, 5)
    plt.plot(buck.time, buck.well, '.')
    buck.smooth_well_cubic(50)
    plt.plot(buck.time, buck.well, '.')
    buck.smooth_well_cubic(30)
    plt.plot(buck.time, buck.well, '.')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    