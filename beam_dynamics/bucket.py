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

#BLonD_Common imports

if __name__ == "__main__":
    import blond_common.rf_functions.potential as pot
else:
    from ..rf_functions import potential as pot


class Bucket:
    
    def __init__(self, time, well, beta, energy, eta):
        
        
        self.time_loaded = time
        self.well_loaded = well
        
        self.beta = beta
        self.energy = energy
        self.eta = eta
    
    
    def calc_separatrix(self):
        
        pot.potential_to_hamiltonian(self.time_loaded, self.well_loaded,
                                     self.beta, self.energy, self.eta)
        
        
if __name__ == '__main__':
    Bucket(1, 2, 3, 4, 5)