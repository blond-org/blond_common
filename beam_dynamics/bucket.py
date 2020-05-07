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
import scipy.interpolate as spInterp
import itertools as itl
import warnings
import functools

#BLonD_Common imports
from ..rf_functions import potential as pot
from ..maths import interpolation as interp
from ..maths import calculus as calc
from ..devtools import exceptions as excpt
from ..devtools import assertions as assrt
from ..devtools import decorators as deco
from ..interfaces.beam import matched_distribution as matchDist
from ..maths import calculus as calc



class Bucket:
    
    def __init__(self, time, well, beta, energy, eta, isSub = False):
        
        self.beta = beta
        self.energy = energy
        self.eta = eta
        self.isSub = isSub
        if self.isSub:
            self.time_loaded = time
            self.well_loaded = well
            self.time = self.time_loaded.copy()
            self.well = self.well_loaded.copy()
            return
        
        try:
            assrt.equal_array_lengths(time, well, 
                              msg = "time and well must have the same length",
                              exception = excpt.InputError)
        except TypeError:
            raise excpt.InputError("time and well must both be iterable")
        
        orderedTime, orderedWell = pot.sort_potential_wells(time, well)
        
        self.time_loaded = np.array(orderedTime[0], dtype=float)
        self.well_loaded = np.array(orderedWell[0], dtype=float)
        
        self.time = self.time_loaded.copy()
        self.well = self.well_loaded.copy()
        
        self.calc_separatrix()
        self.basic_parameters()
        
        self.inner_times = orderedTime[1:]
        self.inner_wells = orderedWell[1:]
        
        self._identify_substructure()


    @classmethod
    def from_dicts(cls, rfDict, machDict, tLeft = None, tRight = None,
                   potential_resolution = 1000):
        
        if tRight is None:
            tRight = 1.05*machDict['t_rev']
        if tLeft is None:
            tLeft = 0 - 0.05*tRight
        
        timeBounds = (tLeft, tRight)
        
        vTime, vWave = pot.rf_voltage_generation(potential_resolution,
                                                 machDict['t_rev'],
                                                 rfDict['voltage'],
                                                 rfDict['harmonic'],
                                                 rfDict['phi_rf_d'],
                                                 time_bounds = timeBounds)
        
        time, well, _ = pot.rf_potential_generation_cubic(vTime, vWave, 
                                                          machDict['eta_0'], 
                                                          machDict['charge'],
                                                          machDict['t_rev'], 
                                                          machDict['delta_E'])
        
        well -= np.min(well)
        maxLocs, _, _, _, _ = pot.find_potential_wells_cubic(time, well,
                                 relative_max_val_precision_limit=1E-4)

        times, wells = pot.potential_well_cut_cubic(time, well, maxLocs)
        
        return cls(times, wells, machDict['beta'], machDict['energy'],
                   machDict['eta_0'])

    def _identify_substructure(self):
        
        contains = [[] for i in range(len(self.inner_times))]
        for times, c in zip(self.inner_times, contains):
            c += [i for i in range(len(self.inner_times)) if self.inner_times[i][0] >= times[0]\
                                                      and self.inner_times[i][-1] <= times[-1]\
                                                      and self.inner_times[i] is not times]
        
        exclude = [[] for i in range(len(self.inner_times))]
        for exc, cont in zip(exclude, contains):
            for c in cont:
                exc += contains[c]
        
        useCont = [[] for i in range(len(self.inner_times))]
        for i in range(len(self.inner_times)):
                useCont[i] += [c for c in contains[i] if c not in exclude[i]]
        
        bucketDict = {i: self.__class__(t, w, self.beta, self.energy,
                                        self.eta, isSub=True) for i, (t, w) in 
                          enumerate(zip(self.inner_times, self.inner_wells))}
        
        nextLayer = []
        for i in range(len(self.inner_times)):
            if not any(i in c for c in useCont):
                nextLayer.append(i)
        
        if len(nextLayer) > 0:
            self.hasSubs = True
        else:
            self.hasSubs = False
        
        self.sub_buckets = [bucketDict[i] for i in nextLayer]
        
        for i, u in enumerate(useCont):
            bucketDict[i].sub_buckets = [bucketDict[c] for c in u]
            if len(u) > 0:
                bucketDict[i].hasSubs = True
            else:
                bucketDict[i].hasSubs = False
        
        self._calc_inner_max()
        self._calc_inner_start()
        self._calc_inner_stop()
        self._calc_minimum()


    @deco.recursive_function
    def _calc_inner_max(self):
        if self.hasSubs:
            self.inner_max = np.max([np.max(b.well) for b in self.sub_buckets])
        else:
            self.inner_max = np.NaN
    
    @deco.recursive_function
    def _calc_inner_start(self):
        if self.hasSubs:
            self.inner_start = np.min([np.min(b.time) for b in self.sub_buckets])
        else:
            self.inner_start = np.NaN
    
    @deco.recursive_function
    def _calc_inner_stop(self):
        if self.hasSubs:
            self.inner_stop = np.max([np.max(b.time) for b in self.sub_buckets])
        else:
            self.inner_stop = np.NaN
    
    @deco.recursive_function
    def _calc_minimum(self):
        self.minimum = np.min(calc.minmax_location_cubic(self.time, 
                                                         self.well)[1][0])
    
    
    def inner_buckets(self):
        
        self.inner_separatrices = []
        for t, w in zip(self.inner_times, self.inner_wells):
            hamil = pot.potential_to_hamiltonian(t, w,
                                             self.beta, self.energy, 
                                             self.eta)

            upper_energy_bound = np.sqrt(hamil)
        
            sepTime = t.tolist() + t[::-1].tolist()
            sepEnergy = upper_energy_bound.tolist() \
                    + (-upper_energy_bound[::-1]).tolist()
        
            self.inner_separatrices.append(np.array([sepTime, sepEnergy]))


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

     #TODO: Test effect with multiple minima of checking if synchronous
     # particle is within sub_bucket before calculating

    @deco.recursive_function
    def _frequency_distribution(self, trapzThresh = 0):
        
        t, f, h, a, _, _ = pot.synchrotron_frequency_hybrid(self.time,
                                                            self.well,
                                                            self.eta, 
                                                            self.beta, 
                                                            self.energy,
                                       min_potential_well = self.minimum,
                                 inner_max_potential_well = self.inner_max,
                                              trapzThresh = trapzThresh)
        
        self.fsTime = t
        self.fsFreq = f
        self.fsHamil = h
        self.fsArea = a


    @deco.recursive_attribute
    def fsTimes(self):
        return self.fsTime


    @deco.recursive_attribute
    def fsFreqs(self):
        return self.fsFreq


    @deco.recursive_attribute
    def fsHamils(self):
        return self.fsHamil


    @deco.recursive_attribute
    def fsAreas(self):
        return self.fsArea


    def frequency_distribution(self, recalculate = False, old = False, 
                         trapzThresh = 1):
        
        if recalculate or not hasattr(self, 'sortedTimes'):
            self._calc_inner_max()
            self._frequency_distribution(trapzThresh)
            
            allTimes = []
            allFreqs = []
            for o in zip(self.fsTimes, self.fsFreqs):
                allTimes += o[0].tolist()
                allFreqs += o[1].tolist()
            args = np.argsort(allTimes)
            
            self.sortedTimes = np.array(allTimes)[args]
            self.sortedFreqs = np.array(allFreqs)[args]
        else:
            return


    @deco.recursive_function
    def contains_time(self, time):
        return (time>self.time[0] and time<self.time[-1])

    @deco.recursive_function
    def contains_potential(self, potential):
        return (potential < np.max(self.well) 
                and potential > np.min(self.well))


    ################################################
    ####Functions for calculating bunch outlines####
    ################################################
    
    
    def _interp_time_from_potential(self, potential, nPts = 0):
        
        if potential > np.max(self.well):
            raise excpt.InputError("Target potential above maximum potential")
        
        if potential < 0:
            raise excpt.InputError("Target potential must be positive")
        
        pts = np.where(self.well <= potential)[0]
        leftPt = pts[0]
        rightPt = pts[-1]

        if leftPt < 2:
            leftPt -= leftPt-2
        if rightPt > len(self.well)-3:
            rightPt += len(self.well) - rightPt - 3

        lTime = np.interp(potential, self.well[leftPt-2:leftPt+2][::-1], 
                          self.time[leftPt-2:leftPt+2][::-1])
        rTime = np.interp(potential, self.well[rightPt-2:rightPt+3],
                          self.time[rightPt-2:rightPt+3])

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

        potential = target_height**2*np.abs(self.eta)\
                        /(2*self.beta**2*self.energy)
        
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
    
    
    def outline_from_emittance(self, target_emittance, nPts = 1000,
                               over_fill = False):

        self.smooth_well()

        if target_emittance > self.area:
            if not over_fill:
                raise excpt.BunchSizeError("target_emittance exceeds "
                                           + "bucket area")
            else:
                target_emittance = self.area
        
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

        try:        
            interpTime = self._interp_time_from_potential(result['x'][0], nPts)
        except excpt.InputError:
            interpTime = self.time.copy()
            interpWell = self.well.copy()
        else:
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


    def outline_from_coordinate(self, dt = None, dE = None):
        
        dE_array = np.linspace(np.min(self.separatrix[1]), 
                               np.max(self.separatrix[1]),
                               len(self.time))
        
        t_grid, dE_grid = np.meshgrid(self.time, dE_array)
        
        H_grid = (np.abs(self.eta)*dE_grid**2/(2*self.beta**2*self.energy)
                  + np.repeat(np.array([self.well]), len(dE_array), axis=0))
        
        interpFunc = spInterp.interp2d(t_grid[0], dE_grid[:,0], H_grid)
        hamVal = interpFunc(dt, dE)
        contour = np.sqrt((hamVal - self.well) * 2*self.beta**2 
                          * self.energy/np.abs(self.eta))
        
        outlineTime = self.time[np.isnan(contour) != True]
        outlineEnergy = contour[np.isnan(contour) != True]
        
        outlineTime = outlineTime.tolist() + outlineTime[::-1].tolist()
        outlineEnergy = outlineEnergy.tolist() \
                        + (-outlineEnergy[::-1]).tolist()
        
        return np.array([outlineTime, outlineEnergy])



    ##################################################
    ####Functions for calculating bunch parameters####
    ##################################################

    def _set_bunch(self, bunch_length = None, bunch_emittance = None,
                           bunch_height = None, over_fill = False):
        
        allowed = ('bunch_length', 'bunch_emittance', 'bunch_height')
        assrt.single_not_none(bunch_length, bunch_emittance, bunch_height,
                              msg = 'Exactly 1 of ' + str(allowed) \
                              + ' should be given', 
                              exception = excpt.InputError)
        
        if bunch_length is not None:
            if bunch_length == 0:
                outline = [[0, 0], [0,0]]
            else:
                outline = self.outline_from_length(bunch_length)
        elif bunch_emittance is not None:
            if bunch_emittance == 0:
                outline = [[0, 0], [0,0]]
            else:
                outline = self.outline_from_emittance(bunch_emittance, 
                                                      over_fill = over_fill)
        elif bunch_height is not None:
            if bunch_height == 0:
                outline = [[0, 0], [0,0]]
            else:
                outline = self.outline_from_dE(bunch_height)
        
        self._bunch_length = np.max(outline[0]) - np.min(outline[0])
        self._bunch_height = np.max(outline[1])
        self._bunch_emittance = np.trapz(outline[1], outline[0])


    @property
    def bunch_length(self):
        return self._bunch_length
    
    @property
    def bunch_height(self):
        return self._bunch_height
    
    @property
    def bunch_emittance(self):
        return self._bunch_emittance
    
    
    @bunch_length.setter
    def bunch_length(self, value):
        self._set_bunch(bunch_length = value)
    
    @bunch_height.setter
    def bunch_height(self, value):
        self._set_bunch(bunch_height = value)
    
    @bunch_emittance.setter
    def bunch_emittance(self, value):
        self._set_bunch(bunch_emittance = value)
        
        
    ###################################################
    ####Functions for generation bunches parameters####
    ###################################################
        
    
    def make_profiles(self, dist_type, length = None, emittance = None, 
                      dE = None, use_action = False, recalculate = False,
                      over_fill = False):
        
        if not recalculate and hasattr(self, 'time_profile'):
            return
        
        if not all(par is None for par in (length, emittance, dE)):
            self._set_bunch(length, emittance, dE, over_fill)
        
        self.dE_array = np.linspace(np.min(self.separatrix[1]), 
                                    np.max(self.separatrix[1]), len(self.time))
        
        self.compute_action()
        
        if use_action:
            size = self.bunch_emittance / (2*np.pi)
        else:
            size = np.interp(self.bunch_emittance / (2*np.pi), 
                             self.J_array[self.J_array.argsort()], 
                             self.well[self.well.argsort()])
        
        profiles = matchDist.matched_profile(dist_type, size, self.time, 
                                             self.well, self.dE_array, 
                                             self.beta, self.energy, self.eta)

        self.time_profile, self.energy_profile = profiles

    def compute_action(self):
    
        J_array = np.zeros(len(self.time))
        for i in range(len(self.time)):
            useWell = self.well[self.well < self.well[i]]
            useTime = self.time[self.well < self.well[i]]
            contour = np.sqrt(np.abs((self.well[i] - useWell)*2
                              *self.beta**2*self.energy/self.eta))
            J_array[i] = np.trapz(contour, useTime)/np.pi
    
        self.J_array = J_array



