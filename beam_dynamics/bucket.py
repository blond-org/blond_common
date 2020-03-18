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
from ..rf_functions import potential as pot
from ..maths import interpolation as interp
from ..devtools import exceptions as excpt
from ..devtools import assertions as assrt
from ..interfaces.beam import matched_distribution as matchDist
from ..maths import calculus as calc

class Bucket:
    
    def __init__(self, time, well, beta, energy, eta):
        
        try:
            assrt.equal_array_lengths(time, well, 
                              msg = "time and well must have the same length",
                              exception = excpt.InputError)
        except TypeError:
            raise excpt.InputError("time and well must both be iterable")
        
        orderedTime, orderedWell = pot.sort_potential_wells(time, well)
        
        self.time_loaded = np.array(orderedTime[0], dtype=float)
        self.well_loaded = np.array(orderedWell[0], dtype=float)
        
        self.beta = beta
        self.energy = energy
        self.eta = eta
        
        self.time = self.time_loaded.copy()
        self.well = self.well_loaded.copy()
        
        self.calc_separatrix()
        self.basic_parameters()
        
        self.inner_times = orderedTime[1:]
        self.inner_wells = orderedWell[1:]
    
    
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
        
    
    def frequency_spread(self):
        
        locs, vals = calc.minmax_location_cubic(self.time, self.well)

        wellBits = [self.well]
        timeBits = [self.time]
        for l, v in zip(locs[1], vals[1]):
            wellBits.append(self.well[(self.time < l)*(self.well<v)])
            timeBits.append(self.time[(self.time < l)*(self.well<v)])
            wellBits.append(self.well[(self.time > l)*(self.well<v)])
            timeBits.append(self.time[(self.time > l)*(self.well<v)])
        
        indices = []
        for i, (t1, w1) in enumerate(zip(timeBits, wellBits)):
            subIndex = []
            for j, (t2, w2) in enumerate(zip(timeBits, wellBits)):
                if i == j:
                    continue
                if t2[0] >= t1[0] and t2[-1] <= t1[-1]:
                    subIndex += np.where((t1 >= t2[0])*(t1 <= t2[-1]))[0].tolist()
            subIndex = list(set(subIndex))
            indices.append(subIndex)
                    
        mainIndices =[]
        for j, (i, t, w) in enumerate(zip(indices, timeBits, wellBits)):
            t = np.delete(t, i)
            mainIndices.append(np.intersect1d(self.time, t, True, True)[1])
        
        # for j, i in enumerate(mainIndices):
        #     plt.plot(self.time[i], self.well[i] + j*50)
        # plt.show()
        self.smooth_well()
        hList, aList = [], []
        for indices in mainIndices:
            if len(indices) == 1:
                hList.append([])
                aList.append([])
                continue
            left = True
            hTemp = []
            aTemp = []
            for i in indices:
                useTime = self.time[self.well<self.well[i]]
                useWell = self.well[self.well<self.well[i]]
                try:
                    fine_time_array = np.linspace(useTime[0], useTime[-1],
                                                  5000)
                except:
                    continue
                fine_potential_well = self._well_smooth_func(fine_time_array)
                # if left:
                #     try:
                #         _, _, h, a, _, _ = pot.trajectory_area_cubic(self.time[i:], 
                #                                                      self.well[i:], 
                #                                                      self.eta, 
                #                                                      self.beta,
                #                                                      self.energy)
                #     except (ValueError, TypeError) as e:
                #         if isinstance(e, TypeError):
                #             print(indices)
                #             raise
                #         left = False
                #         _, _, h, a, _, _ = pot.trajectory_area_cubic(self.time[:i], 
                #                                                      self.well[:i], 
                #                                                      self.eta, 
                #                                                      self.beta,
                #                                                      self.energy)
                # elif not left:
                #         _, _, h, a, _, _ = pot.trajectory_area_cubic(self.time[:i], 
                #                                                      self.well[:i], 
                #                                                      self.eta, 
                #                                                      self.beta,
                #                                                      self.energy)
                try:
                    _, _, h, a, _, _ = pot.trajectory_area_cubic(fine_time_array, 
                                                                 fine_potential_well, 
                                                                 self.eta, 
                                                                 self.beta,
                                                                 self.energy)
                except:
                    continue
                    # print(useWell)
                hTemp.append(h)
                aTemp.append(a)
            
            hList.append(hTemp)
            aList.append(aTemp)
            
        for h, a in zip(hList, aList):
            h = np.array(h)
            a = np.array(a)
            if len(h) <= 1:
                continue
            # print(h, a)
            sorted_area = np.argsort(a[::2])
            # sync_freq = calc.deriv_cubic(
            #     a[::2][sorted_area],
            #     h[::2][sorted_area])[1]
            sync_freq = np.gradient(h[::2][sorted_area])/np.gradient(a[::2][sorted_area])
            plt.plot(sync_freq)
            plt.show()
        
        # maxLocs = locs[1]
        # maxVals = vals[1]
        # minV = np.min(vals[0])
        # maxMax = np.max(maxVals)
        
        # print(maxLocs, maxVals)
        
        # sys.exit()
        
        # try:
        #     fsData = pot.synchrotron_frequency_cubic(self.time, self.well,
        #                                              self.eta, self.beta, 
        #                                              self.energy, minV,
        #                                              maxMax)
        # except TypeError:
        #     fsData = pot.synchrotron_frequency_cubic(self.time, self.well,
        #                                              self.eta, self.beta, 
        #                                              self.energy, minV,
        #                                      np.max(maxVals[maxVals < maxMax]))
        
        # fsData = [fsData]
        
        # for t, w in zip(self.inner_times, self.inner_wells):
        #     # print(t[1], t[-1])
        #     plt.plot(t, w)
        #     for l in locs[1]:
        #         if l > t[1] and l < t[-1]:
        #             print(f'{l} is in {t[1], t[-1]}')
        #             plt.axvline(l)
        #     plt.show()
        


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
    
    
    def outline_from_emittance(self, target_emittance, nPts = 1000):

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


    ##################################################
    ####Functions for calculating bunch parameters####
    ##################################################

    def _set_bunch(self, bunch_length = None, bunch_emittance = None,
                           bunch_height = None):
        
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
                outline = self.outline_from_emittance(bunch_emittance)
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
                      dE = None, use_action = False):
        
        if not all(par is None for par in (length, emittance, dE)):
            self._set_bunch(length, emittance, dE)
        
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

