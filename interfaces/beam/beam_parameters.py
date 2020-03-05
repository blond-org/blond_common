# General_imports
import numpy as np
import matplotlib.pyplot as plt
import sys

# BLonD_Common imports
from ...rf_functions import potential as pot
from ...maths import calculus as calc
from ...datatypes import datatypes as dt
from ...beam_dynamics import bucket as buck


class Beam_Parameters:
    
    def __init__(self, ring, rf, use_samples = None, init_coord = None, 
                 harmonic_divide = 1, potential_resolution = 1000):
        
        self.ring = ring
        self.rf = rf
        
        if use_samples is None:
            use_samples = list(range(len(self.ring.use_turns)))
        
        self.use_samples = use_samples       
        self.n_samples = len(self.use_samples)

        self.init_coord = init_coord
        
        self.harmonic_divide = harmonic_divide
        
        self.potential_resolution = potential_resolution
        
        self.volt_wave_array = np.zeros([self.n_samples, 
                                         self.potential_resolution])
        self.time_window_array = np.zeros([self.n_samples, 
                                           self.potential_resolution])
        self.potential_well_array = np.zeros([self.n_samples, 
                                              self.potential_resolution])
    
        self.calc_potential_wells()
        self.track_synchronous()
        self.buckets = {}
        self.calc_buckets()
        self.bucket_parameters()
        
    
    def calc_potential_wells(self, sample = None):

        '''
        Calculate potential well at all or specified sample

        Parameters
        ----------
        sample : None, int
            if not None:
               well calculated only for specified sample
        '''

        if sample is None:
            for s in range(self.n_samples):
                time, well, vWave = self.sample_potential_well(s)
                self.volt_wave_array[s] = vWave
                self.time_window_array[s] = time
                self.potential_well_array[s] = well

        else:
            time, well, vWave = self.sample_potential_well(sample)
            self.volt_wave_array[sample] = vWave
            self.time_window_array[sample] = time
            self.potential_well_array[sample] = well
    
    
    
    def track_synchronous(self, start_sample = 0):
        
        #If no start points specified create a single particle at the lowest 
        #and leftest minimum
        if self.init_coord is None:
            point = np.where(self.potential_well_array[0]
                             == np.min(self.potential_well_array[0]))[0][0]
            self.init_coord = [self.time_window_array[0][point]]

        if start_sample == 0:
            self.particle_tracks = np.zeros([len(self.init_coord),
                                            self.n_samples])
            if len(self.init_coord) == 1:
                bunching = 'single_bunch'
            else:
                bunching = 'multi_bunch'
            self.particle_tracks \
                    = dt.synchronous_phase.zeros([len(self.init_coord),
                                                  self.n_samples],
                                                {'timebase': 'by_turn',
                                                 'bunching': bunching,
                                                 'units': 's'})

        self.n_particles = len(self.init_coord)

        #Loop over all particles positioning them in closest minimum to 
        #declared start point
        for p in range(self.n_particles):

            startPoint = np.where(self.time_window_array[0]
                                  <= self.init_coord[p])[0][-1]

            self.particle_tracks[p][0] = self.time_window_array[0][startPoint]

            locs, values \
                    = calc.minmax_location_cubic(self.time_window_array[0],
                                                 self.potential_well_array[0])
            locs = locs[0]
            offsets = np.abs(self.particle_tracks[p][0] - locs)
            newLoc = np.where(offsets == np.min(offsets))[0][0]
                
            self.particle_tracks[p][0] = locs[newLoc]

        #Loop over all particles and all but first sample, at each sample new 
        #particle location is nearest minimum in potential well
        for p in range(self.n_particles):
            for t in range(start_sample+1, self.n_samples):
               locs, values \
                   = calc.minmax_location_cubic(self.time_window_array[t], 
                                                self.potential_well_array[t])
               locs = locs[0]
               offsets = np.abs(self.particle_tracks[p][t-1] - locs)
               newLoc = np.where(offsets == np.min(offsets))[0][0]
               self.particle_tracks[p][t] = locs[newLoc]


    
    def calc_buckets(self):

        '''
        Create and store all buckets
        '''


        for s in range(self.n_samples):
            bucket_list = self.create_sample_buckets(s)
            for p in range(self.n_particles):
                self.buckets[(self.ring.use_turns[s], p)] = bucket_list[p]
    

    def sample_potential_well(self, sample, volts = None):

        '''
        Calculate potential well at given sample with existing or passed voltage

        Parameters
        ----------
        sample : int
            sample number to use
        volts : None, np.array or list
            if None:
                Use voltage from self.rfprogram to define potential well
            else:
                Use passed voltage to define potential well
        Returns
        -------
        time : array
            time axis of potential well
        well : array
            potential well amplitude
        vWave : array
            full voltage used to calculate potential well
            returned if volts is None
        '''        
        
        
        ringPars = self.ring.parameters_at_sample(sample)
        rfPars = self.rf.parameters_at_sample(sample)
        
        timeBounds = (0, ringPars['t_rev']/self.harmonic_divide)
        
        if volts is None:
             vTime, vWave = pot.rf_voltage_generation(self.potential_resolution,
                                                      ringPars['t_rev'],
                                                      rfPars['voltage'],
                                                      rfPars['harmonic'],
                                                      rfPars['phi_rf_d'],
                                                      time_bounds = timeBounds)
        else:
            vWave = volts
            vTime = np.linspace(timeBounds[0], timeBounds[1], 
                                self.potential_resolution)

        time, well, _ = pot.rf_potential_generation_cubic(vTime, vWave, 
                                                          ringPars['eta_0'], 
                                                          ringPars['charge'],
                                                          ringPars['t_rev'], 
                                                          ringPars['delta_E'])

        if volts is None:
            return time, well, vWave
        else:
            return time, well
    
    

    def cut_well(self, sample, particle):
        
        '''
        Calculate potential well and inner wells at a specified sample and 
        particle
        
        Parameters
        ----------
        sample : int
            sample number to use
        particle : int
            particle number to use

        Returns
        -------
        time : np.array
            Time component of calculated well
        well : np.array
            Potential well
        '''

        inTime = self.time_window_array[sample]
        inWell = self.potential_well_array[sample]

        maxLocs, _, _, _, _ = pot.find_potential_wells_cubic(inTime, inWell)
        times, wells = pot.potential_well_cut_cubic(inTime, inWell, maxLocs)

        particleLoc = self.particle_tracks[particle][sample]
        
        #Check which subwells contain current particle
        relevant = [i for i, t in enumerate(times) if particleLoc>t[0] and 
                                                      particleLoc<t[-1]]

        subTime = [times[r] for r in relevant]
        subWell = [wells[r] for r in relevant]
        
        biggest = pot.sort_potential_wells(subTime, subWell, by='size')[0][0]
      
        #check which subwells are within the bounds of the largest well 
        #containing the current particle
        relevant = [i for i, t in enumerate(times) if t[0] >= biggest[0] and
                                                      t[-1] <= biggest[-1]]
        
        times = [times[r] for r in relevant]
        wells = [wells[r] for r in relevant]

        mins = [np.min(w) for w in wells]
        wells -= np.min(mins)
        
        times, wells = pot.sort_potential_wells(times, wells, by='t_start')

        return times, wells
    

    def create_particle_bucket(self, sample, particle):

        '''
        Create new bucket at a specified sample for a specified particle.  If time and well are None
        the stored parameters will be used.  If time and well are specified particle becomes the required
        synchronous phase point rather than a counter to the particle_tracks array.

        Parameters
        ----------
        sample : int
            sample number to use
        particle : int
            particle number to use
        time : None, list
            if list:
                used to find bucket
        well : None, list
            if list:
                
        '''

        pars = self.ring.parameters_at_sample(sample)

        time, well = self.cut_well(sample, particle)

        return buck.Bucket(time, well, pars['beta'], pars['energy'],
                           pars['eta_0'])
    
    
    def bucket_parameters(self, update_bunch_parameters = False):

        '''
        Store bucket heights, areas, lengths and centers through the ramp
        '''


        n_pars = self.particle_tracks.shape[0]

        if len(self.init_coord) == 1:
            bunching = 'single_bunch'
        else:
            bunching = 'multi_bunch'

        data_type = {'timebase': 'by_turn', 'bunching': bunching}

        self.heights = dt.height.zeros([n_pars, self.n_samples], 
                                       {**data_type, 'units': 'eV', 
                                        'height_type': 'half_height'})
        self.bunch_heights = dt.height.zeros([n_pars, self.n_samples], 
                                             {**data_type, 'units': 'eV', 
                                              'height_type': 'half_height'})
        self.areas = dt.acceptance.zeros([n_pars, self.n_samples], 
                                         {**data_type, 'units': 'eV'})
        self.bunch_emittances = dt.emittance.zeros([n_pars, self.n_samples], 
                                               {**data_type, 
                                            'emittance_type': 'matched_area',
                                            'units': 'eVs'})
        self.lengths = dt.length.zeros([n_pars, self.n_samples], 
                                               {**data_type, 
                                                'length_type': 'full',
                                                'units': 's'})
        self.bunch_lengths = dt.length.zeros([n_pars, self.n_samples], 
                                               {**data_type, 
                                                'length_type': 'full',
                                                'units': 's'})

        #TODO: add bunch parameters
        for n in range(n_pars):
            buckets = self.buckets_by_particle(n)
            for b in range(len(buckets)):
#                if update_bunch_parameters:
#                    buckets[b].bunch_emittance = buckets[b].bunch_emittance
                    
#                self.bunch_heights[n, b] = buckets[b].bunch_dE
                self.heights[n, b] = buckets[b].half_height
                self.areas[n, b] = buckets[b].area
#                self.bunch_emittances[n, b] = buckets[b].bunch_emittance
                self.lengths[n, b] = buckets[b].length
#                self.bunch_lengths[n, b] = buckets[b].bunch_length

        
        
        
    def create_sample_buckets(self, sample):

        '''
        Create new bucket at a specified sample for all particles.

        Parameters
        ----------
        sample : int
            sample number to use
        '''

        bucket_list = []
        for p in range(self.n_particles):
            bucket_list.append(self.create_particle_bucket(sample, p))

        return bucket_list
    
    
    
    def buckets_by_particle(self, particle):

        '''
        Return list of buckets for a specified particle

        Parameters
        ----------
        particle : int
            particle to return buckets for

        Returns
        -------
        bucket_list : list
            list of buckets through program for specified particle
        '''

        return [self.buckets[key] for key in self.buckets.keys() if 
                                                        key[1] == particle] 