# General_imports
import numpy as np
import matplotlib.pyplot as plt

# BLonD_Common imports
from ...rf_functions import potential as pot
from ...maths import calculus as calc


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

        self.n_particles = len(self.init_coord)

        #Loop over all particles positioning them in closest minimum to 
        #declared start point
        for p in range(self.n_particles):

            startPoint = np.where(self.time_window_array[0]
                                  <= self.init_coord[p])[0][-1]

            self.particle_tracks[p][0] = self.time_window_array[0][startPoint]

            locs, values = calc.minmax_location_cubic(self.time_window_array[0],
                                                      self.potential_well_array[0])
            locs = locs[0]
            offsets = np.abs(self.particle_tracks[p][0] - locs)
            newLoc = np.where(offsets == np.min(offsets))[0][0]
                
            self.particle_tracks[p][0] = locs[newLoc]

        #Loop over all particles and all but first sample, at each sample new 
        #particle location is nearest minimum in potential well
        for p in range(self.n_particles):
            for t in range(start_sample+1, self.n_samples):
               locs, values = calc.minmax_location_cubic(self.time_window_array[t], 
                                                         self.potential_well_array[t])
               locs = locs[0]
               offsets = np.abs(self.particle_tracks[p][t-1] - locs)
               newLoc = np.where(offsets == np.min(offsets))[0][0]
               self.particle_tracks[p][t] = locs[newLoc]