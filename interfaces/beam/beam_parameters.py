# General_imports
import numpy as np
import matplotlib.pyplot as plt

# BLonD_Common imports


class Beam_Parameters:
    
    def __init__(self, ring, rf, use_samples = None, init_coord = None, 
                 harmonic_divide = 1):
        
        self.ring = ring
        self.rf = rf
        
        if use_samples is None:
            use_samples = list(range(len(self.ring.use_turns)))
        
        self.use_samples = use_samples
        self.init_coord = init_coord
        
    
    def track_synchronous(self, start_sample = 0):
        
        #If no start points specified create a single particle at the lowest minimum
        if self.particle_starts is None:
            point = np.where(self.potential_well_array[0]
                             == np.min(self.potential_well_array[0]))[0][0]
            self.particle_starts = [self.time_window_array[0][point]]

        if start_sample == 0:
            self.particle_tracks = np.zeros([len(self.particle_starts),
                                            self.n_samples])

        self.n_particles = len(self.particle_starts)

        #Loop over all particles positioning them in closest minimum to declared start point
        for p in range(self.n_particles):

            startPoint = np.where(self.time_window_array[0]
                                  <= self.particle_starts[p])[0][-1]

            self.particle_tracks[p][0] = self.time_window_array[0][startPoint]

            locs, values = calc.minmax_location(self.time_window_array[0],
                                               self.potential_well_array[0])
            locs = locs[0]
            offsets = np.abs(self.particle_tracks[p][0] - locs)
            newLoc = np.where(offsets == np.min(offsets))[0][0]
                
            self.particle_tracks[p][0] = locs[newLoc]

        #Loop over all particles and all but first sample, at each sample new particle location is nearest minimum in potential well
        for p in range(self.n_particles):
            for t in range(start_sample+1, self.n_samples):
               locs, values = calc.minmax_location(self.time_window_array[t], 
                                                   self.potential_well_array[t])
               locs = locs[0]
               offsets = np.abs(self.particle_tracks[p][t-1] - locs)
               newLoc = np.where(offsets == np.min(offsets))[0][0]
               self.particle_tracks[p][t] = locs[newLoc]