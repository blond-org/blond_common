# General_imports
import numpy as np
import matplotlib.pyplot as plt

# BLonD_Common imports


class Beam_Parameters:
    
    def __init__(self, ring, rf, use_samples = None):
        
        self.ring = ring
        self.rf = rf
        
        if use_samples is None:
            use_samples = list(range(len(self.ring.use_turns)))
        
        self.use_samples = use_samples
        
    
    def track_synchronous(self, init_coord = None):
        
        if init_coord is None:
            init_coord