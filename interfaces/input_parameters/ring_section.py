# General imports
import numbers

# BLonD_Common imports
from ...datatypes import datatypes as dTypes


class ring_section:
    
    def __init__(self, section_length, alpha, momentum = None,
                 kinetic_energy = None, total_energy = None, B_field = None):

        self.section_length = section_length

        self.alpha = dTypes.momentum_compaction(alpha)
