
# BLonD_Common imports
from ...datatypes import datatypes as dTypes

class RF_System:
    
    def __init__(self, voltage, phase, harmonic):
        
        if not isinstance(voltage, dTypes.voltage_program):
            voltage = dTypes.voltage_program(voltage, harmonics = [harmonic])
        self.voltage = voltage
        
        if not isinstance(phase, dTypes.phase_program):
            phase = dTypes.phase_program(phase, harmonics = [harmonic])
        self.phase = phase
        
        self.harmonic = harmonic
    
    def sample(self, use_time = None, use_turns = None):
        
        voltage = self.voltage.reshape([self.harmonic], use_time, use_turns)
        phase = self.phase.reshape([self.harmonic], use_time, use_turns)
        
        return voltage, phase, self.harmonic