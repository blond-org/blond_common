#General imports
import numpy as np

class _function(np.ndarray):
    
    def __new__(cls, input_array, data_type=None):
        
        if data_type is None:
            raise RuntimeError("test")