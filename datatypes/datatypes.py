#General imports
import numpy as np

#Common imports
import data_common.utilities.Exceptions as exceptions

class _function(np.ndarray):
    
    def __new__(cls, input_array, data_type=None):
        
        if data_type is None:
            raise exceptions.InputError("data_type must be specified")
        
        obj = np.asarray(input_array).view(cls)
        
        obj.data_type = data_type
        
        return obj
    
    def __array_finalize__(self, obj):
        
        if obj is None:
            return
        
        self.data_type = getattr(obj, 'data_type', None)



class momentum_program(_function):
    
    def __new__(cls, *args, time = None, n_turns = None):
        
        data_points = []
        data_types = []

        if time is not None and n_turns is not None:
            raise exceptions.InputError("time and n_turns cannot both be specified")

        for arg in args:
            data_point, data_type = cls._check_dims(cls, arg, time, n_turns)
            data_points.append(data_point)
            data_types.append(data_type)
        
        if not all(datType == data_types[0] for datType in data_types):
            raise exceptions.DataDefinitionError("Input momentum programs " \
                                                 + "follow different conventions")
        
        #If single section is passed data_type is unchanged,
        #otherwise _and_section is appended to indicate multiple ring sections
        if len(data_types) == 1:
            return super().__new__(cls, data_points[0], data_types[0])
        else:
            return super().__new__(cls, data_points, data_types[0]+'_and_section')
            
    
    def _check_dims(self, data, time = None, n_turns = None):
        
        #Check and handle single valued data
        #if not single valued coerce to numpy array and continue
        try:
            iter(data)
            data = np.array(data)
        except TypeError:
            if n_turns is None:
                return data, 'momentum_single_valued'
            else:
                return [data]*n_turns, 'momentum_by_turn'

        #If n_turns specified and data is not single valued it should be of len(n_turns)
        if n_turns is not None:
            if len(data) == n_turns:
                return data, 'momentum_by_turn'
            else:
                raise exceptions.InputError("Input length does not match n_turns")
        
        elif time is not None:
            if data.shape[0] == 2:
                raise exceptions.InputError("Data has been passed with " \
                                            + "[time, value] format and time " \
                                            + "defined, only 1 should be given")
            else:
                #If time is passed don't return, use test below avoids duplication
                if len(data) == len(time):
                    data = np.array([time, data])
                else:
                    raise exceptions.InputError("time and data are of unequal" \
                                                + " length")

        #If data has shape (2, n) data[0] is taken as time, which must be increasing
        if data.shape[0] == 2:
            if any(np.diff(data[0]) <= 0):
                raise exceptions.InputError("Time component of input is not " \
                                            + "increasing at all points")
            else:
                return data, 'momentum_by_time'
        #if data has shape (n,) data[0] is taken as momentum by turn
        elif len(data.shape) == 1:
            return data, 'momentum_by_turn'


        raise exceptions.InputError("Input data not understood")


if __name__ == "__main__":
    
    test = momentum_program(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]))
    
    print(test)
    print(type(test))
    
    
    test = momentum_program([1, 2, 3], [4, 5, 6])
    
    print(test)
    print(type(test))    
    